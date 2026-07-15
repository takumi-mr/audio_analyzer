import numpy as np
import librosa
import scipy.ndimage
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any, List
from model.AudioSignal import AudioSignal

class ChorusDetectionBeatSSMStrategy(IAnalysisStrategy):
    """
    ビート同期SSMと対角パス強調(Path Enhancement)を用いた最新のサビ検出戦略 (改良版)。
    「CENSクロマ＋MFCCによる楽曲の繰り返し構造」と、「ボーカル存在度＋リズム存在度＋高音域の明るさ(Spectral Centroid)」
    を融合させてサビ区間を頑健に特定します。

    主な改良点:
    1. 特徴量をchroma_cqtからchroma_censへ変更（音量変化にロバストにし、和音変化にフォーカス）
    2. 音色の高音域の明るさを示すスペクトル重心(Spectral Centroid)を存在感スコアに追加
    3. インスト曲（ボーカルなし）を検知した場合にボーカルウェイトを自動的に排除してリズム・音色を重視する適応型Salience
    4. 対角パス強調窓を16拍(約4小節)に調整し、境界部の減衰影響を低減
    5. スコア統合を掛け算から「重み付き足し算」へ変更し、厳しい足切りを回避して検出数をSSM Structureと同等まで向上
    6. 閾値を0.15σに引き下げ、最小継続拍数を8拍に緩和し、隣接マージの許容ギャップを6秒に拡大
    """
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # 1. 必要なシグナルを取得
        y_vocal = signals["target_vocal"].data if "target_vocal" in signals else signals["target"].data
        y_drums = signals["target_drums"].data if "target_drums" in signals else np.zeros_like(y_vocal)
        y_bass = signals["target_bass"].data if "target_bass" in signals else np.zeros_like(y_vocal)
        y_other = signals["target_other"].data if "target_other" in signals else np.zeros_like(y_vocal)
        
        if "target" in signals:
            y_full = signals["target"].data
            sr = signals["target"].sample_rate
        else:
            y_full = y_vocal + y_drums + y_bass + y_other
            sr = signals["target_vocal"].sample_rate

        print("[Strategy: Chorus] ビート同期SSMと対角パス強調を用いたサビ検出を開始します...")

        try:
            # 2. ビートトラッキング
            y_rhythm = y_drums + y_bass if "target_drums" in signals else y_full
            onset_env = librosa.onset.onset_strength(y=y_rhythm, sr=sr)
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=100.0)
            
            if not isinstance(beat_frames, np.ndarray):
                beat_frames = np.array(beat_frames)
            beat_frames = librosa.util.fix_frames(beat_frames, x_min=0)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            n_beats = len(beat_frames)

            # 曲が短すぎる、または拍が少なすぎる場合の安全なフォールバック
            if n_beats < 16:
                raise ValueError("Too few beats for structural analysis.")

            # 3. 特徴量抽出とビート同期 (chroma_cens + mfcc)
            chroma = librosa.feature.chroma_cens(y=y_other + y_vocal, sr=sr)
            mfcc = librosa.feature.mfcc(y=y_full, sr=sr, n_mfcc=13)
            
            chroma_sync = librosa.util.sync(chroma, beat_frames.tolist(), aggregate=np.median)
            mfcc_sync = librosa.util.sync(mfcc, beat_frames.tolist(), aggregate=np.median)
            
            # 特徴量を結合して正規化
            X = np.vstack([librosa.util.normalize(chroma_sync, axis=0), 
                           librosa.util.normalize(mfcc_sync, axis=0)])

            # 4. 自己類似行列 (SSM) の構築
            R = librosa.segment.recurrence_matrix(X, mode='affinity', metric='cosine', sym=True, width=8)

            # 5. 対角パス強調 (Path Enhancement)
            # 16拍 (約4小節) に戻し、 Hann 窓による端の過度な減衰を防ぎます
            R_enh = librosa.segment.path_enhance(R, 16, window='hann')
            
            # 各ビートの繰り返しスコア
            rep_score = np.max(R_enh, axis=1)

            # 6. 存在感スコア (Salience) の計算
            # ボーカル音圧
            vocal_rms = librosa.feature.rms(y=y_vocal)[0]
            vocal_sync = librosa.util.sync(vocal_rms, beat_frames.tolist(), aggregate=np.max)[0]
            
            # リズム音圧
            rhythm_rms = librosa.feature.rms(y=y_rhythm)[0]
            rhythm_sync = librosa.util.sync(rhythm_rms, beat_frames.tolist(), aggregate=np.max)[0]

            # 明るさ (スペクトル重心)
            centroid = librosa.feature.spectral_centroid(y=y_full, sr=sr)[0]
            centroid_sync = librosa.util.sync(centroid, beat_frames.tolist(), aggregate=np.median)[0]

            # 0-1に正規化するヘルパー関数
            def normalize(arr):
                rng = arr.max() - arr.min()
                return (arr - arr.min()) / (rng if rng > 0 else 1.0)

            rep_score = normalize(rep_score)
            vocal_sync = normalize(vocal_sync)
            rhythm_sync = normalize(rhythm_sync)
            centroid_sync = normalize(centroid_sync)

            # 7. インスト曲 (無ボーカル曲) に応じた適応型 Salience
            full_rms = librosa.feature.rms(y=y_full)[0]
            vocal_ratio = np.mean(vocal_rms) / (np.mean(full_rms) + 1e-8)
            
            if vocal_ratio < 0.08:
                # インスト曲: リズム(0.60) + 明るさ/スペクトル重心(0.40)
                salience = 0.60 * rhythm_sync + 0.40 * centroid_sync
                print(f"[Strategy: Chorus] インスト曲と判定しました。 (Vocal ratio: {vocal_ratio:.3f})")
            else:
                # ボーカルあり曲: ボーカル(0.50) + リズム(0.30) + 明るさ(0.20)
                salience = 0.50 * vocal_sync + 0.30 * rhythm_sync + 0.20 * centroid_sync
            
            # 繰り返し度と存在感の「和」(掛け算による過度な足切りを回避し、検出数を向上)
            chorus_score = 0.5 * rep_score + 0.5 * salience

            # 短いノイズを消すために8拍（約2小節）のメディアンフィルタで平滑化
            chorus_score_smooth = scipy.ndimage.median_filter(chorus_score, size=8)

            # 8. サビ区間の抽出
            # 閾値係数を0.15σに設定 (SSM Structureと同等レベルまで緩和して検出数を確保)
            threshold = np.mean(chorus_score_smooth) + 0.15 * np.std(chorus_score_smooth)
            is_chorus = chorus_score_smooth > threshold

            sections = []
            in_sec = False
            start_b = 0
            
            # サビの最小継続拍数: 8拍 (約2小節。短いサビや境界削れによる消失を防ぐ)
            min_beats_for_chorus = 8

            for b in range(n_beats):
                if is_chorus[b] and not in_sec:
                    in_sec = True
                    start_b = b
                elif not is_chorus[b] and in_sec:
                    in_sec = False
                    if b - start_b >= min_beats_for_chorus:
                        sections.append({"start_sec": float(beat_times[start_b]), "end_sec": float(beat_times[b])})
                        
            if in_sec and n_beats - start_b >= min_beats_for_chorus:
                sections.append({"start_sec": float(beat_times[start_b]), "end_sec": float(beat_times[-1])})

            # 安全策: 何も検出されなかった場合は最もスコアが高い場所を強制出力
            if not sections:
                best_b = int(np.argmax(chorus_score_smooth))
                start_b = max(0, best_b - 16)
                end_b = min(n_beats - 1, best_b + 16)
                sections.append({"start_sec": float(beat_times[start_b]), "end_sec": float(beat_times[end_b])})

            # 9. 隣接する区間のマージ (マージギャップを6.0秒に拡大し、ブレイク等による分断を防止)
            merged_sections = []
            for sec in sections:
                if merged_sections and sec["start_sec"] - merged_sections[-1]["end_sec"] <= 6.0:
                    merged_sections[-1]["end_sec"] = sec["end_sec"]
                else:
                    merged_sections.append(sec)

            return {
                "status": "success",
                "chorus_sections_beat_ssm": [
                    {"start_sec": round(s["start_sec"], 2), "end_sec": round(s["end_sec"], 2)}
                    for s in merged_sections
                ],
                "chorus_confidence_beat_ssm": round(float(np.max(chorus_score_smooth)), 2),
                "chorus_method_beat_ssm": "beat_sync_path_enhanced"
            }

        except Exception as e:
            # エラー時はクラッシュさせず、空の検出結果として安全に返す
            print(f"[Warning] BeatSSM Chorus detection failed: {e}. Falling back to empty result.")
            return {
                "status": "success",
                "chorus_sections_beat_ssm": [],
                "chorus_confidence_beat_ssm": 0.0,
                "chorus_method_beat_ssm": "beat_sync_path_enhanced_fallback"
            }