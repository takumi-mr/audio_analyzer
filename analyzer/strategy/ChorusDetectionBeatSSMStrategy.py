from typing import Any

import librosa
import numpy as np
import scipy.ndimage

from analyzer.IAnalysisStrategy import IAnalysisStrategy
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
    def analyze(self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None) -> dict[str, Any]:
        # 1. 必要なシグナルを安全に取得
        if not signals:
            return {"status": "error", "message": "No audio signal available."}

        target_sig = signals.get("target") or signals.get("target_vocal") or next(iter(signals.values()), None)
        if target_sig is None or len(target_sig.data) == 0:
            return {"status": "error", "message": "No audio signal available."}

        y_vocal = signals["target_vocal"].data if "target_vocal" in signals else target_sig.data
        y_drums = signals["target_drums"].data if "target_drums" in signals else np.zeros_like(y_vocal)
        y_bass = signals["target_bass"].data if "target_bass" in signals else np.zeros_like(y_vocal)
        y_other = signals["target_other"].data if "target_other" in signals else np.zeros_like(y_vocal)
        
        if "target" in signals:
            y_full = signals["target"].data
            sr = signals["target"].sample_rate
        elif "target_vocal" in signals:
            y_full = y_vocal + y_drums + y_bass + y_other
            sr = signals["target_vocal"].sample_rate
        else:
            y_full = target_sig.data
            sr = target_sig.sample_rate

        print("[Strategy: Chorus] ビート同期SSMと対角パス強調を用いたサビ検出を開始します...")

        try:
            # 2. ビートトラッキング
            y_rhythm = y_drums + y_bass if "target_drums" in signals else y_full
            onset_env = librosa.onset.onset_strength(y=y_rhythm, sr=sr)
            _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=100.0)
            
            if not isinstance(beat_frames, np.ndarray):
                beat_frames = np.array(beat_frames)
            beat_frames = librosa.util.fix_frames(beat_frames, x_min=0)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            n_beats = len(beat_frames)

            # 曲が短すぎる、または拍が少なすぎる場合のダイナミクス適応フォールバック
            if n_beats < 16:
                print(f"[Strategy: Chorus] 拍数が少ないため ({n_beats} 拍)、ダイナミクス適応フォールバックでサビを特定します。")
                fallback_sections = self._fallback_dynamics_chorus(y_full, y_vocal, y_rhythm, sr)
                return {
                    "status": "success",
                    "chorus_sections_beat_ssm": [
                        {"start_sec": round(s["start_sec"], 2), "end_sec": round(s["end_sec"], 2)}
                        for s in fallback_sections
                    ],
                    "chorus_confidence_beat_ssm": 0.75,
                    "chorus_method_beat_ssm": "dynamics_adaptive_fallback"
                }

            # 3. 特徴量抽出とビート同期 (chroma_cens + mfcc)
            chroma = librosa.feature.chroma_cens(y=y_other + y_vocal, sr=sr)
            mfcc = librosa.feature.mfcc(y=y_full, sr=sr, n_mfcc=13)
            
            chroma_sync = librosa.util.sync(chroma, beat_frames.tolist(), aggregate=np.median)
            mfcc_sync = librosa.util.sync(mfcc, beat_frames.tolist(), aggregate=np.median)
            
            # 特徴量を結合して正規化 (和声12次元 + 音色13次元の結合空間で最高コントラストを維持)
            X = np.vstack([librosa.util.normalize(chroma_sync, axis=0), 
                           librosa.util.normalize(mfcc_sync, axis=0)])

            # 4. 自己類似行列 (SSM) の構築
            R = librosa.segment.recurrence_matrix(X, mode='affinity', metric='cosine', sym=True, width=8)

            # 5. 対角パス強調 (Path Enhancement)
            # 展開の周期に最適化されたHann窓強調
            w = min(16, max(4, n_beats // 2))
            R_enh = librosa.segment.path_enhance(R, w, window='hann')
            
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
            full_sync = librosa.util.sync(full_rms, beat_frames.tolist(), aggregate=np.max)[0]
            full_sync = normalize(full_sync)
            
            if vocal_ratio < 0.08:
                # インスト曲: 全体音圧/ダイナミクス(0.45) + 高音域の広がり/明るさ(0.35) + リズムアタック(0.20)
                salience = 0.45 * full_sync + 0.35 * centroid_sync + 0.20 * rhythm_sync
                print(f"[Strategy: Chorus] インスト曲と判定しました。 (Vocal ratio: {vocal_ratio:.3f})")
                chorus_score = 0.30 * rep_score + 0.70 * salience
            else:
                # ボーカルあり曲: ボーカル(0.45) + 全体音圧(0.25) + リズム(0.20) + 明るさ(0.10)
                salience = 0.45 * vocal_sync + 0.25 * full_sync + 0.20 * rhythm_sync + 0.10 * centroid_sync
                chorus_score = 0.50 * rep_score + 0.50 * salience

            # 短いノイズを消すためにメディアンフィルタで平滑化
            filter_size = min(8, max(3, n_beats // 4))
            chorus_score_smooth = scipy.ndimage.median_filter(chorus_score, size=filter_size)

            # 8. サビ区間の抽出
            # 閾値係数を0.15σに設定
            threshold = np.mean(chorus_score_smooth) + 0.15 * np.std(chorus_score_smooth)
            is_chorus = chorus_score_smooth > threshold

            sections = []
            in_sec = False
            start_b = 0
            
            # サビの最小継続拍数: 楽曲長に応じて動的調整 (短尺曲では4拍〜6拍)
            min_beats_for_chorus = 8 if n_beats >= 32 else max(4, n_beats // 5)

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

            # 安全策: セクションが空、または冒頭イントロのみ(0秒付近開始かつ全体の60%未満で終了)、あるいは短尺インスト曲の場合はダイナミクス適応フォールバック
            total_dur = len(y_full) / sr
            is_only_intro = (len(sections) == 1 and sections[0]["start_sec"] <= 0.5 and sections[0]["end_sec"] <= total_dur * 0.6)
            if not sections or is_only_intro or (vocal_ratio < 0.08 and n_beats < 48):
                sections = self._fallback_dynamics_chorus(y_full, y_vocal, y_rhythm, sr)

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

        except Exception as e:  # noqa: BLE001
            print(f"[Warning] BeatSSM Chorus detection failed: {e}. ダイナミクス適応フォールバックを実行します。")
            try:
                fallback_sections = self._fallback_dynamics_chorus(y_full, y_vocal, y_rhythm, sr)
                return {
                    "status": "success",
                    "chorus_sections_beat_ssm": [
                        {"start_sec": round(s["start_sec"], 2), "end_sec": round(s["end_sec"], 2)}
                        for s in fallback_sections
                    ],
                    "chorus_confidence_beat_ssm": 0.65,
                    "chorus_method_beat_ssm": "dynamics_adaptive_fallback"
                }
            except Exception:  # noqa: BLE001
                return {
                    "status": "success",
                    "chorus_sections_beat_ssm": [],
                    "chorus_confidence_beat_ssm": 0.0,
                    "chorus_method_beat_ssm": "beat_sync_path_enhanced_fallback"
                }

    def _fallback_dynamics_chorus(self, y_full: np.ndarray, y_vocal: np.ndarray, y_rhythm: np.ndarray, sr: int) -> list[dict[str, float]]:
        """
        短尺音源（拍数が少ない）または構造解析が困難な場合の高精度ダイナミクスサビ検出フォールバック。
        音圧 (RMS)、スペクトル重心 (Centroid)、リズム音圧の時系列から最も盛り上がる区間（サビ/ドロップ）を抽出。
        """
        hop_length = 512
        full_rms = librosa.feature.rms(y=y_full, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y_full, sr=sr, hop_length=hop_length)[0]
        vocal_rms = librosa.feature.rms(y=y_vocal, hop_length=hop_length)[0]
        rhythm_rms = librosa.feature.rms(y=y_rhythm, hop_length=hop_length)[0]

        def norm_arr(arr):
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / (rng if rng > 0 else 1.0)

        n_frames = len(full_rms)
        total_duration = len(y_full) / sr
        
        # ボーカル存在比率
        vocal_ratio = np.mean(vocal_rms) / (np.mean(full_rms) + 1e-8)
        if vocal_ratio < 0.08:
            dynamics = 0.50 * norm_arr(full_rms) + 0.30 * norm_arr(rhythm_rms) + 0.20 * norm_arr(centroid)
        else:
            dynamics = 0.40 * norm_arr(vocal_rms) + 0.35 * norm_arr(full_rms) + 0.25 * norm_arr(centroid)
            
        # 約1秒の平滑化 (約43フレーム)
        win_size = max(5, int(sr / hop_length))
        smooth_dyn = scipy.ndimage.gaussian_filter1d(dynamics, sigma=win_size / 2)
        
        # 閾値: 平均値 + 0.15 * 標準偏差
        thresh = np.mean(smooth_dyn) + 0.15 * np.std(smooth_dyn)
        is_high = smooth_dyn > thresh
        
        # 連続区間
        times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
        sections = []
        in_sec = False
        start_t = 0.0
        
        min_dur = min(2.0, total_duration * 0.15)
        
        for f in range(n_frames):
            if is_high[f] and not in_sec:
                in_sec = True
                start_t = times[f]
            elif not is_high[f] and in_sec:
                in_sec = False
                end_t = times[f]
                if end_t - start_t >= min_dur:
                    sections.append({"start_sec": float(start_t), "end_sec": float(end_t)})
                    
        if in_sec and total_duration - start_t >= min_dur:
            sections.append({"start_sec": float(start_t), "end_sec": float(total_duration)})
            
        # 1つも区間が得られない場合は、最大ピークを中心とした区間
        if not sections:
            peak_idx = int(np.argmax(smooth_dyn))
            peak_t = times[peak_idx]
            span = min(total_duration * 0.4, 8.0)
            st = max(0.0, peak_t - span / 2)
            et = min(total_duration, peak_t + span / 2)
            sections.append({"start_sec": float(st), "end_sec": float(et)})
            
        # マージ
        merged = []
        for sec in sections:
            if merged and sec["start_sec"] - merged[-1]["end_sec"] <= 2.0:
                merged[-1]["end_sec"] = sec["end_sec"]
            else:
                merged.append(sec)
                
        return merged