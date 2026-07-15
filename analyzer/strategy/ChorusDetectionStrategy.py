import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any, List
from model.AudioSignal import AudioSignal

class ChorusDetectionStrategy(IAnalysisStrategy):
    """音圧(RMS)、音の明るさ(スペクトル重心)、および繰り返しの類似性に基づき、楽曲のサビ(Chorus)区間を検出する戦略。
    結果は 'chorus_sections_rms' キーで返す。仙6法との比較用。"""
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # 優先ターゲット
        signal = None
        for key in ["target_vocal", "target_harmonic", "target"]:
            if key in signals:
                signal = signals[key]
                print(f"[Strategy: Chorus] 解析対象として '{key}' シグナルを採用しました。")
                break
                
        if signal is None and signals:
            signal = next(iter(signals.values()))
            
        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print("[Strategy: Chorus] 音響特徴量および構造分析によりサビ(Chorus)区間を検出中...")
        
        sr = signal.sample_rate
        y = signal.data
        
        # 1. 短時間特徴量の計算 (窓幅 2048, ホップ 512 = 約23ms間隔)
        hop_length = 512
        frame_time = hop_length / sr
        
        # RMSエネルギー
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        # スペクトル重心
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # 1秒（約 43 フレーム @22050Hz）ごとの平均値に平滑化
        frames_per_sec = int(sr / hop_length)
        n_seconds = int(len(y) / sr)
        
        if n_seconds < 5:
            return {
                "status": "success",
                "chorus_sections_rms": [{"start_sec": 0.0, "end_sec": round(signal.duration_sec, 2)}],
                "chorus_confidence_rms": 0.5,
                "chorus_method_rms": "rms+centroid"
            }
            
        sec_energy = []
        sec_centroid = []
        
        for s in range(n_seconds):
            start_f = s * frames_per_sec
            end_f = (s + 1) * frames_per_sec
            if end_f <= len(rms):
                sec_energy.append(np.mean(rms[start_f:end_f]))
                sec_centroid.append(np.mean(centroid[start_f:end_f]))
            else:
                sec_energy.append(rms[-1])
                sec_centroid.append(centroid[-1])
                
        sec_energy = np.array(sec_energy)
        sec_centroid = np.array(sec_centroid)
        
        # 2. スコアリング: 音圧と明るさの規格化積
        denom_energy = (np.max(sec_energy) - np.min(sec_energy))
        norm_energy = (sec_energy - np.min(sec_energy)) / (denom_energy if denom_energy > 0 else 1.0)
        
        denom_centroid = (np.max(sec_centroid) - np.min(sec_centroid))
        norm_centroid = (sec_centroid - np.min(sec_centroid)) / (denom_centroid if denom_centroid > 0 else 1.0)
        
        chorus_scores = 0.6 * norm_energy + 0.4 * norm_centroid
        
        # 3. 閾値判定と連続区間（セグメント）の抽出
        threshold = np.mean(chorus_scores) * 1.1
        is_chorus_candidate = chorus_scores > threshold
        
        sections = []
        in_section = False
        start_s = 0
        
        for s in range(n_seconds):
            if is_chorus_candidate[s] and not in_section:
                in_section = True
                start_s = s
            elif not is_chorus_candidate[s] and in_section:
                in_section = False
                end_s = s
                if end_s - start_s >= 3:
                    sections.append((start_s, end_s))
                    
        if in_section:
            if n_seconds - start_s >= 3:
                sections.append((start_s, n_seconds))
                
        # 空の場合のフォールバック
        if len(sections) == 0:
            best_sec = int(np.argmax(chorus_scores))
            start_s = max(0, best_sec - 4)
            end_s = min(n_seconds, best_sec + 4)
            sections.append((start_s, end_s))
            
        # 最もスコア平均が高いトップセクションを選択
        section_scores = []
        for start, end in sections:
            section_scores.append(np.mean(chorus_scores[start:end]))
            
        best_idx = np.argmax(section_scores)
        best_start, best_end = sections[best_idx]
        
        chorus_sections = [{
            "start_sec": float(best_start),
            "end_sec": float(best_end)
        }]
        
        # 4. 繰り返しの類似チェック (クロマ類似度)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
        sec_chroma = []
        for s in range(n_seconds):
            start_f = s * frames_per_sec
            end_f = (s + 1) * frames_per_sec
            if end_f <= chroma.shape[1]:
                sec_chroma.append(np.mean(chroma[:, start_f:end_f], axis=1))
            else:
                sec_chroma.append(chroma[:, -1])
        sec_chroma = np.array(sec_chroma)
        
        ref_chroma = np.mean(sec_chroma[best_start:best_end], axis=0)
        ref_chroma_norm = np.linalg.norm(ref_chroma)
        
        if ref_chroma_norm > 0:
            span_len = best_end - best_start
            for s in range(n_seconds - span_len):
                if s + span_len <= best_start or s >= best_end:
                    test_chroma = np.mean(sec_chroma[s : s + span_len], axis=0)
                    test_norm = np.linalg.norm(test_chroma)
                    if test_norm > 0:
                        sim = np.dot(ref_chroma, test_chroma) / (ref_chroma_norm * test_norm)
                        if sim > 0.92 and np.mean(sec_energy[s : s + span_len]) > np.mean(sec_energy) * 0.8:
                            chorus_sections.append({
                                "start_sec": float(s),
                                "end_sec": float(s + span_len)
                            })
                            break
                            
        chorus_sections.sort(key=lambda x: x["start_sec"])
        confidence = float(section_scores[best_idx])
        
        return {
            "status": "success",
            "chorus_sections_rms": chorus_sections,
            "chorus_confidence_rms": round(confidence, 2),
            "chorus_method_rms": "rms+centroid"
        }
