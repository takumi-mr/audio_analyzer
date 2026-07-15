import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class GenreClassificationStrategy(IAnalysisStrategy):
    """音響特徴量 (BPM, スペクトル重心, ゼロ交差率) に基づきジャンルを分類する戦略"""
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] = None) -> Dict[str, Any]:
        signal = signals.get("target") if signals else None
        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print(f"[Strategy: Genre] 音響特徴量を抽出してジャンルを特定中...")
        
        # 1. 音響特徴量の抽出
        # テンポ (BPM) の検出
        tempo = self._detect_bpm(signal)
        # スペクトル重心 (音の明るさ) の平均値
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal.data, sr=signal.sample_rate)))
        # ゼロ交差率 (激しさ・ノイズ成分) の平均値
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal.data)))
        
        print(f"  - 推定BPM: {tempo}")
        print(f"  - スペクトル重心: {round(centroid, 1)} Hz")
        print(f"  - ゼロ交差率: {round(zcr, 3)}")
        
        # 2. ヒューリスティック判定ルールによる分類
        genre = "Ambient"
        confidence = 0.5
        
        # 高ノイズ成分（ハイハットや歪み）が多く、明るい音質の場合
        if zcr > 0.12 or (zcr > 0.08 and centroid > 2200):
            if tempo >= 115:
                genre = "Dance"
                confidence = float(min(0.98, 0.6 + zcr * 2))
            else:
                genre = "Rock"
                confidence = float(min(0.95, 0.5 + zcr * 2.5))
        # テンポが標準的で明るい音質、かつノイズ感は少ない場合
        elif tempo >= 110 and centroid > 1500:
            genre = "Pop"
            confidence = float(min(0.9, 0.4 + (centroid / 4000.0)))
        # 落ち着いた低域メインでテンポが遅い場合
        elif centroid < 1200 and tempo < 110:
            genre = "Jazz"
            confidence = float(min(0.92, 0.4 + (1.0 - centroid / 1200.0)))
        # 極めて静かなクラシック風
        elif centroid < 800 and zcr < 0.03:
            genre = "Classical"
            confidence = float(min(0.95, 0.5 + (1.0 - zcr * 10)))
            
        return {
            "status": "success",
            "detected_genre": genre,
            "confidence": round(confidence, 2),
            "extracted_features": {
                "tempo_bpm": tempo,
                "spectral_centroid_hz": round(centroid, 1),
                "zero_crossing_rate": round(zcr, 4)
            }
        }
        
    def _detect_bpm(self, signal: AudioSignal) -> float:
        # 簡易テンポ自己相関検出
        sr = signal.sample_rate
        hop_sec = 0.01
        frame_sec = 0.05
        hop_size = int(hop_sec * sr)
        frame_size = int(frame_sec * sr)
        
        cumsum = np.insert(np.cumsum(signal.data ** 2), 0, 0)
        frame_idx = np.arange(0, len(signal.data) - frame_size, hop_size)
        if len(frame_idx) < 10:
            return 120.0
            
        energy = cumsum[frame_idx + frame_size] - cumsum[frame_idx]
        onset = np.diff(energy)
        onset = np.maximum(0, onset)
        
        min_lag = 33
        max_lag = 100
        best_lag = min_lag
        max_corr = -1.0
        for lag in range(min_lag, max_lag + 1):
            corr = np.dot(onset[lag:], onset[:-lag])
            if corr > max_corr:
                max_corr = corr
                best_lag = lag
        return round(60.0 / (best_lag * hop_sec), 1)
