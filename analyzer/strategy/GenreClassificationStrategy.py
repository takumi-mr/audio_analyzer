import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class GenreClassificationStrategy(IAnalysisStrategy):
    """音響特徴量 (BPM, スペクトル重心, ゼロ交差率, 平坦度, ロールオフ周波数) に基づきジャンルを分類する戦略"""
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] = None) -> Dict[str, Any]:
        signal = signals.get("target") if signals else None
        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print(f"[Strategy: Genre] 音響特徴量を抽出してジャンルを特定中...")
        
        # 1. 音響特徴量の抽出
        # テンポ (BPM)
        tempo = self._detect_bpm(signal)
        # スペクトル重心 (明るさ)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal.data, sr=signal.sample_rate)))
        # ゼロ交差率 (アタック感・激しさ)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal.data)))
        # スペクトル平坦度 (ノイズとトーンの度合い。1に近いほどノイズ、0に近いほど純音トーン)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=signal.data)))
        # スペクトルロールオフ (高域の減衰限界周波数)
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=signal.data, sr=signal.sample_rate, roll_percent=0.85)))
        
        print(f"  - 推定BPM: {tempo}")
        print(f"  - スペクトル重心: {round(centroid, 1)} Hz")
        print(f"  - ゼロ交差率: {round(zcr, 3)}")
        print(f"  - スペクトル平坦度: {round(flatness, 4)}")
        print(f"  - ロールオフ周波数: {round(rolloff, 1)} Hz")
        
        # 2. 高度化されたヒューリスティック分類ルール
        genre = "Ambient"
        confidence = 0.5
        
        # 激しい・ノイズが多く（flatness高、zcr高）、高域が強く出ている場合
        if (flatness > 0.005 or zcr > 0.12) and centroid > 2200:
            if tempo >= 115:
                genre = "Dance"
                confidence = float(min(0.98, 0.5 + flatness * 50 + zcr * 1.5))
            else:
                genre = "Rock"
                confidence = float(min(0.95, 0.4 + flatness * 45 + zcr * 2.0))
        # テンポが軽快で、音質は明るいがノイズ感（flatness）が低く抑えられている場合
        elif tempo >= 105 and centroid > 1600 and flatness <= 0.005:
            genre = "Pop"
            confidence = float(min(0.92, 0.3 + (centroid / 3500.0) + (1.0 - flatness * 100) * 0.3))
        # 低音メインで、平坦度が極めて低い（楽器トーンが澄んでいる）、かつテンポが遅い
        elif centroid < 1400 and tempo < 105 and flatness < 0.003:
            genre = "Jazz"
            confidence = float(min(0.92, 0.4 + (1.0 - centroid / 1400.0) * 0.3 + (1.0 - flatness * 200) * 0.2))
        # 高域限界が非常に低く（rolloffが低い）、全体が極めてクリーン（zcr低、flatness超低）
        elif rolloff < 1500 and zcr < 0.03 and flatness < 0.001:
            genre = "Classical"
            confidence = float(min(0.96, 0.5 + (1.0 - flatness * 500) * 0.3 + (1.0 - zcr * 15) * 0.15))
            
        return {
            "status": "success",
            "detected_genre": genre,
            "confidence": round(confidence, 2),
            "extracted_features": {
                "tempo_bpm": tempo,
                "spectral_centroid_hz": round(centroid, 1),
                "zero_crossing_rate": round(zcr, 4),
                "spectral_flatness": round(flatness, 6),
                "spectral_rolloff_hz": round(rolloff, 1)
            }
        }
        
    def _detect_bpm(self, signal: AudioSignal) -> float:
        tempo, _ = librosa.beat.beat_track(y=signal.data, sr=signal.sample_rate)
        return round(float(np.atleast_1d(tempo)[0]), 1)
