import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class KeyDetectionStrategy(IAnalysisStrategy):
    """Krumhansl-Schmucklerプロファイルを用いて楽曲全体の主キー（調）を推定する戦略"""
    def __init__(self):
        # 12音階の名前の定義
        self.pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        # Krumhansl-Schmuckler キープロファイル定義
        # self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        # self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        # Temperley プロファイル: 現代のポピュラー音楽により適合
        self.major_profile = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
        self.minor_profile = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

        # 平均を引いて正規化
        self.major_profile = self.major_profile - np.mean(self.major_profile)
        self.minor_profile = self.minor_profile - np.mean(self.minor_profile)
        
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # 優先キー：target_vocal, target_harmonic, target
        signal = None
        for key in ["target_vocal", "target_harmonic", "target"]:
            if key in signals:
                signal = signals[key]
                print(f"[Strategy: Key] 解析対象として '{key}' シグナルを採用しました。")
                break
                
        if signal is None and signals:
            signal = next(iter(signals.values()))
            
        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print("[Strategy: Key] クロマ特徴量を抽出して主キーを推定中...")
        
        # 1. クロマベクトルの抽出 (CENS)
        chroma = librosa.feature.chroma_cens(y=signal.data, sr=signal.sample_rate)
        # 時間軸の平均を取り12次元ベクトルへ
        mean_chroma = np.mean(chroma, axis=1)
        mean_chroma = mean_chroma - np.mean(mean_chroma)
        
        best_r = -2.0
        best_key = ""
        best_mode = "Major"
        
        # 2. 12音階 × 2モード (Major/Minor) の相関を調査
        for i in range(12):
            # i要素だけ右ローテート
            rotated_major = np.roll(self.major_profile, i)
            rotated_minor = np.roll(self.minor_profile, i)
            
            # コサイン類似度
            denom_major = np.linalg.norm(mean_chroma) * np.linalg.norm(rotated_major)
            r_major = np.dot(mean_chroma, rotated_major) / denom_major if denom_major > 0 else 0.0
            
            denom_minor = np.linalg.norm(mean_chroma) * np.linalg.norm(rotated_minor)
            r_minor = np.dot(mean_chroma, rotated_minor) / denom_minor if denom_minor > 0 else 0.0
            
            if r_major > best_r:
                best_r = r_major
                best_key = self.pitch_classes[i]
                best_mode = "Major"
                
            if r_minor > best_r:
                best_r = r_minor
                best_key = self.pitch_classes[i]
                best_mode = "Minor"
                
        # 信頼度：最も高い相関係数を [0, 1] に正規化
        confidence = (best_r + 1.0) / 2.0
        
        detected_key = f"{best_key} {best_mode}"
        print(f"  - 推定キー: {detected_key} (confidence: {round(confidence, 2)})")
        
        return {
            "status": "success",
            "estimated_key": detected_key,
            "key_tonic": best_key,
            "key_scale": best_mode,
            "confidence": round(float(confidence), 3)
        }
