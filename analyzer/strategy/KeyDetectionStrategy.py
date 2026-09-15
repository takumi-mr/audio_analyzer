from typing import Any

import librosa
import numpy as np

from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal


class KeyDetectionStrategy(IAnalysisStrategy):
    """
    Krumhansl-Schmuckler プロファイルおよびチューニング補正済みハイブリッドクロマ(CQT+CENS)を用いて、
    楽曲全体の主キー（調）を高精度に推定する戦略。
    """

    def __init__(self):
        # 12音階の名前の定義
        self.pitch_classes = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]

        # Krumhansl-Schmuckler キープロファイル定義 (安定した調性検出)
        self.major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        self.minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        # 平均を引いて正規化
        self.major_profile = self.major_profile - np.mean(self.major_profile)
        self.minor_profile = self.minor_profile - np.mean(self.minor_profile)

    def analyze(
        self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # 1. 最適な和声シグナルの選定 (伴奏 target_other または全体 target)
        if "target_other" in signals and "target_vocal" in signals:
            print(
                "[Strategy: Key] 'target_other' と 'target_vocal' を合成して和声と主旋律からキーを推定します。"
            )
            y = signals["target_other"].data + signals["target_vocal"].data
            sr = signals["target_other"].sample_rate
        elif "target_other" in signals:
            print(
                "[Strategy: Key] 解析対象として 'target_other' シグナルを採用しました。"
            )
            y = signals["target_other"].data
            sr = signals["target_other"].sample_rate
        elif "target" in signals:
            print("[Strategy: Key] 解析対象として 'target' シグナルを採用しました。")
            y = signals["target"].data
            sr = signals["target"].sample_rate
        elif "target_harmonic" in signals:
            print(
                "[Strategy: Key] 解析対象として 'target_harmonic' シグナルを採用しました。"
            )
            y = signals["target_harmonic"].data
            sr = signals["target_harmonic"].sample_rate
        elif "target_vocal" in signals:
            print(
                "[Strategy: Key] 解析対象として 'target_vocal' シグナルを採用しました。"
            )
            y = signals["target_vocal"].data
            sr = signals["target_vocal"].sample_rate
        elif signals:
            sig = next(iter(signals.values()))
            y = sig.data
            sr = sig.sample_rate
        else:
            return {"status": "error", "message": "No audio signal available."}

        if len(y) == 0:
            return {"status": "error", "message": "No audio signal available."}

        print(
            "[Strategy: Key] チューニング補正とハイブリッドクロマ(CQT+CENS)により主キーを推定中..."
        )

        # 2. ピッチズレ（チューニング）の推定と補正
        tuning = librosa.estimate_tuning(y=y, sr=sr)

        # 3. ハイブリッドクロマベクトルの抽出 (CQT + CENS)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, tuning=tuning)
        chroma = 0.5 * chroma_cqt + 0.5 * chroma_cens

        mean_chroma = np.mean(chroma, axis=1)
        mean_chroma = mean_chroma - np.mean(mean_chroma)

        best_r = -2.0
        best_key = ""
        best_mode = "Major"

        # 4. 12音階 × 2モード (Major/Minor) の相関を調査
        for i in range(12):
            rotated_major = np.roll(self.major_profile, i)
            rotated_minor = np.roll(self.minor_profile, i)

            denom_major = np.linalg.norm(mean_chroma) * np.linalg.norm(rotated_major)
            r_major = (
                np.dot(mean_chroma, rotated_major) / denom_major
                if denom_major > 0
                else 0.0
            )

            denom_minor = np.linalg.norm(mean_chroma) * np.linalg.norm(rotated_minor)
            r_minor = (
                np.dot(mean_chroma, rotated_minor) / denom_minor
                if denom_minor > 0
                else 0.0
            )

            if r_major > best_r:
                best_r = r_major
                best_key = self.pitch_classes[i]
                best_mode = "Major"

            if r_minor > best_r:
                best_r = r_minor
                best_key = self.pitch_classes[i]
                best_mode = "Minor"

        # 信頼度：最も高い相関係数を [0, 1] に正規化
        confidence = min(1.0, max(0.0, (best_r + 1.0) / 2.0))

        detected_key = f"{best_key} {best_mode}"
        print(f"  - 推定キー: {detected_key} (confidence: {round(confidence, 2)})")

        return {
            "status": "success",
            "estimated_key": detected_key,
            "key_tonic": best_key,
            "key_scale": best_mode,
            "confidence": round(float(confidence), 3),
        }
