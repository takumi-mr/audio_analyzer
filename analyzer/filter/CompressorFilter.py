import numpy as np

from analyzer.filter.IAudioFilter import IAudioFilter
from model.AudioSignal import AudioSignal


class CompressorFilter(IAudioFilter):
    """指定されたキーの音声データに対してのみ、音圧圧縮（コンプレッション）を行うフィルター"""

    def __init__(
        self,
        target_key: str = "target",
        threshold: float = 0.3,
        ratio: float = 4.0,
        gain: float = 1.2,
    ):
        self.target_key = target_key
        self.threshold = threshold
        self.ratio = ratio
        self.gain = gain  # メイクアップゲイン

    def apply(self, signals: dict[str, AudioSignal]) -> dict[str, AudioSignal]:
        result_signals = dict(signals)

        signal = result_signals.get(self.target_key)
        if not signal or len(signal.data) == 0:
            return result_signals

        print(
            f"[Filter: Compressor] '{self.target_key}' に対しコンプレッションを適用中 (threshold={self.threshold}, ratio={self.ratio})..."
        )

        # 簡易コンプレッサーのロジック
        data = np.copy(signal.data)
        abs_data = np.abs(data)

        # 閾値を超えた部分のインデックス
        over_threshold = abs_data > self.threshold

        # 正負の符号を維持しつつ圧縮
        if np.any(over_threshold):
            signs = np.sign(data[over_threshold])
            compressed_vals = (
                self.threshold
                + (abs_data[over_threshold] - self.threshold) / self.ratio
            )
            data[over_threshold] = signs * compressed_vals

        # メイクアップゲインを適用
        data = data * self.gain

        # クリップの防止
        data = np.clip(data, -1.0, 1.0)

        # 辞書の該当キーを上書き
        result_signals[self.target_key] = AudioSignal(
            data=data.astype(np.float32),
            sample_rate=signal.sample_rate,
            duration_sec=signal.duration_sec,
        )

        return result_signals
