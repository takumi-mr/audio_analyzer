import numpy as np
import scipy.signal

from analyzer.filter.IAudioFilter import IAudioFilter
from model.AudioSignal import AudioSignal


class BandSplitterFilter(IAudioFilter):
    """特定の音声データを低域と高域の2つのバンドに分割して辞書に追加するフィルター"""

    def __init__(self, target_key: str = "target", cutoff_hz: float = 150.0):
        self.target_key = target_key
        self.cutoff_hz = cutoff_hz

    def apply(self, signals: dict[str, AudioSignal]) -> dict[str, AudioSignal]:
        # 辞書の浅いコピーを作成して元のオブジェクト破壊を防ぐ
        result_signals = dict(signals)

        signal = result_signals.get(self.target_key)
        if not signal or len(signal.data) == 0:
            return result_signals

        print(
            f"[Filter: BandSplitter] '{self.target_key}' を遮断周波数 {self.cutoff_hz} Hz で低域・高域に分割中..."
        )

        # 1. 2次のデジタルバターワースフィルターの設計
        sr = signal.sample_rate
        nyquist = 0.5 * sr
        normal_cutoff = self.cutoff_hz / nyquist

        # 低周波だけ通すローパスフィルター
        butter_out = scipy.signal.butter(
            2, normal_cutoff, btype="low", analog=False, output="ba"
        )
        assert butter_out is not None
        b_low, a_low = butter_out[0], butter_out[1]
        low_data = scipy.signal.filtfilt(b_low, a_low, signal.data)

        # 高域は「元の信号 - 低域」で得る
        # 元の信号から引くことで、足し合わせれば完璧に元の信号に戻る相補的な関係にする
        high_data = signal.data - low_data

        # 2. 新しいシグナルとして辞書に格納
        result_signals[f"{self.target_key}_low"] = AudioSignal(
            data=low_data.astype(np.float32),
            sample_rate=sr,
            duration_sec=signal.duration_sec,
        )
        result_signals[f"{self.target_key}_high"] = AudioSignal(
            data=high_data.astype(np.float32),
            sample_rate=sr,
            duration_sec=signal.duration_sec,
        )

        return result_signals
