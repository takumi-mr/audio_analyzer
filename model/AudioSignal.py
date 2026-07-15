from dataclasses import dataclass
import numpy as np


@dataclass
class AudioSignal:
    data: np.ndarray  # 音声波形データ（1次元の浮動小数点配列）
    sample_rate: int  # サンプリング周波数 (Hz)
    duration_sec: float  # 音声の長さ (秒)
