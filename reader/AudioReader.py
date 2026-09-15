import wave

import numpy as np

from model.AudioSignal import AudioSignal
from reader.IAudioReader import IAudioReader


class WavAudioReader(IAudioReader):
    """WAVファイル読み取りの実装 (標準ライブラリのみ使用)"""

    def read(self, filepath: str) -> AudioSignal:
        print(f"[Reader] {filepath} から WAV データを読み込みます...")
        with wave.open(filepath, "rb") as w:
            params = w.getparams()
            n_channels = params.nchannels
            sampwidth = params.sampwidth
            framerate = params.framerate
            n_frames = params.nframes

            # 生バイナリデータを読み込む
            raw_data = w.readframes(n_frames)

            # データ型判定
            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            elif sampwidth == 1:
                dtype = np.uint8
            else:
                raise ValueError(f"サポートされていないサンプル幅です: {sampwidth}")

            # numpy 配列に変換
            data = np.frombuffer(raw_data, dtype=dtype)

            # ステレオの場合はモノラルに変換（平均値を取る）
            if n_channels > 1:
                data = data.reshape(-1, n_channels)
                data = data.mean(axis=1)

            # 浮動小数点数（-1.0 ~ 1.0）に正規化
            if dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0

            duration = n_frames / framerate
            return AudioSignal(data=data, sample_rate=framerate, duration_sec=duration)
