import librosa
from reader.IAudioReader import IAudioReader
from model.AudioSignal import AudioSignal

class LibrosaAudioReader(IAudioReader):
    """librosaを使用した汎用的な音声ファイル読み込みの実装 (MP3, WAV, OGG等対応)"""
    def read(self, filepath: str) -> AudioSignal:
        print(f"[Reader] {filepath} から音響データを読み込みます(librosa使用)...")
        # sr=None にすることで元のサンプリングレートを保持し、mono=True でモノラル化して読み込む
        data, sr = librosa.load(filepath, sr=None, mono=True)
        duration = float(librosa.get_duration(y=data, sr=sr))
        return AudioSignal(data=data, sample_rate=sr, duration_sec=duration)
