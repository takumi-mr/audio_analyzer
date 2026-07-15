from abc import ABC, abstractmethod
from model.AudioSignal import AudioSignal

class IAudioReader(ABC):
    """音声ファイル読み込みのインターフェース"""
    @abstractmethod
    def read(self, filepath: str) -> AudioSignal:
        pass