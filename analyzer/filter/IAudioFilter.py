from abc import ABC, abstractmethod
from typing import Dict
from model.AudioSignal import AudioSignal

class IAudioFilter(ABC):
    """音声信号の辞書に対する事前処理（フィルター）のインターフェース"""
    @abstractmethod
    def apply(self, signals: Dict[str, AudioSignal]) -> Dict[str, AudioSignal]:
        """
        音声信号の辞書を入力し、加工・分割・拡張された音声信号の辞書を返却する。
        例: {'target': sig} -> {'target': sig, 'target_low': low_sig, 'target_high': high_sig}
        """
        pass
