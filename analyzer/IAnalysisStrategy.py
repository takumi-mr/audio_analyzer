from abc import ABC, abstractmethod
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class IAnalysisStrategy(ABC):
    """音声解析の戦略インターフェース"""
    @abstractmethod
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] = None) -> Dict[str, Any]:
        pass
