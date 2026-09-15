from abc import ABC, abstractmethod
from typing import Any

from model.AudioSignal import AudioSignal


class IAnalysisStrategy(ABC):
    """音声解析の戦略インターフェース"""

    @abstractmethod
    def analyze(
        self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        pass
