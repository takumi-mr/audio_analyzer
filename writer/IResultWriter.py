from abc import ABC, abstractmethod
from typing import Dict, Any

class IResultWriter(ABC):
    """解析結果書き出しのインターフェース"""
    @abstractmethod
    def write(self, filepath: str, result: Dict[str, Any]) -> None:
        pass