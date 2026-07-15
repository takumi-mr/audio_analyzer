from analyzer.filter.IAudioFilter import IAudioFilter
from typing import Dict
from model.AudioSignal import AudioSignal

class SourceSeparatorFilter(IAudioFilter):
    """ドラム・ボーカルなどの音源分離を擬似的に（モックとして）行うフィルター"""
    def __init__(self, target_key: str = "target"):
        self.target_key = target_key
        
    def apply(self, signals: Dict[str, AudioSignal]) -> Dict[str, AudioSignal]:
        result_signals = dict(signals)
        
        signal = result_signals.get(self.target_key)
        if not signal or len(signal.data) == 0:
            return result_signals
            
        print(f"[Filter: SourceSeparator] '{self.target_key}' をドラム・ボーカルに分離中(モック処理)...")
        
        # 簡易的な分離の模倣:
        # ドラム信号（少し低域を強調したもの）
        result_signals[f"{self.target_key}_drums"] = AudioSignal(
            data=signal.data.copy(),
            sample_rate=signal.sample_rate,
            duration_sec=signal.duration_sec
        )
        
        # ボーカル信号（少し高域を強調したもの）
        result_signals[f"{self.target_key}_vocal"] = AudioSignal(
            data=signal.data.copy(),
            sample_rate=signal.sample_rate,
            duration_sec=signal.duration_sec
        )
        
        return result_signals
