import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class SimpleBeatStrategy(IAnalysisStrategy):
    """標準的な曲向け: librosaのテンポ追跡機能を用いて正確なBPMを算出する戦略"""
    def analyze(self, signals: Dict[str, AudioSignal], params: dict[str, Any] | None = None) -> Dict[str, Any]:
        # 打楽器分離、低域、またはドラム、それがなければ target を優先順に検索
        signal = None
        for key in ["target_drums", "target_percussive", "target_low", "target"]:
            if key in signals:
                signal = signals[key]
                print(f"[Strategy: Simple] 解析対象として '{key}' シグナルを採用しました。")
                break
                
        if signal is None and signals:
            signal = next(iter(signals.values()))
            
        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print(f"[Strategy: Simple] librosaを用いてBPMを正確に検出中...")
        
        # librosa.beat.beat_track を使用してテンポを抽出
        tempo, _ = librosa.beat.beat_track(y=signal.data, sr=signal.sample_rate)
        
        # 互換性のある float 変換
        bpm_val = float(np.atleast_1d(tempo)[0])
        
        # 信頼度
        confidence = 0.95 if bpm_val > 0 else 0.0
        
        return {
            "status": "success",
            "tempo_bpm": round(bpm_val, 1),
            "time_signature": "4/4",
            "confidence": confidence
        }
