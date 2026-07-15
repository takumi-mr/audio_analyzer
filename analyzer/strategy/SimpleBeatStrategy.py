import numpy as np
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class SimpleBeatStrategy(IAnalysisStrategy):
    """標準的な曲向け: 簡易自己相関法を用いてBPMを算出する戦略"""
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] = None) -> Dict[str, Any]:
        # 低域、またはドラム、それがなければ target を優先順に検索
        signal = None
        for key in ["target_low", "target_drums", "target"]:
            if key in signals:
                signal = signals[key]
                print(f"[Strategy: Simple] 解析対象として '{key}' シグナルを採用しました。")
                break
                
        if signal is None and signals:
            signal = next(iter(signals.values()))
            
        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print(f"[Strategy: Simple] 実波形データからBPMを算出中 (サンプル数: {len(signal.data)})...")
        
        # 1. フレームエネルギーの計算 (50ms窓, 10msホップ)
        sr = signal.sample_rate
        hop_sec = 0.01
        frame_sec = 0.05
        hop_size = int(hop_sec * sr)
        frame_size = int(frame_sec * sr)
        
        data_sq = signal.data ** 2
        cumsum = np.insert(np.cumsum(data_sq), 0, 0)
        
        frame_idx = np.arange(0, len(signal.data) - frame_size, hop_size)
        if len(frame_idx) < 10:
            return {"status": "success", "tempo_bpm": 120, "confidence": 0.5, "note": "Too short signal"}
            
        energy = cumsum[frame_idx + frame_size] - cumsum[frame_idx]
        
        # オンセット（エネルギー変化の正の成分）
        onset = np.diff(energy)
        onset = np.maximum(0, onset)
        
        # 2. 自己相関によるBPM検出 (BPM 60〜180 -> ラグ 100〜33)
        min_lag = 33
        max_lag = 100
        
        best_lag = min_lag
        max_corr = -1.0
        
        for lag in range(min_lag, max_lag + 1):
            corr = np.dot(onset[lag:], onset[:-lag])
            if corr > max_corr:
                max_corr = corr
                best_lag = lag
                
        # 最良ラグからBPMを計算
        detected_bpm = round(60.0 / (best_lag * hop_sec), 1)
        
        # 信頼度（簡易計算）
        total_energy = np.dot(onset, onset)
        confidence = float(min(1.0, max_corr / (total_energy + 1e-6) * 3))
        
        return {
            "status": "success",
            "tempo_bpm": detected_bpm,
            "time_signature": "4/4",
            "confidence": round(confidence, 2)
        }
