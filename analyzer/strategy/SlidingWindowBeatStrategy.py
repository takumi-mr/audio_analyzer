import numpy as np
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any, List
from model.AudioSignal import AudioSignal

class SlidingWindowBeatStrategy(IAnalysisStrategy):
    """時間経過で拍子・テンポが変わる曲をスライディングウィンドウで動的に解析する戦略"""
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] = None) -> Dict[str, Any]:
        # 低域、またはドラム、それがなければ target を優先順に検索
        signal = None
        for key in ["target_low", "target_drums", "target"]:
            if key in signals:
                signal = signals[key]
                print(f"[Strategy: SlidingWindow] 解析対象として '{key}' シグナルを採用しました。")
                break
                
        if signal is None and signals:
            signal = next(iter(signals.values()))
            
        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print(f"[Strategy: SlidingWindow] 時間窓で区切って局所的テンポ変化を解析中 (サンプル数: {len(signal.data)})...")
        
        # 全長とサンプリング周波数
        duration = signal.duration_sec
        sr = signal.sample_rate
        
        # スライディングウィンドウ設定 (窓幅 10秒, 移動幅 5秒)
        window_size = 10.0
        step_size = 5.0
        
        win_samples = int(window_size * sr)
        step_samples = int(step_size * sr)
        
        segments = []
        
        for start_idx in range(0, len(signal.data) - win_samples + 1, step_samples):
            sub_data = signal.data[start_idx : start_idx + win_samples]
            sub_signal = AudioSignal(data=sub_data, sample_rate=sr, duration_sec=window_size)
            
            # 部分シグナルのテンポ検出
            bpm = self._detect_local_bpm(sub_signal)
            
            start_sec = start_idx / sr
            end_sec = start_sec + window_size
            
            segments.append({
                "start_sec": round(start_sec, 1),
                "end_sec": round(end_sec, 1),
                "tempo_bpm": bpm,
                # 簡易ルール：BPMが 120 を超えたら 7/8 拍子、それ以下は 4/4 拍子と判定するモック
                "time_signature": "7/8" if bpm > 130 else "4/4"
            })
            
        # 平均テンポ
        base_bpm = round(float(np.mean([s["tempo_bpm"] for s in segments])), 1) if segments else 120.0
        
        return {
            "status": "success",
            "base_tempo_bpm": base_bpm,
            "segments": segments
        }
        
    def _detect_local_bpm(self, signal: AudioSignal) -> float:
        # SimpleBeatStrategy のテンポ検出と同等の簡易実装
        sr = signal.sample_rate
        hop_sec = 0.01
        frame_sec = 0.05
        hop_size = int(hop_sec * sr)
        frame_size = int(frame_sec * sr)
        
        cumsum = np.insert(np.cumsum(signal.data ** 2), 0, 0)
        frame_idx = np.arange(0, len(signal.data) - frame_size, hop_size)
        if len(frame_idx) < 10:
            return 120.0
            
        energy = cumsum[frame_idx + frame_size] - cumsum[frame_idx]
        onset = np.diff(energy)
        onset = np.maximum(0, onset)
        
        min_lag = 33
        max_lag = 100
        best_lag = min_lag
        max_corr = -1.0
        for lag in range(min_lag, max_lag + 1):
            corr = np.dot(onset[lag:], onset[:-lag])
            if corr > max_corr:
                max_corr = corr
                best_lag = lag
        return round(60.0 / (best_lag * hop_sec), 1)
