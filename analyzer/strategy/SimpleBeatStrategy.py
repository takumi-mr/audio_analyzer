import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class SimpleBeatStrategy(IAnalysisStrategy):
    """標準的な曲向け: librosaのテンポ追跡機能を用いて正確なBPMを算出する戦略"""
    def analyze(self, signals: Dict[str, AudioSignal], params: dict[str, Any] | None = None) -> Dict[str, Any]:

        # 1. テンポ解析に最適な「リズムセクション（ドラム＋ベース）」の合成を最優先する
        if "target_drums" in signals and "target_bass" in signals:
            print("[Strategy: Simple] 'target_drums' と 'target_bass' を合成してリズムを解析します (倍検出防止)")
            rhythm_data = signals["target_drums"].data + signals["target_bass"].data
            sr = signals["target_drums"].sample_rate
        elif "target_drums" in signals:
            print("[Strategy: Simple] 解析対象として 'target_drums' シグナルを採用しました。")
            rhythm_data = signals["target_drums"].data
            sr = signals["target_drums"].sample_rate
        elif "target" in signals:
            print("[Strategy: Simple] 解析対象として 'target' シグナルを採用しました。")
            rhythm_data = signals["target"].data
            sr = signals["target"].sample_rate
        else:
            signal = next(iter(signals.values()))
            rhythm_data = signal.data
            sr = signal.sample_rate

        if len(rhythm_data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print(f"[Strategy: Simple] librosaを用いてBPMを正確に検出中...")
        
        # 2. オンセット（音の立ち上がり）エンベロープを計算
        onset_env = librosa.onset.onset_strength(y=rhythm_data, sr=sr)
        
        # 3. ビートトラッキング（BPM倍検出の防止）
        # librosaのデフォルト(120)だと、70BPMが140BPMに引っ張られやすい。
        # start_bpm=100.0 にすることで、70BPMと130BPMの両方に公平な事前確率を与えます。
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=sr,
            start_bpm=100.0  # ★倍検出を防ぐ最大のキモ
        )
        
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
