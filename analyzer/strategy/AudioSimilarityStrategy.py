import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class AudioSimilarityStrategy(IAnalysisStrategy):
    """音階 (Chroma) と音色 (MFCC) のハイブリッド解析による音声類似度計算戦略"""
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] = None) -> Dict[str, Any]:
        target = signals.get("target")
        reference = signals.get("reference")
        
        if not target or not reference:
            raise ValueError("類似度計算には 'target' と 'reference' の両方の音声信号が必要です。")
            
        print(f"[Strategy: Similarity] 音階 (Chroma) と音色 (MFCC) に基づくハイブリッド類似度を計算中...")
        
        # 1. 音色類似度 (MFCC コサイン類似度) の計算
        target_mfcc = np.mean(librosa.feature.mfcc(y=target.data, sr=target.sample_rate, n_mfcc=13), axis=1)
        ref_mfcc = np.mean(librosa.feature.mfcc(y=reference.data, sr=reference.sample_rate, n_mfcc=13), axis=1)
        
        denom_mfcc = np.linalg.norm(target_mfcc) * np.linalg.norm(ref_mfcc)
        timbre_similarity = float(np.dot(target_mfcc, ref_mfcc) / denom_mfcc) if denom_mfcc > 0 else 0.0
        # [-1, 1] -> [0, 1]
        timbre_score = (timbre_similarity + 1.0) / 2.0
        
        # 2. 音階/コード進行類似度 (Chroma CENS コサイン類似度) の計算
        target_chroma = np.mean(librosa.feature.chroma_cens(y=target.data, sr=target.sample_rate), axis=1)
        ref_chroma = np.mean(librosa.feature.chroma_cens(y=reference.data, sr=reference.sample_rate), axis=1)
        
        denom_chroma = np.linalg.norm(target_chroma) * np.linalg.norm(ref_chroma)
        pitch_similarity = float(np.dot(target_chroma, ref_chroma) / denom_chroma) if denom_chroma > 0 else 0.0
        pitch_score = (pitch_similarity + 1.0) / 2.0
        
        # 3. 総合類似度スコアのブレンド (各50%のウェイト)
        weights = params.get("weights", {"timbre": 0.5, "pitch": 0.5}) if params else {"timbre": 0.5, "pitch": 0.5}
        w_timbre = weights.get("timbre", 0.5)
        w_pitch = weights.get("pitch", 0.5)
        
        combined_score = (w_timbre * timbre_score) + (w_pitch * pitch_score)
        
        threshold = params.get("threshold", 0.8) if params else 0.8
        is_match = combined_score >= threshold
        
        return {
            "status": "success",
            "similarity_score": round(combined_score, 3),
            "match": bool(is_match),
            "details": {
                "timbre_similarity_score": round(timbre_score, 3),
                "pitch_similarity_score": round(pitch_score, 3),
                "weights_used": {"timbre": w_timbre, "pitch": w_pitch},
                "threshold": threshold
            }
        }
