import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any
from model.AudioSignal import AudioSignal

class AudioSimilarityStrategy(IAnalysisStrategy):
    """MFCCコサイン類似度を用いて2つの音声の類似度を計算する戦略"""
    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] = None) -> Dict[str, Any]:
        target = signals.get("target")
        reference = signals.get("reference")
        
        if not target or not reference:
            raise ValueError("類似度計算には 'target' と 'reference' の両方の音声信号が必要です。")
            
        print(f"[Strategy: Similarity] 2つの信号の特徴量から類似度を計算中...")
        
        # 1. 各音声から MFCC (13次元) を抽出して時間平均ベクトルを生成
        target_mfcc = np.mean(librosa.feature.mfcc(y=target.data, sr=target.sample_rate, n_mfcc=13), axis=1)
        ref_mfcc = np.mean(librosa.feature.mfcc(y=reference.data, sr=reference.sample_rate, n_mfcc=13), axis=1)
        
        # 2. コサイン類似度の計算
        denom = np.linalg.norm(target_mfcc) * np.linalg.norm(ref_mfcc)
        if denom == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(target_mfcc, ref_mfcc) / denom)
            
        # コサイン類似度 [-1, 1] を [0, 1] にスケーリング
        normalized_score = float((similarity + 1.0) / 2.0)
        
        # 簡易閾値によるマッチ判定
        threshold = 0.8
        is_match = normalized_score >= threshold
        
        return {
            "status": "success",
            "similarity_score": round(normalized_score, 3),
            "match": bool(is_match),
            "details": {
                "raw_cosine_similarity": round(similarity, 3),
                "threshold": threshold
            }
        }
