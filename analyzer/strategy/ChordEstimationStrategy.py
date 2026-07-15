import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any, List
from model.AudioSignal import AudioSignal

class ChordEstimationStrategy(IAnalysisStrategy):
    """フレームごとのクロマ特徴量と主要コードテンプレートを照合し、楽曲全体のコード進行を推定する戦略"""
    def __init__(self):
        self.pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.chord_names = []
        self.templates = []
        
        # 12半音に対するメジャー/マイナーのインデックスシフト
        major_intervals = [0, 4, 7]
        minor_intervals = [0, 3, 7]
        
        # 24通りのコードテンプレートを構築 (12音階 × 2モード)
        for mode_name, intervals in [("Major", major_intervals), ("Minor", minor_intervals)]:
            for i in range(12):
                template = np.zeros(12)
                for interval in intervals:
                    template[(i + interval) % 12] = 1.0
                
                # 正規化
                norm = np.linalg.norm(template)
                if norm > 0:
                    template = template / norm
                    
                root_name = self.pitch_classes[i]
                chord_suffix = "" if mode_name == "Major" else "m"
                self.chord_names.append(f"{root_name}{chord_suffix}")
                self.templates.append(template)
                
        self.templates = np.array(self.templates) # (24, 12)

    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # 優先キー：target_vocal, target_harmonic, target
        signal = None
        for key in ["target_vocal", "target_harmonic", "target"]:
            if key in signals:
                signal = signals[key]
                print(f"[Strategy: Chord] 解析対象として '{key}' シグナルを採用しました。")
                break
                
        if signal is None and signals:
            signal = next(iter(signals.values()))
            
        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print("[Strategy: Chord] 音声データからコード進行を推定中...")
        
        sr = signal.sample_rate
        # クロマ特徴量の抽出 (CENS)
        hop_length = 512
        chroma = librosa.feature.chroma_cens(y=signal.data, sr=sr, hop_length=hop_length)
        
        n_frames = chroma.shape[1]
        raw_chords = []
        
        # 各フレームに対してテンプレートマッチング
        for f in range(n_frames):
            frame_vec = chroma[:, f]
            norm = np.linalg.norm(frame_vec)
            if norm > 0:
                frame_vec = frame_vec / norm
                
            # 24テンプレートとの内積（コサイン類似度）
            similarities = np.dot(self.templates, frame_vec)
            best_idx = np.argmax(similarities)
            raw_chords.append(self.chord_names[best_idx])
            
        # 連続する同一コード区間の圧縮
        chords_sequence = []
        if len(raw_chords) > 0:
            current_chord = raw_chords[0]
            start_frame = 0
            
            for f in range(1, len(raw_chords)):
                if raw_chords[f] != current_chord:
                    start_sec = start_frame * hop_length / sr
                    end_sec = f * hop_length / sr
                    chords_sequence.append({
                        "start_sec": round(float(start_sec), 2),
                        "end_sec": round(float(end_sec), 2),
                        "chord": current_chord
                    })
                    current_chord = raw_chords[f]
                    start_frame = f
                    
            # 最後の区間
            start_sec = start_frame * hop_length / sr
            end_sec = len(raw_chords) * hop_length / sr
            chords_sequence.append({
                "start_sec": round(float(start_sec), 2),
                "end_sec": round(float(end_sec), 2),
                "chord": current_chord
            })
            
        return {
            "status": "success",
            "chords": chords_sequence
        }
