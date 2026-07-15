import numpy as np
import librosa
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from typing import Dict, Any, List
from model.AudioSignal import AudioSignal

class ChordEstimationStrategy(IAnalysisStrategy):
    """
    ベース音のルート音強調とViterbiデコーディングを用いた、高精度なコード進行推定戦略。
    
    改良点:
    1. ベース音源 (target_bass) からベースのルート音高をビート同期で抽出。
       ルート音高と一致するコードテンプレートの類似度を動的にブーストすることで、ルート誤判定を大幅に低減。
    2. Viterbi自己遷移確率 (p_self) を 0.90 に引き上げ、チャタリング（1ビート内での細かなコード変化）を抑制。
    3. 特徴量をCQTだけでなく、高周波の和音ノイズを抑えるために平滑化されたクロマ (chroma_cens) で相補。
    """
    def __init__(self) -> None:
        self.pitch_classes: List[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.chord_names: List[str] = []
        
        # 事前確率 (Prior) の定義: (Mode, Intervals, Suffix, Prior)
        # 基本の和音は 1.0、複雑なテンションコード等は誤検出を防ぐために出にくくします
        chord_types = [
            ("Major",        [0, 4, 7],       "",       1.0),
            ("Minor",        [0, 3, 7],       "m",      1.0),
            ("Major7",       [0, 4, 7, 11],   "M7",     0.85),
            ("Minor7",       [0, 3, 7, 10],   "m7",     0.85),
            ("Dominant7",    [0, 4, 7, 10],   "7",      0.80),
            ("Diminished",   [0, 3, 6],       "dim",    0.50),
            ("Diminished7",  [0, 3, 6, 9],    "dim7",   0.50),
            ("Augmented",    [0, 4, 8],       "aug",    0.50),
            ("Sus4",         [0, 5, 7],       "sus4",   0.50),
            ("Major7(9)",    [0, 4, 7, 11, 2], "M7(9)", 0.20),
            ("Minor7(9)",    [0, 3, 7, 10, 2], "m7(9)", 0.20), 
            ("Dominant7(9)", [0, 4, 7, 10, 2], "7(9)",  0.20),
        ]
        
        templates_list: List[np.ndarray] = []
        priors_list: List[float] = []
        # 各コードテンプレートのルート音のインデックス (0-11) を記録
        self.chord_roots: List[int] = []
        
        for mode_name, intervals, suffix, prior in chord_types:
            for i in range(12):
                template = np.zeros(12)
                for idx, interval in enumerate(intervals):
                    if idx == 0: weight = 1.3  # ルート音の重みを強調
                    elif idx == 1: weight = 1.0
                    elif idx == 2: weight = 0.8
                    elif idx == 3: weight = 0.9
                    else: weight = 0.6
                    template[(i + interval) % 12] = weight
                    
                norm = np.linalg.norm(template)
                if norm > 0:
                    template = template / norm
                    
                root_name = self.pitch_classes[i]
                self.chord_names.append(f"{root_name}{suffix}")
                templates_list.append(template)
                priors_list.append(prior)
                self.chord_roots.append(i)
                
        self.templates: np.ndarray = np.array(templates_list)
        self.chord_priors: np.ndarray = np.array(priors_list)[:, np.newaxis]
        self.chord_roots_arr: np.ndarray = np.array(self.chord_roots) # (144,)

    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        signal = None
        has_bass = "target_bass" in signals

        # 1. 音源分離の「その他(ピアノ/ギター等)」をメインの和音検出用に使用
        if "target_other" in signals:
            print("[Strategy: Chord] 音源分離の 'target_other' を中心に伴奏解析を行います。")
            signal = signals["target_other"]
        elif "target_harmonic" in signals:
            signal = signals["target_harmonic"]
            print("[Strategy: Chord] 解析対象として 'target_harmonic' シグナルを採用しました。")
        elif "target" in signals:
            signal = signals["target"]
            print("[Strategy: Chord] 事前フィルターなしの 'target' シグナルを採用しました。")
            
        if signal is None or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print("[Strategy: Chord] ルートブースト＆Viterbiデコーディングを用いたコード進行を推定中...")
        
        sr = signal.sample_rate
        hop_length = 512
        
        # チューニング（A=440Hzからのズレ）を自動推定して補正
        tuning = librosa.estimate_tuning(y=signal.data, sr=sr)
        print(f"  - 推定ピッチズレ(Tuning): {tuning:+.2f} セント")
        
        # 1. クロマ特徴量の抽出 (CENSクロマとCQTクロマを合成してノイズ低減)
        chroma_cqt = librosa.feature.chroma_cqt(y=signal.data, sr=sr, hop_length=hop_length, tuning=tuning)
        chroma_cens = librosa.feature.chroma_cens(y=signal.data, sr=sr, hop_length=hop_length, tuning=tuning)
        chroma = 0.6 * chroma_cqt + 0.4 * chroma_cens
        
        # 2. ビート検出
        tempo, beat_frames = librosa.beat.beat_track(y=signal.data, sr=sr, hop_length=hop_length, start_bpm=100.0)
        if not isinstance(beat_frames, np.ndarray):
            beat_frames = np.array(beat_frames)
            
        beat_frames = librosa.util.fix_frames(beat_frames, x_min=0, x_max=chroma.shape[1])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        
        # 3. ビート同期
        chroma_sync = librosa.util.sync(chroma, beat_frames.tolist(), aggregate=np.median)
        
        # 4. 観測確率（Emission Probability）の計算
        chroma_norm = librosa.util.normalize(chroma_sync, norm=2, axis=0)
        similarities = np.dot(self.templates, chroma_norm) # (144, n_beats)
        
        # 5. 【ルート音ブースト】ベース音源がある場合、ベースの低音から得た音高をコードのルートに反映
        if has_bass:
            print("[Strategy: Chord] ベース音源 (target_bass) からルート音高を動的にブーストします。")
            y_bass = signals["target_bass"].data
            # 低域成分にフォーカスしたCQTクロマ
            chroma_bass_raw = librosa.feature.chroma_cqt(y=y_bass, sr=sr, hop_length=hop_length, tuning=tuning)
            chroma_bass_sync = librosa.util.sync(chroma_bass_raw, beat_frames.tolist(), aggregate=np.median)
            
            # 各ビートにおいて、ベースラインで顕著に強いトップ2の音高を検出してブースト
            n_beats = chroma_sync.shape[1]
            for b in range(n_beats):
                bass_vec = chroma_bass_sync[:, b]
                # トップ2音高のインデックス
                top_roots = np.argsort(bass_vec)[-2:]
                for root_idx in top_roots:
                    # その音高の強さに応じてブースト量を調整 (最大 3.0)
                    boost_val = 3.0 * (bass_vec[root_idx] / (bass_vec.max() + 1e-8))
                    # そのルートを持つコード (self.chord_roots_arr == root_idx) にブーストを加算
                    similarities[self.chord_roots_arr == root_idx, b] += boost_val

        scale_factor = 25.0
        
        # 事前確率(Prior)を対数(log)にしてから、類似度に足し合わせる
        log_priors = np.log(self.chord_priors)
        log_prob = (similarities * scale_factor) + log_priors
        
        # 数値のオーバーフローを防ぎつつ確率(Softmax)に変換
        prob = np.exp(log_prob - np.max(log_prob, axis=0))
        prob = prob / np.sum(prob, axis=0, keepdims=True)
        
        # 6. 遷移確率行列（Transition Matrix）の定義
        n_states = len(self.chord_names)
        
        # 自己遷移確率を高めに設定 (0.90) して、コード変化を滑らかに安定化
        p_self = 0.90
        p_other = (1.0 - p_self) / (n_states - 1)
        
        transition_matrix = np.full((n_states, n_states), p_other)
        np.fill_diagonal(transition_matrix, p_self)
        
        # 7. Viterbiデコーディング
        path = librosa.sequence.viterbi(prob, transition_matrix)
        raw_chords: List[str] = [self.chord_names[idx] for idx in path]
        
        # 8. 連続区間の圧縮
        chords_sequence: List[Dict[str, Any]] = []
        if len(raw_chords) > 0:
            current_chord = raw_chords[0]
            start_time = beat_times[0]
            for i in range(1, len(raw_chords)):
                if raw_chords[i] != current_chord:
                    chords_sequence.append({
                        "start_sec": round(float(start_time), 2),
                        "end_sec": round(float(beat_times[i]), 2),
                        "chord": current_chord
                    })
                    current_chord = raw_chords[i]
                    start_time = beat_times[i]
                    
            chords_sequence.append({
                "start_sec": round(float(start_time), 2),
                "end_sec": round(float(beat_times[-1]), 2),
                "chord": current_chord
            })
            
        return {
            "status": "success",
            "tempo": round(float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo), 1),
            "chords": chords_sequence
        }