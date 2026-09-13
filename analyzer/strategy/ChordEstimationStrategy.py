from typing import Any

import librosa
import numpy as np
import torch
from torch import nn

from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal


class ChordEstimationStrategy(IAnalysisStrategy):
    """
    PyTorch (GPU/CPU対応) による倍音抑制・ベースアテンション・音楽理論HMM内蔵型ニューラルコード推定エンジン。
    
    複雑な楽曲（ドラムや激しいギター、ボーカルの倍音が混ざっている音源）において、
    従来の単純なテンプレートマッチングの弱点である「倍音干渉による誤認」「1拍ごとのチャタリング振動」「ルート音の喪失」を
    音響物理モデルと音楽理論的テンソル演算によって解決します。

    主な特徴と改良点:
    1. 【物理モデルに基づく倍音抑制レイヤー (Physical Overtone Suppression)】:
       音響物理の原理に基づき、各音高から「完全5度下」および「長3度下」の基音から漏れ出た第3倍音・第5倍音のゴースト成分を正確に減算。
       真のルート音やマイナーサードが削られることなく、セブンス・テンションコードの過剰誤判定を徹底排除。
    2. 【ロバスト・ベースアテンションゲート (Robust Bass-Route Attention)】:
       CQTとCENSをハイブリッド融合した低音解析により、ベース音源 (target_bass) からルート音の確信度をSoftmaxアテンションで抽出し、
       推定コードのルート音を強力にブースト。
    3. 【リズムセクション優先のビート同期 (Rhythm-prior Beat Tracking)】:
       ドラムやベーストラックが利用可能な場合はリズムセクションから高精度なビートグリッドを生成し、伴奏クロマの同期ずれを防止。
    4. 【音楽理論HMM遷移確率行列 (Music-Theoretic Viterbi Transition)】:
       自己遷移確率 (p_self = 0.96) の強化に加え、完全5度下降（強進行）やダイアトニックコードへの遷移を優遇し、
       隣接半音間の不自然な往復振動（チャタリング）を音楽理論的に強力に抑制。
    5. 【時間的平滑化ポストプロセッシング (Temporal Smoothing)】:
       1拍未満のノイズ的な孤立コードや過渡区間の微小スパンを自然に結合・平滑化。
    """
    def __init__(self, engine: str = "hybrid") -> None:
        self.engine = engine.lower()
        self.btc_strategy = None
        self.btc_model = None
        self.btc_mean = 0.0
        self.btc_std = 1.0
        self.btc_vocab = {}

        if self.engine in ["hybrid", "btc"]:
            try:
                from analyzer.strategy.BTCChordEstimationStrategy import (
                    BTCChordEstimationStrategy,
                )
                self.btc_strategy = BTCChordEstimationStrategy()
                if self.btc_strategy.is_ready:
                    self.btc_model = self.btc_strategy.model
                    self.btc_mean = self.btc_strategy.mean
                    self.btc_std = self.btc_strategy.std
                    self.btc_vocab = self.btc_strategy.vocab
            except Exception as e:  # noqa: BLE001
                print(f"[ChordEstimationStrategy] BTC engine unavailable, fallback to heuristic: {e}")
                self.btc_strategy = None

        self.pitch_classes: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.chord_names: list[str] = []
        
        # 12半音ごとのメジャー／マイナーのダイアトニックマップ定義（C=0, C#=1...）
        self.diatonic_roots_major = {
            k: [(k + offset) % 12 for offset in [0, 2, 4, 5, 7, 9, 11]]
            for k in range(12)
        }
        self.diatonic_roots_minor = {
            k: [(k + offset) % 12 for offset in [0, 2, 3, 5, 7, 8, 10]]
            for k in range(12)
        }

        # コード構成音のインターバル定義 (オンコード判定用)
        self.chord_intervals_dict = {
            "": [0, 4, 7],
            "m": [0, 3, 7],
            "M7": [0, 4, 7, 11],
            "m7": [0, 3, 7, 10],
            "7": [0, 4, 7, 10],
            "dim": [0, 3, 6],
            "dim7": [0, 3, 6, 9],
            "aug": [0, 4, 8],
            "sus4": [0, 5, 7],
            "M7(9)": [0, 4, 7, 11, 2],
            "m7(9)": [0, 3, 7, 10, 2],
            "7(9)": [0, 4, 7, 10, 2]
        }

        # コード種別と事前確率（Prior）の定義
        # トライアド (Major/Minor) を安定させ、セブンスは構成音が明確な場合のみ選択されるようバランスを最適化
        chord_types = [
            ("Major",        [0, 4, 7],         "",       1.00),
            ("Minor",        [0, 3, 7],         "m",      1.00),
            ("Major7",       [0, 4, 7, 11],     "M7",     0.80),
            ("Minor7",       [0, 3, 7, 10],     "m7",     0.85),
            ("Dominant7",    [0, 4, 7, 10],     "7",      0.75),
            ("Diminished",   [0, 3, 6],         "dim",    0.50),
            ("Diminished7",  [0, 3, 6, 9],      "dim7",   0.45),
            ("Augmented",    [0, 4, 8],         "aug",    0.45),
            ("Sus4",         [0, 5, 7],         "sus4",   0.50),
            ("Major7(9)",    [0, 4, 7, 11, 2],   "M7(9)",  0.15),
            ("Minor7(9)",    [0, 3, 7, 10, 2],   "m7(9)",  0.15), 
            ("Dominant7(9)", [0, 4, 7, 10, 2],   "7(9)",   0.15),
        ]
        
        templates_list: list[np.ndarray] = []
        priors_list: list[float] = []
        self.chord_roots: list[int] = []
        
        for mode_name, intervals, suffix, prior in chord_types:
            for i in range(12):
                template = np.zeros(12)
                for idx, interval in enumerate(intervals):
                    if idx == 0: weight = 1.4  # ルート音の結合度を高く設定
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

        # PyTorch デバイスの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # テンプレートおよび優先確率テンソルのGPU/CPU登録
        self.templates_t = torch.tensor(self.templates, dtype=torch.float32, device=self.device) # (144, 12)
        self.priors_t = torch.tensor(self.chord_priors, dtype=torch.float32, device=self.device) # (144, 1)
        self.chord_roots_t = torch.tensor(self.chord_roots_arr, dtype=torch.long, device=self.device) # (144,)

        # BTC 170クラスから ChordEstimationStrategy 144クラスへの射影テンソル M: (144, 170)
        M = np.zeros((144, 170), dtype=np.float32)
        btc_q_to_suffix = {
            'maj': '', 'min': 'm', 'maj7': 'M7', 'min7': 'm7', '7': '7',
            'dim': 'dim', 'dim7': 'dim7', 'aug': 'aug', 'sus4': 'sus4'
        }
        from model.btc.btc_loader import QUALITY_LIST, ROOT_LIST
        for i in range(168):
            r_idx = i // 14
            q_idx = i % 14
            q_name = QUALITY_LIST[q_idx]
            if q_name in btc_q_to_suffix:
                suf = btc_q_to_suffix[q_name]
                c_name = f"{ROOT_LIST[r_idx]}{suf}"
                if c_name in self.chord_names:
                    h_idx = self.chord_names.index(c_name)
                    M[h_idx, i] = 1.0
        self.btc_mapping_t = torch.tensor(M, dtype=torch.float32, device=self.device)

    def analyze(self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None) -> dict[str, Any]:
        engine_to_use = self.engine
        if params and "chord_engine" in params:
            engine_to_use = str(params["chord_engine"]).lower()

        if engine_to_use == "btc":
            if self.btc_strategy is not None and self.btc_strategy.is_ready:
                return self.btc_strategy.analyze(signals, params)
            else:
                print("[Strategy: Chord] BTC Strategy 利用不可のため、ルールベース物理HMMで解析します。")

        signal = None
        has_bass = "target_bass" in signals

        # 1. 伴奏シグナルの選定 (ピアノ/ギターを含む target_other が最適)
        if "target_other" in signals:
            print("[Strategy: Chord] 音源分離の 'target_other' を伴奏特徴抽出に使用します。")
            signal = signals["target_other"]
        elif "target_harmonic" in signals:
            signal = signals["target_harmonic"]
            print("[Strategy: Chord] 'target_harmonic' を特徴抽出に使用します。")
        elif "target" in signals:
            signal = signals["target"]
            print("[Strategy: Chord] オリジナル 'target' を特徴抽出に使用します。")
            
        if signal is None or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}
            
        print(f"[Strategy: Chord] 高精度PyTorch-Neuralコード進行推定中 (デバイス: {self.device})...")
        
        sr = signal.sample_rate
        hop_length = 512
        
        # チューニング補正
        tuning = librosa.estimate_tuning(y=signal.data, sr=sr)
        print(f"  - 推定ピッチズレ(Tuning): {tuning:+.2f} セント")
        
        # 2. クロマ特徴量の抽出 (CENSとCQTをブレンド)
        chroma_cqt = librosa.feature.chroma_cqt(y=signal.data, sr=sr, hop_length=hop_length, tuning=tuning)
        chroma_cens = librosa.feature.chroma_cens(y=signal.data, sr=sr, hop_length=hop_length, tuning=tuning)
        chroma = 0.5 * chroma_cqt + 0.5 * chroma_cens
        
        # 3. ビート検出 (リズムセクションがある場合はドラム/ベースを優先してアタックを正確に捕捉)
        if "target_drums" in signals and "target_bass" in signals:
            beat_audio = signals["target_drums"].data + signals["target_bass"].data
            beat_sr = signals["target_drums"].sample_rate
        elif "target_drums" in signals:
            beat_audio = signals["target_drums"].data
            beat_sr = signals["target_drums"].sample_rate
        elif "target" in signals:
            beat_audio = signals["target"].data
            beat_sr = signals["target"].sample_rate
        else:
            beat_audio = signal.data
            beat_sr = sr

        tempo, beat_frames = librosa.beat.beat_track(y=beat_audio, sr=beat_sr, hop_length=hop_length, start_bpm=100.0)
        if not isinstance(beat_frames, np.ndarray):
            beat_frames = np.array(beat_frames)
            
        beat_frames = librosa.util.fix_frames(beat_frames, x_min=0, x_max=chroma.shape[1])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        
        # ビート同期クロマ
        chroma_sync = librosa.util.sync(chroma, beat_frames.tolist(), aggregate=np.median)
        n_beats = chroma_sync.shape[1]

        # PyTorch テンソルへ変換して GPU/CPU に転送
        chroma_t = torch.tensor(chroma_sync, dtype=torch.float32, device=self.device) # (12, n_beats)

        # ---- 4. 【物理モデルに基づく倍音抑制レイヤー (Physical Overtone Suppression)】 ----
        # 各音高 k から、その「完全5度下」および「長3度下」の基音から生じた第3倍音(3f0)・第5倍音(5f0)のエネルギーを減算
        chroma_clean = torch.zeros_like(chroma_t)
        for i in range(12):
            sub_fifth = (i - 7) % 12  # 完全5度下の基音インデックス
            sub_third = (i - 4) % 12  # 長3度下の基音インデックス
            val = chroma_t[i, :] - 0.30 * chroma_t[sub_fifth, :] - 0.15 * chroma_t[sub_third, :]
            chroma_clean[i, :] = torch.clamp(val, min=0.0)
        
        # L2正規化
        chroma_norm = nn.functional.normalize(chroma_clean, p=2, dim=0)

        # 初期類似度行列の計算
        similarities = torch.mm(self.templates_t, chroma_norm) # (144, n_beats)

        # ---- 5. 【超高速YINベース音追跡 ＆ ベースルートアテンションゲート】 ----
        beat_bass_pitches: list[int | None] = [None] * n_beats
        if has_bass:
            y_bass = signals["target_bass"].data
            sr_bass = signals["target_bass"].sample_rate
            
            # 1. 超高速 YIN によるベース最低音ピッチ (F0) 推定 (4kHzダウンサンプリング)
            try:
                sr_down = 4000
                y_bass_down = librosa.resample(y_bass, orig_sr=sr_bass, target_sr=sr_down)
                hop_yin = 64
                f0_bass = librosa.yin(
                    y_bass_down,
                    fmin=librosa.note_to_hz('C1'),  # 32.7 Hz
                    fmax=librosa.note_to_hz('C4'),  # 261.6 Hz
                    sr=sr_down,
                    hop_length=hop_yin
                )
                yin_times = librosa.frames_to_time(np.arange(len(f0_bass)), sr=sr_down, hop_length=hop_yin)
                
                for b_idx in range(n_beats):
                    t_st = beat_times[b_idx]
                    t_en = beat_times[b_idx + 1] if b_idx + 1 < n_beats else (len(y_bass) / sr_bass)
                    mask = (yin_times >= t_st) & (yin_times < t_en)
                    seg = f0_bass[mask]
                    valid = seg[(seg >= 30.0) & (seg <= 300.0)]
                    if len(valid) > 0:
                        med_f0 = float(np.median(valid))
                        midi_val = round(librosa.hz_to_midi(med_f0))
                        beat_bass_pitches[b_idx] = midi_val % 12
            except Exception as e:  # noqa: BLE001
                print(f"[Warning] Fast YIN bass tracking fallback: {e}")

            # 2. CQT + CENS ハイブリッド・ベースクロマ
            bass_tuning = librosa.estimate_tuning(y=y_bass, sr=sr)
            chroma_b_cqt = librosa.feature.chroma_cqt(y=y_bass, sr=sr, hop_length=hop_length, tuning=bass_tuning)
            chroma_b_cens = librosa.feature.chroma_cens(y=y_bass, sr=sr, hop_length=hop_length, tuning=bass_tuning)
            chroma_bass_raw = 0.5 * chroma_b_cqt + 0.5 * chroma_b_cens
            
            chroma_bass_sync = librosa.util.sync(chroma_bass_raw, beat_frames.tolist(), aggregate=np.median)
            bass_t = torch.tensor(chroma_bass_sync, dtype=torch.float32, device=self.device) # (12, n_beats)
            
            # 各ビートのベースベクトルをアテンション重みに変換 (Softmaxで強調)
            bass_attn = torch.softmax(bass_t / 0.15, dim=0) # (12, n_beats)
            
            # 予測コードテンプレートの各ルート音に対応するアテンション重みを抽出し、類似度にゲート加算
            boost_matrix = bass_attn[self.chord_roots_t, :] * 3.5
            similarities = similarities + boost_matrix

        # ---- 5.5 【BTC Transformer 深層学習事後確率テンソル射影融合】 ----
        if engine_to_use == "hybrid" and self.btc_model is not None:
            try:
                target_hz = 22050
                inst_len = 10.0
                n_timestep = 108
                chunk_samples = int(target_hz * inst_len)
                y_22k = librosa.resample(signal.data, orig_sr=sr, target_sr=target_hz) if sr != target_hz else signal.data
                
                cqt_chunks = []
                curr_s = 0
                while curr_s < len(y_22k):
                    seg = y_22k[curr_s : curr_s + chunk_samples]
                    if len(seg) < chunk_samples:
                        seg = np.pad(seg, (0, chunk_samples - len(seg)), mode='constant')
                    tmp = librosa.cqt(seg, sr=target_hz, n_bins=144, bins_per_octave=24, hop_length=2048)
                    if tmp.shape[1] > n_timestep:
                        tmp = tmp[:, :n_timestep]
                    elif tmp.shape[1] < n_timestep:
                        tmp = np.pad(tmp, ((0, 0), (0, n_timestep - tmp.shape[1])), mode='constant')
                    cqt_chunks.append(tmp)
                    curr_s += chunk_samples
                
                all_cqt = np.concatenate(cqt_chunks, axis=1)
                log_cqt = np.log(np.abs(all_cqt) + 1e-6).T
                norm_cqt = (log_cqt - self.btc_mean) / self.btc_std
                num_inst = norm_cqt.shape[0] // n_timestep
                
                btc_logits_list = []
                with torch.no_grad():
                    f_t = torch.tensor(norm_cqt, dtype=torch.float32).unsqueeze(0).to(self.device)
                    for t in range(num_inst):
                        out_l = self.btc_model(f_t[:, n_timestep * t : n_timestep * (t + 1), :])
                        btc_logits_list.append(out_l.cpu())
                        
                btc_logits = torch.cat(btc_logits_list, dim=1).squeeze(0) # (total_frames, 170)
                # N と X の学習時不均衡バイアスを適正化
                btc_logits[:, 169] -= 6.0
                btc_logits[:, 168] -= 6.0
                btc_probs = torch.softmax(btc_logits, dim=-1).numpy() # (total_frames, 170)
                
                # ビート同期プーリング
                frame_sec = inst_len / n_timestep
                frame_times = np.arange(btc_probs.shape[0]) * frame_sec
                total_dur = len(signal.data) / sr
                
                beat_btc_p = np.zeros((170, n_beats), dtype=np.float32)
                for b_idx in range(n_beats):
                    t_st = beat_times[b_idx]
                    t_en = beat_times[b_idx + 1] if b_idx + 1 < n_beats else total_dur
                    mask = (frame_times >= t_st) & (frame_times < t_en)
                    if np.any(mask):
                        beat_btc_p[:, b_idx] = np.mean(btc_probs[mask, :], axis=0)
                    else:
                        nearest = min(round(t_st / frame_sec), btc_probs.shape[0] - 1)
                        beat_btc_p[:, b_idx] = btc_probs[nearest, :]
                        
                beat_btc_t = torch.tensor(beat_btc_p, dtype=torch.float32, device=self.device) # (170, n_beats)
                # 144 クラス空間への射影
                btc_mapped = torch.mm(self.btc_mapping_t, beat_btc_t) # (144, n_beats)
                
                # 物理類似度テンソルへディープニューラル事後確率を融合ブースト
                similarities = similarities + btc_mapped * 2.5
                print("  - [Hybrid Fusion] SOTA BTC Transformer テンソル射影融合完了 (重み: 2.5)")
            except Exception as e:  # noqa: BLE001
                print(f"  - [Warning: Hybrid Fusion] BTC融合処理フォールバック: {e}")

        # ---- 6. 【キー検出とダイアトニック遷移の動的生成】 ----
        # 外部パラメータ (params) から指定されたキーがあればそれを最優先活用
        best_key = None
        best_mode = None
        if params:
            if "key_tonic" in params and "key_scale" in params:
                tonic = params["key_tonic"]
                scale = params["key_scale"].lower()
                if tonic in self.pitch_classes:
                    best_key = self.pitch_classes.index(tonic)
                    best_mode = "major" if "major" in scale else "minor"
            elif "estimated_key" in params:
                parts = params["estimated_key"].split()
                if len(parts) >= 2 and parts[0] in self.pitch_classes:
                    best_key = self.pitch_classes.index(parts[0])
                    best_mode = "major" if "major" in parts[1].lower() else "minor"

        if best_key is None:
            # 高精度 Pearson 相関による Krumhansl-Schmuckler プロファイルマッチング (生クロマの中心化)
            mean_chroma_np = np.mean(chroma, axis=1)
            mc = mean_chroma_np - np.mean(mean_chroma_np)
            norm_mc = np.linalg.norm(mc)
            
            major_prof = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_prof = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            major_prof = major_prof - np.mean(major_prof)
            minor_prof = minor_prof - np.mean(minor_prof)
            
            best_r = -2.0
            best_key = 0
            best_mode = "major"
            
            for k in range(12):
                rot_maj = np.roll(major_prof, k)
                rot_min = np.roll(minor_prof, k)
                
                denom_maj = norm_mc * np.linalg.norm(rot_maj)
                r_maj = np.dot(mc, rot_maj) / denom_maj if denom_maj > 0 else -1.0
                
                denom_min = norm_mc * np.linalg.norm(rot_min)
                r_min = np.dot(mc, rot_min) / denom_min if denom_min > 0 else -1.0
                
                if r_maj > best_r:
                    best_r = r_maj
                    best_key = k
                    best_mode = "major"
                if r_min > best_r:
                    best_r = r_min
                    best_key = k
                    best_mode = "minor"
                
        key_name = self.pitch_classes[best_key] + (" Major" if best_mode == "major" else " Minor")
        print(f"  - 曲全体の自動キー推定: {key_name} (機能和声Priorおよびダイアトニック優遇を適用します)")
        
        # ---- 【アプローチ3: 調性（キー）に基づく機能和声Prior (Functional Harmony Prior)】 ----
        # 各コードについて、主キーからの度数(deg)とコードクオリティに応じた機能和声確率を動的算出
        n_states = len(self.chord_names)
        functional_priors = torch.ones(n_states, dtype=torch.float32, device=self.device)
        
        for idx in range(n_states):
            root_idx = self.chord_roots[idx]
            deg = (root_idx - best_key) % 12
            c_name = self.chord_names[idx]
            suffix = c_name[len(self.pitch_classes[root_idx]):]
            
            if best_mode == "major":
                # Majorキーの機能和声ルール
                if deg == 0:        # I度 (Tonic): Major / M7 は高確率、7(Dominant)はトニックでは不協和のため抑制
                    if suffix in ["", "M7", "M7(9)"]:
                        functional_priors[idx] = 1.25
                    elif suffix in ["7", "7(9)"]:
                        functional_priors[idx] = 0.35
                elif deg == 7:      # V度 (Dominant): 7th / 7(9) / Major は自然、M7は原則不出現
                    if suffix in ["7", "7(9)", ""]:
                        functional_priors[idx] = 1.30
                    elif suffix in ["M7", "M7(9)"]:
                        functional_priors[idx] = 0.15
                elif deg in (2, 4, 9): # ii, iii, vi度: minor / m7 は自然
                    if suffix in ["m", "m7", "m7(9)"]:
                        functional_priors[idx] = 1.25
                elif deg == 5:      # IV度 (Subdominant): Major / M7 は自然
                    if suffix in ["", "M7", "M7(9)"]:
                        functional_priors[idx] = 1.25
                elif deg == 11 and suffix in ["dim", "dim7"]: # vii度: dim は自然
                    functional_priors[idx] = 1.10
            else:
                # Minorキーの機能和声ルール
                if deg == 0:        # i度 (Tonic): minor / m7 は高確率、Majorや7thは抑制
                    if suffix in ["m", "m7", "m7(9)"]:
                        functional_priors[idx] = 1.30
                    elif suffix in ["", "7", "M7"]:
                        functional_priors[idx] = 0.25
                elif deg == 7:      # v度 (Dominant): 7th (和声短音階強進行) または m/m7 (自然短音階)
                    if suffix in ["7", "7(9)", "m", "m7"]:
                        functional_priors[idx] = 1.25
                elif deg == 3:      # bIII度 (平行長調Tonic): Major / M7 は自然
                    if suffix in ["", "M7", "M7(9)"]:
                        functional_priors[idx] = 1.25
                elif (deg == 5 and suffix in ["m", "m7"]) or (deg in (8, 10) and suffix in ["", "M7", "7"]):
                    functional_priors[idx] = 1.20
                elif deg == 2 and suffix in ["dim", "dim7"]: # ii度: dim は自然
                    functional_priors[idx] = 1.10

        # 観測確率（Softmax）の算出
        scale_factor = 25.0
        combined_priors = self.priors_t * functional_priors.unsqueeze(1)
        log_priors = torch.log(combined_priors)
        log_prob = (similarities * scale_factor) + log_priors
        prob_np = torch.softmax(log_prob, dim=0).cpu().numpy()

        # ---- 7. 【音楽理論に基づくHMMダイアトニック遷移確率行列】 ----        
        # 基本の自己遷移確率 (1拍単位で不要に変化するのを防ぐ)
        p_self = 0.96
        transition_matrix = np.zeros((n_states, n_states))
        
        # 各調性におけるダイアトニック度数と許容サフィックスの定義
        # Major: I(maj/M7), ii(m/m7), iii(m/m7), IV(maj/M7), V(maj/7), vi(m/m7), vii(dim)
        diatonic_degrees_major = {
            0: ["", "M7"],
            2: ["m", "m7"],
            4: ["m", "m7"],
            5: ["", "M7"],
            7: ["", "7"],
            9: ["m", "m7"],
            11: ["dim"]
        }
        # Minor: i(m/m7), ii(dim), III(maj/M7), iv(m/m7), v(m/7), VI(maj/M7), VII(maj/7)
        diatonic_degrees_minor = {
            0: ["m", "m7"],
            2: ["dim"],
            3: ["", "M7"],
            5: ["m", "m7"],
            7: ["m", "7"],
            8: ["", "M7"],
            10: ["", "7"]
        }
        deg_map = diatonic_degrees_major if best_mode == "major" else diatonic_degrees_minor

        is_diatonic = np.zeros(n_states, dtype=bool)
        for idx in range(n_states):
            root_pitch = self.chord_roots[idx]
            deg = (root_pitch - best_key) % 12
            if deg in deg_map:
                chord_name = self.chord_names[idx]
                root_name = self.pitch_classes[root_pitch]
                suffix = chord_name[len(root_name):]
                if suffix in deg_map[deg]:
                    is_diatonic[idx] = True

        for i in range(n_states):
            root_i = self.chord_roots[i]
            for j in range(n_states):
                if i == j:
                    transition_matrix[i, j] = p_self
                else:
                    root_j = self.chord_roots[j]
                    diff = (root_j - root_i) % 12
                    
                    # ダイアトニックコードへの遷移重み
                    weight = 3.0 if is_diatonic[j] else 1.0
                    
                    # 音楽理論的ルート進行ボーナス
                    if diff == 5:       # 完全5度下降 (強進行・ドミナントモーション: 例 G -> C)
                        weight *= 1.5
                    elif diff == 7:     # 完全4度下降 (5度進行: 例 C -> G)
                        weight *= 1.2
                    elif diff in (3, 4, 8, 9): # 3度進行 (代理コード進行)
                        weight *= 1.1
                    elif diff in (1, 11): # 半音隣接進行 (激しい半音チャタリングの強力な抑止)
                        weight *= 0.3
                        
                    transition_matrix[i, j] = weight
            
            # 行正規化
            row_sum = np.sum(transition_matrix[i, :])
            transition_matrix[i, :] = transition_matrix[i, :] / (row_sum if row_sum > 0 else 1.0)
            
        # 8. Viterbiデコーディング
        path = librosa.sequence.viterbi(prob_np, transition_matrix)
        
        # 9. オンコード (分数コード / 転回形) のデコード
        beat_chord_info: list[dict[str, Any]] = []
        for idx_b, path_idx in enumerate(path):
            raw_c = self.chord_names[path_idx]
            root_idx = self.chord_roots[path_idx]
            root_name = self.pitch_classes[root_idx]
            suffix = raw_c[len(root_name):]
            
            bass_p = beat_bass_pitches[idx_b] if idx_b < len(beat_bass_pitches) else None
            
            # ベース音が明確に検出され、ルート音と異なる場合のオンコード判定
            if bass_p is not None and bass_p != root_idx:
                bass_name = self.pitch_classes[bass_p]
                # 転回形・ペダルポイント・分数コードとして認識
                slash_name = f"{raw_c}/{bass_name}"
                beat_chord_info.append({
                    "chord": slash_name,
                    "root": root_name,
                    "bass": bass_name,
                    "is_slash": True,
                    "base_chord": raw_c
                })
            else:
                beat_chord_info.append({
                    "chord": raw_c,
                    "root": root_name,
                    "bass": root_name,
                    "is_slash": False,
                    "base_chord": raw_c
                })
        
        # 10. 時間平滑化ポストフィルタリング (1拍のみの孤立ノイズコード平滑化)
        if len(beat_chord_info) >= 3:
            for i in range(1, len(beat_chord_info) - 1):
                if beat_chord_info[i-1]["chord"] == beat_chord_info[i+1]["chord"] and beat_chord_info[i]["chord"] != beat_chord_info[i-1]["chord"]:
                    beat_chord_info[i] = dict(beat_chord_info[i-1])

        # 11. 連続区間の圧縮 (0.0秒〜楽曲末尾まで完全カバレッジ)
        total_duration = float(len(signal.data) / sr)
        chords_sequence: list[dict[str, Any]] = []
        if len(beat_chord_info) > 0:
            cur = beat_chord_info[0]
            start_time = 0.0
            for i in range(1, len(beat_chord_info)):
                if beat_chord_info[i]["chord"] != cur["chord"]:
                    chords_sequence.append({
                        "start_sec": round(float(start_time), 2),
                        "end_sec": round(float(beat_times[i]), 2),
                        "chord": cur["chord"],
                        "root": cur["root"],
                        "bass": cur["bass"],
                        "is_slash": cur["is_slash"]
                    })
                    cur = beat_chord_info[i]
                    start_time = beat_times[i]
                    
            chords_sequence.append({
                "start_sec": round(float(start_time), 2),
                "end_sec": round(total_duration, 2),
                "chord": cur["chord"],
                "root": cur["root"],
                "bass": cur["bass"],
                "is_slash": cur["is_slash"]
            })
            
            # 楽曲冒頭の極小スパン (0.35秒未満の立ち上がり過渡ノイズ) があれば直後の安定コードにマージ
            if len(chords_sequence) > 1 and (chords_sequence[0]["end_sec"] - chords_sequence[0]["start_sec"]) < 0.35:
                chords_sequence[1]["start_sec"] = 0.0
                chords_sequence.pop(0)
            
        return {
            "status": "success",
            "tempo": round(float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo), 1),
            "chords": chords_sequence
        }