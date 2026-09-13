import os
import math
from typing import Dict, Any, List, Optional
import numpy as np
import librosa
import torch

from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal
from model.btc.btc_loader import load_btc_model

class BTCChordEstimationStrategy(IAnalysisStrategy):
    """
    SOTA Bi-directional Transformer (BTC) と音響物理 YIN ベース追跡による
    ハイブリッド深層学習コード進行推定エンジン。

    特徴:
    1. 【BTC Bi-directional Transformer (ISMIR 2019)】:
       前後文脈のアテンションにより 170 クラス（Major, Minor, 7, maj7, min7, dim, aug, sus4, sus2, 6, mM7, m7b5 等）
       のテンション・セブンスコードを高精度に識別。
    2. 【ビート同期プーリング (Beat-synchronous Pooling)】:
       フレーム単位（約0.093s）の事後確率分布を楽曲のビート単位に統合し、微小振動・チャタリングを完全解消。
    3. 【超高速 YIN 最低音F0追跡 & オンコード合成】:
       ベース音源から 4kHz 超高速 YIN により拍ごとの最低音ピッチを抽出し、
       BTC の和音トライアドと統合してオンコード（分数コード `C/E`, `G/B` 等）を自動合成。
    4. 【堅牢なフォールバック設計】:
       モデルロードや推論に万一問題が生じた場合は、自動で高精度物理モデル（ChordEstimationStrategy）へフォールバック。
    """
    def __init__(self, device: Optional[torch.device] = None, use_gpu: bool = True) -> None:
        self.device = device or torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.mean = 0.0
        self.std = 1.0
        self.vocab = {}
        self.is_ready = False
        self.pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # コード構成音のインターバル定義 (オンコード判定用)
        self.chord_intervals_dict = {
            "": [0, 4, 7],
            "m": [0, 3, 7],
            "M7": [0, 4, 7, 11],
            "m7": [0, 3, 7, 10],
            "7": [0, 4, 7, 10],
            "dim": [0, 3, 6],
            "dim7": [0, 3, 6, 9],
            "m7b5": [0, 3, 6, 10],
            "aug": [0, 4, 8],
            "sus4": [0, 5, 7],
            "sus2": [0, 2, 7],
            "6": [0, 4, 7, 9],
            "m6": [0, 3, 7, 9],
            "mM7": [0, 3, 7, 11]
        }

        self._init_model()

    def _init_model(self) -> None:
        try:
            self.model, self.mean, self.std, self.vocab = load_btc_model(
                device=self.device, voca_large=True
            )
            self.is_ready = True
        except Exception as e:
            print(f"[Warning: BTCChord] BTC モデルの初期化に失敗しました。フォールバックを使用します: {e}")
            self.is_ready = False

    def analyze(self, signals: Dict[str, AudioSignal], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if not self.is_ready or self.model is None:
            print("[Strategy: BTCChord] BTCモデルが利用できないため、物理ルールベースHMMにフォールバックします。")
            from analyzer.strategy.ChordEstimationStrategy import ChordEstimationStrategy
            fallback_strat = ChordEstimationStrategy(engine="heuristic")
            return fallback_strat.analyze(signals, params)

        # 1. 解析対象シグナルの選定
        # 伴奏楽器（ピアノ、ギター等）が分離された target_other が最適
        signal = None
        has_bass = "target_bass" in signals
        if "target_other" in signals:
            print("[Strategy: BTCChord] 音源分離 'target_other' を伴奏特徴抽出に使用します。")
            signal = signals["target_other"]
        elif "target_harmonic" in signals:
            signal = signals["target_harmonic"]
            print("[Strategy: BTCChord] 'target_harmonic' を特徴抽出に使用します。")
        elif "target" in signals:
            signal = signals["target"]
            print("[Strategy: BTCChord] オリジナル 'target' を特徴抽出に使用します。")

        if signal is None or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}

        print(f"[Strategy: BTCChord] SOTA BTC Transformer による深層学習コード進行推定中 (デバイス: {self.device})...")

        orig_sr = signal.sample_rate
        y_orig = signal.data

        # 2. リズムセクションによる高精度ビート検出
        hop_length_beat = 512
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
            beat_audio = y_orig
            beat_sr = orig_sr

        tempo, beat_frames = librosa.beat.beat_track(
            y=beat_audio, sr=beat_sr, hop_length=hop_length_beat, start_bpm=100.0
        )
        if not isinstance(beat_frames, np.ndarray):
            beat_frames = np.array(beat_frames)
        beat_times = librosa.frames_to_time(beat_frames, sr=beat_sr, hop_length=hop_length_beat)
        n_beats = len(beat_times)

        # 3. BTC 特徴量抽出 (22,050Hz, 10秒チャンク CQT: 144 bins / 24 bins/oct / hop 2048)
        target_hz = 22050
        inst_len = 10.0
        n_bins = 144
        bins_per_octave = 24
        hop_cqt = 2048
        n_timestep = 108

        # 22,050Hz へのリサンプリング
        if orig_sr != target_hz:
            y_22k = librosa.resample(y_orig, orig_sr=orig_sr, target_sr=target_hz)
        else:
            y_22k = y_orig

        total_duration = float(len(y_orig) / orig_sr)
        chunk_samples = int(target_hz * inst_len)

        cqt_chunks = []
        curr_sample = 0
        while curr_sample < len(y_22k):
            seg = y_22k[curr_sample : curr_sample + chunk_samples]
            if len(seg) < chunk_samples:
                # 最終チャンクをゼロパディングして10秒長にする
                seg = np.pad(seg, (0, chunk_samples - len(seg)), mode='constant')

            tmp = librosa.cqt(
                seg, sr=target_hz, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_cqt
            )
            # 各10秒チャンクは正確に 108 フレームに切り詰める/整える
            if tmp.shape[1] > n_timestep:
                tmp = tmp[:, :n_timestep]
            elif tmp.shape[1] < n_timestep:
                tmp = np.pad(tmp, ((0, 0), (0, n_timestep - tmp.shape[1])), mode='constant')

            cqt_chunks.append(tmp)
            curr_sample += chunk_samples

        all_cqt = np.concatenate(cqt_chunks, axis=1) # (144, num_chunks * 108)
        log_cqt = np.log(np.abs(all_cqt) + 1e-6).T # (total_frames, 144)
        norm_cqt = (log_cqt - self.mean) / self.std

        num_instances = norm_cqt.shape[0] // n_timestep
        frame_sec = inst_len / n_timestep # 約 0.09259 秒

        # 4. BTC Transformer によるフレーム単位推論
        logits_list = []
        with torch.no_grad():
            feat_t = torch.tensor(norm_cqt, dtype=torch.float32).unsqueeze(0).to(self.device)
            for t in range(num_instances):
                chunk_inp = feat_t[:, n_timestep * t : n_timestep * (t + 1), :]
                out_logits = self.model(chunk_inp) # (1, 108, 170)
                logits_list.append(out_logits.cpu())

        all_logits = torch.cat(logits_list, dim=1).squeeze(0) # (total_frames, 170)

        # N (No chord, idx 169) と X (Unknown, idx 168) の学習時不均衡バイアスを適正化
        # 楽曲が明確に演奏されている区間で過剰な N 判定を抑制し、テンション・和音の事後確率を正常化
        all_logits_calib = all_logits.clone()
        all_logits_calib[:, 169] -= 6.0
        all_logits_calib[:, 168] -= 6.0

        probs_all = torch.softmax(all_logits_calib, dim=-1).numpy() # (total_frames, 170)
        total_frames = probs_all.shape[0]
        frame_timestamps = np.arange(total_frames) * frame_sec

        # 5. 【超高速 YIN ベース最低音F0追跡】(オンコード判定用)
        beat_bass_pitches: List[Optional[int]] = [None] * n_beats
        if has_bass:
            y_bass = signals["target_bass"].data
            sr_bass = signals["target_bass"].sample_rate
            try:
                sr_down = 4000
                y_bass_down = librosa.resample(y_bass, orig_sr=sr_bass, target_sr=sr_down)
                hop_yin = 64
                f0_bass = librosa.yin(
                    y_bass_down,
                    fmin=float(librosa.note_to_hz('C1')), # 32.7 Hz
                    fmax=float(librosa.note_to_hz('C4')), # 261.6 Hz
                    sr=sr_down,
                    hop_length=hop_yin
                )
                yin_times = librosa.frames_to_time(np.arange(len(f0_bass)), sr=sr_down, hop_length=hop_yin)

                for b_idx in range(n_beats):
                    t_st = beat_times[b_idx]
                    t_en = beat_times[b_idx + 1] if b_idx + 1 < n_beats else total_duration
                    mask = (yin_times >= t_st) & (yin_times < t_en)
                    seg = f0_bass[mask]
                    valid = seg[(seg >= 30.0) & (seg <= 300.0)]
                    if len(valid) > 0:
                        med_f0 = float(np.median(valid))
                        midi_val = int(round(librosa.hz_to_midi(med_f0)))
                        beat_bass_pitches[b_idx] = midi_val % 12
            except Exception as e:
                print(f"[Warning: BTCChord] Fast YIN bass tracking fallback: {e}")

        # 6. 【ビート同期プーリング (Beat-synchronous Pooling)】
        # 各ビート区間のフレーム事後確率を平均し、最も確信度の高いコードを選択
        beat_chord_info: List[Dict[str, Any]] = []

        for b_idx in range(n_beats):
            t_st = beat_times[b_idx]
            t_en = beat_times[b_idx + 1] if b_idx + 1 < n_beats else total_duration

            mask = (frame_timestamps >= t_st) & (frame_timestamps < t_en)
            if np.any(mask):
                beat_probs = np.mean(probs_all[mask, :], axis=0) # (170,)
            else:
                # 最も近いフレームを採用
                nearest_idx = min(int(round(t_st / frame_sec)), total_frames - 1)
                beat_probs = probs_all[nearest_idx, :]

            # N (No Chord) の確率が高すぎる場合でも、音楽が鳴っていれば第2候補を採用
            best_idx = int(np.argmax(beat_probs))
            if best_idx == 169: # N
                sorted_indices = np.argsort(beat_probs)[::-1]
                # 有意なコード（トップ2が確信度15%以上）があれば採用
                if beat_probs[sorted_indices[1]] > 0.12:
                    best_idx = int(sorted_indices[1])

            chord_meta = self.vocab.get(best_idx, {"chord": "N", "root": None, "suffix": "", "triad": "none"})
            raw_chord = chord_meta["chord"]
            root_name = chord_meta.get("root")
            root_pitch = chord_meta.get("root_pitch")
            suffix = chord_meta.get("suffix", "")

            # ベース音との照合によるオンコード（分数コード）合成
            bass_p = beat_bass_pitches[b_idx] if b_idx < len(beat_bass_pitches) else None

            if raw_chord not in ["N", "X"] and root_pitch is not None and bass_p is not None and bass_p != root_pitch:
                bass_name = self.pitch_classes[bass_p]
                slash_name = f"{raw_chord}/{bass_name}"
                beat_chord_info.append({
                    "chord": slash_name,
                    "root": root_name,
                    "bass": bass_name,
                    "is_slash": True,
                    "base_chord": raw_chord
                })
            else:
                beat_chord_info.append({
                    "chord": raw_chord,
                    "root": root_name or "N",
                    "bass": root_name or "N",
                    "is_slash": False,
                    "base_chord": raw_chord
                })

        # 7. 時間平滑化ポストプロセッシング (1拍のみの孤立ノイズ除去)
        if len(beat_chord_info) >= 3:
            for i in range(1, len(beat_chord_info) - 1):
                prev_c = beat_chord_info[i - 1]["chord"]
                next_c = beat_chord_info[i + 1]["chord"]
                curr_c = beat_chord_info[i]["chord"]
                if prev_c == next_c and curr_c != prev_c:
                    beat_chord_info[i] = dict(beat_chord_info[i - 1])

        # 8. 連続区間の圧縮 (0.0秒から末尾まで完全カバレッジ)
        chords_sequence: List[Dict[str, Any]] = []
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

        # もし最初が "N" で曲全体が埋まっている場合の安全フォールバック
        all_n = all(c["chord"] in ["N", "X"] for c in chords_sequence)
        if all_n and len(signals) > 0:
            print("[Warning: BTCChord] BTC出力が全てNとなったため、物理ルールベースHMMにフォールバックします。")
            from analyzer.strategy.ChordEstimationStrategy import ChordEstimationStrategy
            fallback_strat = ChordEstimationStrategy(engine="heuristic")
            return fallback_strat.analyze(signals, params)

        return {
            "status": "success",
            "tempo": round(float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo), 1),
            "chords": chords_sequence
        }
