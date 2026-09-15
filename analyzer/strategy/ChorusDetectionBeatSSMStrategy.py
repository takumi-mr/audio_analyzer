from typing import Any

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal

from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal


class ChorusDetectionBeatSSMStrategy(IAnalysisStrategy):
    """
    ビート同期SSMと対角パス強調(Path Enhancement)を用いた最新のサビ検出戦略 (改良版)。
    「CENSクロマ＋MFCCによる楽曲の繰り返し構造」と、「ボーカル存在度＋リズム存在度＋高音域の明るさ(Spectral Centroid)」
    を融合させてサビ区間を頑健に特定します。

    主な改良点:
    1. 特徴量をchroma_cqtからchroma_censへ変更（音量変化にロバストにし、和音変化にフォーカス）
    2. 音色の高音域の明るさを示すスペクトル重心(Spectral Centroid)を存在感スコアに追加
    3. インスト曲（ボーカルなし）を検知した場合にボーカルウェイトを自動的に排除してリズム・音色を重視する適応型Salience
    4. 対角パス強調窓を16拍(約4小節)に調整し、境界部の減衰影響を低減
    5. スコア統合を掛け算から「重み付き足し算」へ変更し、厳しい足切りを回避して検出数をSSM Structureと同等まで向上
    6. 閾値を0.15σに引き下げ、最小継続拍数を8拍に緩和し、隣接マージの許容ギャップを6秒に拡大
    """

    def analyze(
        self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # 1. 必要なシグナルを安全に取得
        if not signals:
            return {"status": "error", "message": "No audio signal available."}

        target_sig = (
            signals.get("target")
            or signals.get("target_vocal")
            or next(iter(signals.values()), None)
        )
        if target_sig is None or len(target_sig.data) == 0:
            return {"status": "error", "message": "No audio signal available."}

        y_vocal = (
            signals["target_vocal"].data
            if "target_vocal" in signals
            else target_sig.data
        )
        y_drums = (
            signals["target_drums"].data
            if "target_drums" in signals
            else np.zeros_like(y_vocal)
        )
        y_bass = (
            signals["target_bass"].data
            if "target_bass" in signals
            else np.zeros_like(y_vocal)
        )
        y_other = (
            signals["target_other"].data
            if "target_other" in signals
            else np.zeros_like(y_vocal)
        )

        if "target" in signals:
            y_full = signals["target"].data
            sr = signals["target"].sample_rate
        elif "target_vocal" in signals:
            y_full = y_vocal + y_drums + y_bass + y_other
            sr = signals["target_vocal"].sample_rate
        else:
            y_full = target_sig.data
            sr = target_sig.sample_rate

        print(
            "[Strategy: Chorus] ビート同期SSMと対角パス強調を用いたサビ検出を開始します..."
        )

        try:
            # 2. ビートトラッキング
            y_rhythm = y_drums + y_bass if "target_drums" in signals else y_full
            onset_env = librosa.onset.onset_strength(y=y_rhythm, sr=sr)
            _, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, start_bpm=100.0
            )

            if not isinstance(beat_frames, np.ndarray):
                beat_frames = np.array(beat_frames)
            beat_frames = librosa.util.fix_frames(beat_frames, x_min=0)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            n_beats = len(beat_frames)

            # 曲が極端に短すぎる（8拍未満）場合のダイナミクス適応フォールバック
            if n_beats < 8:
                print(
                    f"[Strategy: Chorus] 拍数が極めて少ないため ({n_beats} 拍)、ダイナミクス適応フォールバックでサビを特定します。"
                )
                fallback_sections = self._fallback_dynamics_chorus(
                    y_full, y_vocal, y_rhythm, sr
                )
                return {
                    "status": "success",
                    "chorus_sections_beat_ssm": [
                        {
                            "start_sec": round(s["start_sec"], 2),
                            "end_sec": round(s["end_sec"], 2),
                        }
                        for s in fallback_sections
                    ],
                    "chorus_confidence_beat_ssm": 0.75,
                    "chorus_method_beat_ssm": "dynamics_adaptive_fallback",
                }

            # 3. 特徴量抽出とビート同期 (chroma_cens + mfcc)
            chroma = librosa.feature.chroma_cens(y=y_other + y_vocal, sr=sr)
            mfcc = librosa.feature.mfcc(y=y_full, sr=sr, n_mfcc=13)

            chroma_sync = librosa.util.sync(
                chroma, beat_frames.tolist(), aggregate=np.median
            )
            mfcc_sync = librosa.util.sync(
                mfcc, beat_frames.tolist(), aggregate=np.median
            )

            # 特徴量を結合して正規化 (和声12次元 + 音色13次元の結合空間で最高コントラストを維持)
            X = np.vstack(
                [
                    librosa.util.normalize(chroma_sync, axis=0),
                    librosa.util.normalize(mfcc_sync, axis=0),
                ]
            )

            # 4. 自己類似行列 (SSM) の構築
            rec_width = min(8, max(2, n_beats // 4))
            R = librosa.segment.recurrence_matrix(
                X, mode="affinity", metric="cosine", sym=True, width=rec_width
            )

            # 5. 対角パス強調 (Path Enhancement)
            # 展開の周期に最適化されたHann窓強調
            w = min(16, max(2, n_beats // 2))
            R_enh = librosa.segment.path_enhance(R, w, window="hann")

            # 各ビートの繰り返しスコア
            rep_score = np.max(R_enh, axis=1)

            # 6. 存在感スコア (Salience) の計算
            # ボーカル音圧
            vocal_rms = librosa.feature.rms(y=y_vocal)[0]
            vocal_sync = librosa.util.sync(
                vocal_rms.reshape(1, -1), beat_frames.tolist(), aggregate=np.max
            )[0]

            # リズム音圧
            rhythm_rms = librosa.feature.rms(y=y_rhythm)[0]
            rhythm_sync = librosa.util.sync(
                rhythm_rms.reshape(1, -1), beat_frames.tolist(), aggregate=np.max
            )[0]

            # 伴奏シンセ/リード音圧 (インスト曲やダンス曲の主旋律・コードリフ検出に重要)
            other_rms = librosa.feature.rms(y=y_other)[0]
            other_sync = librosa.util.sync(
                other_rms.reshape(1, -1), beat_frames.tolist(), aggregate=np.median
            )[0]

            # 明るさ (スペクトル重心)
            centroid = librosa.feature.spectral_centroid(y=y_full, sr=sr)[0]
            centroid_sync = librosa.util.sync(
                centroid.reshape(1, -1), beat_frames.tolist(), aggregate=np.median
            )[0]

            # 全体音圧
            full_rms = librosa.feature.rms(y=y_full)[0]
            full_sync = librosa.util.sync(
                full_rms.reshape(1, -1), beat_frames.tolist(), aggregate=np.max
            )[0]

            # 【アプローチ1: サブベース (<80Hz) 急上昇＆ドロップイン検知】
            S_full = np.abs(librosa.stft(y_full))
            freqs = librosa.fft_frequencies(sr=sr)
            sub_energy = np.sum(S_full[freqs <= 80.0, :], axis=0)
            sub_sync = librosa.util.sync(
                sub_energy.reshape(1, -1), beat_frames.tolist(), aggregate=np.median
            )[0]

            # 0-1に正規化するヘルパー関数
            def normalize(arr: np.ndarray) -> np.ndarray:
                rng = arr.max() - arr.min()
                return (arr - arr.min()) / (rng if rng > 0 else 1.0)

            rep_score = normalize(rep_score)
            vocal_sync = normalize(vocal_sync)
            rhythm_sync = normalize(rhythm_sync)
            other_sync = normalize(other_sync)
            centroid_sync = normalize(centroid_sync)
            full_sync = normalize(full_sync)
            sub_sync = normalize(sub_sync)

            # サブベースの局所微分（急上昇コントラスト / ドロップイン）
            sub_diff = np.diff(sub_sync, prepend=sub_sync[0])
            sub_drop = normalize(sub_sync + 0.6 * np.clip(sub_diff, 0.0, None))

            # 7. 適応型 Salience (インスト/ボーカル曲の自動切り替え)
            vocal_ratio = np.mean(vocal_rms) / (np.mean(full_rms) + 1e-8)
            if vocal_ratio < 0.08:
                # インスト曲: 伴奏シンセ/リード(0.35) + サブベース(0.25) + 全体音圧(0.20) + リズム(0.10) + 明るさ(0.10)
                salience = (
                    0.35 * other_sync
                    + 0.25 * sub_drop
                    + 0.20 * full_sync
                    + 0.10 * rhythm_sync
                    + 0.10 * centroid_sync
                )
                print(
                    f"[Strategy: Chorus] インスト曲と判定しました。 (Vocal ratio: {vocal_ratio:.3f})"
                )
                chorus_score = 0.40 * rep_score + 0.60 * salience
            else:
                # ボーカルあり曲: ボーカル(0.40) + 全体音圧(0.25) + サブベース(0.15) + リズム(0.10) + 明るさ(0.10)
                salience = (
                    0.40 * vocal_sync
                    + 0.25 * full_sync
                    + 0.15 * sub_drop
                    + 0.10 * rhythm_sync
                    + 0.10 * centroid_sync
                )
                chorus_score = 0.50 * rep_score + 0.50 * salience

            # 短いノイズを消すためにメディアンフィルタで平滑化
            filter_size = min(8, max(3, n_beats // 4))
            chorus_score_smooth = scipy.ndimage.median_filter(
                chorus_score, size=filter_size
            )

            # 8. サビ区間の抽出
            threshold = np.mean(chorus_score_smooth) + 0.15 * np.std(
                chorus_score_smooth
            )
            is_chorus = chorus_score_smooth > threshold

            sections = []
            in_sec = False
            start_b = 0

            # サビの最小継続拍数: 楽曲長に応じて動的調整
            min_beats_for_chorus = 6 if n_beats >= 24 else max(3, n_beats // 5)

            for b in range(n_beats):
                if is_chorus[b] and not in_sec:
                    in_sec = True
                    start_b = b
                elif not is_chorus[b] and in_sec:
                    in_sec = False
                    if b - start_b >= min_beats_for_chorus:
                        sections.append(
                            {
                                "start_sec": float(beat_times[start_b]),
                                "end_sec": float(beat_times[b]),
                            }
                        )

            if in_sec and n_beats - start_b >= min_beats_for_chorus:
                sections.append(
                    {
                        "start_sec": float(beat_times[start_b]),
                        "end_sec": float(beat_times[-1]),
                    }
                )

            # 冒頭イントロ補正: 0秒付近から始まっていて、途中で伴奏や反復の急上昇がある場合は真のサビ開始位置へスナップ
            if sections and sections[0]["start_sec"] <= 0.5:
                search_limit = min(8, n_beats // 2)
                diff_curve = np.diff(
                    chorus_score_smooth[:search_limit], prepend=chorus_score_smooth[0]
                )
                max_rise_b = int(np.argmax(diff_curve))
                if max_rise_b >= 2 and chorus_score_smooth[max_rise_b] > threshold:
                    sections[0]["start_sec"] = float(beat_times[max_rise_b])

            # セクションが完全に空の場合のみダイナミクス適応フォールバック
            if not sections:
                sections = self._fallback_dynamics_chorus(y_full, y_vocal, y_rhythm, sr)

            # 9. 隣接する区間のマージ (マージギャップを5.0秒に設定)
            merged_sections = []
            for sec in sections:
                if (
                    merged_sections
                    and sec["start_sec"] - merged_sections[-1]["end_sec"] <= 5.0
                ):
                    merged_sections[-1]["end_sec"] = sec["end_sec"]
                else:
                    merged_sections.append(sec)

            # 10. Foote Novelty Checkerboard Kernel によるセクション・小節境界スナップ (BPM連動適応窓)
            final_sections = self._snap_to_section_boundaries(
                merged_sections, beat_times, X
            )

            return {
                "status": "success",
                "chorus_sections_beat_ssm": [
                    {
                        "start_sec": round(s["start_sec"], 2),
                        "end_sec": round(s["end_sec"], 2),
                    }
                    for s in final_sections
                ],
                "chorus_confidence_beat_ssm": round(
                    float(np.max(chorus_score_smooth)), 2
                ),
                "chorus_method_beat_ssm": "beat_sync_path_enhanced",
            }

        except Exception as e:
            print(
                f"[Warning] BeatSSM Chorus detection failed: {e}. ダイナミクス適応フォールバックを実行します。"
            )
            try:
                fallback_sections = self._fallback_dynamics_chorus(
                    y_full, y_vocal, y_rhythm, sr
                )
                return {
                    "status": "success",
                    "chorus_sections_beat_ssm": [
                        {
                            "start_sec": round(s["start_sec"], 2),
                            "end_sec": round(s["end_sec"], 2),
                        }
                        for s in fallback_sections
                    ],
                    "chorus_confidence_beat_ssm": 0.65,
                    "chorus_method_beat_ssm": "dynamics_adaptive_fallback",
                }
            except Exception:
                return {
                    "status": "success",
                    "chorus_sections_beat_ssm": [],
                    "chorus_confidence_beat_ssm": 0.0,
                    "chorus_method_beat_ssm": "beat_sync_path_enhanced_fallback",
                }

    def _fallback_dynamics_chorus(
        self, y_full: np.ndarray, y_vocal: np.ndarray, y_rhythm: np.ndarray, sr: int
    ) -> list[dict[str, float]]:
        """
        短尺音源（拍数が少ない）または構造解析が困難な場合の高精度ダイナミクスサビ検出フォールバック。
        音圧 (RMS)、スペクトル重心 (Centroid)、リズム音圧の時系列から最も盛り上がる区間（サビ/ドロップ）を抽出。
        """
        hop_length = 512
        full_rms = librosa.feature.rms(y=y_full, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(
            y=y_full, sr=sr, hop_length=hop_length
        )[0]
        vocal_rms = librosa.feature.rms(y=y_vocal, hop_length=hop_length)[0]
        rhythm_rms = librosa.feature.rms(y=y_rhythm, hop_length=hop_length)[0]

        def norm_arr(arr):
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / (rng if rng > 0 else 1.0)

        n_frames = len(full_rms)
        total_duration = len(y_full) / sr

        # ボーカル存在比率
        vocal_ratio = np.mean(vocal_rms) / (np.mean(full_rms) + 1e-8)
        if vocal_ratio < 0.08:
            dynamics = (
                0.50 * norm_arr(full_rms)
                + 0.30 * norm_arr(rhythm_rms)
                + 0.20 * norm_arr(centroid)
            )
        else:
            dynamics = (
                0.40 * norm_arr(vocal_rms)
                + 0.35 * norm_arr(full_rms)
                + 0.25 * norm_arr(centroid)
            )

        # 約1秒の平滑化 (約43フレーム)
        win_size = max(5, int(sr / hop_length))
        smooth_dyn = scipy.ndimage.gaussian_filter1d(dynamics, sigma=win_size / 2)

        # 閾値: 平均値 + 0.15 * 標準偏差
        thresh = np.mean(smooth_dyn) + 0.15 * np.std(smooth_dyn)
        is_high = smooth_dyn > thresh

        # 連続区間
        times = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=hop_length
        )
        sections = []
        in_sec = False
        start_t = 0.0

        min_dur = min(2.0, total_duration * 0.15)

        for f in range(n_frames):
            if is_high[f] and not in_sec:
                in_sec = True
                start_t = times[f]
            elif not is_high[f] and in_sec:
                in_sec = False
                end_t = times[f]
                if end_t - start_t >= min_dur:
                    sections.append(
                        {"start_sec": float(start_t), "end_sec": float(end_t)}
                    )

        if in_sec and total_duration - start_t >= min_dur:
            sections.append(
                {"start_sec": float(start_t), "end_sec": float(total_duration)}
            )

        # 1つも区間が得られない場合は、最大ピークを中心とした区間
        if not sections:
            peak_idx = int(np.argmax(smooth_dyn))
            peak_t = times[peak_idx]
            span = min(total_duration * 0.4, 8.0)
            st = max(0.0, peak_t - span / 2)
            et = min(total_duration, peak_t + span / 2)
            sections.append({"start_sec": float(st), "end_sec": float(et)})

        # マージ
        merged = []
        for sec in sections:
            if merged and sec["start_sec"] - merged[-1]["end_sec"] <= 2.0:
                merged[-1]["end_sec"] = sec["end_sec"]
            else:
                merged.append(sec)

        return merged

    def _compute_foote_novelty(self, X: np.ndarray, L: int = 4) -> np.ndarray:
        """
        2D Gaussian-tapered checkerboard kernel (Foote 2000) をビート同期自己類似行列 (SSM) の対角線上に畳み込み、
        楽曲のセクション境界（イントロ/サビ/Aメロ/アウトロ等）を示す Foote Novelty カーブを算出します。
        """
        N = X.shape[1]
        if N < 2 * L:
            L = max(1, N // 4)
        if L < 1:
            return np.zeros(N)

        norms = np.linalg.norm(X, axis=0, keepdims=True) + 1e-8
        X_norm = X / norms
        S = np.dot(X_norm.T, X_norm)

        t = np.arange(-L, L)
        i, j = np.meshgrid(t, t, indexing="ij")
        sign_mat = np.where((i < 0) == (j < 0), 1.0, -1.0)
        sigma = max(1.0, L / 2.0)
        gauss = np.exp(-(i**2 + j**2) / (2.0 * sigma**2))
        kernel = sign_mat * gauss
        pos_mask = kernel > 0
        neg_mask = kernel < 0
        if np.any(pos_mask) and np.sum(kernel[pos_mask]) > 0:
            kernel[pos_mask] /= np.sum(kernel[pos_mask])
        if np.any(neg_mask) and np.abs(np.sum(kernel[neg_mask])) > 0:
            kernel[neg_mask] /= np.abs(np.sum(kernel[neg_mask]))

        S_pad = np.pad(S, L, mode="edge")
        nov = np.zeros(N)
        for n in range(N):
            nov[n] = np.sum(S_pad[n : n + 2 * L, n : n + 2 * L] * kernel)
        nov = np.clip(nov, 0.0, None)
        ptp = np.ptp(nov)
        if ptp > 0:
            nov = (nov - np.min(nov)) / ptp
        return nov

    def _snap_to_section_boundaries(
        self,
        sections: list[dict[str, float]],
        beat_times: np.ndarray,
        X: np.ndarray,
        max_tol: float | None = None,
    ) -> list[dict[str, float]]:
        """
        Foote Novelty カーブのピーク（セクション境界）および音楽的小節境界（4拍周期）へ
        サビ区間の start_sec / end_sec を吸着（スナップ）させて境界ジッターを解消します。
        許容窓 max_tol が未指定の場合は、楽曲の拍間隔（テンポ）に応じた BPM連動型適応窓を動的算出します。
        """
        if not sections or len(beat_times) < 2:
            return sections

        n_beats = len(beat_times)

        # BPM連動型適応スナップ許容窓の計算 (1拍の長さに応じてスケール)
        if max_tol is None:
            beat_intervals = np.diff(beat_times)
            median_beat_dur = (
                float(np.median(beat_intervals)) if len(beat_intervals) > 0 else 0.5
            )
            # 1拍の約0.85倍（最小0.35秒〜最大1.2秒にリミット）
            effective_tol = float(np.clip(0.85 * median_beat_dur, 0.35, 1.2))
        else:
            effective_tol = max_tol

        L = min(4, max(2, n_beats // 8))
        nov = self._compute_foote_novelty(X, L=L)
        pks, _ = scipy.signal.find_peaks(nov, prominence=0.10, distance=2)
        foote_boundaries = [float(beat_times[p]) for p in pks if p < len(beat_times)]

        # 4拍（1小節）周期の候補
        bar_boundaries = [float(beat_times[b]) for b in range(0, n_beats, 4)]
        if n_beats > 5:
            bar_boundaries += [float(beat_times[b]) for b in range(1, n_beats, 4)]

        candidate_boundaries = sorted(list(set(foote_boundaries + bar_boundaries)))

        def snap(t_sec: float) -> float:
            best = t_sec
            min_diff = effective_tol
            for b in candidate_boundaries:
                d = abs(t_sec - b)
                if d < min_diff:
                    min_diff = d
                    best = b
            return best

        snapped = []
        for sec in sections:
            st = snap(sec["start_sec"])
            et = snap(sec["end_sec"])
            if et > st + 1.0:
                snapped.append({"start_sec": st, "end_sec": et})
            else:
                snapped.append(sec)
        return snapped
