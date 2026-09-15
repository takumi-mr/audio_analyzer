from typing import Any

import librosa
import numpy as np

from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal


class ChorusDetectionSSMStrategy(IAnalysisStrategy):
    """
    アプローチ1 (改): Self-Similarity Matrix (SSM) ＋ ラグ変換によるサビ検出。
    librosa.segment.recurrence_matrix + lag_to_recurrence を用いて楽曲全体の
    繰り返しパターン（サビらしい区間）をグローバルに検出する。

    調整ポイント:
    - width=5: 対角線近傍（自己自身との類似）をより広く除去
    - 閾値: mean + 0.15σ (以前の 0.3σ から緩和 → 短すぎる検出を改善)
    - マージギャップ: 4秒 (以前の 2秒 → 区間の分断を防ぐ)
    - 最小セグメント長: 2秒 (以前の 3秒 → 頭・尾の切れを防ぐ)
    - ラグ行列: SSMのラグドメインで繰り返し強度を算出し繰り返し区間を強調
    - 結果は 'chorus_sections_ssm' キーで返す
    """

    def analyze(
        self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # 全体シグナルを優先（構造解析はミックス全体で行う）
        signal = signals.get("target") or signals.get("target_vocal")
        if signal is None or len(signal.data) == 0:
            return {
                "status": "error",
                "message": "No audio signal available for SSM chorus detection.",
            }

        print("[Strategy: ChorusSSM] 自己類似行列(SSM) + ラグ変換で楽曲構造を解析中...")

        try:
            sr = signal.sample_rate
            y = signal.data
            hop_length = 512
            frames_per_sec = int(sr / hop_length)
            n_seconds = int(len(y) / sr)

            if n_seconds < 8:
                return {
                    "status": "success",
                    "chorus_sections_ssm": [
                        {"start_sec": 0.0, "end_sec": round(signal.duration_sec, 2)}
                    ],
                    "chorus_confidence_ssm": 0.5,
                    "chorus_method_ssm": "ssm_segmentation",
                }

            # ---- 1. 特徴量行列の構築（クロマ + MFCC を連結） ----
            chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

            # 正規化してから連結
            chroma_norm = librosa.util.normalize(chroma, axis=0)
            mfcc_norm = librosa.util.normalize(mfcc, axis=0)
            features = np.vstack([chroma_norm, mfcc_norm])  # (25, T)

            # 1秒ごとの特徴量ベクトルに変換
            sec_features = []
            for s in range(n_seconds):
                sf, ef = s * frames_per_sec, (s + 1) * frames_per_sec
                if ef <= features.shape[1]:
                    sec_features.append(np.mean(features[:, sf:ef], axis=1))
                else:
                    sec_features.append(features[:, -1])
            sec_features = np.array(sec_features).T  # (25, n_seconds)

            # ---- 2. Self-Similarity Matrix (SSM) の構築 ----
            # width=5: より広い対角線帯を除去して「隣接が似ている」だけの秒を弾く
            R = librosa.segment.recurrence_matrix(
                sec_features, mode="affinity", sym=True, width=5
            )

            # sparse → dense 変換
            R_dense = (
                np.asarray(R.todense()) if hasattr(R, "todense") else np.asarray(R)
            )

            # ---- 3. ラグ行列による繰り返し強度の強調 ----
            # SSM をラグドメインに変換すると、一定の「繰り返し間隔」を持つ区間が
            # 強いパターンとして現れる（サビが同じ間隔で繰り返すことを利用）
            try:
                L = librosa.segment.recurrence_to_lag(R_dense, pad=False)
                # ラグ方向の最大類似度（最も類似する繰り返しを代表値として使用）
                lag_score = np.max(L, axis=0)  # (n_seconds,)
                # ラグ行列が使えない場合（短い曲等）は repeat_score のみ使用
            except Exception:
                lag_score = np.mean(R_dense, axis=1).flatten()

            # 各行の平均類似度（全区間との類似の平均）
            repeat_score = np.mean(R_dense, axis=1).flatten()

            # lag_score と repeat_score を統合（どちらも 0-1 レンジに正規化）
            def normalize(arr: np.ndarray) -> np.ndarray:
                rng = np.max(arr) - np.min(arr)
                return (arr - np.min(arr)) / (rng if rng > 0 else 1.0)

            norm_lag = normalize(lag_score)
            norm_repeat = normalize(repeat_score)

            # ---- 4. 全体音圧でエネルギーフィルタリング ----
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            sec_rms = []
            for s in range(n_seconds):
                sf, ef = s * frames_per_sec, (s + 1) * frames_per_sec
                sec_rms.append(np.mean(rms[sf:ef]) if ef <= len(rms) else rms[-1])
            sec_rms = np.array(sec_rms)
            norm_rms = normalize(sec_rms)

            # ---- 5. スコア統合: ラグ繰り返し（0.45）+ 平均繰り返し（0.25）+ 音圧（0.30）----
            chorus_scores = 0.45 * norm_lag + 0.25 * norm_repeat + 0.30 * norm_rms

            # ---- 6. 適応的閾値でセグメント抽出 ----
            # 0.15σ: 以前の 0.3σ から緩和して短すぎる検出を改善
            threshold = np.mean(chorus_scores) + 0.15 * np.std(chorus_scores)
            is_candidate = chorus_scores > threshold

            sections = []
            in_sec, start_s = False, 0
            for s in range(n_seconds):
                if is_candidate[s] and not in_sec:
                    in_sec, start_s = True, s
                elif not is_candidate[s] and in_sec:
                    in_sec = False
                    # 最小セグメント長: 2秒（以前の3秒から緩和）
                    if s - start_s >= 2:
                        sections.append((start_s, s))
            if in_sec and n_seconds - start_s >= 2:
                sections.append((start_s, n_seconds))

            if not sections:
                best = int(np.argmax(chorus_scores))
                sections.append((max(0, best - 5), min(n_seconds, best + 5)))

            # ---- 7. 隣接する区間をマージ（ギャップ 4秒以内）----
            # 以前の 2秒から 4秒に拡大し、区間の不自然な分断を防ぐ
            merged: list[list[int]] = []
            for seg in sorted(sections):
                if merged and seg[0] - merged[-1][1] <= 4:
                    merged[-1] = [merged[-1][0], seg[1]]
                else:
                    merged.append(list(seg))

            # スコア最大のセクションを特定
            sec_scores = [float(np.mean(chorus_scores[s:e])) for s, e in merged]
            best_idx = int(np.argmax(sec_scores))

            chorus_sections = [
                {"start_sec": float(s), "end_sec": float(e)} for s, e in merged
            ]
            chorus_sections.sort(key=lambda x: x["start_sec"])

            return {
                "status": "success",
                "chorus_sections_ssm": chorus_sections,
                "chorus_confidence_ssm": round(sec_scores[best_idx], 3),
                "chorus_method_ssm": "ssm_lag_segmentation",
            }

        except Exception as e:
            print(
                f"[Warning] SSM Chorus detection failed: {e}. Falling back to empty result."
            )
            return {
                "status": "success",
                "chorus_sections_ssm": [],
                "chorus_confidence_ssm": 0.0,
                "chorus_method_ssm": "ssm_lag_segmentation_fallback",
            }
