from typing import Any

import librosa
import numpy as np

from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal


class ChorusDetectionVocalStrategy(IAnalysisStrategy):
    """
    アプローチ2+4: Demucs分離済みボーカル信号のRMS + 適応的閾値によるサビ検出。
    - target_vocal の RMS をメイン特徴として活用（ボーカルが最も活発な区間 = サビ）
    - 閾値を固定乗数ではなく (mean + 0.5 * std) による適応閾値に変更
    - 結果は 'chorus_sections_vocal' キーで返す
    """

    def analyze(
        self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # ボーカルシグナルを優先採用
        vocal = signals.get("target_vocal") or signals.get("target")
        full = signals.get("target") or vocal

        if vocal is None or len(vocal.data) == 0:
            return {
                "status": "error",
                "message": "No audio signal available for vocal chorus detection.",
            }

        print("[Strategy: ChorusVocal] ボーカル信号 + 適応閾値でサビ区間を検出中...")

        sr = vocal.sample_rate
        y_vocal = vocal.data
        y_full = full.data if full is not None else y_vocal

        hop_length = 512
        frames_per_sec = int(sr / hop_length)
        n_seconds = int(len(y_vocal) / sr)

        if n_seconds < 5:
            return {
                "status": "success",
                "chorus_sections_vocal": [
                    {"start_sec": 0.0, "end_sec": round(vocal.duration_sec, 2)}
                ],
                "chorus_confidence_vocal": 0.5,
                "chorus_method_vocal": "vocal+adaptive_threshold",
            }

        # ---- 特徴量の計算 ----
        # 1. ボーカルRMS（分離済みなので純粋にボーカルの音量）
        vocal_rms = librosa.feature.rms(y=y_vocal, hop_length=hop_length)[0]

        # 2. 全体RMS（ボーカルだけでは静かなサビを見逃すことへの補完）
        full_rms = librosa.feature.rms(y=y_full, hop_length=hop_length)[0]

        # 3. スペクトル重心（明るさ）
        centroid = librosa.feature.spectral_centroid(
            y=y_full, sr=sr, hop_length=hop_length
        )[0]

        # 1秒ごとの平均に平滑化
        sec_vocal_rms, sec_full_rms, sec_centroid = [], [], []
        for s in range(n_seconds):
            sf, ef = s * frames_per_sec, (s + 1) * frames_per_sec
            if ef <= len(vocal_rms):
                sec_vocal_rms.append(np.mean(vocal_rms[sf:ef]))
                sec_full_rms.append(np.mean(full_rms[sf:ef]))
                sec_centroid.append(np.mean(centroid[sf:ef]))
            else:
                sec_vocal_rms.append(vocal_rms[-1])
                sec_full_rms.append(full_rms[-1])
                sec_centroid.append(centroid[-1])

        sec_vocal_rms = np.array(sec_vocal_rms)
        sec_full_rms = np.array(sec_full_rms)
        sec_centroid = np.array(sec_centroid)

        def normalize(arr: np.ndarray) -> np.ndarray:
            rng = np.max(arr) - np.min(arr)
            return (arr - np.min(arr)) / (rng if rng > 0 else 1.0)

        norm_vocal = normalize(sec_vocal_rms)
        norm_full = normalize(sec_full_rms)
        norm_centroid = normalize(sec_centroid)

        # ---- スコアリング: ボーカル重視（0.5）+ 全体音圧（0.3）+ 明るさ（0.2）----
        chorus_scores = 0.5 * norm_vocal + 0.3 * norm_full + 0.2 * norm_centroid

        # ---- 適応的閾値: mean + 0.5 * std ----
        threshold = np.mean(chorus_scores) + 0.5 * np.std(chorus_scores)
        is_candidate = chorus_scores > threshold

        sections = []
        in_sec, start_s = False, 0
        for s in range(n_seconds):
            if is_candidate[s] and not in_sec:
                in_sec, start_s = True, s
            elif not is_candidate[s] and in_sec:
                in_sec = False
                if s - start_s >= 3:
                    sections.append((start_s, s))
        if in_sec and n_seconds - start_s >= 3:
            sections.append((start_s, n_seconds))

        if not sections:
            best = int(np.argmax(chorus_scores))
            sections.append((max(0, best - 4), min(n_seconds, best + 4)))

        # 最高スコアセクションを選択
        sec_scores = [np.mean(chorus_scores[s:e]) for s, e in sections]
        best_idx = int(np.argmax(sec_scores))
        best_start, best_end = sections[best_idx]

        chorus_sections = [{"start_sec": float(best_start), "end_sec": float(best_end)}]

        # クロマ類似度で繰り返し区間も追加
        chroma = librosa.feature.chroma_cens(y=y_full, sr=sr, hop_length=hop_length)
        sec_chroma = []
        for s in range(n_seconds):
            sf, ef = s * frames_per_sec, (s + 1) * frames_per_sec
            sec_chroma.append(
                np.mean(chroma[:, sf:ef], axis=1)
                if ef <= chroma.shape[1]
                else chroma[:, -1]
            )
        sec_chroma = np.array(sec_chroma)

        ref = np.mean(sec_chroma[best_start:best_end], axis=0)
        ref_norm = np.linalg.norm(ref)
        span = best_end - best_start
        if ref_norm > 0:
            for s in range(n_seconds - span):
                if s + span <= best_start or s >= best_end:
                    test = np.mean(sec_chroma[s : s + span], axis=0)
                    t_norm = np.linalg.norm(test)
                    if t_norm > 0:
                        sim = np.dot(ref, test) / (ref_norm * t_norm)
                        if (
                            sim > 0.90
                            and np.mean(sec_vocal_rms[s : s + span])
                            > np.mean(sec_vocal_rms) * 0.7
                        ):
                            chorus_sections.append(
                                {"start_sec": float(s), "end_sec": float(s + span)}
                            )
                            break

        chorus_sections.sort(key=lambda x: x["start_sec"])
        return {
            "status": "success",
            "chorus_sections_vocal": chorus_sections,
            "chorus_confidence_vocal": round(float(sec_scores[best_idx]), 2),
            "chorus_method_vocal": "vocal+adaptive_threshold",
        }
