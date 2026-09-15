from typing import Any

import librosa
import numpy as np

from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal


class KeyDetectionStrategy(IAnalysisStrategy):
    """
    ベース成分の統合（target_bass）、低音/高音2階層クロマ評価、
    およびコード進行解析結果との音楽理論連動（Diatonic Fit / Cadence スコアリング）を用い、
    楽曲全体の主キー（調）を高精度に推定する戦略。
    """

    def __init__(self):
        # 12音階の名前の定義
        self.pitch_classes = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]

        # Krumhansl-Schmuckler キープロファイル定義 (安定した調性検出)
        self.major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        self.minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        # 平均を引いて正規化
        self.major_profile = self.major_profile - np.mean(self.major_profile)
        self.minor_profile = self.minor_profile - np.mean(self.minor_profile)

    def _parse_chord_name(self, chord_str: str) -> tuple[int, str] | None:
        """
        コード名文字列（例: 'C', 'G7', 'Am', 'F#m7', 'Bb', 'N'）から
        ルート音のピッチクラスインデックス (0~11) と 基本タイプ ('maj', 'min', 'dim') を返す。
        """
        if not chord_str or chord_str in ["N", "X"]:
            return None

        # スラッシュコード (オンコード: C/G など) の場合は分子コードを採用
        base_chord = chord_str.split("/")[0].strip()
        if not base_chord:
            return None

        enharmonics = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
        root_name = ""
        rest = ""

        if len(base_chord) >= 2 and base_chord[1] in ["#", "b"]:
            root_candidate = base_chord[:2]
            root_name = enharmonics.get(root_candidate, root_candidate)
            rest = base_chord[2:]
        else:
            root_name = base_chord[0]
            rest = base_chord[1:]

        if root_name not in self.pitch_classes:
            return None

        root_idx = self.pitch_classes.index(root_name)

        # 品質（Quality）の分類
        rest_lower = rest.lower()
        if "min" in rest_lower or (rest.startswith("m") and not rest.startswith("maj")):
            quality = "min"
        elif "dim" in rest_lower or "hdim" in rest_lower or "m7-5" in rest_lower:
            quality = "dim"
        elif "aug" in rest_lower:
            quality = "aug"
        else:
            quality = "maj"

        return root_idx, quality

    def _score_chords_for_keys(
        self, chords: list[dict[str, Any]]
    ) -> dict[tuple[int, str], float]:
        """
        推定されたコード進行列から各調 (12 Major + 12 Minor) の適合スコアを算出。
        Diatonic Fit（構成音・機能和声一致度）および Cadence（V->I などの終止形）を評価。
        """
        scores: dict[tuple[int, str], float] = {
            (i, mode): 0.0 for i in range(12) for mode in ["Major", "Minor"]
        }
        if not chords:
            return scores

        parsed_chords: list[tuple[int, str, float]] = []
        total_duration = 0.0

        for c_info in chords:
            chord_name = c_info.get("chord", "")
            duration = float(c_info.get("end_sec", 0.0) - c_info.get("start_sec", 0.0))
            if duration <= 0.0:
                duration = 1.0
            parsed = self._parse_chord_name(chord_name)
            if parsed is not None:
                parsed_chords.append((parsed[0], parsed[1], duration))
                total_duration += duration

        if not parsed_chords or total_duration <= 0.0:
            return scores

        # 1. ダイアトニック適合度（Diatonic Fit）の積算
        for root_idx, quality, dur in parsed_chords:
            weight_dur = dur / total_duration

            for tonic_idx in range(12):
                interval = (root_idx - tonic_idx) % 12

                # --- Major 調 ---
                if interval == 0:
                    score = 1.5 if quality == "maj" else -0.8
                elif interval == 2:
                    score = 1.0 if quality == "min" else 0.5
                elif interval == 4:
                    score = 0.9 if quality == "min" else 0.5
                elif interval == 5:
                    score = 1.3 if quality == "maj" else 0.3
                elif interval == 7:
                    score = 1.4 if quality == "maj" else -0.5
                elif interval == 9:
                    score = 1.2 if quality == "min" else 0.4
                elif interval == 11:
                    score = 0.8 if quality == "dim" else 0.2
                elif interval == 10:
                    score = 0.6 if quality == "maj" else 0.0
                else:
                    score = -0.4
                scores[(tonic_idx, "Major")] += score * weight_dur

                # --- Minor 調 ---
                if interval == 0:
                    score = 1.5 if quality == "min" else -0.8
                elif interval == 2:
                    score = 0.8 if quality == "dim" else 0.4
                elif interval == 3:
                    score = 1.2 if quality == "maj" else 0.2
                elif interval == 5:
                    score = 1.2 if quality == "min" else 0.2
                elif interval == 7:
                    score = 1.4
                elif interval == 8:
                    score = 1.1 if quality == "maj" else 0.3
                elif interval == 10:
                    score = 1.0 if quality == "maj" else 0.1
                elif interval == 11:
                    score = 0.8 if quality == "dim" else 0.2
                else:
                    score = -0.4
                scores[(tonic_idx, "Minor")] += score * weight_dur

        # 2. カデンツおよび定型コード進行（Cadence & Progression）のボーナス
        n_chords = len(parsed_chords)
        for idx in range(n_chords - 1):
            r1, q1, _ = parsed_chords[idx]
            r2, q2, _ = parsed_chords[idx + 1]

            for tonic_idx in range(12):
                int1 = (r1 - tonic_idx) % 12
                int2 = (r2 - tonic_idx) % 12

                # ドミナントモーション (V -> I / V -> i)
                if int1 == 7 and int2 == 0:
                    if q1 == "maj" and q2 == "maj":
                        scores[(tonic_idx, "Major")] += 0.4
                    if q2 == "min":
                        scores[(tonic_idx, "Minor")] += 0.4

                # サブドミナント終止 (IV -> I)
                if int1 == 5 and int2 == 0 and q1 == "maj" and q2 == "maj":
                    scores[(tonic_idx, "Major")] += 0.25

                # 王道進行 (IV -> V -> iii -> vi) の推進力 (IV -> V)
                if int1 == 5 and int2 == 7 and q1 == "maj" and q2 == "maj":
                    scores[(tonic_idx, "Major")] += 0.25

        return scores

    def analyze(
        self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # 1. 最適な和声シグナルの選定（低音、中高音、全体の3層）
        y_bass: np.ndarray | None = None
        y_treble: np.ndarray | None = None
        y_full: np.ndarray | None = None
        sr: int = 22050

        has_bass = "target_bass" in signals and len(signals["target_bass"].data) > 0
        has_other = "target_other" in signals and len(signals["target_other"].data) > 0
        has_vocal = "target_vocal" in signals and len(signals["target_vocal"].data) > 0

        if has_other and has_bass:
            print(
                "[Strategy: Key] 分離信号 ('target_bass', 'target_other', 'target_vocal') から低音と和声・旋律を統合してキーを推定します。"
            )
            y_bass = signals["target_bass"].data
            sr = signals["target_bass"].sample_rate
            if has_vocal:
                min_len_tr = min(
                    len(signals["target_other"].data), len(signals["target_vocal"].data)
                )
                y_treble = (
                    signals["target_other"].data[:min_len_tr]
                    + signals["target_vocal"].data[:min_len_tr]
                )
            else:
                y_treble = signals["target_other"].data

            min_len = min(len(y_bass), len(y_treble))
            y_full = y_treble[:min_len] + y_bass[:min_len]
        elif "target" in signals and len(signals["target"].data) > 0:
            print("[Strategy: Key] 解析対象として 'target' シグナルを採用しました。")
            y_full = signals["target"].data
            sr = signals["target"].sample_rate
            y_treble = y_full
        elif "target_harmonic" in signals and len(signals["target_harmonic"].data) > 0:
            print(
                "[Strategy: Key] 解析対象として 'target_harmonic' シグナルを採用しました。"
            )
            y_full = signals["target_harmonic"].data
            sr = signals["target_harmonic"].sample_rate
            y_treble = y_full
        elif signals:
            sig = next(iter(signals.values()))
            y_full = sig.data
            sr = sig.sample_rate
            y_treble = y_full
        else:
            return {"status": "error", "message": "No audio signal available."}

        if len(y_full) == 0:
            return {"status": "error", "message": "No audio signal available."}

        print(
            "[Strategy: Key] 低音クロマ(Bass)と高音クロマ(Treble)の2階層評価により主キーを推定中..."
        )

        # 2. ピッチズレ（チューニング）の推定と補正
        tuning = float(librosa.estimate_tuning(y=y_full, sr=sr))

        # 3. 2階層クロマ抽出 (Bass Chroma & Treble/Full Chroma)
        # (a) 低音クロマ (Bass Chroma: C1〜B3 / 約32Hz〜250Hz の基底帯域)
        target_bass_data = y_bass if y_bass is not None else y_full
        chroma_bass = librosa.feature.chroma_cqt(
            y=target_bass_data,
            sr=sr,
            tuning=tuning,
            fmin=float(librosa.note_to_hz("C1")),
            n_octaves=3,
        )
        mean_bass = np.mean(chroma_bass, axis=1)
        bass_sum = float(np.sum(mean_bass))
        if bass_sum > 0:
            norm_bass = mean_bass / bass_sum
        else:
            norm_bass = np.ones(12) / 12.0

        # (b) 中高音クロマ (Treble Chroma: C3以上でコードの3度音・テンションを鮮明に抽出)
        target_treble_data = y_treble if y_treble is not None else y_full
        chroma_cqt_tr = librosa.feature.chroma_cqt(
            y=target_treble_data,
            sr=sr,
            tuning=tuning,
            fmin=float(librosa.note_to_hz("C3")),
            n_octaves=5,
        )
        chroma_cens_tr = librosa.feature.chroma_cens(
            y=target_treble_data,
            sr=sr,
            tuning=tuning,
            fmin=float(librosa.note_to_hz("C3")),
            n_octaves=5,
        )
        mean_treble = 0.5 * np.mean(chroma_cqt_tr, axis=1) + 0.5 * np.mean(
            chroma_cens_tr, axis=1
        )
        norm_treble = mean_treble - np.mean(mean_treble)

        # (c) 全体クロマ (Full Chroma)
        chroma_full_cqt = librosa.feature.chroma_cqt(y=y_full, sr=sr, tuning=tuning)
        chroma_full_cens = librosa.feature.chroma_cens(y=y_full, sr=sr, tuning=tuning)
        mean_full = 0.5 * np.mean(chroma_full_cqt, axis=1) + 0.5 * np.mean(
            chroma_full_cens, axis=1
        )
        norm_full = mean_full - np.mean(mean_full)

        # 4. コード進行情報の抽出（連携されている場合）
        chord_theory_scores: dict[tuple[int, str], float] | None = None
        chords_input = None
        if params:
            if "chords" in params and isinstance(params["chords"], list):
                chords_input = params["chords"]
            elif "chord" in params and isinstance(params["chord"], list):
                chords_input = params["chord"]

        if chords_input:
            print(
                f"[Strategy: Key] コード進行 ({len(chords_input)} 区間) から音楽理論適合度 (Diatonic Fit / Cadence) を算出中..."
            )
            chord_theory_scores = self._score_chords_for_keys(chords_input)

        best_score = -999.0
        best_key = ""
        best_mode = "Major"

        # 5. 12音階 × 2モード (Major/Minor) のハイブリッドスコアリング
        for i in range(12):
            rotated_major = np.roll(self.major_profile, i)
            rotated_minor = np.roll(self.minor_profile, i)

            # (1) 中高音および全体の和声 KS 相関
            d_full_maj = float(
                np.linalg.norm(norm_full) * np.linalg.norm(rotated_major)
            )
            r_full_maj = (
                float(np.dot(norm_full, rotated_major) / d_full_maj)
                if d_full_maj > 0
                else 0.0
            )

            d_tr_maj = float(
                np.linalg.norm(norm_treble) * np.linalg.norm(rotated_major)
            )
            r_tr_maj = (
                float(np.dot(norm_treble, rotated_major) / d_tr_maj)
                if d_tr_maj > 0
                else 0.0
            )

            r_maj = 0.6 * r_full_maj + 0.4 * r_tr_maj

            d_full_min = float(
                np.linalg.norm(norm_full) * np.linalg.norm(rotated_minor)
            )
            r_full_min = (
                float(np.dot(norm_full, rotated_minor) / d_full_min)
                if d_full_min > 0
                else 0.0
            )

            d_tr_min = float(
                np.linalg.norm(norm_treble) * np.linalg.norm(rotated_minor)
            )
            r_tr_min = (
                float(np.dot(norm_treble, rotated_minor) / d_tr_min)
                if d_tr_min > 0
                else 0.0
            )

            r_min = 0.6 * r_full_min + 0.4 * r_tr_min

            # (2) 低音クロマによる主音（根音）・属音の支持度 (Bass Tonic Support)
            tonic_bass = float(norm_bass[i] * 12.0)
            dominant_bass = float(norm_bass[(i + 7) % 12] * 12.0)
            bass_support = 0.75 * (tonic_bass - 1.0) + 0.25 * (dominant_bass - 1.0)

            # (3) 総合スコア統合: 和声相関 + 低音主音支持
            score_maj = r_maj + 0.30 * bass_support
            score_min = r_min + 0.30 * bass_support

            # (4) コード進行理論スコアの統合（利用可能な場合）
            if chord_theory_scores:
                score_maj += 0.45 * chord_theory_scores.get((i, "Major"), 0.0)
                score_min += 0.45 * chord_theory_scores.get((i, "Minor"), 0.0)

            if score_maj > best_score:
                best_score = score_maj
                best_key = self.pitch_classes[i]
                best_mode = "Major"

            if score_min > best_score:
                best_score = score_min
                best_key = self.pitch_classes[i]
                best_mode = "Minor"

        # 信頼度：スコアを [0, 1] に正規化
        confidence = min(1.0, max(0.0, (best_score + 1.0) / 2.0))

        detected_key = f"{best_key} {best_mode}"
        print(f"  - 推定キー: {detected_key} (confidence: {round(confidence, 2)})")

        return {
            "status": "success",
            "estimated_key": detected_key,
            "key_tonic": best_key,
            "key_scale": best_mode,
            "confidence": round(float(confidence), 3),
        }
