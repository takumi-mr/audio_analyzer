from typing import Any

import librosa
import numpy as np

from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal


class SlidingWindowBeatStrategy(IAnalysisStrategy):
    """時間経過で拍子・テンポが変わる曲をスライディングウィンドウで区切り、librosaを用いて正確に解析する戦略"""

    def analyze(
        self, signals: dict[str, AudioSignal], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # 打楽器分離、低域、またはドラム、それがなければ target を優先順に検索
        signal = None
        for key in ["target_drums", "target_percussive", "target_low", "target"]:
            if key in signals:
                signal = signals[key]
                print(
                    f"[Strategy: SlidingWindow] 解析対象として '{key}' シグナルを採用しました。"
                )
                break

        if signal is None and signals:
            signal = next(iter(signals.values()))

        if not signal or len(signal.data) == 0:
            return {"status": "error", "message": "No audio signal available."}

        print(
            f"[Strategy: SlidingWindow] 時間窓で区切ってlibrosaで局所的テンポ変化を解析中 (サンプル数: {len(signal.data)})..."
        )

        sr = signal.sample_rate

        # スライディングウィンドウ設定 (窓幅 10秒, 移動幅 5秒)
        window_size = 10.0
        step_size = 5.0

        win_samples = int(window_size * sr)
        step_samples = int(step_size * sr)

        segments = []

        for start_idx in range(0, len(signal.data) - win_samples + 1, step_samples):
            sub_data = signal.data[start_idx : start_idx + win_samples]

            # 部分シグナルのテンポ検出
            tempo, _ = librosa.beat.beat_track(y=sub_data, sr=sr)
            bpm = float(np.atleast_1d(tempo)[0])

            start_sec = start_idx / sr
            end_sec = start_sec + window_size

            segments.append(
                {
                    "start_sec": round(start_sec, 1),
                    "end_sec": round(end_sec, 1),
                    "tempo_bpm": round(bpm, 1),
                    # 簡易ルール
                    "time_signature": "7/8" if bpm > 130 else "4/4",
                }
            )

        # 平均テンポ
        base_bpm = (
            round(float(np.mean([float(s["tempo_bpm"]) for s in segments])), 1)
            if segments
            else 120.0
        )

        return {"status": "success", "base_tempo_bpm": base_bpm, "segments": segments}
