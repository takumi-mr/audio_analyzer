import unittest

import numpy as np

from analyzer.strategy.KeyDetectionStrategy import KeyDetectionStrategy
from model.AudioSignal import AudioSignal


class TestKeyDetectionTwoTier(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        self.duration = 2.0
        self.t = np.linspace(
            0, self.duration, int(self.sr * self.duration), endpoint=False
        )

    def _synth_tone(self, freq: float) -> np.ndarray:
        return 0.3 * np.sin(2 * np.pi * freq * self.t)

    def test_c_major_with_bass(self):
        """C Major の和声（C4, E4, G4）と C2 ベース音から C Major が正しく検出されるか"""
        treble_data = (
            self._synth_tone(261.63)
            + self._synth_tone(329.63)
            + self._synth_tone(392.00)
        )
        bass_data = self._synth_tone(65.41)

        signals = {
            "target_other": AudioSignal(treble_data, self.sr, self.duration),
            "target_bass": AudioSignal(bass_data, self.sr, self.duration),
        }

        strategy = KeyDetectionStrategy()
        result = strategy.analyze(signals)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["key_tonic"], "C")
        self.assertEqual(result["key_scale"], "Major")

    def test_a_minor_with_bass(self):
        """A Minor の和声（A3, C4, E4）と A1 ベース音から A Minor が正しく検出されるか"""
        treble_data = (
            self._synth_tone(220.00)
            + self._synth_tone(261.63)
            + self._synth_tone(329.63)
        )
        bass_data = self._synth_tone(55.00)

        signals = {
            "target_other": AudioSignal(treble_data, self.sr, self.duration),
            "target_bass": AudioSignal(bass_data, self.sr, self.duration),
        }

        strategy = KeyDetectionStrategy()
        result = strategy.analyze(signals)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["key_tonic"], "A")
        self.assertEqual(result["key_scale"], "Minor")

    def test_key_detection_with_chord_progression_c_major(self):
        """コード進行 (F -> G -> Em -> Am, G -> C) と連携して C Major が強固に判定されるか"""
        # 微弱なノイズ波形
        data = self._synth_tone(261.63)
        signals = {"target": AudioSignal(data, self.sr, self.duration)}

        # ポップス定番進行 (王道進行 + ドミナント終止)
        params = {
            "chords": [
                {"chord": "F", "start_sec": 0.0, "end_sec": 1.0},
                {"chord": "G", "start_sec": 1.0, "end_sec": 2.0},
                {"chord": "Em", "start_sec": 2.0, "end_sec": 3.0},
                {"chord": "Am", "start_sec": 3.0, "end_sec": 4.0},
                {"chord": "G7", "start_sec": 4.0, "end_sec": 5.0},
                {"chord": "C", "start_sec": 5.0, "end_sec": 6.0},
            ]
        }

        strategy = KeyDetectionStrategy()
        result = strategy.analyze(signals, params)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["key_tonic"], "C")
        self.assertEqual(result["key_scale"], "Major")

    def test_key_detection_with_chord_progression_a_minor(self):
        """コード進行 (Am -> Dm -> E7 -> Am) と連携して A Minor が強固に判定されるか"""
        data = self._synth_tone(220.00)
        signals = {"target": AudioSignal(data, self.sr, self.duration)}

        params = {
            "chords": [
                {"chord": "Am", "start_sec": 0.0, "end_sec": 2.0},
                {"chord": "Dm", "start_sec": 2.0, "end_sec": 4.0},
                {"chord": "E7", "start_sec": 4.0, "end_sec": 6.0},
                {"chord": "Am", "start_sec": 6.0, "end_sec": 8.0},
            ]
        }

        strategy = KeyDetectionStrategy()
        result = strategy.analyze(signals, params)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["key_tonic"], "A")
        self.assertEqual(result["key_scale"], "Minor")


if __name__ == "__main__":
    unittest.main()
