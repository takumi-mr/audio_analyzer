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
        # 和声: C4 (261.63), E4 (329.63), G4 (392.00)
        treble_data = (
            self._synth_tone(261.63)
            + self._synth_tone(329.63)
            + self._synth_tone(392.00)
        )
        # ベース: C2 (65.41 Hz)
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
        # 和声: A3 (220.00), C4 (261.63), E4 (329.63)
        treble_data = (
            self._synth_tone(220.00)
            + self._synth_tone(261.63)
            + self._synth_tone(329.63)
        )
        # ベース: A1 (55.00 Hz)
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


if __name__ == "__main__":
    unittest.main()
