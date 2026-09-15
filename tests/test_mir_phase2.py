import unittest

import numpy as np

from analyzer.strategy.ChordEstimationStrategy import ChordEstimationStrategy
from model.AudioSignal import AudioSignal


class TestMIRPhase2(unittest.TestCase):
    def setUp(self):
        # 1秒分の C Major 和音 (C4=261.63Hz, E4=329.63Hz, G4=392.00Hz)
        self.sr = 22050
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.data = (
            0.3 * np.sin(2 * np.pi * 261.63 * t)
            + 0.3 * np.sin(2 * np.pi * 329.63 * t)
            + 0.3 * np.sin(2 * np.pi * 392.00 * t)
        )
        self.signal = AudioSignal(
            data=self.data, sample_rate=self.sr, duration_sec=self.duration
        )

    def test_chord_estimation_c_major(self):
        strategy = ChordEstimationStrategy()
        signals = {"target": self.signal}

        result = strategy.analyze(signals)

        self.assertEqual(result["status"], "success")
        self.assertIn("chords", result)
        self.assertGreater(len(result["chords"]), 0)

        # 最初のコード判定が "C" であることを期待
        first_span = result["chords"][0]
        self.assertEqual(first_span["chord"], "C")
        self.assertIn("start_sec", first_span)
        self.assertIn("end_sec", first_span)

        # タイムスパンの整合性
        self.assertEqual(first_span["start_sec"], 0.0)
        self.assertAlmostEqual(first_span["end_sec"], self.duration, delta=0.1)


if __name__ == "__main__":
    unittest.main()
