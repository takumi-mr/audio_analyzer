import unittest

import numpy as np
import torch

from analyzer.strategy.BTCChordEstimationStrategy import BTCChordEstimationStrategy
from analyzer.strategy.ChordEstimationStrategy import ChordEstimationStrategy
from model.AudioSignal import AudioSignal
from model.btc.btc_loader import get_btc_vocabulary
from model.btc.btc_model import BTC_model


class TestBTCChordRecognition(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        self.duration = 4.0
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        # C Major トライアド (C4=261.63Hz, E4=329.63Hz, G4=392.00Hz)
        y = (
            0.3 * np.sin(2 * np.pi * 261.63 * t)
            + 0.3 * np.sin(2 * np.pi * 329.63 * t)
            + 0.3 * np.sin(2 * np.pi * 392.00 * t)
        )
        self.signal = AudioSignal(
            data=y, sample_rate=self.sr, duration_sec=self.duration
        )

    def test_btc_vocabulary(self):
        vocab = get_btc_vocabulary()
        self.assertEqual(len(vocab), 170)
        self.assertEqual(vocab[169]["chord"], "N")
        self.assertEqual(vocab[168]["chord"], "X")

        # Cメジャー (root=0, q=1 -> C)
        self.assertEqual(vocab[1]["chord"], "C")
        self.assertEqual(vocab[1]["triad"], "maj")

        # Cマイナー (root=0, q=0 -> Cm)
        self.assertEqual(vocab[0]["chord"], "Cm")
        self.assertEqual(vocab[0]["triad"], "min")

        # Cセブンス (root=0, q=9 -> C7)
        self.assertEqual(vocab[9]["chord"], "C7")

        # Cメジャーセブンス (root=0, q=8 -> CM7)
        self.assertEqual(vocab[8]["chord"], "CM7")

    def test_btc_model_forward(self):
        cfg = {
            "feature_size": 144,
            "timestep": 108,
            "num_chords": 170,
            "input_dropout": 0.0,
            "layer_dropout": 0.0,
            "attention_dropout": 0.0,
            "relu_dropout": 0.0,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_size": 32,
            "total_key_depth": 32,
            "total_value_depth": 32,
            "filter_size": 32,
            "probs_out": True,
        }
        model = BTC_model(cfg)
        model.eval()
        dummy_inp = torch.randn(1, 108, 144)
        with torch.no_grad():
            out = model(dummy_inp)
        self.assertEqual(out.shape, (1, 108, 170))

    def test_btc_strategy_analyze(self):
        strat = BTCChordEstimationStrategy()
        signals = {"target": self.signal}
        res = strat.analyze(signals)
        self.assertEqual(res["status"], "success")
        self.assertIn("chords", res)
        self.assertGreater(len(res["chords"]), 0)
        first_c = res["chords"][0]
        self.assertIn("chord", first_c)
        self.assertIn("root", first_c)
        self.assertIn("bass", first_c)
        self.assertIn("is_slash", first_c)

    def test_chord_strategy_engine_switch(self):
        # hybrid モード
        strat_hybrid = ChordEstimationStrategy(engine="hybrid")
        signals = {"target": self.signal}
        res_hybrid = strat_hybrid.analyze(signals)
        self.assertEqual(res_hybrid["status"], "success")

        # heuristic モード
        strat_heuristic = ChordEstimationStrategy(engine="heuristic")
        res_heuristic = strat_heuristic.analyze(signals)
        self.assertEqual(res_heuristic["status"], "success")


if __name__ == "__main__":
    unittest.main()
