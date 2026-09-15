import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from analyzer.filter.SourceSeparatorFilter import SourceSeparatorFilter
from analyzer.strategy.KeyDetectionStrategy import KeyDetectionStrategy
from model.AudioSignal import AudioSignal


class TestMIRPhase1(unittest.TestCase):
    def setUp(self):
        # 1秒分のテスト波形 (C Majorの構成音: C4=261.63Hz, E4=329.63Hz, G4=392.00Hz の重畳信号)
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

    @patch("demucs.api.Separator")
    def test_demucs_separator_filter(self, mock_separator_cls):
        # demucs の API 呼び出しのモック
        mock_separator = MagicMock()
        mock_separator_cls.return_value = mock_separator
        mock_separator.samplerate = 44100

        # モックの separate_tensor 戻り値
        import torch

        dummy_tensor = torch.zeros(2, 44100)  # 1秒分
        mock_separated = {
            "vocals": dummy_tensor,
            "drums": dummy_tensor,
            "bass": dummy_tensor,
            "other": dummy_tensor,
        }
        mock_separator.separate_tensor.return_value = (None, mock_separated)

        filter_sep = SourceSeparatorFilter(target_key="target")
        signals = {"target": self.signal}

        result = filter_sep.apply(signals)

        # 分離結果の検証
        self.assertIn("target_vocal", result)
        self.assertIn("target_drums", result)
        self.assertIn("target_bass", result)
        self.assertIn("target_other", result)

    def test_key_detection_strategy(self):
        strategy = KeyDetectionStrategy()
        signals = {"target": self.signal}

        result = strategy.analyze(signals)

        self.assertEqual(result["status"], "success")
        self.assertIn("estimated_key", result)
        self.assertIn("key_tonic", result)
        self.assertIn("key_scale", result)
        self.assertIn("confidence", result)

        self.assertEqual(result["key_tonic"], "C")
        self.assertEqual(result["key_scale"], "Major")


if __name__ == "__main__":
    unittest.main()
