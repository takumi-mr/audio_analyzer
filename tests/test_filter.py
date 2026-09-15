import unittest

import numpy as np

from analyzer.filter.BandSplitterFilter import BandSplitterFilter
from analyzer.filter.CompressorFilter import CompressorFilter
from analyzer.filter.SourceSeparatorFilter import SourceSeparatorFilter
from model.AudioSignal import AudioSignal


class TestAudioFilters(unittest.TestCase):
    def setUp(self):
        # 1秒分のテスト波形 (50Hz と 1000Hz の重畳信号)
        self.sr = 8000  # テスト用に低めのサンプリングレート
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        # 低音 50Hz (振幅 0.5) + 高音 1000Hz (振幅 0.5)
        self.data = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(
            2 * np.pi * 1000 * t
        )
        self.signal = AudioSignal(
            data=self.data, sample_rate=self.sr, duration_sec=self.duration
        )

    def test_band_splitter_filter(self):
        # 150Hz 遮断の分割フィルター
        filter_splitter = BandSplitterFilter(target_key="target", cutoff_hz=150.0)
        signals = {"target": self.signal}

        result = filter_splitter.apply(signals)

        # 検証
        self.assertIn("target_low", result)
        self.assertIn("target_high", result)

        low_sig = result["target_low"]
        high_sig = result["target_high"]

        self.assertEqual(len(low_sig.data), len(self.data))
        self.assertEqual(len(high_sig.data), len(self.data))

        # ローパスされた低音は 1000Hz がカットされているはず
        # 周波数の簡易推定としてゼロ交差数を数える
        # 50Hz の正弦波は1秒間に100回ゼロ交差する。1000Hzは2000回。
        zcr_low = np.sum(np.diff(np.sign(low_sig.data)) != 0)
        zcr_high = np.sum(np.diff(np.sign(high_sig.data)) != 0)

        self.assertTrue(zcr_low < 150)  # 低周波支配のため交差数は少ない
        self.assertTrue(zcr_high > 1500)  # 高周波支配のため交差数は多い

    def test_compressor_filter(self):
        # 閾値 0.3, ratio 4.0, gain 1.0 (ゲインなし) のコンプレッサー
        filter_comp = CompressorFilter(
            target_key="target", threshold=0.3, ratio=4.0, gain=1.0
        )
        signals = {"target": self.signal}

        result = filter_comp.apply(signals)
        comp_sig = result["target"]

        # 圧縮されているので最大振幅は 1.0 未満になっている
        max_val = np.max(np.abs(comp_sig.data))
        self.assertTrue(max_val < 0.6)
        self.assertTrue(max_val > 0.3)

    def test_source_separator_filter(self):
        filter_sep = SourceSeparatorFilter(target_key="target")
        signals = {"target": self.signal}

        result = filter_sep.apply(signals)

        self.assertIn("target_drums", result)
        self.assertIn("target_vocal", result)
        self.assertEqual(len(result["target_drums"].data), len(self.data))


if __name__ == "__main__":
    unittest.main()
