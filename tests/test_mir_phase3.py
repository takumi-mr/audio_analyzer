import unittest
import numpy as np
from model.AudioSignal import AudioSignal
from analyzer.strategy.ChorusDetectionStrategy import ChorusDetectionStrategy

class TestMIRPhase3(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        self.duration = 10.0
        
        # 前半 5秒: 小音量の 100Hz 正弦波
        t_half = np.linspace(0, 5.0, int(self.sr * 5.0), endpoint=False)
        data_low = 0.05 * np.sin(2 * np.pi * 100 * t_half)
        
        # 後伴 5秒: 大音量の 1000Hz 正弦波 + ノイズ (サビ想定)
        t_half_high = np.linspace(5.0, 10.0, int(self.sr * 5.0), endpoint=False)
        data_high = 0.6 * np.sin(2 * np.pi * 1000 * t_half_high) + np.random.normal(0, 0.1, len(t_half_high))
        
        # 結合
        self.data = np.concatenate([data_low, data_high])
        self.data = np.clip(self.data, -1.0, 1.0)
        
        self.signal = AudioSignal(data=self.data, sample_rate=self.sr, duration_sec=self.duration)
        
    def test_chorus_detection_success(self):
        strategy = ChorusDetectionStrategy()
        signals = {"target": self.signal}
        
        result = strategy.analyze(signals)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("chorus_sections", result)
        self.assertGreater(len(result["chorus_sections"]), 0)
        
        # 盛り上がっている後半部分 (5.0秒〜10.0秒) がサビとして特定されているかを検証
        found_chorus = result["chorus_sections"][0]
        # 開始秒数は 5.0秒付近 (誤差1秒以内)
        self.assertAlmostEqual(found_chorus["start_sec"], 5.0, delta=1.0)
        self.assertAlmostEqual(found_chorus["end_sec"], 10.0, delta=1.0)
        self.assertIn("confidence", result)

if __name__ == "__main__":
    unittest.main()
