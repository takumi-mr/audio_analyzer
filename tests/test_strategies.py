import unittest
import numpy as np
from model.AudioSignal import AudioSignal
from analyzer.strategy.SimpleBeatStrategy import SimpleBeatStrategy
from analyzer.strategy.SlidingWindowBeatStrategy import SlidingWindowBeatStrategy
from analyzer.strategy.GenreClassificationStrategy import GenreClassificationStrategy
from analyzer.strategy.AudioSimilarityStrategy import AudioSimilarityStrategy

class TestAnalysisStrategies(unittest.TestCase):
    def setUp(self):
        # 2秒分のダミー波形シグナル
        self.sr = 22050
        self.duration = 2.0
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        
        # 120 BPMのビートを模倣した波形 (0.5秒おきにクリック)
        data = np.random.normal(0, 0.01, len(t))
        for beat_t in [0.0, 0.5, 1.0, 1.5]:
            idx = int(beat_t * self.sr)
            click_len = int(0.05 * self.sr)
            if idx + click_len <= len(data):
                data[idx : idx + click_len] += np.sin(2 * np.pi * 1000 * np.linspace(0, 0.05, click_len, endpoint=False))
                
        self.signal_120bpm = AudioSignal(data=data, sample_rate=self.sr, duration_sec=self.duration)
        
        # もう1つ別のダミー波形 (BPM 150 -> 0.4秒おきにクリック)
        data_150 = np.random.normal(0, 0.01, len(t))
        for beat_t in [0.0, 0.4, 0.8, 1.2, 1.6]:
            idx = int(beat_t * self.sr)
            click_len = int(0.05 * self.sr)
            if idx + click_len <= len(data_150):
                data_150[idx : idx + click_len] += np.sin(2 * np.pi * 1000 * np.linspace(0, 0.05, click_len, endpoint=False))
                
        self.signal_150bpm = AudioSignal(data=data_150, sample_rate=self.sr, duration_sec=self.duration)
        
    def test_simple_beat_strategy(self):
        strategy = SimpleBeatStrategy()
        signals = {"target": self.signal_120bpm}
        result = strategy.analyze(signals)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("tempo_bpm", result)
        self.assertIsInstance(result["tempo_bpm"], float)
        self.assertAlmostEqual(result["tempo_bpm"], 120.0, delta=10.0) # 簡易検出の許容誤差
        self.assertIn("time_signature", result)
        self.assertIsInstance(result["confidence"], float)
        
    def test_sliding_window_beat_strategy(self):
        strategy = SlidingWindowBeatStrategy()
        # 長めのシグナル (10秒)
        long_t = np.linspace(0, 10.0, int(self.sr * 10.0), endpoint=False)
        long_data = np.random.normal(0, 0.01, len(long_t))
        # 120 BPM
        for beat_t in np.arange(0, 10.0, 0.5):
            idx = int(beat_t * self.sr)
            click_len = int(0.05 * self.sr)
            if idx + click_len <= len(long_data):
                long_data[idx : idx + click_len] += np.sin(2 * np.pi * 1000 * np.linspace(0, 0.05, click_len, endpoint=False))
                
        long_signal = AudioSignal(data=long_data, sample_rate=self.sr, duration_sec=10.0)
        
        signals = {"target": long_signal}
        result = strategy.analyze(signals)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("base_tempo_bpm", result)
        self.assertIn("segments", result)
        self.assertGreater(len(result["segments"]), 0)
        
        # 各セグメントの構造検証
        segment = result["segments"][0]
        self.assertIn("start_sec", segment)
        self.assertIn("end_sec", segment)
        self.assertIn("tempo_bpm", segment)
        self.assertIn("time_signature", segment)
        
    def test_genre_classification_strategy(self):
        strategy = GenreClassificationStrategy()
        signals = {"target": self.signal_120bpm}
        result = strategy.analyze(signals)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("detected_genre", result)
        self.assertIn("confidence", result)
        self.assertIn("extracted_features", result)
        self.assertIn("tempo_bpm", result["extracted_features"])
        self.assertIn("spectral_centroid_hz", result["extracted_features"])
        
    def test_audio_similarity_strategy(self):
        strategy = AudioSimilarityStrategy()
        signals = {
            "target": self.signal_120bpm,
            "reference": self.signal_150bpm
        }
        result = strategy.analyze(signals)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("similarity_score", result)
        self.assertIn("match", result)
        self.assertIsInstance(result["match"], bool)
        self.assertIsInstance(result["similarity_score"], float)
        
    def test_audio_similarity_missing_inputs(self):
        strategy = AudioSimilarityStrategy()
        # 片方のシグナルしか渡さない場合は例外を期待
        with self.assertRaises(ValueError):
            strategy.analyze({"target": self.signal_120bpm})

    def test_chorus_detection_beat_ssm_strategy(self):
        from analyzer.strategy.ChorusDetectionBeatSSMStrategy import ChorusDetectionBeatSSMStrategy
        strategy = ChorusDetectionBeatSSMStrategy()
        
        # 16ビート以上の長さのダミー信号を作成 (約15秒)
        long_duration = 15.0
        t = np.linspace(0, long_duration, int(self.sr * long_duration), endpoint=False)
        data = np.random.normal(0, 0.01, len(t))
        # 120 BPMの拍を入れる
        for beat_t in np.arange(0.0, long_duration, 0.5):
            idx = int(beat_t * self.sr)
            click_len = int(0.05 * self.sr)
            if idx + click_len <= len(data):
                data[idx : idx + click_len] += np.sin(2 * np.pi * 1000 * np.linspace(0, 0.05, click_len, endpoint=False))
                
        signal = AudioSignal(data=data, sample_rate=self.sr, duration_sec=long_duration)
        signals = {"target": signal}
        
        result = strategy.analyze(signals)
        self.assertEqual(result["status"], "success")
        self.assertIn("chorus_sections_beat_ssm", result)
        self.assertIn("chorus_confidence_beat_ssm", result)

if __name__ == "__main__":
    unittest.main()
