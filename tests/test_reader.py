import unittest
import os
import tempfile
import numpy as np
import scipy.io.wavfile as wav
from reader.AudioReader import WavAudioReader
from reader.LibrosaAudioReader import LibrosaAudioReader
from model.AudioSignal import AudioSignal

class TestAudioReader(unittest.TestCase):
    def setUp(self):
        # テスト用の極小WAVファイルを一時ディレクトリ内に作成 (1秒分)
        self.test_dir = tempfile.TemporaryDirectory()
        self.sample_rate = 22050
        self.duration = 1.0
        t = np.linspace(0, self.duration, self.sample_rate, endpoint=False)
        self.dummy_data = np.sin(2 * np.pi * 440 * t)
        
        # 16-bit PCM WAV
        self.wav_path = os.path.join(self.test_dir.name, "test_audio.wav")
        data_int16 = (self.dummy_data * 32767).astype(np.int16)
        wav.write(self.wav_path, self.sample_rate, data_int16)
        
    def tearDown(self):
        self.test_dir.cleanup()
        
    def test_wav_audio_reader_success(self):
        reader = WavAudioReader()
        signal = reader.read(self.wav_path)
        
        self.assertIsInstance(signal, AudioSignal)
        self.assertEqual(signal.sample_rate, self.sample_rate)
        self.assertEqual(len(signal.data), self.sample_rate)
        self.assertAlmostEqual(signal.duration_sec, self.duration, places=2)
        
    def test_librosa_audio_reader_success(self):
        reader = LibrosaAudioReader()
        signal = reader.read(self.wav_path)
        
        self.assertIsInstance(signal, AudioSignal)
        self.assertEqual(signal.sample_rate, self.sample_rate)
        self.assertEqual(len(signal.data), self.sample_rate)
        self.assertAlmostEqual(signal.duration_sec, self.duration, places=2)
        
    def test_reader_file_not_found(self):
        reader = WavAudioReader()
        with self.assertRaises(FileNotFoundError):
            reader.read("non_existent_file.wav")
            
        reader_librosa = LibrosaAudioReader()
        with self.assertRaises(FileNotFoundError):
            reader_librosa.read("non_existent_file.wav")

if __name__ == "__main__":
    unittest.main()
