import unittest
import os
import tempfile
import numpy as np
from main import main, InMemoryResultWriter
from model.AudioSignal import AudioSignal
from reader.IAudioReader import IAudioReader

class DummyAudioReader(IAudioReader):
    """ダミーのオーディオリーダー"""
    def read(self, filepath: str) -> AudioSignal:
        sr = 22050
        duration = 1.0
        data = np.zeros(int(sr * duration))
        return AudioSignal(data=data, sample_rate=sr, duration_sec=duration)

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.temp_dir.name, "dummy_input.wav")
        self.output_path = os.path.join(self.temp_dir.name, "report.json")
        
        # ダミー入力ファイルの作成
        with open(self.input_path, "wb") as f:
            f.write(b"dummy wav data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_parsing_and_execution_all_no_sep(self):
        # 実際に main() を呼び出して、BPM, key を一括実行できるか検証
        # AI分離は無効（--no-separation）にする
        from unittest.mock import patch
        
        # ダミー波形
        sr = 22050
        duration = 5.0 # サビ検出のために最低5秒以上
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        data = 0.5 * np.sin(2 * np.pi * 440.0 * t) # A4 440Hz
        dummy_signal = AudioSignal(data=data, sample_rate=sr, duration_sec=duration)
        
        with patch("reader.LibrosaAudioReader.LibrosaAudioReader.read", return_value=dummy_signal):
            # main を呼び出し
            main([
                "-i", self.input_path,
                "-o", self.output_path,
                "-s", "beat", "key",
                "--no-separation"
            ])
            
            # 出力ファイルが存在することを確認
            self.assertTrue(os.path.exists(self.output_path))
            
            # JSONの中身を検証
            import json
            with open(self.output_path, "r") as f:
                result = json.load(f)
                
            self.assertEqual(result["status"], "success")
            self.assertIn("tempo_bpm", result)
            self.assertIn("estimated_key", result)

    def test_in_memory_writer_merging(self):
        writer = InMemoryResultWriter()
        writer.write("dummy", {"status": "success", "tempo_bpm": 120.0})
        writer.write("dummy", {"status": "success", "estimated_key": "C Major"})
        
        # エラー発生時のマージ
        writer.write("dummy", {"status": "error", "message": "Failed key"})
        
        self.assertEqual(writer.result["status"], "error")
        self.assertEqual(writer.result["tempo_bpm"], 120.0)
        self.assertEqual(writer.result["estimated_key"], "C Major")
        self.assertEqual(writer.result["message"], "Failed key")

if __name__ == "__main__":
    unittest.main()
