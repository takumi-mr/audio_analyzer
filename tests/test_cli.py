import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from main import main
from model.AudioSignal import AudioSignal
from reader.IAudioReader import IAudioReader
from analyzer.AudioAnalyzer import AudioAnalyzer
from writer.IResultWriter import IResultWriter

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

    def test_process_multi_single_read_filter_pass(self):
        """process_multi はフィルターと読込を1回のみ実行することを確認"""
        from writer.JsonResultWriter import JsonResultWriter
        from analyzer.strategy.SimpleBeatStrategy import SimpleBeatStrategy
        from analyzer.strategy.KeyDetectionStrategy import KeyDetectionStrategy

        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        data = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        dummy_signal = AudioSignal(data=data, sample_rate=sr, duration_sec=duration)

        mock_reader = MagicMock()
        mock_reader.read.return_value = dummy_signal

        mock_filter = MagicMock()
        mock_filter.apply.return_value = {"target": dummy_signal}

        output_path = os.path.join(self.temp_dir.name, "multi_result.json")
        writer = JsonResultWriter()

        analyzer = AudioAnalyzer(mock_reader, writer, SimpleBeatStrategy(), filters=[mock_filter])
        analyzer.process_multi(
            {"target": self.input_path},
            [SimpleBeatStrategy(), KeyDetectionStrategy()],
            output_path
        )

        # 読込とフィルターが1回のみ呼ばれていることを確認
        mock_reader.read.assert_called_once()
        mock_filter.apply.assert_called_once()

        # 出力ファイルが存在してキーが含まれていることを確認
        self.assertTrue(os.path.exists(output_path))
        import json
        with open(output_path) as f:
            result = json.load(f)
        self.assertIn("tempo_bpm", result)
        self.assertIn("estimated_key", result)

if __name__ == "__main__":
    unittest.main()
