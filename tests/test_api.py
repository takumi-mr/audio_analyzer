import unittest
import os
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# テストターゲット
from server import app
from model.AudioSignal import AudioSignal

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("server.LibrosaAudioReader.read")
    @patch("server.SourceSeparatorFilter.apply")
    def test_analyze_audio_endpoint(self, mock_separator_apply, mock_reader_read):
        # 1. 各モックの構成
        sr = 22050
        duration = 15.0 # サビ検出に必要な最低長 (16拍以上)
        dummy_signal = AudioSignal(
            data=np.zeros(int(sr * duration)),
            sample_rate=sr,
            duration_sec=duration
        )
        mock_reader_read.return_value = dummy_signal

        # separator.apply のモック戻り値
        # target_vocal / target_drums などのキーマッピングをして返す
        def fake_apply(signals):
            res = dict(signals)
            res["target_vocal"] = dummy_signal
            res["target_drums"] = dummy_signal
            res["target_bass"] = dummy_signal
            res["target_other"] = dummy_signal
            return res
        mock_separator_apply.side_effect = fake_apply

        # 2. ダミーファイルを送信
        dummy_file_content = b"RIFF....WAVEfmt ...."
        
        response = self.client.post(
            "/api/analyze",
            files={"file": ("test.wav", dummy_file_content, "audio/wav")}
        )
        
        # 3. アサーション
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        self.assertEqual(result["status"], "success")
        self.assertIn("tempo_bpm", result)
        self.assertIn("estimated_key", result)
        self.assertIn("chords", result)
        self.assertIn("chorus_sections_rms", result)
        self.assertIn("chorus_sections_vocal", result)
        self.assertIn("chorus_sections_ssm", result)
        self.assertIn("chorus_sections_beat_ssm", result)
        self.assertIn("audio_url", result)
        self.assertEqual(result["filename"], "test.wav")

    def test_unsupported_format(self):
        response = self.client.post(
            "/api/analyze",
            files={"file": ("test.txt", b"plain text", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file format", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
