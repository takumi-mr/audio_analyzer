import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# パス追加
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from server import app
from model.AudioSignal import AudioSignal

client = TestClient(app)

# モック構成
sr = 22050
duration = 15.0  # サビ検出に必要な最低長
dummy_signal = AudioSignal(
    data=np.zeros(int(sr * duration)),
    sample_rate=sr,
    duration_sec=duration
)

def fake_apply(signals):
    res = dict(signals)
    res["target_vocal"] = dummy_signal
    res["target_drums"] = dummy_signal
    res["target_bass"] = dummy_signal
    res["target_other"] = dummy_signal
    res["target_harmonic"] = dummy_signal
    return res

@patch("server.LibrosaAudioReader.read", return_value=dummy_signal)
@patch("server.SourceSeparatorFilter.apply", side_effect=fake_apply)
def debug_call(mock_sep, mock_read):
    dummy_file_content = b"RIFF....WAVEfmt ...."
    try:
        response = client.post(
            "/api/analyze",
            files={"file": ("test.wav", dummy_file_content, "audio/wav")}
        )
        print("STATUS CODE:", response.status_code)
        print("RESPONSE JSON:", response.json())
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_call()
