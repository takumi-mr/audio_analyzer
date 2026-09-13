import os
import urllib.request
from typing import Tuple, Dict, Any, Optional
import torch
from model.btc.btc_model import BTC_model

BTC_LARGE_VOCA_URL = "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/master/test/btc_model_large_voca.pt"
BTC_MAJMIN_URL = "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/master/test/btc_model.pt"

ROOT_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
QUALITY_LIST = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

QUALITY_MAP = {
    'maj': '',
    'min': 'm',
    'maj7': 'M7',
    'min7': 'm7',
    '7': '7',
    'dim': 'dim',
    'dim7': 'dim7',
    'hdim7': 'm7b5',
    'aug': 'aug',
    'sus4': 'sus4',
    'sus2': 'sus2',
    'min6': 'm6',
    'maj6': '6',
    'minmaj7': 'mM7'
}

def get_btc_vocabulary() -> Dict[int, Dict[str, Any]]:
    """
    BTC 170クラスのインデックスからコード表記・ルート・クオリティへの対応マップを生成
    """
    vocab = {}
    vocab[169] = {"chord": "N", "root": None, "suffix": "", "quality": "no_chord", "triad": "none"}
    vocab[168] = {"chord": "X", "root": None, "suffix": "", "quality": "unknown", "triad": "unknown"}

    for i in range(168):
        root_idx = i // 14
        quality_idx = i % 14
        root_name = ROOT_LIST[root_idx]
        q_raw = QUALITY_LIST[quality_idx]
        suffix = QUALITY_MAP.get(q_raw, q_raw)
        chord_name = f"{root_name}{suffix}"

        triad = "maj"
        if q_raw in ["min", "min7", "min6", "minmaj7"]:
            triad = "min"
        elif q_raw in ["dim", "dim7", "hdim7"]:
            triad = "dim"
        elif q_raw == "aug":
            triad = "aug"
        elif q_raw == "sus4":
            triad = "sus4"

        vocab[i] = {
            "chord": chord_name,
            "root": root_name,
            "root_pitch": root_idx,
            "suffix": suffix,
            "quality": q_raw,
            "triad": triad
        }
    return vocab

def download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    print(f"[BTC Loader] Downloading weights from: {url}")
    print(f"             Target path: {dest_path}")
    urllib.request.urlretrieve(url, temp_path)
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    print(f"[BTC Loader] Download complete ({os.path.getsize(dest_path)} bytes).")

def load_btc_model(
    device: Optional[torch.device] = None,
    voca_large: bool = True,
    model_dir: Optional[str] = None
) -> Tuple[BTC_model, float, float, Dict[int, Dict[str, Any]]]:
    """
    BTC Transformer モデルと正規化統計量 (mean, std)、語彙マップをロード
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is None:
        # プロジェクトルート基準の model/btc
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = base_dir

    filename = "btc_model_large_voca.pt" if voca_large else "btc_model.pt"
    ckpt_path = os.path.join(model_dir, filename)
    url = BTC_LARGE_VOCA_URL if voca_large else BTC_MAJMIN_URL

    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1000000:
        download_file(url, ckpt_path)

    config_model = {
        'feature_size': 144,
        'timestep': 108,
        'num_chords': 170 if voca_large else 25,
        'input_dropout': 0.2,
        'layer_dropout': 0.2,
        'attention_dropout': 0.2,
        'relu_dropout': 0.2,
        'num_layers': 8,
        'num_heads': 4,
        'hidden_size': 128,
        'total_key_depth': 128,
        'total_value_depth': 128,
        'filter_size': 128,
        'probs_out': True
    }

    model = BTC_model(config_model).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    mean = float(checkpoint['mean'])
    std = float(checkpoint['std'])
    vocab = get_btc_vocabulary()

    print(f"[BTC Loader] BTC Transformer initialized successfully on {device} (Vocabulary: {len(vocab)} classes)")
    return model, mean, std, vocab
