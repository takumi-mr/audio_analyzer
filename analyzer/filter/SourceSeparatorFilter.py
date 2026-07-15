import torch
import demucs.api
import librosa
import numpy as np
from analyzer.filter.IAudioFilter import IAudioFilter
from typing import Dict
from model.AudioSignal import AudioSignal

class SourceSeparatorFilter(IAudioFilter):
    """Demucsを用いた高品質なドラム・ボーカル・ベース等の音源分離を行うフィルター"""
    def __init__(self, target_key: str = "target", model_name: str = "htdemucs"):
        self.target_key = target_key
        self.model_name = model_name
        self._separator = None

    def apply(self, signals: Dict[str, AudioSignal]) -> Dict[str, AudioSignal]:
        result_signals = dict(signals)
        
        signal = result_signals.get(self.target_key)
        if not signal or len(signal.data) == 0:
            return result_signals
            
        print(f"[Filter: SourceSeparator] Demucs ({self.model_name}) を用いて '{self.target_key}' の音源分離を開始します...")
        
        # 1. Separatorの遅延初期化（初回適用時にモデルのロード/ダウンロードを行う）
        if self._separator is None:
            print(f"[Filter: SourceSeparator] モデル '{self.model_name}' をロード中 (初回実行時はダウンロードが発生します)...")
            self._separator = demucs.api.Separator(model=self.model_name)
            print("[Filter: SourceSeparator] モデルのロードが完了しました。")
            
        # 2. 入力信号(モノラル)を2チャンネル(ステレオ)のPyTorchテンソルに変換
        # shape: (samples,) -> (2, samples)
        wav_tensor = torch.from_numpy(signal.data).float()
        wav_tensor = wav_tensor.unsqueeze(0).repeat(2, 1)
        
        # 3. Demucsで音源分離を実行
        # separate_tensorは自動的に内部でリサンプリングを実行して分離処理を行います
        # 戻り値: (resampled_input_tensor, separated_sources_dict)
        _, separated = self._separator.separate_tensor(wav_tensor, sr=signal.sample_rate)
        
        # 4. 各ステムをモノラル化し、元のサンプリング周波数にリサンプリングしてAudioSignalとして格納
        # Demucsのデフォルトの出力stem: "vocals", "drums", "bass", "other"
        # 互換性のため "vocals" は "vocal" としてキーを設定します
        stem_key_map = {
            "vocals": f"{self.target_key}_vocal",
            "drums": f"{self.target_key}_drums",
            "bass": f"{self.target_key}_bass",
            "other": f"{self.target_key}_other"
        }
        
        target_sr = signal.sample_rate
        demucs_sr = self._separator.samplerate
        
        for stem_name, output_key in stem_key_map.items():
            if stem_name in separated:
                # 2チャンネルから平均を取ってモノラルに戻す (shape: (samples,))
                stem_tensor = separated[stem_name]
                stem_mono = stem_tensor.mean(dim=0).cpu().numpy()
                
                # 元のサンプリング周波数にリサンプリング
                if demucs_sr != target_sr:
                    stem_resampled = librosa.resample(
                        stem_mono, 
                        orig_sr=demucs_sr, 
                        target_sr=target_sr
                    )
                else:
                    stem_resampled = stem_mono
                
                # AudioSignalオブジェクトを作成して追加
                duration = len(stem_resampled) / target_sr
                result_signals[output_key] = AudioSignal(
                    data=stem_resampled.astype(np.float32),
                    sample_rate=target_sr,
                    duration_sec=duration
                )
                print(f"[Filter: SourceSeparator] 分離完了: '{output_key}' (長さ: {duration:.2f}秒)")
                
        return result_signals
