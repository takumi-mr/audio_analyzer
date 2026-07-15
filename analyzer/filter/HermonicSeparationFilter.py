from typing import Dict
import librosa
from analyzer.filter.IAudioFilter import IAudioFilter
from model.AudioSignal import AudioSignal

class HarmonicSeparationFilter(IAudioFilter):
    """
    HPSS (Harmonic-Percussive Source Separation) を用いて、
    楽曲から打楽器成分を分離し、調波成分（コードやメロディ）を抽出するフィルター
    """
    def apply(self, signals: Dict[str, AudioSignal]) -> Dict[str, AudioSignal]:
        # ベースとなる対象シグナルを取得
        target_key = "target"
        if target_key not in signals:
            return signals  # 処理対象がなければそのまま返す

        signal = signals[target_key]
        print("[Filter: HPSS] 打楽器成分を分離し、調波成分(Harmonic)を抽出中...")

        # HPSSの実行。marginを少し(例:1.2)設定すると分離がより強めにかかります
        harmonic_data, percussive_data = librosa.effects.hpss(signal.data, margin=1.2)

        # 抽出した調波成分を 'target_harmonic' としてシグナル辞書に追加
        signals["target_harmonic"] = AudioSignal(
            data=harmonic_data,
            sample_rate=signal.sample_rate,
            duration_sec=signal.duration_sec
        )
        
        # (オプション) 必要であればパーカッシブ成分も保持しておく
        # signals["target_percussive"] = AudioSignal(data=percussive_data, sample_rate=signal.sample_rate)

        return signals