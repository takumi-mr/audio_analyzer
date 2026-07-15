from analyzer.AudioAnalyzer import AudioAnalyzer
from analyzer.strategy.SimpleBeatStrategy import SimpleBeatStrategy
from analyzer.strategy.SlidingWindowBeatStrategy import SlidingWindowBeatStrategy
from analyzer.strategy.GenreClassificationStrategy import GenreClassificationStrategy
from analyzer.strategy.AudioSimilarityStrategy import AudioSimilarityStrategy
from analyzer.strategy.KeyDetectionStrategy import KeyDetectionStrategy
from analyzer.strategy.ChordEstimationStrategy import ChordEstimationStrategy
from analyzer.strategy.ChorusDetectionStrategy import ChorusDetectionStrategy
from reader.LibrosaAudioReader import LibrosaAudioReader
from writer.JsonResultWriter import JsonResultWriter

# フィルターのインポート
from analyzer.filter.BandSplitterFilter import BandSplitterFilter
from analyzer.filter.CompressorFilter import CompressorFilter
from analyzer.filter.SourceSeparatorFilter import SourceSeparatorFilter

if __name__ == "__main__":
    # インフラストラクチャ層のセットアップ
    reader = LibrosaAudioReader()
    writer = JsonResultWriter()
    
    # 事前処理フィルターのセットアップ
    # 1. 信号を旋律成分(Harmonic)と打楽器成分(Percussive)に実音源分離
    separator = SourceSeparatorFilter(target_key="target")
    # 2. 分離されたドラム成分のみに対してさらに低音域を抽出し、アタック音を強調
    splitter = BandSplitterFilter(target_key="target_drums", cutoff_hz=150.0)
    # 3. 低音ドラム成分(target_drums_low)に対してコンプレッサーをかける
    compressor = CompressorFilter(target_key="target_drums_low", threshold=0.2, ratio=3.0)
    
    # analyzerにフィルターを追加して初期化
    analyzer = AudioAnalyzer(reader, writer, SimpleBeatStrategy(), filters=[separator, splitter, compressor])
    
    print("--- 4つ打ちのダンスミュージックを解析 (実音源分離 + 低域強調フィルター適用) ---")
    analyzer.process_file("standard_dance.wav", "result_simple.json")
    
    print("\n--- 楽曲の主キー（調）推定を実行 (旋律成分を優先使用) ---")
    analyzer.set_strategy(KeyDetectionStrategy())
    analyzer.process_file("standard_dance.wav", "result_key.json")
    
    print("\n--- コード進行特定を実行 (旋律成分を優先使用) ---")
    analyzer.set_strategy(ChordEstimationStrategy())
    analyzer.process_file("standard_dance.wav", "result_chords.json")
    
    print("\n--- 楽曲のサビ(Chorus)検出を実行 (旋律成分を優先使用) ---")
    analyzer.set_strategy(ChorusDetectionStrategy())
    analyzer.process_file("standard_dance.wav", "result_chorus.json")
    
    print("\n--- プログレッシブ・ロックを解析 (実音源分離 + 低域強調フィルター適用) ---")
    # 実行時にアルゴリズムを変拍子対応のものに切り替える
    analyzer.set_strategy(SlidingWindowBeatStrategy())
    analyzer.process_file("complex_prog_rock.wav", "result_complex.json")
    
    print("\n--- 音楽ジャンル判定を実行 ---")
    analyzer.set_strategy(GenreClassificationStrategy())
    analyzer.process_file("standard_dance.wav", "result_genre.json")
    
    print("\n--- 音声類似度計算を実行 ---")
    analyzer.set_strategy(AudioSimilarityStrategy())
    analyzer.process(
        input_paths={
            "target": "standard_dance.wav",
            "reference": "complex_prog_rock.wav"
        },
        output_path="result_similarity.json"
    )