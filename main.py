from analyzer.AudioAnalyzer import AudioAnalyzer
from analyzer.strategy.SimpleBeatStrategy import SimpleBeatStrategy
from analyzer.strategy.SlidingWindowBeatStrategy import SlidingWindowBeatStrategy
from analyzer.strategy.GenreClassificationStrategy import GenreClassificationStrategy
from analyzer.strategy.AudioSimilarityStrategy import AudioSimilarityStrategy
from reader.LibrosaAudioReader import LibrosaAudioReader
from writer.JsonResultWriter import JsonResultWriter


if __name__ == "__main__":
    # インフラストラクチャ層のセットアップ
    reader = LibrosaAudioReader()
    writer = JsonResultWriter()
    
    # 1. シンプルな曲を解析する場合
    analyzer = AudioAnalyzer(reader, writer, SimpleBeatStrategy())
    print("--- 4つ打ちのダンスミュージックを解析 ---")
    analyzer.process_file("standard_dance.wav", "result_simple.json")
    
    print("\n--- プログレッシブ・ロックを解析 ---")
    # 2. 実行時にアルゴリズムを変拍子対応のものに切り替える
    analyzer.set_strategy(SlidingWindowBeatStrategy())
    analyzer.process_file("complex_prog_rock.wav", "result_complex.json")
    
    print("\n--- 音楽ジャンル判定を実行 ---")
    # 3. 音楽ジャンル判定戦略に切り替える
    analyzer.set_strategy(GenreClassificationStrategy())
    analyzer.process_file("standard_dance.wav", "result_genre.json")
    
    print("\n--- 音声類似度計算を実行 ---")
    # 4. 2つのファイルの類似度を計算する（process を直接呼ぶ）
    analyzer.set_strategy(AudioSimilarityStrategy())
    analyzer.process(
        input_paths={
            "target": "standard_dance.wav",
            "reference": "complex_prog_rock.wav"
        },
        output_path="result_similarity.json"
    )