import argparse
import sys
from typing import List, Optional

from analyzer.AudioAnalyzer import AudioAnalyzer
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from analyzer.strategy.SimpleBeatStrategy import SimpleBeatStrategy
from analyzer.strategy.SlidingWindowBeatStrategy import SlidingWindowBeatStrategy
from analyzer.strategy.GenreClassificationStrategy import GenreClassificationStrategy
from analyzer.strategy.AudioSimilarityStrategy import AudioSimilarityStrategy
from analyzer.strategy.KeyDetectionStrategy import KeyDetectionStrategy
from analyzer.strategy.ChordEstimationStrategy import ChordEstimationStrategy
from analyzer.strategy.ChorusDetectionStrategy import ChorusDetectionStrategy
from analyzer.strategy.ChorusDetectionBeatSSMStrategy import ChorusDetectionBeatSSMStrategy
from reader.LibrosaAudioReader import LibrosaAudioReader
from writer.JsonResultWriter import JsonResultWriter

# フィルターのインポート
from analyzer.filter.IAudioFilter import IAudioFilter
from analyzer.filter.BandSplitterFilter import BandSplitterFilter
from analyzer.filter.CompressorFilter import CompressorFilter
from analyzer.filter.SourceSeparatorFilter import SourceSeparatorFilter

def main(cli_args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audio Analyzer Command Line Interface - BPM、キー、コード進行、サビ、類似度、ジャンル等の統合音声解析ツール"
    )
    parser.add_argument("-i", "--input", required=True, help="解析対象の音声ファイルパス (WAV, MP3, FLAC, OGG等)")
    parser.add_argument("-o", "--output", default="result.json", help="解析結果を出力するJSONファイルパス (デフォルト: result.json)")
    parser.add_argument(
        "-s", "--strategies", 
        nargs="+", 
        choices=["beat", "sliding-beat", "key", "chord", "chorus", "chorus-beat-ssm", "genre", "similarity", "all"],
        default=["all"],
        help="実行する解析戦略を指定します (複数指定可能、デフォルト: all)"
    )
    parser.add_argument("-r", "--reference", help="類似度計算(similarity)時に比較対象とする参照音声ファイルパス")
    parser.add_argument("--no-separation", action="store_true", help="DemucsによるAI音源分離フィルターをスキップします")
    parser.add_argument("--cutoff", type=float, default=150.0, help="ドラム低域抽出用フィルターのカットオフ周波数 (Hz, デフォルト: 150.0)")
    parser.add_argument("--threshold", type=float, default=0.2, help="アタック音強調用コンプレッサーの閾値 (デフォルト: 0.2)")
    parser.add_argument("--ratio", type=float, default=3.0, help="アタック音強調用コンプレッサーのレシオ (デフォルト: 3.0)")

    args = parser.parse_args(cli_args)

    # インフラ層のセットアップ
    reader = LibrosaAudioReader()
    file_writer = JsonResultWriter()

    # フィルターチェーンの構築
    filters: List[IAudioFilter] = []
    if not args.no_separation:
        # Demucsによる音源分離を適用
        filters.append(SourceSeparatorFilter(target_key="target"))
        
    # ドラム/低域アタック強調フィルター
    filters.append(BandSplitterFilter(target_key="target_drums", cutoff_hz=args.cutoff))
    filters.append(CompressorFilter(target_key="target_drums_low", threshold=args.threshold, ratio=args.ratio))

    # 解析戦略の解決
    strategy_mapping = {
        "beat": SimpleBeatStrategy,
        "sliding-beat": SlidingWindowBeatStrategy,
        "key": KeyDetectionStrategy,
        "chord": ChordEstimationStrategy,
        "chorus": ChorusDetectionStrategy,
        "chorus-beat-ssm": ChorusDetectionBeatSSMStrategy,
        "genre": GenreClassificationStrategy,
    }

    selected_keys = list(args.strategies)
    if "all" in selected_keys:
        # similarity 以外のすべてを有効化
        selected_keys = ["beat", "key", "chord", "chorus-beat-ssm", "genre"]

    # 重複排除（順序保持）
    seen: set = set()
    selected_keys = [k for k in selected_keys if not (k in seen or seen.add(k))]

    input_paths = {"target": args.input}

    # 類似度計算用（reference）の指定がある場合
    if "similarity" in selected_keys:
        if not args.reference:
            print("Error: similarity 戦略の実行には --reference (-r) オプションが必須です。", file=sys.stderr)
            sys.exit(1)
        input_paths["reference"] = args.reference

    # 実行する戦略リストを構築
    strategies: List[IAnalysisStrategy] = []
    for key in selected_keys:
        if key == "similarity":
            strategies.append(AudioSimilarityStrategy())
        else:
            strategies.append(strategy_mapping[key]())

    # 解析器の初期化
    analyzer = AudioAnalyzer(reader, file_writer, SimpleBeatStrategy(), filters=filters)

    # 一括実行 (読込・フィルタが1回のみ走る)
    print(f"\n>>> 解析を一括実行中: {selected_keys}")
    try:
        analyzer.process_multi(input_paths, strategies, args.output)
    except Exception as e:
        print(f"Error: 解析の実行中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n>>> 解析結果を {args.output} に書き出しました。")

if __name__ == "__main__":
    main()