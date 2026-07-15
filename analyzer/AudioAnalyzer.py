from typing import Dict, Any
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from reader.IAudioReader import IAudioReader
from writer.IResultWriter import IResultWriter

class AudioAnalyzer:
    """入出力と解析戦略を統合するコンテキストクラス"""
    def __init__(self, reader: IAudioReader, writer: IResultWriter, strategy: IAnalysisStrategy):
        # 依存性の注入 (DI)
        self._reader = reader
        self._writer = writer
        self._strategy = strategy

    def set_strategy(self, strategy: IAnalysisStrategy) -> None:
        """実行時に解析アルゴリズムを動的に切り替える"""
        self._strategy = strategy

    def process_file(self, input_path: str, output_path: str) -> None:
        """単一ファイル処理用の後方互換メソッド"""
        self.process({"target": input_path}, output_path)

    def process(self, input_paths: Dict[str, str], output_path: str, params: Dict[str, Any] = None) -> None:
        """複数ファイルやパラメータの指定に対応した汎用処理パイプライン"""
        # 1. すべてのファイルを読み込んで AudioSignal の辞書を作成
        signals = {key: self._reader.read(path) for key, path in input_paths.items()}
        
        # 2. 解析 (Strategyへ委譲)
        result = self._strategy.analyze(signals, params)
        
        # 3. 出力
        self._writer.write(output_path, result)