from typing import Dict, Any, List
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from reader.IAudioReader import IAudioReader
from writer.IResultWriter import IResultWriter
from analyzer.filter.IAudioFilter import IAudioFilter

class AudioAnalyzer:
    """入出力と解析戦略を統合するコンテキストクラス"""
    def __init__(self, reader: IAudioReader, writer: IResultWriter, strategy: IAnalysisStrategy, filters: List[IAudioFilter] = None):
        # 依存性の注入 (DI)
        self._reader = reader
        self._writer = writer
        self._strategy = strategy
        self._filters = filters or []

    def set_strategy(self, strategy: IAnalysisStrategy) -> None:
        """実行時に解析アルゴリズムを動的に切り替える"""
        self._strategy = strategy

    def add_filter(self, audio_filter: IAudioFilter) -> None:
        """事前処理フィルターをパイプラインに追加"""
        self._filters.append(audio_filter)

    def process_file(self, input_path: str, output_path: str) -> None:
        """単一ファイル処理用の後方互換メソッド"""
        self.process({"target": input_path}, output_path)

    def process(self, input_paths: Dict[str, str], output_path: str, params: Dict[str, Any] = None) -> None:
        """複数ファイルやパラメータの指定に対応した汎用処理パイプライン"""
        # 1. 読み込み
        signals = {key: self._reader.read(path) for key, path in input_paths.items()}
        
        # 1.5. 事前処理の適用 (フィルターチェーンの実行)
        for audio_filter in self._filters:
            signals = audio_filter.apply(signals)
        
        # 2. 解析 (Strategyへ委譲)
        result = self._strategy.analyze(signals, params)
        
        # 3. 出力
        self._writer.write(output_path, result)