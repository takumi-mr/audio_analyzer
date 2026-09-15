from typing import Any

from analyzer.filter.IAudioFilter import IAudioFilter
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from reader.IAudioReader import IAudioReader
from writer.IResultWriter import IResultWriter


class AudioAnalyzer:
    """入出力と解析戦略を統合するコンテキストクラス"""

    def __init__(
        self,
        reader: IAudioReader,
        writer: IResultWriter,
        strategy: IAnalysisStrategy,
        filters: list[IAudioFilter] | None = None,
    ):
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

    def process(
        self,
        input_paths: dict[str, str],
        output_path: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """複数ファイルやパラメータの指定に対応した汎用処理パイプライン"""
        # 1. 読み込み
        signals = {key: self._reader.read(path) for key, path in input_paths.items()}

        # 1.5. 事前処理の適用 (フィルターチェーンの実行)
        for audio_filter in self._filters:
            signals = audio_filter.apply(signals)

        # 2. 解析 (Strategyへ委譲)
        actual_params = params or {}
        result = self._strategy.analyze(signals, actual_params)

        # 3. 出力
        self._writer.write(output_path, result)

    def process_multi(
        self,
        input_paths: dict[str, str],
        strategies: list[IAnalysisStrategy],
        output_path: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """複数の解析戦略を一括実行し、結果をマージして出力する (性能最適化版)"""
        # 1. 読み込み (1回のみ)
        signals = {key: self._reader.read(path) for key, path in input_paths.items()}

        # 1.5. 事前処理の適用 (1回のみ)
        for audio_filter in self._filters:
            signals = audio_filter.apply(signals)

        # 2. 複数戦略の順次解析と結果のマージ
        merged_result = {"status": "success"}
        actual_params = params or {}
        for strategy in strategies:
            result = strategy.analyze(signals, actual_params)

            # 各結果のインテリジェントマージ
            for k, v in result.items():
                if k == "status":
                    if v == "error":
                        merged_result["status"] = "error"
                elif k == "message":
                    merged_result["message"] = (
                        merged_result.get("message", "") + "; " + v
                    ).strip("; ")
                else:
                    merged_result[k] = v

        # 3. マージ結果を出力
        self._writer.write(output_path, merged_result)
