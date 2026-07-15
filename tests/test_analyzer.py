import unittest
from unittest.mock import MagicMock
from analyzer.AudioAnalyzer import AudioAnalyzer
from reader.IAudioReader import IAudioReader
from writer.IResultWriter import IResultWriter
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from analyzer.filter.IAudioFilter import IAudioFilter
from model.AudioSignal import AudioSignal

class TestAudioAnalyzer(unittest.TestCase):
    def test_analyzer_pipeline_success(self):
        # モックの作成
        mock_reader = MagicMock(spec=IAudioReader)
        mock_writer = MagicMock(spec=IResultWriter)
        mock_strategy = MagicMock(spec=IAnalysisStrategy)
        mock_filter = MagicMock(spec=IAudioFilter)
        
        # ダミーの音声シグナルと解析結果をセットアップ
        dummy_signal = MagicMock(spec=AudioSignal)
        filtered_signal = MagicMock(spec=AudioSignal)
        dummy_result = {"status": "success", "tempo_bpm": 120}
        
        mock_reader.read.return_value = dummy_signal
        # フィルターが入力辞書を受け取り、加工した辞書を返す挙動を定義
        mock_filter.apply.return_value = {"target": filtered_signal}
        mock_strategy.analyze.return_value = dummy_result
        
        # フィルター付きで Analyzer を初期化
        analyzer = AudioAnalyzer(mock_reader, mock_writer, mock_strategy, filters=[mock_filter])
        
        # 1. process_file の呼び出し検証 (フィルターが適用されるか)
        analyzer.process_file("input.wav", "output.json")
        
        # verify
        mock_reader.read.assert_called_once_with("input.wav")
        # フィルターが呼ばれたか
        mock_filter.apply.assert_called_once_with({"target": dummy_signal})
        # strategy にはフィルター処理後の filtered_signal が渡されたか
        mock_strategy.analyze.assert_called_once_with({"target": filtered_signal}, None)
        mock_writer.write.assert_called_once_with("output.json", dummy_result)
        
        # モックの呼び出し履歴をリセット
        mock_reader.reset_mock()
        mock_writer.reset_mock()
        mock_strategy.reset_mock()
        mock_filter.reset_mock()
        
        # 2. process の呼び出し検証 (複数ファイルパイプライン)
        input_paths = {"target": "input1.wav", "reference": "input2.wav"}
        params = {"threshold": 0.8}
        
        sig1 = MagicMock(spec=AudioSignal)
        sig2 = MagicMock(spec=AudioSignal)
        
        def mock_read_side_effect(path):
            if path == "input1.wav":
                return sig1
            elif path == "input2.wav":
                return sig2
            return None
            
        mock_reader.read.side_effect = mock_read_side_effect
        
        # フィルターが signals 辞書を受け取り、一部を拡張して返す
        def mock_filter_side_effect(signals):
            signals_copy = dict(signals)
            signals_copy["target_low"] = MagicMock(spec=AudioSignal)
            return signals_copy
            
        mock_filter.apply.side_effect = mock_filter_side_effect
        mock_strategy.analyze.return_value = {"similarity": 0.85}
        
        analyzer.process(input_paths, "similarity.json", params=params)
        
        # verify
        self.assertEqual(mock_reader.read.call_count, 2)
        
        # フィルターが 2つのインプットで呼ばれたか
        mock_filter.apply.assert_called_once()
        args, _ = mock_filter.apply.call_args
        self.assertIn("target", args[0])
        self.assertIn("reference", args[0])
        
        # analyze にはフィルター適用で追加された 'target_low' も渡されているか
        mock_strategy.analyze.assert_called_once()
        strategy_args, _ = mock_strategy.analyze.call_args
        self.assertIn("target_low", strategy_args[0])
        
        mock_writer.write.assert_called_once_with("similarity.json", {"similarity": 0.85})

if __name__ == "__main__":
    unittest.main()
