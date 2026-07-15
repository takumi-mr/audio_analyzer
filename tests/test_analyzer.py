import unittest
from unittest.mock import MagicMock
from analyzer.AudioAnalyzer import AudioAnalyzer
from reader.IAudioReader import IAudioReader
from writer.IResultWriter import IResultWriter
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from model.AudioSignal import AudioSignal

class TestAudioAnalyzer(unittest.TestCase):
    def test_analyzer_pipeline_success(self):
        # モックの作成
        mock_reader = MagicMock(spec=IAudioReader)
        mock_writer = MagicMock(spec=IResultWriter)
        mock_strategy = MagicMock(spec=IAnalysisStrategy)
        
        # ダミーの音声シグナルと解析結果をセットアップ
        dummy_signal = MagicMock(spec=AudioSignal)
        dummy_result = {"status": "success", "tempo_bpm": 120}
        
        mock_reader.read.return_value = dummy_signal
        mock_strategy.analyze.return_value = dummy_result
        
        analyzer = AudioAnalyzer(mock_reader, mock_writer, mock_strategy)
        
        # 1. process_file の呼び出し検証 (単一ファイルパイプライン)
        analyzer.process_file("input.wav", "output.json")
        
        # モックが正しい引数で呼び出されたことを検証
        mock_reader.read.assert_called_once_with("input.wav")
        # signals として辞書 {'target': dummy_signal} が strategy に渡されたか
        mock_strategy.analyze.assert_called_once_with({"target": dummy_signal}, None)
        mock_writer.write.assert_called_once_with("output.json", dummy_result)
        
        # モックの呼び出し履歴をリセット
        mock_reader.reset_mock()
        mock_writer.reset_mock()
        mock_strategy.reset_mock()
        
        # 2. process の呼び出し検証 (複数ファイルパイプライン)
        input_paths = {"target": "input1.wav", "reference": "input2.wav"}
        params = {"threshold": 0.8}
        
        # 各呼び出しに対する return_value の挙動をモックで表現
        sig1 = MagicMock(spec=AudioSignal)
        sig2 = MagicMock(spec=AudioSignal)
        
        # read の呼び出し引数によって異なる return_value を返す
        def mock_read_side_effect(path):
            if path == "input1.wav":
                return sig1
            elif path == "input2.wav":
                return sig2
            return None
            
        mock_reader.read.side_effect = mock_read_side_effect
        mock_strategy.analyze.return_value = {"similarity": 0.85}
        
        analyzer.process(input_paths, "similarity.json", params=params)
        
        # verify
        self.assertEqual(mock_reader.read.call_count, 2)
        mock_reader.read.assert_any_call("input1.wav")
        mock_reader.read.assert_any_call("input2.wav")
        
        # analyze が signals 辞書を受け取ってパラメータと共に呼ばれたか
        mock_strategy.analyze.assert_called_once_with(
            {"target": sig1, "reference": sig2},
            params
        )
        mock_writer.write.assert_called_once_with("similarity.json", {"similarity": 0.85})

if __name__ == "__main__":
    unittest.main()
