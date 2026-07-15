import unittest
import os
import tempfile
import json
from writer.JsonResultWriter import JsonResultWriter

class TestJsonResultWriter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.test_dir.name, "test_result.json")
        self.test_data = {
            "status": "success",
            "tempo_bpm": 120.0,
            "genre": "Dance",
            "match": True
        }
        
    def tearDown(self):
        self.test_dir.cleanup()
        
    def test_json_writer_success(self):
        writer = JsonResultWriter()
        writer.write(self.output_path, self.test_data)
        
        # ファイルが存在するか
        self.assertTrue(os.path.exists(self.output_path))
        
        # 中身が一致するか
        with open(self.output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.assertEqual(data, self.test_data)

if __name__ == "__main__":
    unittest.main()
