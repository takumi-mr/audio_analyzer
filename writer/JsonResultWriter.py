import json
from writer.IResultWriter import IResultWriter
from typing import Dict, Any

class JsonResultWriter(IResultWriter):
    """JSON形式での書き出しの実装"""
    def write(self, filepath: str, result: Dict[str, Any]) -> None:
        print(f"[Writer] {filepath} に解析結果を出力します...")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
