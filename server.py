from analyzer.filter.HermonicSeparationFilter import HarmonicSeparationFilter
import os
import shutil
import uuid
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# コア解析コンポーネントのインポート
from analyzer.AudioAnalyzer import AudioAnalyzer
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from analyzer.strategy.SimpleBeatStrategy import SimpleBeatStrategy
from analyzer.strategy.GenreClassificationStrategy import GenreClassificationStrategy
from analyzer.strategy.KeyDetectionStrategy import KeyDetectionStrategy
from analyzer.strategy.ChordEstimationStrategy import ChordEstimationStrategy
from analyzer.strategy.ChorusDetectionStrategy import ChorusDetectionStrategy
from analyzer.strategy.ChorusDetectionVocalStrategy import ChorusDetectionVocalStrategy
from analyzer.strategy.ChorusDetectionSSMStrategy import ChorusDetectionSSMStrategy
from analyzer.strategy.ChorusDetectionBeatSSMStrategy import ChorusDetectionBeatSSMStrategy
from reader.LibrosaAudioReader import LibrosaAudioReader
from writer.IResultWriter import IResultWriter

from analyzer.filter.IAudioFilter import IAudioFilter
from analyzer.filter.BandSplitterFilter import BandSplitterFilter
from analyzer.filter.CompressorFilter import CompressorFilter
from analyzer.filter.SourceSeparatorFilter import SourceSeparatorFilter

UPLOAD_DIR = os.path.abspath("temp_uploads")
TEMPLATES_DIR = os.path.abspath("templates")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# lifespan ハンドラでスタートアップのクリーンアップ処理
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(UPLOAD_DIR):
        try:
            shutil.rmtree(UPLOAD_DIR)
        except Exception as e:
            print(f"[Warning] Failed to clean up upload dir on startup: {e}")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    yield

app = FastAPI(title="Audio Analyzer API Server", lifespan=lifespan)

# CORSの許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InMemoryResultWriter(IResultWriter):
    """process_multi の結果をメモリ上で受け取るためのライター"""
    def __init__(self):
        self.result: Dict[str, Any] = {"status": "success"}

    def write(self, filepath: str, result: Dict[str, Any]) -> None:
        self.result.update(result)

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend HTML file not found in 'templates/' directory.")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    # 1. サポートフォーマットの検証
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")

    # 2. 一時保存用のファイルを作成
    unique_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary upload: {e}")

    # 3. AudioAnalyzerでフル解析を一括実行 (読込・Demucs分離は1回のみ)
    try:
        reader = LibrosaAudioReader()
        mem_writer = InMemoryResultWriter()
        
        # 音源分離およびドラムアタック強調前処理フィルター
        filters: List[IAudioFilter] = [
            SourceSeparatorFilter(target_key="target"),
            BandSplitterFilter(target_key="target_drums", cutoff_hz=150.0),
            CompressorFilter(target_key="target_drums_low", threshold=0.2, ratio=3.0),
            HarmonicSeparationFilter()
        ]
        
        analyzer = AudioAnalyzer(reader, mem_writer, SimpleBeatStrategy(), filters=filters)
        
        input_paths = {"target": temp_path}
        
        # 全ての解析戦略を一括実行 (process_multi で読込・分離が1回のみ)
        # サビ検出は3手法を並走させ、比較できるようにする
        strategies: List[IAnalysisStrategy] = [
            SimpleBeatStrategy(),
            KeyDetectionStrategy(),
            ChordEstimationStrategy(),
            ChorusDetectionStrategy(),       # 手法A: RMS + スペクトル重心
            ChorusDetectionVocalStrategy(),  # 手法B: ボーカル分離 + 適応閾値
            ChorusDetectionSSMStrategy(),    # 手法C: 自己類似行列(SSM)構造解析
            ChorusDetectionBeatSSMStrategy(), # 手法D: ビート同期SSM + 対角パス強調
            GenreClassificationStrategy()
        ]
        
        analyzer.process_multi(input_paths, strategies, "dummy_path")
            
        result_data = dict(mem_writer.result)
        result_data["audio_url"] = f"/api/audio/{unique_filename}"
        result_data["filename"] = filename
        
        return result_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")


@app.get("/api/audio/{filename}")
def get_audio_stream(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
