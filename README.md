# Audio Analyzer (高精度音声解析ツール)

本ツールは、市販の楽曲や音声ファイルを読み込み、AI音源分離と多様な音声分析技術を組み合わせて、**BPM（テンポ）検出、キー（調）推定、コード（和音）進行特定、サビ（Chorus）自動検出、曲の類似度判定、音楽ジャンル推定**を一括で行うことができる音声解析エンジンです。

---

## 🌟 主な機能と特徴

### 1. AI音源分離 & フィルターパイプライン (前処理)
正確な解析を行うため、解析の前に以下の前処理を実行してドラムやボーカルを抽出します。
- **Demucs音源分離**: Meta社開発のAI分離技術 `Demucs` を使用し、音声を「ボーカル」「ドラム」「ベース」「その他」の4つに完全に分離。
- **ドラム強調処理**: 分離したドラムトラックからキック音（低音域）のみを抽出し、さらにコンプレッサーをかけてビートのアタックを最大化します。これらを優先的にBPM検出へ引き渡すことで、テンポ判定のブレを徹底的に防ぎます。

### 2. 多角的な楽曲解析 (解析戦略)
- **テンポ・BPM分析**: 正確なBPMを推定します。また、曲を数秒ごとにスキャンしてテンポチェンジや変拍子の位置を捉えるスライディングウィンドウ解析にも対応。
- **主キー（調）判定**: ボーカルや旋律成分を分析し、楽曲全体のキー（例: `G Minor`, `C Major` 等）を特定。
- **コード進行特定**: 楽曲の音階情報をフレーム単位で解析し、秒数ごとのコード推移（C, Am, F, Gなど）をタイムスパン（開始秒・終了秒）で結合して出力します。
- **サビ（Chorus）自動特定**: 音圧の盛り上がり、音の明るさに加え、「同じメロディやコード構成が曲中で繰り返される」構造的特徴をクロマ類似度分析で捉え、サビの開始・終了秒数を自動的に特定。
- **類似度計算・ジャンル判定**: 音色とメロディ構成を総合評価する類似度比較や、明るさ・ノイズ感によるジャンル自動推定。

---

## 📂 フォルダ構成

```text
audio_analyzer/
├── analyzer/                 # 解析コア
│   ├── AudioAnalyzer.py      # 解析パイプラインの統合コントローラー
│   ├── filter/               # 前処理用フィルター (Demucs音源分離、帯域分割、コンプレッサ)
│   └── strategy/             # 各解析アルゴリズムの実装 (BPM, キー, コード, サビ, 類似度, ジャンル)
├── model/                    # 音声データの共通オブジェクトモデル
├── reader/                   # 音声ファイルデコーダー (WAV, MP3, OGG等対応)
├── writer/                   # 解析結果のJSONファイル出力
├── tests/                    # 自動単体テストスイート
├── main.py                   # サンプル実行エントリポイント
└── requirements.txt          # 依存ライブラリの定義
```

---

## 🚀 セットアップ (導入方法)

1. **仮想環境の作成と有効化**:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **依存関係のインストール**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(※ 初回の音源分離実行時のみ、Demucs AIモデルの自動ダウンロードが発生します)*

---

## 💻 使用方法と出力結果

付属の `main.py` を実行することで、フィルター適用からすべての解析を一括で検証できます。

```bash
python main.py
```

実行が完了すると、ルートディレクトリに以下の解析結果JSONファイルが出力されます。

### ① `result_simple.json` (BPM解析結果)
```json
{
    "status": "success",
    "tempo_bpm": 120.0,
    "time_signature": "4/4",
    "confidence": 0.95
}
```

### ② `result_key.json` (キー推定結果)
```json
{
    "status": "success",
    "estimated_key": "G Minor",
    "key_tonic": "G",
    "key_scale": "Minor",
    "confidence": 0.692
}
```

### ③ `result_chords.json` (コード進行判定結果)
楽曲全体のコード（和音）の推移を秒数付きでリスト化して出力します。
```json
{
    "status": "success",
    "chords": [
        { "start_sec": 0.0, "end_sec": 1.25, "chord": "Em" },
        { "start_sec": 1.25, "end_sec": 2.35, "chord": "C" },
        { "start_sec": 2.35, "end_sec": 4.25, "chord": "G" }
    ]
}
```

### ④ `result_chorus.json` (サビ区間特定結果)
最も盛り上がり、かつ繰り返し出現する「サビ」の位置を自動検出します。サビの繰り返しがある場合、複数の区間が出力されます。
```json
{
    "status": "success",
    "chorus_sections": [
        { "start_sec": 12.5, "end_sec": 28.0 },
        { "start_sec": 55.0, "end_sec": 70.5 }
    ],
    "confidence": 0.85
}
```

---

## 🧪 テストの実行

各モジュールが正常に動作しているかを確認するテストを実行できます。

```bash
python -m unittest discover -s tests -p "test_*.py"
```
