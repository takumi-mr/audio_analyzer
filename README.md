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
├── reader/                   # 音声ファイルデコーダー (WAV, MP3, FLAC, OGG等対応)
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

## 💻 使用方法

コマンドラインから解析対象のファイルや実行したい解析戦略（BPM, キー, コード, サビ等）を動的に指定して実行できます。

### 基本コマンド

```bash
python main.py -i <入力音声ファイル> -o <出力結果JSON> -s <解析戦略>
```

#### オプション一覧:
- `-i`, `--input` (必須): 解析対象の音声ファイルへのパス (WAV, MP3, FLAC, OGG等)。
- `-o`, `--output` (デフォルト: `result.json`): 解析結果を保存するJSONファイル。
- `-s`, `--strategies` (複数指定可能, デフォルト: `all`): 実行する解析戦略を指定。
  - `beat`: BPM・テンポ検出
  - `sliding-beat`: 時系列テンポ遷移トラッキング
  - `key`: 主キー（調）判定
  - `chord`: 秒数ごとのコード進行特定
  - `chorus`: サビ（Chorus）区間自動特定
  - `genre`: 音楽ジャンル推定
  - `similarity`: 音声類似度計算 *(※ `--reference` が必須)*
  - `all`: `similarity` を除くすべての解析を一括実行し、結果を1つのファイルにマージして出力します。
- `-r`, `--reference`: 類似度計算用の比較対象音声ファイルのパス。
- `--no-separation`: AI音源分離 (Demucs) をスキップして直接解析を行います (処理が高速化します)。
- `--cutoff`: ドラム低域抽出用フィルターの遮断周波数 (Hz, デフォルト: 150.0)。
- `--threshold`: アタック音強調コンプレッサーの閾値 (デフォルト: 0.2)。
- `--ratio`: コンプレッサーの比率 (デフォルト: 3.0)。

### 💡 実行例

#### 1. 1つのファイルに対してBPM、キー、サビを一括解析し、結果をマージして出力
```bash
python main.py -i standard_dance.wav -o report.json -s beat key chorus
```

#### 2. 音源分離をスキップして高速にコード進行のみを特定
```bash
python main.py -i standard_dance.wav -o chords.json -s chord --no-separation
```

#### 3. 2つのファイルの音声類似度を比較
```bash
python main.py -i standard_dance.wav -r complex_prog_rock.wav -s similarity
```

---

## 📄 出力結果（マージ例）

複数の戦略を一括実行した場合、以下のようにマージされた単一の JSON ファイルとして結果が得られます。

```json
{
    "status": "success",
    "tempo_bpm": 117.5,
    "time_signature": "4/4",
    "estimated_key": "G Minor",
    "key_tonic": "G",
    "key_scale": "Minor",
    "chorus_sections": [
        { "start_sec": 0.0, "end_sec": 6.0 }
    ]
}
```

---

## 🧪 テストの実行

各モジュールが正常に動作しているかを確認するテストを実行できます。

```bash
python -m unittest discover -s tests -p "test_*.py"
```
