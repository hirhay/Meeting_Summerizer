# Meeting Summarizer (Standalone Antigravity)

音声ファイルをAI (OpenAI Whisper & GPT Models) を使って文字起こしし、要約をMarkdownファイルとして保存するCLIツールです。

## 特徴

*   **高精度な文字起こし**: OpenAI Whisper API (`whisper-1`) を使用。
*   **柔軟な要約**: GPT-4o や **GPT-5 / O1系モデル** に対応。
*   **自動チャンク分割**: `pydub` を使用し、API制限を超える大きなファイルも安全に分割処理。
*   **ファイル整理機能**:
    *   音声ファイルごとにフォルダを自動作成。
    *   音声ファイルをそのフォルダにコピー。
    *   文字起こし結果 (`_transcript_<model>.txt`) をキャッシュし、再実行時はAPIコストを節約。
    *   要約結果 (`_summary_<model>.md`) も同フォルダに保存。
*   **GPT-5 / O1 対応**: 新しいモデル (`gpt-5`, `o1`) では自動的に `max_completion_tokens` を使用し、トークン制限を緩和。
*   **専門用語辞書**: `special_terms.txt` に用語を定義することで、文字起こしや要約の精度を向上。

## 必要要件

*   Python 3.8+
*   `ffmpeg` がシステムパスに通っていること (pydub用)
*   OpenAI API Key

## インストール

1.  **リポジトリをクローン** (またはファイルをダウンロード)

2.  **仮想環境の作成と有効化** (推奨)
    ```bash
    python -m venv venv
    
    # Windows
    .\venv\Scripts\activate
    
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **依存パッケージのインストール**
    ```bash
    pip install -r requirements.txt
    ```

4.  **環境変数の設定**
    `OPENAI_API_KEY` 環境変数にAPIキーを設定してください。
    ```bash
    # Windows PowerShell
    $env:OPENAI_API_KEY="sk-..."
    
    # Bash
    export OPENAI_API_KEY="sk-..."
    ```

## 使い方

基本コマンド:
```bash
python meeting_summarizer.py <音声ファイルパス> [オプション]
```

### オプション引数

*   `audio_file`: (必須) 処理対象の音声ファイルパス。
*   `--prompt-type`: 要約のタイプを指定 (デフォルト: `general`)
    *   `general`: 一般的な要約
    *   `meeting`: 会議議事録向け（目的、決定事項、アクションなど）
    *   `presentation`: プレゼン・講演向け（テーマ、結論、Q&Aなど）
*   `--model-transcribe`: 文字起こしモデル (デフォルト: `whisper-1`)
*   `--model-summarize`: 要約モデル (デフォルト: `gpt-4o`)。`gpt-5.1` などを指定可能。

### 実行例

**会議の要約を GPT-4o で作成:**
```bash
python meeting_summarizer.py meeting_recording.mp3 --prompt-type meeting
```

**GPT-5.1 を指定して実行:**
```bash
python meeting_summarizer.py lecture.mp3 --prompt-type presentation --model-summarize gpt-5.1
```

### 出力構造

コマンドを実行すると、カレントディレクトリに音声ファイル名のフォルダが作成され、すべてのファイルがそこに格納されます。

例: `MyMeeting.mp3` を処理した場合
```
./MyMeeting/
├── MyMeeting.mp3                  # コピーされた音声ファイル
├── MyMeeting_transcript_whisper-1.txt  # 文字起こしテキスト (キャッシュ)
└── MyMeeting_summary_gpt-4o.md    # 作成された要約
```

## 開発・テスト

開発用ライブラリ (`pytest`, `pytest-mock`) がインストールされていれば、テストを実行できます。

```bash
pip install pytest pytest-mock
pytest tests/
```

## ライセンス

MIT License
