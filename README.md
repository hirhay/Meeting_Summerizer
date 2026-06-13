# Meeting Summarizer

音声ファイルをOpenAI APIで文字起こしし、最新版GPTモデルで要約してMarkdownファイルとして保存するCLIツールです。

## 特徴

*   **選べる文字起こしモデル**: 高精度な `gpt-4o-transcribe` と、長尺・大容量音声で扱いやすい `whisper-1` を選択可能。
*   **最新版GPTモデルでの要約**: 要約はResponses APIと `gpt-5.5` をデフォルトで使用。
*   **シンプルなResponses API実装**: `instructions` / `input` / `max_output_tokens` に統一し、古いモデル向けの `max_tokens` / `max_completion_tokens` 分岐は行いません。
*   **モデル別の自動チャンク分割**: `pydub` を使用し、`gpt-4o-transcribe` は20分単位、`whisper-1` は主に25MB上限を基準に分割します。
*   **ファイル整理機能**:
    *   音声ファイルごとにフォルダを自動作成。
    *   音声ファイルをそのフォルダにコピー。
    *   文字起こし結果 (`_transcript_<model>.txt`) をキャッシュし、再実行時はAPIコストを節約。
    *   要約結果 (`_summary_<model>.md`) も同フォルダに保存。
*   **専門用語辞書**: `special_terms.txt` に用語を定義することで、文字起こしや要約の精度を向上。

## 必要要件

*   Python 3.12 推奨
    *   このリポジトリでは `.python-version` で `3.12` を指定しています。
    *   Python 3.13以降では標準ライブラリの `audioop` が削除されているため、`pydub` 利用時に互換レイヤーが必要になる場合があります。
    *   `pyaudioop` の代替として、`requirements.txt` には `audioop-lts; python_version >= '3.13'` を条件付きで追加しています。
*   `ffmpeg` がシステムパスに通っていること (pydub用)
*   OpenAI API Key
*   OpenAI Python SDK: Responses API対応版 (`requirements.txt` では `openai>=1.78.1` とし、古い固定バージョンに縛られないようにしています)

## インストール

1.  **リポジトリをクローン** (またはファイルをダウンロード)

2.  **Python 3.12の仮想環境を作成して有効化** (推奨)
    ```bash
    python3.12 -m venv venv
    
    # Windows
    .\venv\Scripts\activate
    
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **依存パッケージのインストール**
    ```bash
    pip install --upgrade -r requirements.txt
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
*   `--model-transcribe`: 文字起こしモデル (デフォルト: `gpt-4o-transcribe`)
    *   `gpt-4o-transcribe`: 精度を優先する場合に推奨。長尺音声は20分単位に分割します。
    *   `whisper-1`: 25MB以内なら長めのMP3を一括処理しやすいモデル。25MBを超える場合は分割します。
*   `--model-summarize`: 要約モデル (デフォルト: `gpt-5.5`)


### 文字起こしモデルの使い分け

| モデル | 向いている用途 | 分割方針 |
| --- | --- | --- |
| `gpt-4o-transcribe` | 精度を優先したい会議・プレゼン録音 | 25MB上限に加え、長尺音声を避けるため20分単位で分割 |
| `whisper-1` | 長めのMP3をなるべく一括で処理したい場合、既存運用との互換性を優先する場合 | 25MB以内なら一括、25MB超過時は音声全体を分割 |

### 実行例

**会議の要約をデフォルトモデルで作成:**
```bash
python meeting_summarizer.py meeting_recording.mp3 --prompt-type meeting
```

**プレゼンテーション録音をデフォルトの最新版GPTモデルで要約:**
```bash
python meeting_summarizer.py lecture.mp3 --prompt-type presentation
```

**長めのMP3を `whisper-1` で文字起こし:**
```bash
python meeting_summarizer.py long_meeting.mp3 --prompt-type meeting --model-transcribe whisper-1
```

### 出力構造

コマンドを実行すると、カレントディレクトリに音声ファイル名のフォルダが作成され、すべてのファイルがそこに格納されます。

例: `MyMeeting.mp3` を処理した場合
```
./MyMeeting/
├── MyMeeting.mp3                             # コピーされた音声ファイル
├── MyMeeting_transcript_gpt-4o-transcribe.txt # 文字起こしテキスト (キャッシュ)
└── MyMeeting_summary_gpt-5.5.md              # 作成された要約
```

## 開発・テスト

開発用ライブラリ (`pytest`, `pytest-mock`) がインストールされていれば、テストを実行できます。

```bash
pip install --upgrade -r requirements.txt
pytest tests/
```

## ライセンス

MIT License
