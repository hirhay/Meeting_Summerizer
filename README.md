# Transcribe and Summarize

音声ファイルを文字起こしし、要約をMarkdownファイルとして保存するCLIツールです。

## 機能

* OpenAI Whisper API (`whisper-1`) で音声→テキスト
* GPT-4o でテキスト要約
* 大きいファイルは自動チャンク分割
* 専門用語リスト（`special_terms.txt`）の利用に対応
* 要約結果を処理日時フォルダに保存

## 必要要件

* Python 3.8+
* ffmpegコマンドがパスに通っていること
* 環境変数 `OPENAI_API_KEY` に OpenAI API キーを設定

## インストール

1. リポジトリをクローン
2. 仮想環境を作成・有効化

   * `python -m venv venv`
   * `source venv/bin/activate`
     （Windows PowerShell: `.\venv\Scripts\Activate.ps1`）
3. 依存パッケージをインストール

   * `pip install -r requirements.txt`

## 使い方

* `python meeting_summarizer.py path/to/audio.mp3 --prompt-type meeting`

  * `--prompt-type` は `general` / `meeting` / `presentation`

出力は `YYYYMMDD_HHMM/` フォルダ内に `<basename>.md` として保存されます。

## special_terms.txt の例

以下のように、1行ずつ専門用語を記述しておくと、要約時に適切にプロンプトに組み込まれます。コードと同じフォルダに置いてください。
  ```
  #コメント行（# 以降は無視されます）
  機械学習
  ディープラーニング
  アンサンブル学習
  最適化
  ハイパーパラメータ
  ```

## ライセンス

This project is licensed under the **MIT License**.
