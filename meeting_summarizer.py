#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声ファイルを文字起こしし、要約をMarkdownファイルに保存するスタンドアロンスクリプト。

使用方法:
    python transcribe_and_summarize.py <audio_file> [--prompt-type {general,meeting,presentation}]

要約結果は処理実行日時(YYYYMMDD_HHMM)のフォルダに格納されます。
"""
import os
import sys
import argparse
import logging
import datetime
import tempfile
from openai import OpenAI
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# --- 設定 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("環境変数 OPENAI_API_KEY が設定されていません。スクリプトを終了します。")
    sys.exit(1)

MAX_FILE_SIZE_MB = 25               # Whisper APIのサイズ制限（MB）
AUDIO_CHUNK_LENGTH_MS = 30 * 60 * 1000  # 1チャンクあたり30分
SPECIAL_TERMS_FILE = os.path.join(os.path.dirname(__file__), "special_terms.txt")

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 文字起こし関数 ---
def _transcribe_chunk(client: OpenAI, path: str) -> str:
    """Whisper APIで単一チャンクを文字起こしする"""
    with open(path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )


def transcribe_audio(path: str) -> str:
    """音声ファイルを文字起こし。サイズ超過時はチャンク分割して処理"""
    if not os.path.isfile(path):
        logger.error(f"ファイルが見つかりません: {path}")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"文字起こし開始: {path} ({size_mb:.2f}MB)")

    # 小ファイルはそのまま
    if size_mb <= MAX_FILE_SIZE_MB:
        return _transcribe_chunk(client, path)

    # 大ファイルはチャンク分割
    try:
        audio = AudioSegment.from_file(path)
    except CouldntDecodeError:
        logger.error(f"音声ファイルをデコードできません: {path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"音声読み込みエラー: {e}")
        sys.exit(1)

    transcripts = []
    base, ext = os.path.splitext(os.path.basename(path))
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, start in enumerate(range(0, len(audio), AUDIO_CHUNK_LENGTH_MS)):
            chunk = audio[start:start + AUDIO_CHUNK_LENGTH_MS]
            chunk_path = os.path.join(tmpdir, f"chunk_{idx}{ext}")
            chunk.export(chunk_path, format=ext.lstrip('.'))
            logger.info(f"チャンク{idx+1}を文字起こし中...")
            transcripts.append(_transcribe_chunk(client, chunk_path))
    return "\n\n--- (次のチャンク) ---\n\n".join(transcripts)

# --- 要約関数 ---
def load_special_terms() -> list[str]:
    """専門用語リストを読み込む"""
    if not os.path.exists(SPECIAL_TERMS_FILE):
        return []
    with open(SPECIAL_TERMS_FILE, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]


def build_prompt(transcript: str, terms: list[str], prompt_type: str) -> tuple[str, str]:
    """要約のプロンプトとシステムメッセージを生成"""
    types = {
        "meeting": (
            "あなたは優秀な会議書記担当AIです。以下の会議の文字起こし内容を日本語で要約してください。\n会議の主な目的、議論された主要なポイント、参加者の発言の要点、決定事項、およびネクストアクション（担当者と期限が明確な場合はそれも含む）を、構造化された箇条書き（Markdown形式）で示してください。",
            "会議の要点を正確に抽出し、議事録として分かりやすくまとめてください。"
        ),
        "presentation": (
            "あなたは優秀なレポート作成AIです。以下の発表、講演、または講義の文字起こし内容を日本語で要約してください。\n発表の主要なテーマ、背景、提唱されている中心的なアイデアや議論、重要な論点や発見、そして結論や聴衆への主なメッセージを明確に箇条書き（Markdown形式）で示してください。",
            "発表内容の核心を捉えた要約を作成してください。"
        ),
        "general": (
            "あなたは優秀な要約AIです。以下の文字起こし内容を日本語で簡潔に要約してください。\nテキスト全体の主要な情報を抽出し、最も重要なポイントやトピックを箇条書き（Markdown形式）で分かりやすく示してください。",
            "与えられたテキストの内容を正確に把握し、簡潔な要約を作成してください。"
        )
    }
    core, sys_msg = types.get(prompt_type, types['general'])
    term_section = '' if not terms else (
        "以下の専門用語を適切に使用してください:\n" + '\n'.join(f"- {t}" for t in terms)
    )
    prompt = f"{core}\n{term_section}\n文字起こし:\n---\n{transcript}\n---\n要約 (Markdown):"
    return prompt, sys_msg


def summarize_transcript(transcript: str, terms: list[str], prompt_type: str) -> str:
    """文字起こしテキストをGPT-4oで要約"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt, sys_msg = build_prompt(transcript, terms, prompt_type)
    try:
        logger.info("要約生成リクエスト中...")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"要約中にエラー: {e}")
        sys.exit(1)

# --- エントリポイント ---
def main():
    """引数解析、文字起こし→要約、Markdown保存処理"""
    parser = argparse.ArgumentParser(
        description="音声ファイルを文字起こしし、要約をMarkdownファイルに保存するスクリプト。"
    )
    parser.add_argument(
        "audio_file", help="処理対象の音声ファイルへのパス"
    )
    parser.add_argument(
        "--prompt-type", choices=["general", "meeting", "presentation"],
        default="general", help="要約のプロンプトタイプを指定"
    )
    args = parser.parse_args()

    # 文字起こし
    transcript = transcribe_audio(args.audio_file)

    # 要約
    terms = load_special_terms()
    summary = summarize_transcript(transcript, terms, args.prompt_type)

    # 結果表示
    print("=== 文字起こし ===\n", transcript)
    print("\n=== 要約 ===\n", summary)

    # 出力ディレクトリ作成
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = now
    os.makedirs(output_dir, exist_ok=True)

    # Markdown保存
    base = os.path.splitext(os.path.basename(args.audio_file))[0]
    md_path = os.path.join(output_dir, f"{base}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 要約: {base}\n\n")
        f.write(summary)
    logger.info(f"要約を'{md_path}'に保存しました。")

if __name__ == "__main__":
    main()
