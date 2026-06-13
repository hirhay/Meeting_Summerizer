#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声ファイルを文字起こしし、要約をMarkdownファイルに保存するスタンドアロンスクリプト。

使用方法:
    python meeting_summarizer.py <audio_file> [--prompt-type {general,meeting,presentation}] [--model-transcribe {gpt-4o-transcribe,whisper-1}] [--model-summarize gpt-5.5]

要約結果は処理実行日時(YYYYMMDD_HHMM)のフォルダに格納されます。
"""
import os
import sys
import argparse
import logging
import datetime
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# --- 設定クラス ---
@dataclass(frozen=True)
class TranscriptionModelProfile:
    """文字起こしモデルごとの安全な分割設定"""
    max_file_size_mb: int
    max_chunk_sec: Optional[int]
    check_duration_before_upload: bool


@dataclass
class Config:
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    max_file_size_mb: int = 25
    audio_chunk_length_ms: int = 30 * 60 * 1000  # 30分
    special_terms_file: str = os.path.join(os.path.dirname(__file__), "special_terms.txt")
    default_transcribe_model: str = "gpt-4o-transcribe"
    default_summarize_model: str = "gpt-5.5"
    transcription_profiles: Optional[Dict[str, TranscriptionModelProfile]] = None

    def __post_init__(self):
        if self.transcription_profiles is None:
            self.transcription_profiles = {
                # Whisperは主に25MBのアップロード上限を意識し、長さ制限は設けない。
                "whisper-1": TranscriptionModelProfile(
                    max_file_size_mb=25,
                    max_chunk_sec=None,
                    check_duration_before_upload=False,
                ),
                # gpt-4o-transcribeは精度面のメリットがある一方、長尺音声は安全側で20分単位に分割する。
                "gpt-4o-transcribe": TranscriptionModelProfile(
                    max_file_size_mb=25,
                    max_chunk_sec=20 * 60,
                    check_duration_before_upload=True,
                ),
            }

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED_TRANSCRIBE_MODELS = ("gpt-4o-transcribe", "whisper-1")

class MeetingSummarizer:
    def __init__(self, config: Config):
        self.config = config
        if not self.config.api_key:
            logger.error("環境変数 OPENAI_API_KEY が設定されていません。")
            sys.exit(1)
        self.client = OpenAI(api_key=self.config.api_key)

    def get_transcription_profile(self, model: str) -> TranscriptionModelProfile:
        """サポート対象の文字起こしモデル設定を取得する"""
        if model not in self.config.transcription_profiles:
            supported = ", ".join(sorted(self.config.transcription_profiles))
            raise ValueError(f"未対応の文字起こしモデルです: {model}。対応モデル: {supported}")
        return self.config.transcription_profiles[model]

    def load_special_terms(self) -> List[str]:
        """専門用語リストを読み込む"""
        if not os.path.exists(self.config.special_terms_file):
            return []
        try:
            with open(self.config.special_terms_file, 'r', encoding='utf-8') as f:
                return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
        except Exception as e:
            logger.warning(f"専門用語ファイルの読み込みに失敗しました: {e}")
            return []

    def build_prompt(self, transcript: str, terms: List[str], prompt_type: str) -> Tuple[str, str]:
        """要約のプロンプトとシステムメッセージを生成"""
        types = {
            "meeting": (
                "あなたは優秀な会議書記担当AIです。以下の会議の文字起こし内容を日本語で**極めて簡潔に**要約してください。\n"
                "会議の主な目的、前回会議のまとめ、今回議論された主要なポイント、およびネクストアクション"
                "（担当者と期限が明確な場合はそれも含む）を、構造化された箇条書き（Markdown形式）で示してください。\n"
                "**【重要】詳細は省き、本質的な意思決定とアクションアイテムのみを抽出してください。各項目は短くまとめてください。**",
                "前後の会話から文脈を理解し、会議の要点を正確かつ簡潔に抽出し、議事録としてまとめてください。"
            ),
            "presentation": (
                "あなたは優秀なレポート作成AIです。以下の発表、講演、または講義の文字起こし内容を日本語で**簡潔に**要約してください。\n"
                "発表の主要なテーマ、背景、提唱されている中心的なアイデアや議論、重要な論点や発見、"
                "そして結論や聴衆への主なメッセージを明確に箇条書き（Markdown形式）で示してください。\n"
                "**【重要】細かい説明は省略し、核心部分のみを抽出してください。**",
                "質疑応答は質問と回答を端的にまとめてください。発表内容の核心を捉えた簡潔な要約を作成してください。"
            ),
            "general": (
                "あなたは優秀な要約AIです。以下の文字起こし内容を日本語で**簡潔に**要約してください。\n"
                "テキスト全体の主要な情報を抽出し、最も重要なポイントやトピックを箇条書き（Markdown形式）で分かりやすく示してください。\n"
                "**【重要】冗長な表現を避け、要点のみをリストアップしてください。**",
                "与えられたテキストの内容を正確に把握し、非常に簡潔な要約を作成してください。"
            )
        }
        core, sys_msg = types.get(prompt_type, types['general'])
        
        term_section = ''
        if terms:
            term_list = '\n'.join(f"- {t}" for t in terms)
            term_section = f"元素名は元素記号で書いてください。また、以下の専門用語を適切に使用してください。:\n{term_list}"

        prompt = f"{core}\n{term_section}\n文字起こし:\n---\n{transcript}\n---\n要約 (Markdown):"
        return prompt, sys_msg

    def _transcribe_chunk(self, path: str, model: str, language: str = "ja") -> str:
        """指定モデルで単一チャンクを文字起こしする"""
        try:
            with open(path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="text",
                    language=language
                )
            return str(transcription)
        except Exception as e:
            logger.error(f"チャンク文字起こしエラー ({path}): {e}")
            raise

    def transcribe_audio(self, path: str, model: str) -> str:
        """音声ファイルを文字起こしする。サイズ超過時はチャンク分割して処理"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        profile = self.get_transcription_profile(model)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"文字起こし開始: {path} ({size_mb:.2f}MB, model={model})")

        # 小ファイルかつモデル側の時間チェックが不要な場合は一括処理
        if size_mb <= profile.max_file_size_mb and not profile.check_duration_before_upload:
            return self._transcribe_chunk(path, model=model)

        # gpt-4o-transcribeのように長尺音声を安全側で分割したいモデル、または大ファイルは音声長を見て分割する
        try:
            audio = AudioSegment.from_file(path)
        except CouldntDecodeError:
            logger.error(f"音声ファイルをデコードできません: {path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"音声読み込みエラー: {e}")
            sys.exit(1)

        duration_ms = len(audio)
        max_chunk_ms = profile.max_chunk_sec * 1000 if profile.max_chunk_sec else duration_ms

        # サイズ・長さのどちらもモデル設定内なら一括処理
        if size_mb <= profile.max_file_size_mb and duration_ms <= max_chunk_ms:
            return self._transcribe_chunk(path, model=model)

        transcripts = []
        _, ext = os.path.splitext(os.path.basename(path))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, start in enumerate(range(0, duration_ms, max_chunk_ms)):
                chunk = audio[start:start + max_chunk_ms]
                # 拡張子のドット処理を安全に
                fmt = ext.lstrip('.') if ext else "mp3"
                chunk_path = os.path.join(tmpdir, f"chunk_{idx}.{fmt}")
                chunk.export(chunk_path, format=fmt)
                
                logger.info(f"チャンク {idx+1} を文字起こし中 ({model}) ...")
                transcripts.append(self._transcribe_chunk(chunk_path, model=model))
        
        return "\n\n--- (次のチャンク) ---\n\n".join(transcripts)

    def summarize_transcript(self, transcript: str, terms: List[str], prompt_type: str, model: str) -> str:
        """文字起こしテキストを要約"""
        prompt, sys_msg = self.build_prompt(transcript, terms, prompt_type)
        try:
            logger.info(f"要約生成リクエスト中 (model={model})...")
            response = self.client.responses.create(
                model=model,
                instructions=sys_msg,
                input=prompt,
                max_output_tokens=1500,
            )

            content = getattr(response, "output_text", "")
            if content:
                return content.strip()
            return ""
        except Exception as e:
            logger.error(f"要約中にエラーが発生しました: {e}")
            sys.exit(1)

    def run(self, audio_file: str, prompt_type: str, transcribe_model: str, summarize_model: str):
        # 1. オーディオファイル名のフォルダ作成 (カレントディレクトリ)
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        # 拡張子を取得（ドット付き）
        _, ext = os.path.splitext(audio_file)
        
        target_dir = os.path.join(os.getcwd(), base_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 2. オーディオファイルをコピー
        target_audio_path = os.path.join(target_dir, os.path.basename(audio_file))
        if not os.path.exists(target_audio_path):
            try:
                import shutil
                shutil.copy2(audio_file, target_audio_path)
                logger.info(f"オーディオファイルをコピーしました: {target_audio_path}")
            except Exception as e:
                logger.error(f"オーディオファイルのコピーに失敗: {e}")
                # コピー失敗でも続行するか、エラーにするか。ここでは元ファイルを使うようにフォールバックも可能だが、
                # 要件に従いターゲットパスを正とするならエラーが良いが、柔軟に元パスも考慮。
                # ただし「そこに生成」とあるので、ターゲットパスで処理を進めるべき。
                # コピー失敗自体は fatal なので exit
                sys.exit(1)
        
        # 3. 文字起こし (既存チェック)
        # ファイル名にモデル名を含める
        transcript_filename = f"{base_name}_transcript_{transcribe_model}.txt"
        transcript_path = os.path.join(target_dir, transcript_filename)
        transcript = ""
        
        if os.path.exists(transcript_path):
            logger.info(f"既存の文字起こしファイルを使用します: {transcript_path}")
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            except Exception as e:
                logger.error(f"文字起こしファイルの読み込み失敗: {e}")
                sys.exit(1)
        else:
            # ターゲットディレクトリ内のオーディオファイルを使って文字起こし
            transcript = self.transcribe_audio(target_audio_path, model=transcribe_model)
            # 保存
            try:
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                logger.info(f"文字起こしを保存しました: {transcript_path}")
            except Exception as e:
                logger.error(f"文字起こし保存エラー: {e}")

        # 4. 要約
        terms = self.load_special_terms()
        summary = self.summarize_transcript(transcript, terms, prompt_type, model=summarize_model)

        # 結果表示
        print("=== 文字起こし (先頭500文字) ===\n", transcript[:500], "...(略)..." if len(transcript) > 500 else "")
        print("\n=== 要約 ===\n", summary)

        # 5. Markdown保存 (ターゲットディレクトリ内)
        # ファイル名にモデル名を含める
        md_filename = f"{base_name}_summary_{summarize_model}.md"
        md_path = os.path.join(target_dir, md_filename)
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# 要約: {base_name} ({summarize_model})\n\n")
                f.write(summary)
            logger.info(f"要約を'{md_path}'に保存しました。")
        except Exception as e:
            logger.error(f"ファイル保存エラー: {e}")

    def _save_results(self, audio_file: str, summary: str):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = now
        os.makedirs(output_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(audio_file))[0]
        md_path = os.path.join(output_dir, f"{base}.md")
        
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# 要約: {base}\n\n")
                f.write(summary)
            logger.info(f"要約を'{md_path}'に保存しました。")
        except Exception as e:
            logger.error(f"ファイル保存エラー: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルを文字起こしし、要約をMarkdownファイルに保存するスクリプト。"
    )
    parser.add_argument("audio_file", help="処理対象の音声ファイルへのパス")
    parser.add_argument(
        "--prompt-type",
        choices=["general", "meeting", "presentation"],
        default="general",
        help="要約のプロンプトタイプを指定"
    )
    parser.add_argument(
        "--model-transcribe",
        choices=["gpt-4o-transcribe", "whisper-1"],
        default="gpt-4o-transcribe",
        help="文字起こしモデル (default: gpt-4o-transcribe)"
    )
    parser.add_argument("--model-summarize", default="gpt-5.5", help="要約モデル (default: gpt-5.5)")

    args = parser.parse_args()

    config = Config()
    if args.model_transcribe:
        config.default_transcribe_model = args.model_transcribe
    if args.model_summarize:
        config.default_summarize_model = args.model_summarize

    app = MeetingSummarizer(config)
    app.run(
        args.audio_file,
        prompt_type=args.prompt_type,
        transcribe_model=config.default_transcribe_model,
        summarize_model=config.default_summarize_model
    )

if __name__ == "__main__":
    main()
