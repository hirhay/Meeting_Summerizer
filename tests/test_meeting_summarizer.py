import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from meeting_summarizer import (
    Config,
    DEFAULT_SUMMARIZE_MODEL,
    DEFAULT_TRANSCRIBE_MODEL,
    MeetingSummarizer,
    SUPPORTED_SUMMARIZE_MODELS,
    SUPPORTED_TRANSCRIBE_MODELS,
)

# --- Fixtures ---
@pytest.fixture
def mock_config():
    return Config(
        api_key="test-api-key",
        max_file_size_mb=10,
        audio_chunk_length_ms=1000,
        special_terms_file="test_terms.txt",
        default_transcribe_model=DEFAULT_TRANSCRIBE_MODEL,
        default_summarize_model=DEFAULT_SUMMARIZE_MODEL
    )

@pytest.fixture
def summarizer(mock_config):
    with patch("meeting_summarizer.OpenAI"):
        return MeetingSummarizer(mock_config)

# --- Tests ---

def test_config_initialization():
    config = Config(api_key="key")
    assert config.api_key == "key"
    assert config.default_transcribe_model == DEFAULT_TRANSCRIBE_MODEL
    assert config.default_transcribe_model in SUPPORTED_TRANSCRIBE_MODELS
    assert set(SUPPORTED_TRANSCRIBE_MODELS) == set(config.transcription_profiles)
    assert config.default_summarize_model == DEFAULT_SUMMARIZE_MODEL
    assert SUPPORTED_SUMMARIZE_MODELS == (DEFAULT_SUMMARIZE_MODEL,)

def test_unsupported_transcribe_model_is_rejected(summarizer):
    with pytest.raises(ValueError, match="未対応の文字起こしモデルです"):
        summarizer.get_transcription_profile("unsupported-model")

def test_build_prompt_general(summarizer):
    transcript = "これはテストです。"
    terms = ["用語A", "用語B"]
    prompt, sys_msg = summarizer.build_prompt(transcript, terms, "general")
    
    assert "あなたは優秀な要約AIです" in sys_msg or "あなたは優秀な要約AIです" in prompt
    assert "これはテストです。" in prompt
    assert "冗長な表現を避け" in prompt or "簡潔に" in prompt
    assert "- 用語A" in prompt
    assert "- 用語B" in prompt
    assert "2000文字以内" in prompt
    assert "## 概要" in prompt
    assert "## 重要ポイント" in prompt
    assert "## 補足" in prompt

def test_build_prompt_meeting(summarizer):
    prompt, _ = summarizer.build_prompt("会議本文", [], "meeting")

    assert "日本語のMarkdown" in prompt
    assert "Discord投稿" in prompt
    assert "## 概要" in prompt
    assert "## 決定事項" in prompt
    assert "## アクションアイテム" in prompt
    assert "## 論点・保留事項" in prompt
    assert "推測で補完しない" in prompt

def test_build_prompt_presentation(summarizer):
    prompt, _ = summarizer.build_prompt("発表本文", [], "presentation")

    assert "日本語のMarkdown" in prompt
    assert "## 要旨" in prompt
    assert "## 主要メッセージ" in prompt
    assert "## 根拠・重要ポイント" in prompt
    assert "## Q&A/補足" in prompt

def test_transcribe_audio_small_gpt_file_checks_duration(summarizer):
    # Mock os.path.isfile and os.path.getsize
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.getsize", return_value=100), \
         patch("builtins.open", mock_open(read_data=b"audio data")):
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 60 * 1000

        with patch("meeting_summarizer.AudioSegment.from_file", return_value=mock_audio):
            summarizer.client.audio.transcriptions.create.return_value = "Transcribed Text"

            result = summarizer.transcribe_audio("dummy.mp3", model="gpt-4o-transcribe")

            assert result == "Transcribed Text"
            summarizer.client.audio.transcriptions.create.assert_called_once()
            call_args = summarizer.client.audio.transcriptions.create.call_args
            assert call_args.kwargs['model'] == "gpt-4o-transcribe"

def test_transcribe_audio_small_whisper_file_skips_duration_check(summarizer):
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.getsize", return_value=100), \
         patch("builtins.open", mock_open(read_data=b"audio data")), \
         patch("meeting_summarizer.AudioSegment.from_file") as mock_from_file:
        summarizer.client.audio.transcriptions.create.return_value = "Whisper Text"

        result = summarizer.transcribe_audio("dummy.mp3", model="whisper-1")

        assert result == "Whisper Text"
        mock_from_file.assert_not_called()
        call_args = summarizer.client.audio.transcriptions.create.call_args
        assert call_args.kwargs['model'] == "whisper-1"

def test_transcribe_audio_large_file(summarizer):
    # Mock large file size
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.getsize", return_value=100 * 1024 * 1024): # 100MB
        
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 2000 * 1000 # 2000 seconds total
        
        # Slicing mock
        mock_chunk = MagicMock()
        mock_audio.__getitem__.return_value = mock_chunk
        
        with patch("meeting_summarizer.AudioSegment.from_file", return_value=mock_audio):
             # Mock _transcribe_chunk internally to avoid full chain simulation
             with patch.object(summarizer, "_transcribe_chunk", side_effect=["Chunk 1", "Chunk 2"]) as mock_chunk_transcribe:
                 
                 result = summarizer.transcribe_audio("large.mp3", model="gpt-4o-transcribe")
                 
                 assert "Chunk 1" in result
                 assert "Chunk 2" in result
                 assert "--- (次のチャンク) ---" in result
                 assert mock_chunk_transcribe.call_count == 2

def test_summarize_transcript(summarizer):
    summarizer.client.responses.create.return_value.output_text = "Summary Result"
    
    result = summarizer.summarize_transcript("Transc", [], "general", DEFAULT_SUMMARIZE_MODEL)
    assert result == "Summary Result"
    summarizer.client.responses.create.assert_called_once()
    call_kwargs = summarizer.client.responses.create.call_args.kwargs
    assert call_kwargs['model'] == DEFAULT_SUMMARIZE_MODEL
    assert "instructions" in call_kwargs
    assert "input" in call_kwargs
    assert call_kwargs["max_output_tokens"] == 1500
    assert "temperature" not in call_kwargs
    assert "max_tokens" not in call_kwargs
    assert "max_completion_tokens" not in call_kwargs

def test_summarize_transcript_uses_latest_gpt_model(summarizer):
    summarizer.client.responses.create.return_value.output_text = "Summary Result"
    
    result = summarizer.summarize_transcript("Transc", [], "general", DEFAULT_SUMMARIZE_MODEL)
    assert result == "Summary Result"
    summarizer.client.responses.create.assert_called_once()
    
    call_kwargs = summarizer.client.responses.create.call_args.kwargs
    assert call_kwargs['model'] == DEFAULT_SUMMARIZE_MODEL
    assert "instructions" in call_kwargs
    assert "input" in call_kwargs
    assert call_kwargs["max_output_tokens"] == 1500
    assert "temperature" not in call_kwargs
    assert "max_completion_tokens" not in call_kwargs
    assert "max_tokens" not in call_kwargs

def test_summarize_transcript_compresses_summary_for_discord(summarizer):
    long_summary = "あ" * (summarizer.config.summary_char_limit + 1)
    compressed_summary = "## 概要\n短い要約"
    summarizer.client.responses.create.side_effect = [
        MagicMock(output_text=long_summary),
        MagicMock(output_text=compressed_summary),
    ]

    result = summarizer.summarize_transcript("Transc", [], "general", DEFAULT_SUMMARIZE_MODEL)

    assert result == compressed_summary
    assert summarizer.client.responses.create.call_count == 2
    compression_kwargs = summarizer.client.responses.create.call_args_list[1].kwargs
    assert "再圧縮" in compression_kwargs["input"]
    assert compression_kwargs["max_output_tokens"] == 1200

def test_summarize_transcript_does_not_compress_short_summary(summarizer):
    summarizer.client.responses.create.return_value.output_text = "## 概要\n短い要約"

    result = summarizer.summarize_transcript("Transc", [], "general", DEFAULT_SUMMARIZE_MODEL)

    assert result == "## 概要\n短い要約"
    summarizer.client.responses.create.assert_called_once()
