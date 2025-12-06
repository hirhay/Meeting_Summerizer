import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from meeting_summarizer import MeetingSummarizer, Config

# --- Fixtures ---
@pytest.fixture
def mock_config():
    return Config(
        api_key="test-api-key",
        max_file_size_mb=10,
        audio_chunk_length_ms=1000,
        special_terms_file="test_terms.txt",
        default_transcribe_model="whisper-1",
        default_summarize_model="gpt-4o"
    )

@pytest.fixture
def summarizer(mock_config):
    with patch("meeting_summarizer.OpenAI"):
        return MeetingSummarizer(mock_config)

# --- Tests ---

def test_config_initialization():
    config = Config(api_key="key")
    assert config.api_key == "key"
    assert config.default_transcribe_model == "whisper-1"

def test_build_prompt_general(summarizer):
    transcript = "これはテストです。"
    terms = ["用語A", "用語B"]
    prompt, sys_msg = summarizer.build_prompt(transcript, terms, "general")
    
    assert "あなたは優秀な要約AIです" in sys_msg or "あなたは優秀な要約AIです" in prompt
    assert "これはテストです。" in prompt
    assert "冗長な表現を避け" in prompt or "簡潔に" in prompt
    assert "- 用語A" in prompt
    assert "- 用語B" in prompt

def test_transcribe_audio_small_file(summarizer):
    # Mock os.path.isfile and os.path.getsize
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.getsize", return_value=100), \
         patch("builtins.open", mock_open(read_data=b"audio data")):
        
        # Mock client response
        summarizer.client.audio.transcriptions.create.return_value = "Transcribed Text"
        
        result = summarizer.transcribe_audio("dummy.mp3", model="whisper-1")
        
        assert result == "Transcribed Text"
        summarizer.client.audio.transcriptions.create.assert_called_once()
        call_args = summarizer.client.audio.transcriptions.create.call_args
        assert call_args.kwargs['model'] == "whisper-1"

def test_transcribe_audio_large_file(summarizer):
    # Mock large file size
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.getsize", return_value=100 * 1024 * 1024): # 100MB
        
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 2000 # 2 seconds total
        
        # Slicing mock
        mock_chunk = MagicMock()
        mock_audio.__getitem__.return_value = mock_chunk
        
        with patch("meeting_summarizer.AudioSegment.from_file", return_value=mock_audio):
             # Mock _transcribe_chunk internally to avoid full chain simulation
             with patch.object(summarizer, "_transcribe_chunk", side_effect=["Chunk 1", "Chunk 2"]) as mock_chunk_transcribe:
                 
                 # Set config chunk size small to force loop
                 summarizer.config.chunk_sec_default = 1 # 1 second chunk
                 
                 result = summarizer.transcribe_audio("large.mp3", model="whisper-1")
                 
                 assert "Chunk 1" in result
                 assert "Chunk 2" in result
                 assert "--- (次のチャンク) ---" in result
                 assert mock_chunk_transcribe.call_count == 2

def test_summarize_transcript(summarizer):
    summarizer.client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Summary Result"))
    ]
    
    result = summarizer.summarize_transcript("Transc", [], "general", "gpt-4o")
    assert result == "Summary Result"
    summarizer.client.chat.completions.create.assert_called_once()
    # gpt-4o should use max_tokens
    assert summarizer.client.chat.completions.create.call_args.kwargs['model'] == "gpt-4o"
    assert "max_tokens" in summarizer.client.chat.completions.create.call_args.kwargs
    assert "max_completion_tokens" not in summarizer.client.chat.completions.create.call_args.kwargs

def test_summarize_transcript_gpt5(summarizer):
    summarizer.client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Summary Result"))
    ]
    
    result = summarizer.summarize_transcript("Transc", [], "general", "gpt-5.1")
    assert result == "Summary Result"
    summarizer.client.chat.completions.create.assert_called_once()
    
    call_kwargs = summarizer.client.chat.completions.create.call_args.kwargs
    assert call_kwargs['model'] == "gpt-5.1"
    # gpt-5 should use max_completion_tokens and NOT max_tokens
    assert "max_completion_tokens" in call_kwargs
    assert call_kwargs["max_completion_tokens"] == 20000
    assert "max_tokens" not in call_kwargs

