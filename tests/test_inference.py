"""Unit tests for inference.py helper functions.

Validates: Requirements 8.5
"""

import os
import sys
import pytest

from inference import build_prompt, load_env_vars, parse_llm_response


# ---------------------------------------------------------------------------
# Tests for load_env_vars — missing env vars cause sys.exit(1)
# ---------------------------------------------------------------------------

def test_missing_all_env_vars_exits(monkeypatch):
    """HF_TOKEN missing (API_BASE_URL and MODEL_NAME have defaults) → sys.exit(1)."""
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        load_env_vars()
    assert exc_info.value.code == 1


def test_missing_one_env_var_exits(monkeypatch):
    """HF_TOKEN missing → sys.exit(1)."""
    monkeypatch.setenv("API_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        load_env_vars()
    assert exc_info.value.code == 1


def test_all_env_vars_present_returns_values(monkeypatch):
    """All env vars set → returns the three values."""
    monkeypatch.setenv("API_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("HF_TOKEN", "hf_abc123")

    api_base_url, model_name, hf_token = load_env_vars()
    assert api_base_url == "http://localhost:8000"
    assert model_name == "test-model"
    assert hf_token == "hf_abc123"


def test_missing_env_var_error_message_is_descriptive(monkeypatch, capsys):
    """Error message names HF_TOKEN as the missing required variable."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit):
        load_env_vars()

    captured = capsys.readouterr()
    assert "HF_TOKEN" in captured.err


# ---------------------------------------------------------------------------
# Tests for build_prompt — returns non-empty string
# ---------------------------------------------------------------------------

SAMPLE_OBSERVATION = {
    "pr_title": "Fix off-by-one error in loop",
    "pr_description": "Fixes the loop boundary condition",
    "diff": "- for i in range(n):\n+ for i in range(n + 1):",
    "file_count": 1,
    "additions": 1,
    "deletions": 1,
    "step_number": 0,
    "task_difficulty": "easy",
    "episode_id": "test-episode-id",
}


def test_build_prompt_returns_non_empty_string():
    """build_prompt returns a non-empty string."""
    prompt = build_prompt(SAMPLE_OBSERVATION)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_prompt_includes_pr_title():
    """build_prompt includes the PR title in the output."""
    prompt = build_prompt(SAMPLE_OBSERVATION)
    assert SAMPLE_OBSERVATION["pr_title"] in prompt


def test_build_prompt_includes_diff():
    """build_prompt includes the diff content."""
    prompt = build_prompt(SAMPLE_OBSERVATION)
    assert SAMPLE_OBSERVATION["diff"] in prompt


# ---------------------------------------------------------------------------
# Tests for parse_llm_response — handles malformed JSON gracefully
# ---------------------------------------------------------------------------

def test_parse_valid_json_response():
    """Valid JSON response is parsed correctly."""
    response = '{"decision": "request_changes", "review_body": "Found a bug."}'
    result = parse_llm_response(response)
    assert result is not None
    assert result["decision"] == "request_changes"
    assert result["review_body"] == "Found a bug."


def test_parse_malformed_json_returns_none():
    """Malformed JSON returns None without raising."""
    result = parse_llm_response("this is not json at all")
    assert result is None


def test_parse_empty_string_returns_none():
    """Empty string returns None."""
    result = parse_llm_response("")
    assert result is None


def test_parse_json_in_markdown_code_fence():
    """JSON wrapped in markdown code fences is parsed correctly."""
    response = '```json\n{"decision": "approve", "review_body": "LGTM"}\n```'
    result = parse_llm_response(response)
    assert result is not None
    assert result["decision"] == "approve"


def test_parse_invalid_decision_returns_none():
    """Response with invalid decision value returns None."""
    response = '{"decision": "invalid_decision", "review_body": "some review"}'
    result = parse_llm_response(response)
    assert result is None


def test_parse_missing_required_fields_returns_none():
    """Response missing required fields returns None."""
    result = parse_llm_response('{"decision": "approve"}')
    assert result is None


def test_parse_oversized_review_body_is_truncated():
    """review_body exceeding 4000 chars is truncated to 4000."""
    long_body = "x" * 5000
    response = json_str = f'{{"decision": "approve", "review_body": "{long_body}"}}'
    result = parse_llm_response(response)
    assert result is not None
    assert len(result["review_body"]) == 4000


def test_parse_with_inline_comments():
    """Response with inline_comments is parsed correctly."""
    response = (
        '{"decision": "request_changes", "review_body": "See inline.", '
        '"inline_comments": [{"file_path": "main.py", "line_number": 5, "body": "Bug here"}]}'
    )
    result = parse_llm_response(response)
    assert result is not None
    assert len(result["inline_comments"]) == 1
    assert result["inline_comments"][0]["file_path"] == "main.py"
