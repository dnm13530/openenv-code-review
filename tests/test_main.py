"""Tests for the FastAPI application endpoints.

# Feature: openenv-code-review
"""

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from src.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset(difficulty=None):
    body = {}
    if difficulty is not None:
        body["difficulty"] = difficulty
    return client.post("/reset", json=body)


def _step(decision="request_changes", review_body="looks good", inline_comments=None):
    payload = {"decision": decision, "review_body": review_body}
    if inline_comments is not None:
        payload["inline_comments"] = inline_comments
    return client.post("/step", json=payload)


# ---------------------------------------------------------------------------
# Unit tests — 6.2
# Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6
# ---------------------------------------------------------------------------

def test_reset_returns_valid_observation():
    """POST /reset returns a valid Observation."""
    resp = _reset()
    assert resp.status_code == 200
    data = resp.json()
    assert "pr_title" in data
    assert "diff" in data
    assert data["step_number"] == 0
    assert data["task_difficulty"] in ("easy", "medium", "hard")
    assert "episode_id" in data


def test_reset_with_difficulty_returns_matching_task():
    """POST /reset with difficulty returns a task of that difficulty."""
    for diff in ("easy", "medium", "hard"):
        resp = _reset(difficulty=diff)
        assert resp.status_code == 200
        assert resp.json()["task_difficulty"] == diff


def test_reset_with_invalid_difficulty_returns_422():
    """POST /reset with an unknown difficulty returns HTTP 422."""
    resp = _reset(difficulty="impossible")
    assert resp.status_code == 422


def test_step_returns_step_response():
    """POST /step returns a StepResponse with correct shape."""
    _reset()
    resp = _step(decision="request_changes", review_body="off-by-one error found")
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert "info" in data
    assert "score" in data["reward"]
    assert "rationale" in data["reward"]
    assert "breakdown" in data["reward"]


def test_step_done_after_terminal_decision():
    """POST /step with a terminal decision sets done=True."""
    _reset()
    resp = _step(decision="approve", review_body="looks fine")
    assert resp.status_code == 200
    assert resp.json()["done"] is True


def test_step_after_done_returns_400():
    """POST /step after episode is done returns HTTP 400."""
    _reset()
    _step(decision="approve", review_body="looks fine")
    resp = _step(decision="approve", review_body="looks fine again")
    assert resp.status_code == 400


def test_get_state_returns_state_snapshot():
    """GET /state returns a StateSnapshot."""
    _reset()
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "episode_id" in data
    assert "step_number" in data
    assert "done" in data
    assert "accumulated_score" in data
    assert "current_task_difficulty" in data


def test_root_returns_info():
    """GET / returns a JSON info page."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


# ---------------------------------------------------------------------------
# Property 9: Action validation rejects oversized review body
# Feature: openenv-code-review, Property 9: Action validation rejects oversized review body
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------

@given(
    extra=st.integers(min_value=1, max_value=1000),
    decision=st.sampled_from(["approve", "request_changes", "comment"]),
)
@settings(max_examples=50)
def test_oversized_review_body_returns_422(extra, decision):
    """POST /step with review_body longer than 4000 chars returns HTTP 422."""
    oversized_body = "x" * (4000 + extra)
    resp = client.post("/step", json={"decision": decision, "review_body": oversized_body})
    assert resp.status_code == 422
