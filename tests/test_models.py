"""Unit tests for Pydantic data model validation."""

import pytest
from pydantic import ValidationError
from src.models import (
    DecisionEnum,
    InlineComment,
    Observation,
    Action,
    Reward,
    StepResponse,
    StateSnapshot,
)


# --- DecisionEnum / Action.decision validation ---

def test_invalid_decision_raises_validation_error():
    """Invalid decision value should raise ValidationError."""
    with pytest.raises(ValidationError):
        Action(decision="merge", review_body="Looks good")


def test_valid_decision_approve():
    action = Action(decision="approve", review_body="LGTM")
    assert action.decision == DecisionEnum.approve


def test_valid_decision_request_changes():
    action = Action(decision="request_changes", review_body="Please fix the bug")
    assert action.decision == DecisionEnum.request_changes


def test_valid_decision_comment():
    action = Action(decision="comment", review_body="Just a note")
    assert action.decision == DecisionEnum.comment


# --- review_body length validation ---

def test_review_body_over_4000_chars_raises_validation_error():
    """review_body exceeding 4000 chars should raise ValidationError."""
    with pytest.raises(ValidationError):
        Action(decision="approve", review_body="x" * 4001)


def test_review_body_exactly_4000_chars_is_valid():
    action = Action(decision="approve", review_body="x" * 4000)
    assert len(action.review_body) == 4000


def test_review_body_empty_is_valid():
    action = Action(decision="comment", review_body="")
    assert action.review_body == ""


# --- Valid model construction ---

def test_inline_comment_constructs():
    comment = InlineComment(file_path="src/main.py", line_number=42, body="Off-by-one here")
    assert comment.file_path == "src/main.py"
    assert comment.line_number == 42


def test_observation_constructs():
    obs = Observation(
        pr_title="Fix login bug",
        pr_description="Fixes the login issue",
        diff="- old\n+ new",
        file_count=1,
        additions=1,
        deletions=1,
        step_number=0,
        task_difficulty="easy",
        episode_id="abc-123",
    )
    assert obs.step_number == 0
    assert obs.task_difficulty == "easy"


def test_reward_constructs():
    reward = Reward(
        score=0.75,
        rationale="Good review",
        breakdown={"decision_correctness": 1.0, "issue_identification": 0.3, "review_quality": 0.2},
    )
    assert reward.score == 0.75


def test_reward_score_out_of_range_raises():
    with pytest.raises(ValidationError):
        Reward(score=1.5, rationale="Too high", breakdown={})


def test_state_snapshot_constructs():
    snap = StateSnapshot(
        episode_id="ep-1",
        step_number=2,
        done=False,
        accumulated_score=0.4,
        current_task_difficulty="medium",
    )
    assert snap.done is False
