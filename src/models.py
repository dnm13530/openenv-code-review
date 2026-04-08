"""Pydantic v2 data models for the OpenEnv code review environment."""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


class DecisionEnum(str, Enum):
    """Valid review decision types."""
    approve = "approve"
    request_changes = "request_changes"
    comment = "comment"


class InlineComment(BaseModel):
    """An inline comment on a specific file and line."""
    file_path: str
    line_number: int
    body: str


class Observation(BaseModel):
    """What the agent observes at each step."""
    pr_title: str
    pr_description: str
    diff: str
    file_count: int
    additions: int
    deletions: int
    step_number: int
    task_difficulty: Literal["easy", "medium", "hard"]
    episode_id: str


class Action(BaseModel):
    """An action submitted by the agent."""
    decision: DecisionEnum
    review_body: str = Field(..., max_length=4000)
    inline_comments: Optional[list[InlineComment]] = None


class Reward(BaseModel):
    """Reward signal with score, rationale, and breakdown."""
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    breakdown: dict[str, float]


class StepResponse(BaseModel):
    """Response from a step action."""
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class StateSnapshot(BaseModel):
    """Current episode state snapshot."""
    episode_id: str
    step_number: int
    done: bool
    accumulated_score: float
    current_task_difficulty: str
