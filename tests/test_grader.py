"""Property-based and unit tests for the Grader.

# Feature: openenv-code-review
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.grader import grade, _EPSILON
from src.models import Action, DecisionEnum, InlineComment
from src.tasks import get_all_tasks, TaskDefinition

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_ALL_TASKS = get_all_tasks()

# Strategy: pick any task from the registry
tasks_st = st.sampled_from(_ALL_TASKS)

# Strategy: generate a valid review_body (0–4000 chars)
review_body_st = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=0,
    max_size=4000,
)

# Strategy: generate any valid decision
decision_st = st.sampled_from(list(DecisionEnum))

# Strategy: generate optional inline comments
inline_comment_st = st.builds(
    InlineComment,
    file_path=st.text(min_size=1, max_size=100),
    line_number=st.integers(min_value=1, max_value=10000),
    body=st.text(min_size=1, max_size=200),
)

inline_comments_st = st.one_of(
    st.none(),
    st.lists(inline_comment_st, min_size=0, max_size=5),
)

# Full random Action strategy
action_st = st.builds(
    Action,
    decision=decision_st,
    review_body=review_body_st,
    inline_comments=inline_comments_st,
)

# Step number strategy
step_number_st = st.integers(min_value=0, max_value=20)


# ---------------------------------------------------------------------------
# Property 1: Grader determinism
# Validates: Requirements 4.5
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 1: Grader determinism
@given(task=tasks_st, action=action_st, step_number=step_number_st)
@settings(max_examples=100)
def test_grader_determinism(task, action, step_number):
    """Calling grade twice with the same inputs returns identical Reward."""
    reward1 = grade(task, action, step_number)
    reward2 = grade(task, action, step_number)
    assert reward1.score == reward2.score
    assert reward1.breakdown == reward2.breakdown
    assert reward1.rationale == reward2.rationale


# ---------------------------------------------------------------------------
# Property 2: Score range invariant
# Validates: Requirements 4.1
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 2: Score range invariant
@given(task=tasks_st, action=action_st, step_number=step_number_st)
@settings(max_examples=100)
def test_score_range_invariant(task, action, step_number):
    """Score is always strictly inside (0.0, 1.0).

    The OpenEnv validator rejects task scores of exactly 0.0 or exactly 1.0,
    so the grader must never emit either boundary — see src/grader.py
    _strict_clamp and _EPSILON.
    """
    reward = grade(task, action, step_number)
    assert 0.0 < reward.score < 1.0


# ---------------------------------------------------------------------------
# Property 3: Wrong decision yields the minimum (epsilon) score
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 3: Wrong decision yields the minimum (epsilon) score
@given(
    task=st.sampled_from(
        [t for t in _ALL_TASKS if t.ground_truth_decision == "request_changes"]
    ),
    review_body=review_body_st,
    inline_comments=inline_comments_st,
    step_number=step_number_st,
)
@settings(max_examples=100)
def test_wrong_decision_yields_minimum_score(task, review_body, inline_comments, step_number):
    """Submitting 'approve' when ground truth is 'request_changes' yields the minimum score.

    The score is clamped to _EPSILON rather than 0.0 to satisfy the OpenEnv
    validator's strict (0, 1) range requirement.
    """
    action = Action(
        decision=DecisionEnum.approve,
        review_body=review_body,
        inline_comments=inline_comments,
    )
    reward = grade(task, action, step_number)
    assert reward.score == _EPSILON


# ---------------------------------------------------------------------------
# Property 4: Correct decision with all keywords yields maximum score
# Validates: Requirements 4.2, 5.4
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 4: Correct decision with all keywords yields maximum score
@given(task=tasks_st, step_number=step_number_st)
@settings(max_examples=100)
def test_perfect_action_yields_maximum_score(task, step_number):
    """Correct decision + all required keywords in review_body yields score >= 0.9."""
    # Build a review body that contains all required keywords
    review_body = " ".join(task.required_keywords) + " " + "detailed review " * 20
    review_body = review_body[:4000]  # stay within limit

    # Build inline comments targeting the required file if needed
    inline_comments = None
    if task.required_inline_file:
        inline_comments = [
            InlineComment(
                file_path=task.required_inline_file,
                line_number=1,
                body="Issue found here",
            )
        ]

    action = Action(
        decision=DecisionEnum(task.ground_truth_decision),
        review_body=review_body,
        inline_comments=inline_comments,
    )
    reward = grade(task, action, step_number)
    assert reward.score >= 0.9, (
        f"Expected score >= 0.9 for perfect action on task {task.task_id}, "
        f"got {reward.score}. Rationale: {reward.rationale}"
    )


# ---------------------------------------------------------------------------
# Property 5: Partial keyword match yields intermediate score
# Validates: Requirements 4.3, 5.3
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 5: Partial keyword match yields intermediate score
@given(
    task=st.sampled_from(
        [t for t in _ALL_TASKS if len(t.required_keywords) >= 2]
    ),
    step_number=step_number_st,
)
@settings(max_examples=100)
def test_partial_keywords_yield_intermediate_score(task, step_number):
    """Correct decision + strict subset of keywords yields 0.0 < score < 0.9."""
    # Use only the first keyword (strict subset)
    partial_body = task.required_keywords[0] + " some review text"

    action = Action(
        decision=DecisionEnum(task.ground_truth_decision),
        review_body=partial_body,
        inline_comments=None,
    )
    reward = grade(task, action, step_number)
    assert 0.0 < reward.score < 0.9, (
        f"Expected 0.0 < score < 0.9 for partial keywords on task {task.task_id}, "
        f"got {reward.score}. Rationale: {reward.rationale}"
    )


# ---------------------------------------------------------------------------
# Property 6: Breakdown sub-scores sum to total score
# Validates: Requirements 4.6, 5.6
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 6: Breakdown sub-scores sum to total score
@given(task=tasks_st, action=action_st, step_number=step_number_st)
@settings(max_examples=100)
def test_breakdown_sums_to_total_score(task, action, step_number):
    """Weighted sum of breakdown values equals reward.score, up to the clamp offset.

    When the raw weighted sum lands at exactly 0.0 (wrong decision) or 1.0
    (perfect answer), the grader clamps the stored score to _EPSILON or
    1 - _EPSILON to keep scores strictly inside (0, 1). That shifts score
    relative to the unclamped weighted sum by at most _EPSILON, so the
    allowed tolerance is _EPSILON plus a small float-drift margin.
    """
    # Only applies to terminal decisions (not comment intermediate rewards)
    assume(action.decision != DecisionEnum.comment)

    reward = grade(task, action, step_number)
    dc = reward.breakdown.get("decision_correctness", 0.0)
    ii = reward.breakdown.get("issue_identification", 0.0)
    rq = reward.breakdown.get("review_quality", 0.0)
    weighted_sum = dc * 0.4 + ii * 0.4 + rq * 0.2
    # Clamped score may differ from the raw sum by up to _EPSILON.
    tolerance = _EPSILON + 1e-6
    assert abs(reward.score - weighted_sum) <= tolerance, (
        f"score={reward.score}, weighted_sum={weighted_sum}, "
        f"breakdown={reward.breakdown}"
    )


# ---------------------------------------------------------------------------
# Property 10: Reward rationale is always non-empty
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 10: Reward rationale is always non-empty
@given(task=tasks_st, action=action_st, step_number=step_number_st)
@settings(max_examples=100)
def test_reward_rationale_is_non_empty(task, action, step_number):
    """Reward.rationale is always a non-empty string."""
    reward = grade(task, action, step_number)
    assert isinstance(reward.rationale, str)
    assert len(reward.rationale) > 0
