"""Property-based tests for the Episode Manager.

# Feature: openenv-code-review
"""

import pytest
from fastapi import HTTPException
from hypothesis import given, settings
from hypothesis import strategies as st

from src.episode import EpisodeManager
from src.models import Action, DecisionEnum, InlineComment

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Valid difficulty values (plus None for round-robin)
difficulty_st = st.one_of(
    st.none(),
    st.sampled_from(["easy", "medium", "hard"]),
)

# Terminal decisions that end an episode
terminal_decision_st = st.sampled_from([DecisionEnum.approve, DecisionEnum.request_changes])

# Valid review body
review_body_st = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=200,
)

# A minimal valid terminal action
terminal_action_st = st.builds(
    Action,
    decision=terminal_decision_st,
    review_body=review_body_st,
    inline_comments=st.none(),
)


# ---------------------------------------------------------------------------
# Property 7: Episode done flag is monotonic
# Validates: Requirements 1.5, 1.6
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 7: Episode done flag is monotonic
@given(
    difficulty=difficulty_st,
    terminal_action=terminal_action_st,
    followup_action=terminal_action_st,
)
@settings(max_examples=100)
def test_done_flag_is_monotonic(difficulty, terminal_action, followup_action):
    """Once done=True, any subsequent step() raises HTTP 400."""
    em = EpisodeManager()
    em.reset(difficulty)

    # Drive episode to done
    _, _, done, _ = em.step(terminal_action)
    assert done is True

    # Any further step must raise HTTP 400
    with pytest.raises(HTTPException) as exc_info:
        em.step(followup_action)
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Property 8: Reset produces valid observation
# Validates: Requirements 1.4, 3.4, 3.5
# ---------------------------------------------------------------------------

# Feature: openenv-code-review, Property 8: Reset produces valid observation
@given(difficulty=difficulty_st)
@settings(max_examples=100)
def test_reset_produces_valid_observation(difficulty):
    """reset() always returns an Observation with step_number==0, non-empty diff,
    and task_difficulty in ['easy', 'medium', 'hard']."""
    em = EpisodeManager()
    obs = em.reset(difficulty)

    assert obs.step_number == 0
    assert len(obs.diff) > 0
    assert obs.task_difficulty in ("easy", "medium", "hard")
