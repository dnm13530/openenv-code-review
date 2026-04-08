"""Stateless, deterministic grader for the OpenEnv code review environment."""

from src.models import Action, DecisionEnum, Reward
from src.tasks import TaskDefinition


def grade(task: TaskDefinition, action: Action, step_number: int) -> Reward:
    """Grade an agent's action against the task rubric.

    Returns a Reward with score in [0.0, 1.0], a rationale string, and a
    breakdown dict with keys: decision_correctness, issue_identification,
    review_quality.
    """
    # ------------------------------------------------------------------
    # Intermediate reward for 'comment' actions (not a terminal decision)
    # ------------------------------------------------------------------
    if action.decision == DecisionEnum.comment:
        return _grade_comment(task, action, step_number)

    # ------------------------------------------------------------------
    # Terminal decision scoring
    # ------------------------------------------------------------------
    decision_correctness = _score_decision(task, action)

    if decision_correctness == 0.0:
        breakdown = {
            "decision_correctness": 0.0,
            "issue_identification": 0.0,
            "review_quality": 0.0,
        }
        rationale = (
            f"Incorrect decision '{action.decision.value}'; "
            f"expected '{task.ground_truth_decision}'. Score: 0.0."
        )
        return Reward(score=0.0, rationale=rationale, breakdown=breakdown)

    issue_identification = _score_issue_identification(task, action)
    review_quality = _score_review_quality(task, action)

    score = (
        decision_correctness * 0.4
        + issue_identification * 0.4
        + review_quality * 0.2
    )
    # Clamp to [0.0, 1.0] to guard against floating-point drift
    score = max(0.0, min(1.0, score))

    breakdown = {
        "decision_correctness": decision_correctness,
        "issue_identification": issue_identification,
        "review_quality": review_quality,
    }

    matched = _matched_keywords(task, action)
    rationale_parts = [
        f"Decision '{action.decision.value}' is correct (score contribution: "
        f"{decision_correctness * 0.4:.2f}).",
    ]
    if matched:
        rationale_parts.append(
            f"Identified {len(matched)}/{len(task.required_keywords)} required "
            f"keywords ({', '.join(matched)}); issue_identification={issue_identification:.2f}."
        )
    else:
        rationale_parts.append(
            "No required keywords found in review body; issue_identification=0.0."
        )
    rationale_parts.append(
        f"Review quality score: {review_quality:.2f} "
        f"(body length={len(action.review_body)}, "
        f"inline_file_hit={'yes' if _has_inline_on_required_file(task, action) else 'no'})."
    )
    rationale_parts.append(f"Final score: {score:.4f}.")
    rationale = " ".join(rationale_parts)

    return Reward(score=score, rationale=rationale, breakdown=breakdown)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_decision(task: TaskDefinition, action: Action) -> float:
    """Return 1.0 if decision matches ground truth, else 0.0."""
    return 1.0 if action.decision.value == task.ground_truth_decision else 0.0


def _matched_keywords(task: TaskDefinition, action: Action) -> list[str]:
    """Return the list of required_keywords found (case-insensitive) in review_body.

    Uses flexible matching: a keyword matches if ANY of its words appear in the
    review body, or if the full phrase appears. This rewards partial/synonym usage.
    """
    body_lower = action.review_body.lower()
    matched = []
    for kw in task.required_keywords:
        kw_lower = kw.lower()
        # Full phrase match
        if kw_lower in body_lower:
            matched.append(kw)
            continue
        # Any individual word in the keyword phrase (length > 3 to skip noise words)
        words = [w for w in kw_lower.split() if len(w) > 3]
        if words and any(w in body_lower for w in words):
            matched.append(kw)
    return matched


def _score_issue_identification(task: TaskDefinition, action: Action) -> float:
    """Proportional keyword match score in [0.0, 1.0]."""
    if not task.required_keywords:
        return 1.0
    matched = _matched_keywords(task, action)
    ratio = len(matched) / len(task.required_keywords)
    return round(ratio, 6)


def _has_inline_on_required_file(task: TaskDefinition, action: Action) -> bool:
    """True if the action has an inline comment targeting the required file."""
    if not task.required_inline_file:
        return False
    if not action.inline_comments:
        return False
    return any(
        ic.file_path == task.required_inline_file for ic in action.inline_comments
    )


def _score_review_quality(task: TaskDefinition, action: Action) -> float:
    """Score review quality in [0.0, 1.0].

    Scoring strategy:
    - If all required keywords are matched, award full quality (1.0). This
      satisfies Requirement 4.2: correct decision + all keywords → score >= 0.9.
    - Otherwise, combine body-length contribution (up to 0.5) and an inline
      comment bonus (0.5) for tasks that require a specific file.
    """
    matched = _matched_keywords(task, action)
    all_keywords_matched = (
        len(task.required_keywords) > 0
        and len(matched) == len(task.required_keywords)
    )

    # Full quality when all keywords are present
    if all_keywords_matched:
        return 1.0

    # Length contribution: 0.0 at 0 chars, 0.5 at >= 500 chars
    length_score = min(len(action.review_body) / 500.0, 1.0) * 0.5

    # Inline comment bonus
    inline_score = 0.5 if _has_inline_on_required_file(task, action) else 0.0

    # For tasks without a required_inline_file, the full 1.0 comes from length
    if task.required_inline_file is None:
        quality = min(len(action.review_body) / 500.0, 1.0)
    else:
        quality = length_score + inline_score

    return round(min(quality, 1.0), 6)


def _grade_comment(task: TaskDefinition, action: Action, step_number: int) -> Reward:
    """Intermediate reward for 'comment' actions (0.1–0.4)."""
    has_inline = _has_inline_on_required_file(task, action)
    keyword_ratio = (
        len(_matched_keywords(task, action)) / len(task.required_keywords)
        if task.required_keywords
        else 0.0
    )

    # Base partial score
    if has_inline and keyword_ratio > 0:
        score = 0.4
        rationale = (
            "Comment action correctly targets the required file and mentions "
            f"relevant keywords (ratio={keyword_ratio:.2f}). Partial score: 0.4."
        )
    elif has_inline:
        score = 0.3
        rationale = (
            "Comment action correctly targets the required file but does not "
            "mention required keywords. Partial score: 0.3."
        )
    elif keyword_ratio > 0:
        score = 0.2
        rationale = (
            f"Comment action mentions relevant keywords (ratio={keyword_ratio:.2f}) "
            "but does not target the required file. Partial score: 0.2."
        )
    else:
        score = 0.1
        rationale = (
            "Comment action does not target the required file and contains no "
            "required keywords. Partial score: 0.1."
        )

    breakdown = {
        "decision_correctness": 0.0,  # comment is not a terminal decision
        "issue_identification": round(keyword_ratio, 6),
        "review_quality": 0.5 if has_inline else 0.0,
    }

    return Reward(score=score, rationale=rationale, breakdown=breakdown)
