"""Baseline inference script for the openenv-code-review environment.

Environment variables:
  API_BASE_URL  - LLM proxy endpoint (injected by validator, default: https://api.openai.com/v1)
  MODEL_NAME    - Model identifier (default: gpt-4.1-mini)
  API_KEY       - API key for LLM proxy (injected by validator)
  HF_TOKEN      - Fallback API key for local use

Output format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.XXXX> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Reward formatting note: rewards are printed with 4-decimal precision. The
OpenEnv validator requires every task score to fall *strictly* inside the
open interval (0, 1). Printing with .2f would round the grader's clamped
bounds (0.0001 and 0.9999) back to "0.00" and "1.00", re-introducing the
forbidden boundary values at the print layer.

Note: The environment is used IN-PROCESS via EpisodeManager (no HTTP). This
removes network dependency on a remote env server and guarantees that LLM
calls through the validator's proxy are reliably exercised.
"""

import json
import os
import sys
from typing import Optional

# Ensure the repo root is on sys.path so `src` is importable regardless of cwd
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from openai import OpenAI

from src.episode import EpisodeManager
from src.grader import _EPSILON  # strict-clamp floor used for fallback rewards
from src.models import Action, DecisionEnum, InlineComment, Observation


# Format string for rewards. Must use enough precision that the grader's
# clamped bounds (_EPSILON and 1 - _EPSILON, currently 1e-4) do not round
# to the forbidden exact boundaries 0.00 or 1.00 when printed.
_REWARD_FMT = ".4f"
# Fallback reward emitted for failed / exceptional steps. Must be strictly
# inside (0, 1) since the validator parses these printed values.
_FALLBACK_REWARD_STR = f"{_EPSILON:{_REWARD_FMT}}"


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

def load_env_vars() -> tuple[str, str, str]:
    """Return (llm_base_url, model_name, api_key)."""
    # LLM proxy — validator injects this
    llm_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")

    # Validator injects API_KEY; fall back to HF_TOKEN for local use
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")

    if not api_key:
        print(
            "Error: API_KEY (or HF_TOKEN) environment variable is required.\n"
            "  API_BASE_URL  - LLM proxy endpoint (injected by validator)\n"
            "  MODEL_NAME    - Model identifier (default: gpt-4.1-mini)\n"
            "  API_KEY       - API key injected by the validator\n"
            "  HF_TOKEN      - Hugging Face API token (fallback for local use)",
            file=sys.stderr,
        )
        sys.exit(1)

    return llm_base_url, model_name, api_key


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(observation: Observation) -> str:
    """Build a prompt string from an Observation."""
    return (
        f"You are a code reviewer. Review the following pull request and provide feedback.\n\n"
        f"PR Title: {observation.pr_title}\n"
        f"PR Description: {observation.pr_description}\n"
        f"Difficulty: {observation.task_difficulty}\n"
        f"Files changed: {observation.file_count} "
        f"(+{observation.additions} -{observation.deletions})\n\n"
        f"Diff:\n{observation.diff}\n\n"
        f"Respond with a JSON object with these fields:\n"
        f"  decision: one of 'approve', 'request_changes', or 'comment'\n"
        f"  review_body: your review text (max 4000 chars)\n"
        f"  inline_comments: optional list of {{file_path, line_number, body}}\n\n"
        f"Example:\n"
        f'{{"decision": "request_changes", "review_body": "Found a bug on line 5.", '
        f'"inline_comments": [{{"file_path": "main.py", "line_number": 5, "body": "Off-by-one error here."}}]}}'
    )


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response text into an action dict. Returns None on failure."""
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            data = json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            return None

    if not isinstance(data, dict):
        return None
    if "decision" not in data or "review_body" not in data:
        return None

    valid_decisions = {"approve", "request_changes", "comment"}
    if data.get("decision") not in valid_decisions:
        return None

    if len(data.get("review_body", "")) > 4000:
        data["review_body"] = data["review_body"][:4000]

    return data


def action_from_dict(data: dict) -> Action:
    """Construct an Action model from a parsed dict."""
    inline_raw = data.get("inline_comments") or []
    inline_comments: list[InlineComment] = []
    if isinstance(inline_raw, list):
        for item in inline_raw:
            if not isinstance(item, dict):
                continue
            try:
                inline_comments.append(
                    InlineComment(
                        file_path=str(item.get("file_path", "")),
                        line_number=int(item.get("line_number", 0)),
                        body=str(item.get("body", "")),
                    )
                )
            except (ValueError, TypeError):
                continue
    return Action(
        decision=DecisionEnum(data["decision"]),
        review_body=data.get("review_body", "")[:4000],
        inline_comments=inline_comments or None,
    )


# ---------------------------------------------------------------------------
# Episode runner (in-process environment)
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, model_name: str, difficulty: str) -> None:
    """Run a single episode, printing required [START]/[STEP]/[END] output lines."""
    task_name = f"code-review-{difficulty}"
    step_rewards: list[float] = []
    step_count = 0
    success = False
    last_error: Optional[str] = None

    print(f"[START] task={task_name} env=openenv-code-review model={model_name}")

    # Use a fresh EpisodeManager per episode — avoids any cross-episode state
    manager = EpisodeManager()

    try:
        observation = manager.reset(difficulty)
        done = False

        while not done:
            step_count += 1
            prompt = build_prompt(observation)

            # LLM call goes through the validator's proxy via the OpenAI client
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            response_text = completion.choices[0].message.content or ""

            parsed = parse_llm_response(response_text)
            if parsed is None:
                last_error = "Failed to parse LLM response"
                parsed = {
                    "decision": "request_changes",
                    "review_body": "Unable to parse LLM response. Requesting changes as a precaution.",
                }

            action = action_from_dict(parsed)
            action_str = action.decision.value

            # Env step runs in-process via EpisodeManager
            observation, reward, done, _info = manager.step(action)
            reward_val = reward.score
            step_rewards.append(reward_val)

            done_str = "true" if done else "false"
            error_str = last_error if last_error else "null"
            print(
                f"[STEP]  step={step_count} action={action_str} "
                f"reward={reward_val:{_REWARD_FMT}} done={done_str} error={error_str}"
            )
            last_error = None

        success = True

    except Exception as exc:  # noqa: BLE001
        # step_count is incremented at the top of the while-loop before the LLM
        # call, so by the time an exception fires it already reflects the
        # in-flight step. For errors thrown by reset() (before any step ran),
        # fall back to 1 so we never emit "step=0".
        last_error = str(exc)
        print(
            f"[STEP]  step={step_count or 1} action=null "
            f"reward={_FALLBACK_REWARD_STR} done=true error={last_error}"
        )

    # Validator requires every printed reward to be strictly inside (0, 1),
    # so the empty fallback cannot be "0.00" (parses as exact zero).
    rewards_str = (
        ",".join(f"{r:{_REWARD_FMT}}" for r in step_rewards)
        if step_rewards
        else _FALLBACK_REWARD_STR
    )
    success_str = "true" if success else "false"
    print(f"[END]   success={success_str} steps={step_count} rewards={rewards_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    llm_base_url, model_name, api_key = load_env_vars()

    # OpenAI client points at the validator's LLM proxy
    llm_client = OpenAI(base_url=llm_base_url, api_key=api_key)

    for difficulty in ["easy", "medium", "hard"]:
        run_episode(llm_client, model_name, difficulty)


if __name__ == "__main__":
    main()
