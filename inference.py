"""Baseline inference script for the openenv-code-review environment.

Environment variables:
  API_BASE_URL  - LLM proxy endpoint (injected by validator, default: https://api.openai.com/v1)
  MODEL_NAME    - Model identifier (default: gpt-4.1-mini)
  API_KEY       - API key for LLM proxy (injected by validator)
  HF_TOKEN      - Fallback API key for local use
  ENV_BASE_URL  - URL of the OpenEnv server (default: https://dnm13530-openenv-code-review.hf.space)

Output format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import urllib.request
import urllib.error
from typing import Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

def load_env_vars() -> tuple[str, str, str, str]:
    """Return (llm_base_url, env_base_url, model_name, api_key)."""
    # LLM proxy — validator injects this
    llm_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

    # OpenEnv server — our HF Space
    env_base_url = os.getenv(
        "ENV_BASE_URL", "https://dnm13530-openenv-code-review.hf.space"
    )

    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")

    # Validator injects API_KEY; fall back to HF_TOKEN for local use
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")

    if not api_key:
        print(
            "Error: API_KEY (or HF_TOKEN) environment variable is required.\n"
            "  API_BASE_URL  - LLM proxy endpoint (injected by validator)\n"
            "  ENV_BASE_URL  - OpenEnv server URL (default: HF Space URL)\n"
            "  MODEL_NAME    - Model identifier (default: gpt-4.1-mini)\n"
            "  API_KEY       - API key injected by the validator\n"
            "  HF_TOKEN      - Hugging Face API token (fallback for local use)",
            file=sys.stderr,
        )
        sys.exit(1)

    return llm_base_url, env_base_url, model_name, api_key


# ---------------------------------------------------------------------------
# HTTP helpers using stdlib urllib (no external dependencies)
# ---------------------------------------------------------------------------

def _http_post(url: str, body: dict, timeout: int = 30) -> dict:
    """POST JSON to url, return parsed response dict."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(observation: dict) -> str:
    """Build a prompt string from an Observation dict."""
    return (
        f"You are a code reviewer. Review the following pull request and provide feedback.\n\n"
        f"PR Title: {observation['pr_title']}\n"
        f"PR Description: {observation['pr_description']}\n"
        f"Difficulty: {observation['task_difficulty']}\n"
        f"Files changed: {observation['file_count']} "
        f"(+{observation['additions']} -{observation['deletions']})\n\n"
        f"Diff:\n{observation['diff']}\n\n"
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
    """Parse LLM response text into an Action dict. Returns None on failure."""
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


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env_base_url: str, client: OpenAI, model_name: str, difficulty: str
) -> None:
    """Run a single episode, printing required [START]/[STEP]/[END] output lines."""
    task_name = f"code-review-{difficulty}"
    step_rewards: list[float] = []
    step_count = 0
    success = False
    last_error: Optional[str] = None

    print(f"[START] task={task_name} env=openenv-code-review model={model_name}")

    try:
        observation = _http_post(f"{env_base_url}/reset", {"difficulty": difficulty})
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

            action = parse_llm_response(response_text)
            if action is None:
                last_error = "Failed to parse LLM response"
                action = {
                    "decision": "request_changes",
                    "review_body": "Unable to parse LLM response. Requesting changes as a precaution.",
                }

            action_str = action["decision"]
            # Env step goes to our HF Space server
            step_data = _http_post(f"{env_base_url}/step", action)

            observation = step_data["observation"]
            reward_val = step_data["reward"]["score"]
            done = step_data["done"]
            step_rewards.append(reward_val)

            done_str = "true" if done else "false"
            error_str = last_error if last_error else "null"
            print(
                f"[STEP]  step={step_count} action={action_str} "
                f"reward={reward_val:.2f} done={done_str} error={error_str}"
            )
            last_error = None

        success = True

    except Exception as exc:  # noqa: BLE001
        last_error = str(exc)
        print(
            f"[STEP]  step={step_count + 1} action=null "
            f"reward=0.00 done=true error={last_error}"
        )

    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.00"
    success_str = "true" if success else "false"
    print(f"[END]   success={success_str} steps={step_count} rewards={rewards_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    llm_base_url, env_base_url, model_name, api_key = load_env_vars()

    # OpenAI client points to the LLM proxy
    llm_client = OpenAI(base_url=llm_base_url, api_key=api_key)

    for difficulty in ["easy", "medium", "hard"]:
        run_episode(env_base_url, llm_client, model_name, difficulty)


if __name__ == "__main__":
    main()
