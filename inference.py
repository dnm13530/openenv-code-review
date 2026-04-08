"""Baseline inference script for the openenv-code-review environment.

Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables and runs
one episode per difficulty level using an OpenAI-compatible LLM client.

Output format (per hackathon spec):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
from typing import Optional

import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# Environment variable loading — defaults required for API_BASE_URL, MODEL_NAME
# ---------------------------------------------------------------------------

def load_env_vars() -> tuple[str, str, str]:
    """Return (API_BASE_URL, MODEL_NAME, HF_TOKEN). Exits with code 1 if HF_TOKEN missing."""
    missing = []
    api_base_url = os.environ.get("API_BASE_URL", "http://localhost:7860")
    model_name = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not hf_token:
        missing.append("HF_TOKEN")

    if missing:
        print(
            f"Error: The following required environment variables are not set: "
            f"{', '.join(missing)}\n"
            f"Please set them before running this script.\n"
            f"  API_BASE_URL  - Base URL of the OpenAI-compatible LLM endpoint (default: http://localhost:7860)\n"
            f"  MODEL_NAME    - Name of the model to use for inference (default: gpt-4.1-mini)\n"
            f"  HF_TOKEN      - Hugging Face API token for authentication (required)",
            file=sys.stderr,
        )
        sys.exit(1)

    return api_base_url, model_name, hf_token


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
    """Parse LLM response text into an Action dict.

    Returns None if the response cannot be parsed into a valid action.
    Handles malformed JSON gracefully.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
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

    # Truncate review_body if it exceeds the limit
    if len(data.get("review_body", "")) > 4000:
        data["review_body"] = data["review_body"][:4000]

    return data


# ---------------------------------------------------------------------------
# Episode runner — emits [START], [STEP], [END] lines
# ---------------------------------------------------------------------------

def run_episode(env_base_url: str, client: OpenAI, model_name: str, difficulty: str) -> None:
    """Run a single episode for the given difficulty level, printing required output lines."""
    task_name = f"code-review-{difficulty}"
    step_rewards: list[float] = []
    step_count = 0
    success = False
    last_error: Optional[str] = None

    print(f"[START] task={task_name} env=openenv-code-review model={model_name}")

    try:
        reset_resp = requests.post(
            f"{env_base_url}/reset", json={"difficulty": difficulty}, timeout=30
        )
        reset_resp.raise_for_status()
        observation = reset_resp.json()

        done = False

        while not done:
            step_count += 1
            prompt = build_prompt(observation)

            # Call the LLM
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

            step_resp = requests.post(f"{env_base_url}/step", json=action, timeout=30)
            step_resp.raise_for_status()
            step_data = step_resp.json()

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
            last_error = None  # reset after reporting

        success = True

    except Exception as exc:  # noqa: BLE001
        last_error = str(exc)
        error_str = last_error if last_error else "null"
        print(
            f"[STEP]  step={step_count + 1} action=null "
            f"reward=0.00 done=true error={error_str}"
        )

    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.00"
    success_str = "true" if success else "false"
    print(f"[END]   success={success_str} steps={step_count} rewards={rewards_str}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_base_url, model_name, hf_token = load_env_vars()
    llm_client = OpenAI(base_url=api_base_url, api_key=hf_token)

    for difficulty in ["easy", "medium", "hard"]:
        run_episode(api_base_url, llm_client, model_name, difficulty)


if __name__ == "__main__":
    main()
