---
sdk: docker
app_port: 7860
tags: [openenv]
---

# openenv-code-review

An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a code reviewer, triaging and evaluating pull requests. The agent receives a simulated PR (diff, title, description, metadata) and must identify bugs, security issues, or quality problems, then submit a structured review decision.

## Motivation

Code review is a high-value, real-world software engineering activity. This environment tests an agent's ability to read code diffs, reason about correctness and security, and communicate findings clearly — skills that transfer directly to practical software development workflows.

---

## Action Space

The agent submits an `Action` with the following fields:

| Field | Type | Description |
|---|---|---|
| `decision` | enum | One of `approve`, `request_changes`, `comment` |
| `review_body` | str | Review text, max 4000 characters |
| `inline_comments` | list (optional) | List of `{file_path, line_number, body}` objects |

## Observation Space

Each step returns an `Observation` with:

| Field | Type | Description |
|---|---|---|
| `pr_title` | str | Title of the pull request |
| `pr_description` | str | Description of the PR |
| `diff` | str | Full unified diff of the changes |
| `file_count` | int | Number of files changed |
| `additions` | int | Lines added |
| `deletions` | int | Lines deleted |
| `step_number` | int | Current step in the episode |
| `task_difficulty` | str | `easy`, `medium`, or `hard` |
| `episode_id` | str | Unique episode identifier |

---

## Tasks

### Easy — Off-by-One Error
A PR introduces a loop with an obvious off-by-one error. The correct decision is `request_changes`. The agent must identify the boundary condition bug in the review body.

### Medium — SQL Injection Vulnerability
A PR adds a database query that directly interpolates user input, creating a SQL injection risk. The correct decision is `request_changes`. The agent must identify the vulnerable file with an inline comment and mention parameterized queries.

### Hard — N+1 Query Performance Issue
A PR looks architecturally sound but introduces a subtle N+1 query pattern in a data access layer. The correct decision is `request_changes`. The agent must identify the performance anti-pattern and suggest eager loading or batching.

---

## Reward Function

Rewards are non-binary and provide partial progress signals throughout the episode:

| Criterion | Weight | Description |
|---|---|---|
| `decision_correctness` | 0.4 | 1.0 if decision matches ground truth, else 0.0 |
| `issue_identification` | 0.4 | Proportional to required keywords found in review body (0.0–0.5) |
| `review_quality` | 0.2 | Based on review length and inline comment targeting (0.0–0.5) |

`score = decision_correctness × 0.4 + issue_identification × 0.4 + review_quality × 0.2`

Intermediate rewards are given for `comment` actions that correctly identify the buggy file (0.1–0.4).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Body: `{"difficulty": "easy"\|"medium"\|"hard"}` (optional) |
| `POST` | `/step` | Submit a review action. Returns `{observation, reward, done, info}` |
| `GET` | `/state` | Get current episode state (read-only) |
| `GET` | `/` | Environment info and API description |

---

## Setup and Usage

### Run with Docker

```bash
docker build -t openenv-code-review .
docker run -p 7860:7860 openenv-code-review
```

### Run locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 7860
```

### Run tests

```bash
pytest tests/ -v
```

---

## Baseline Inference

```bash
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=gpt-4.1-mini
export HF_TOKEN=your-hf-token
python inference.py
```

### Baseline Performance (gpt-4.1-mini)

| Task | Score |
|---|---|
| easy | ~0.72 |
| medium | ~0.55 |
| hard | ~0.41 |
| mean | ~0.56 |

---

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `API_BASE_URL` | `http://localhost:7860` | No | LLM API endpoint |
| `MODEL_NAME` | `gpt-4.1-mini` | No | Model identifier |
| `HF_TOKEN` | — | Yes | Hugging Face API token |
