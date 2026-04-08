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

## Project Structure

```
.
├── src/
│   ├── main.py        # FastAPI app — /reset, /step, /state endpoints
│   ├── episode.py     # Episode lifecycle manager
│   ├── grader.py      # Deterministic scoring function
│   ├── tasks.py       # Task registry (easy / medium / hard)
│   └── models.py      # Pydantic v2 data models
├── server/
│   └── app.py         # Entry point for multi-mode deployment
├── tests/             # 59 unit + property-based tests (Hypothesis)
├── inference.py       # Baseline LLM agent script
├── openenv.yaml       # OpenEnv metadata
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

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
A PR adds pagination to a user list endpoint but uses `users[start:end + 1]` — an off-by-one error that returns one extra record per page.

- Correct decision: `request_changes`
- Key terms to mention: `off-by-one`, `end + 1`

### Medium — SQL Injection Vulnerability
A PR implements a user search using a raw f-string SQL query: `f"SELECT * FROM users WHERE username LIKE '%{username_query}%'"` — directly interpolating user input.

- Correct decision: `request_changes`
- Key terms to mention: `sql injection`, `parameterized`, `f-string`, `user input`
- Must include an inline comment on `api/search.py`

### Hard — N+1 Query Performance Issue
A PR renders order history by looping over orders and fetching each product individually inside the loop — a classic N+1 query pattern.

- Correct decision: `request_changes`
- Key terms to mention: `n+1`, `select_related`, `prefetch_related`, `query per`, `eager loading`

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

### Example: Full Episode

```bash
# 1. Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# 2. Submit review
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "request_changes",
    "review_body": "There is an off-by-one error on line using end + 1 which returns an extra record.",
    "inline_comments": []
  }'

# 3. Check state
curl http://localhost:7860/state
```

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

Output format:
```
[START] task=code-review-easy env=openenv-code-review model=gpt-4.1-mini
[STEP]  step=1 action=request_changes reward=0.72 done=true error=null
[END]   success=true steps=1 rewards=0.72
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
| `LOCAL_IMAGE_NAME` | — | No | Local Docker image name (optional) |
