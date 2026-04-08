"""FastAPI application for the OpenEnv code review environment."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from src.episode import EpisodeManager
from src.models import Action, Observation, StateSnapshot, StepResponse

app = FastAPI(title="openenv-code-review", version="1.0.0")

# Global episode manager instance (single-session model for HF Spaces)
_episode_manager = EpisodeManager()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: Optional[str] = None


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected server errors."""
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def info() -> JSONResponse:
    """Return a JSON info page describing the environment and its API."""
    return JSONResponse(content={
        "name": "openenv-code-review",
        "version": "1.0.0",
        "description": (
            "An OpenEnv-compliant code review environment where an AI agent "
            "acts as a code reviewer, triaging and evaluating pull requests."
        ),
        "endpoints": {
            "POST /reset": "Initialize a new episode. Optional body: {\"difficulty\": \"easy|medium|hard\"}",
            "POST /step": "Submit a review action. Body: Action model.",
            "GET /state": "Get the current episode state (read-only).",
            "GET /": "This info page.",
        },
    })


@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest = ResetRequest()) -> Observation:
    """Initialize a new episode and return the initial Observation."""
    try:
        return _episode_manager.reset(request.difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    """Advance the episode by one step and return the result."""
    observation, reward, done, info = _episode_manager.step(action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateSnapshot)
async def state() -> StateSnapshot:
    """Return the current episode state without modifying it."""
    return _episode_manager.get_state()
