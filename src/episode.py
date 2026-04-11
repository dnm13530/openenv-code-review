"""Episode Manager for the OpenEnv code review environment."""

import uuid

from src.models import Action, DecisionEnum, Observation, Reward, StateSnapshot
from src.tasks import TaskDefinition, get_task
from src.grader import grade


class EpisodeStateError(RuntimeError):
    """Raised when an episode operation is invalid for the current state.

    Kept as a plain runtime error (no fastapi dependency) so this module can
    be imported in lightweight environments such as the validator harness
    running inference.py. The FastAPI layer in src/main.py translates this
    to HTTP 400.
    """


class EpisodeManager:
    """Manages the lifecycle of a single code review episode."""

    def __init__(self) -> None:
        self.current_task: TaskDefinition | None = None
        self.step_number: int = 0
        self.done: bool = False
        self.episode_id: str = ""
        self.accumulated_score: float = 0.0

    def reset(self, difficulty: str | None = None) -> Observation:
        """Start a new episode, optionally filtered by difficulty.

        Returns the initial Observation with step_number=0.
        Raises ValueError if difficulty is invalid (caller should map to HTTP 422).
        """
        self.current_task = get_task(difficulty)
        self.step_number = 0
        self.done = False
        self.episode_id = str(uuid.uuid4())
        self.accumulated_score = 0.0

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Advance the episode by one step.

        Raises HTTP 400 if the episode is already done.
        Sets done=True when the action is a terminal decision (approve / request_changes).
        """
        if self.done:
            raise EpisodeStateError(
                "Episode is already done. Call reset to start a new episode."
            )

        if self.current_task is None:
            raise EpisodeStateError(
                "No active episode. Call reset to start a new episode."
            )

        reward = grade(self.current_task, action, self.step_number)
        self.accumulated_score += reward.score
        self.step_number += 1

        # Terminal decisions end the episode
        if action.decision in (DecisionEnum.approve, DecisionEnum.request_changes):
            self.done = True

        observation = self._build_observation()
        info: dict = {"step_number": self.step_number, "accumulated_score": self.accumulated_score}

        return observation, reward, self.done, info

    def get_state(self) -> StateSnapshot:
        """Return a read-only snapshot of the current episode state."""
        return StateSnapshot(
            episode_id=self.episode_id,
            step_number=self.step_number,
            done=self.done,
            accumulated_score=self.accumulated_score,
            current_task_difficulty=self.current_task.difficulty if self.current_task else "",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Build an Observation from the current episode state."""
        task = self.current_task
        assert task is not None
        return Observation(
            pr_title=task.pr_title,
            pr_description=task.pr_description,
            diff=task.diff,
            file_count=task.file_count,
            additions=task.additions,
            deletions=task.deletions,
            step_number=self.step_number,
            task_difficulty=task.difficulty,
            episode_id=self.episode_id,
        )
