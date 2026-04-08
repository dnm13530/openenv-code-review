"""Unit tests for the Task Registry."""

import pytest
import importlib
from src import tasks as tasks_module
from src.tasks import TaskDefinition, get_task, get_all_tasks


def _reset_registry():
    """Reset round-robin state between tests by reloading the module."""
    importlib.reload(tasks_module)
    # Re-import the functions from the reloaded module
    return tasks_module.get_task


def test_get_task_easy_returns_easy_task():
    get_task_fn = _reset_registry()
    task = get_task_fn("easy")
    assert task.difficulty == "easy"


def test_get_task_medium_returns_medium_task():
    get_task_fn = _reset_registry()
    task = get_task_fn("medium")
    assert task.difficulty == "medium"


def test_get_task_hard_returns_hard_task():
    get_task_fn = _reset_registry()
    task = get_task_fn("hard")
    assert task.difficulty == "hard"


def test_get_task_invalid_difficulty_raises():
    get_task_fn = _reset_registry()
    with pytest.raises(ValueError, match="difficulty must be one of"):
        get_task_fn("extreme")


def test_get_task_round_robin_cycles_across_difficulties():
    """get_task() with no argument should cycle easy → medium → hard → easy ..."""
    get_task_fn = _reset_registry()
    difficulties = [get_task_fn().difficulty for _ in range(6)]
    # First full cycle
    assert difficulties[:3] == ["easy", "medium", "hard"]
    # Second full cycle
    assert difficulties[3:6] == ["easy", "medium", "hard"]


def test_task_definition_has_required_fields():
    get_task_fn = _reset_registry()
    for diff in ("easy", "medium", "hard"):
        task = get_task_fn(diff)
        assert hasattr(task, "task_id")
        assert task.task_id
        assert task.pr_title
        assert task.diff
        assert task.ground_truth_decision == "request_changes"
        assert len(task.required_keywords) >= 2



def test_medium_task_has_required_inline_file():
    get_task_fn = _reset_registry()
    task = get_task_fn("medium")
    assert task.required_inline_file is not None


def test_hard_task_has_four_or_more_keywords():
    get_task_fn = _reset_registry()
    task = get_task_fn("hard")
    assert len(task.required_keywords) >= 4


def test_get_all_tasks_returns_all_three():
    all_tasks = get_all_tasks()
    assert len(all_tasks) == 9
    difficulties = {t.difficulty for t in all_tasks}
    assert difficulties == {"easy", "medium", "hard"}
