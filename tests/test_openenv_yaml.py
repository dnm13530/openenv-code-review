"""Unit tests for openenv.yaml structure and required fields."""

import yaml
import pytest
from pathlib import Path


OPENENV_YAML_PATH = Path(__file__).parent.parent / "openenv.yaml"
REQUIRED_TOP_LEVEL_FIELDS = [
    "name", "version", "description", "task_type",
    "action_space", "observation_space", "difficulty_levels", "tags",
]
REQUIRED_DECISION_VALUES = {"approve", "request_changes", "comment"}
REQUIRED_OBSERVATION_FIELDS = [
    "pr_title", "pr_description", "diff", "file_count",
    "additions", "deletions", "step_number", "task_difficulty", "episode_id",
]


@pytest.fixture(scope="module")
def openenv_config():
    with open(OPENENV_YAML_PATH) as f:
        return yaml.safe_load(f)


def test_all_required_fields_present(openenv_config):
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        assert field in openenv_config, f"Missing required field: {field}"


def test_tags_contains_openenv(openenv_config):
    assert "openenv" in openenv_config["tags"]


def test_action_space_contains_all_decisions(openenv_config):
    action_space = set(openenv_config["action_space"])
    assert REQUIRED_DECISION_VALUES == action_space


def test_observation_space_contains_all_fields(openenv_config):
    obs_space = set(openenv_config["observation_space"])
    for field in REQUIRED_OBSERVATION_FIELDS:
        assert field in obs_space, f"Missing observation field: {field}"


def test_difficulty_levels_are_valid(openenv_config):
    assert set(openenv_config["difficulty_levels"]) == {"easy", "medium", "hard"}
