from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent_flow.engine import TaskEngine, TaskExecutionError


def _write_task(tmp_path: Path, steps: List[Dict[str, Any]]) -> Path:
    task = {"task_name": "Test Task", "steps": steps}
    path = tmp_path / "task.yaml"
    path.write_text(yaml.dump(task), encoding="utf-8")
    return path


@patch("agent_flow.engine.OllamaClient")
def test_engine_runs_successful_step(mock_client_cls: MagicMock, tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["llama3:latest"]
    mock_client.generate.return_value = "A valid, sufficiently long response."
    mock_client_cls.return_value = mock_client

    task_path = _write_task(
        tmp_path, [{"id": 1, "action": "Do something.", "expected_output": "Something."}]
    )
    engine = TaskEngine(task_path=task_path, verbose=False)
    results = engine.run()

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].attempts == 1


@patch("agent_flow.engine.OllamaClient")
def test_engine_retries_then_succeeds(mock_client_cls: MagicMock, tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["llama3:latest"]
    mock_client.generate.side_effect = ["", "bad", "Finally a valid response here."]
    mock_client_cls.return_value = mock_client

    task_path = _write_task(tmp_path, [{"id": 1, "action": "Do something."}])
    engine = TaskEngine(task_path=task_path, verbose=False)
    results = engine.run()

    assert results[0].attempts == 3
    assert results[0].success is True
    assert mock_client.generate.call_count == 3

    # The third prompt should carry the failure reason from attempt 2.
    third_prompt = mock_client.generate.call_args_list[2].args[0]
    assert "previous attempt failed" in third_prompt


@patch("agent_flow.engine.OllamaClient")
def test_engine_raises_after_exhausting_retries(mock_client_cls: MagicMock, tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["llama3:latest"]
    mock_client.generate.return_value = ""
    mock_client_cls.return_value = mock_client

    task_path = _write_task(tmp_path, [{"id": 1, "action": "Do something."}])
    engine = TaskEngine(task_path=task_path, max_retries=2, verbose=False)

    with pytest.raises(TaskExecutionError):
        engine.run()

    assert mock_client.generate.call_count == 2


@patch("agent_flow.engine.OllamaClient")
def test_engine_threads_prior_step_context(mock_client_cls: MagicMock, tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["llama3:latest"]
    mock_client.generate.side_effect = [
        "Title A, Title B, Title C",
        "A two sentence summary of Title A.",
    ]
    mock_client_cls.return_value = mock_client

    task_path = _write_task(
        tmp_path,
        [
            {"id": 1, "action": "Generate 3 titles."},
            {"id": 2, "action": "Summarize the best title from step 1."},
        ],
    )
    engine = TaskEngine(task_path=task_path, verbose=False)
    engine.run()

    second_prompt = mock_client.generate.call_args_list[1].args[0]
    assert "Title A, Title B, Title C" in second_prompt


@patch("agent_flow.engine.OllamaClient")
def test_engine_falls_back_to_mistral_when_llama3_unavailable(
    mock_client_cls: MagicMock, tmp_path: Path
) -> None:
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["mistral:latest"]
    mock_client_cls.return_value = mock_client

    task_path = _write_task(tmp_path, [{"id": 1, "action": "Do something."}])
    TaskEngine(task_path=task_path, verbose=False)

    assert mock_client.model == "mistral"


@patch("agent_flow.engine.OllamaClient")
def test_engine_raises_when_no_candidate_model_available(
    mock_client_cls: MagicMock, tmp_path: Path
) -> None:
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["phi3:latest"]
    mock_client_cls.return_value = mock_client

    task_path = _write_task(tmp_path, [{"id": 1, "action": "Do something."}])
    with pytest.raises(RuntimeError):
        TaskEngine(task_path=task_path, verbose=False)


def test_missing_task_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        TaskEngine(task_path=missing)


def test_invalid_task_definition_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump({"task_name": "Missing steps"}), encoding="utf-8")
    with pytest.raises(ValueError):
        TaskEngine(task_path=path)
