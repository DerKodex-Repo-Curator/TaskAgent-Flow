from unittest.mock import MagicMock, patch

import pytest
import requests

from agent_flow.llm_client import OllamaClient


def test_generate_returns_stripped_response() -> None:
    client = OllamaClient(model="llama3")
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "  hello world  "}
    mock_response.raise_for_status.return_value = None

    with patch("agent_flow.llm_client.requests.post", return_value=mock_response) as mock_post:
        result = client.generate("hi")

    assert result == "hello world"
    mock_post.assert_called_once()


def test_generate_raises_connection_error_when_unreachable() -> None:
    client = OllamaClient(model="llama3")
    with patch(
        "agent_flow.llm_client.requests.post",
        side_effect=requests.exceptions.ConnectionError(),
    ):
        with pytest.raises(ConnectionError):
            client.generate("hi")


def test_generate_raises_timeout_error() -> None:
    client = OllamaClient(model="llama3")
    with patch(
        "agent_flow.llm_client.requests.post",
        side_effect=requests.exceptions.Timeout(),
    ):
        with pytest.raises(TimeoutError):
            client.generate("hi")


def test_list_models_parses_full_and_base_names() -> None:
    client = OllamaClient(model="llama3")
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
    mock_response.raise_for_status.return_value = None

    with patch("agent_flow.llm_client.requests.get", return_value=mock_response):
        names = client.list_models()

    assert names is not None
    assert "llama3:latest" in names
    assert "llama3" in names


def test_list_models_returns_none_when_unreachable() -> None:
    client = OllamaClient(model="llama3")
    with patch(
        "agent_flow.llm_client.requests.get",
        side_effect=requests.exceptions.ConnectionError(),
    ):
        assert client.list_models() is None
