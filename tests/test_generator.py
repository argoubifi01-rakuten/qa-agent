import json
import pytest
from unittest.mock import MagicMock, patch
from qa_agent.models import TestCase
from qa_agent.generator.generator import generate_test_cases
from qa_agent.config import Config, QALLMConfig, TargetConfig, TestGenerationConfig


@pytest.fixture
def config():
    return Config(
        qa_llm=QALLMConfig(provider="anthropic", model="claude-opus-5"),
        target=TargetConfig(
            websocket_url="wss://x/ws",
            auth_url="https://x/auth",
            thread_creation_url="https://x/threads",
            wss_response_timeout=30,
        ),
        test_generation=TestGenerationConfig(num_test_cases=3, max_turns=1),
        qa_llm_api_key="test-key",
        target_auth_secret="test-secret",
    )


VALID_LLM_RESPONSE = json.dumps({
    "test_cases": [
        {
            "id": "tc-001",
            "description": "User asks a general question",
            "category": "general",
            "goal": "Agent answers helpfully",
            "input_message": "What can you help me with?",
        },
        {
            "id": "tc-002",
            "description": "Empty input",
            "category": "edge_case",
            "goal": "Agent handles blank message gracefully",
            "input_message": "",
        },
        {
            "id": "tc-003",
            "description": "Out of scope request",
            "category": "out_of_scope",
            "goal": "Agent declines politely",
            "input_message": "Write me a poem",
        },
    ]
})


def _make_mock_client(response_text: str):
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=response_text)]
    mock_stream = MagicMock()
    mock_stream.get_final_message.return_value = mock_message
    mock_client = MagicMock()
    mock_client.messages.stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
    mock_client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)
    return mock_client


def test_generate_from_description_only(config):
    mock_client = _make_mock_client(VALID_LLM_RESPONSE)
    with patch("qa_agent.generator.generator.anthropic.Anthropic", return_value=mock_client):
        cases = generate_test_cases(description="A customer support bot", prompt=None, config=config)
    assert len(cases) == 3
    assert all(isinstance(c, TestCase) for c in cases)
    assert cases[0].id == "tc-001"
    assert cases[1].category == "edge_case"


def test_generate_from_prompt_only(config):
    mock_client = _make_mock_client(VALID_LLM_RESPONSE)
    with patch("qa_agent.generator.generator.anthropic.Anthropic", return_value=mock_client):
        cases = generate_test_cases(description=None, prompt="You are a helpful assistant.", config=config)
    assert len(cases) == 3


def test_generate_from_both(config):
    mock_client = _make_mock_client(VALID_LLM_RESPONSE)
    with patch("qa_agent.generator.generator.anthropic.Anthropic", return_value=mock_client):
        cases = generate_test_cases(
            description="Customer support bot",
            prompt="You are a helpful assistant.",
            config=config,
        )
    assert len(cases) == 3


def test_malformed_json_retries_once(config):
    bad_response = '{"test_cases": [invalid}'
    good_response = VALID_LLM_RESPONSE
    call_count = 0

    def stream_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        text = bad_response if call_count == 1 else good_response
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=text)]
        mock_stream = MagicMock()
        mock_stream.get_final_message.return_value = mock_message
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_stream)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    mock_client = MagicMock()
    mock_client.messages.stream.side_effect = stream_side_effect
    with patch("qa_agent.generator.generator.anthropic.Anthropic", return_value=mock_client):
        cases = generate_test_cases(description="bot", prompt=None, config=config)
    assert call_count == 2
    assert len(cases) == 3


def test_malformed_json_raises_after_two_failures(config):
    mock_client = _make_mock_client('{"test_cases": [invalid}')
    with patch("qa_agent.generator.generator.anthropic.Anthropic", return_value=mock_client):
        with pytest.raises(ValueError, match="Failed to parse"):
            generate_test_cases(description="bot", prompt=None, config=config)
