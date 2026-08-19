import json
import pytest
from unittest.mock import patch
from qa_agent.models import TestCase
from qa_agent.generator.generator import generate_test_cases
from qa_agent.config import Config, QALLMConfig, TargetConfig, TestGenerationConfig


@pytest.fixture(autouse=True)
def no_override():
    """Prevent test_cases_override.yaml from being picked up during tests."""
    with patch("qa_agent.generator.generator._load_override", return_value=None):
        yield


@pytest.fixture
def config():
    return Config(
        qa_llm=QALLMConfig(provider="openai", model="gpt-4o"),
        target=TargetConfig(
            websocket_url="wss://x/ws",
            auth_url="https://x/auth",
            thread_creation_url="https://x/threads",
            scenario_id="test-scenario",
            wss_response_timeout=30,
        ),
        test_generation=TestGenerationConfig(num_test_cases=3, max_turns=1),
        qa_llm_api_key="test-key",
        eval_mock_secret="mock-secret",
        mongodb_uri="mongodb://localhost/testdb",
    )


AGENT_CONTEXT = """\
# Agent: Shopping Assistant

## Purpose
Helps users find and purchase products on Rakuten.

## Capabilities
- Product search
- Price filtering
- Recommendations
"""

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

MULTI_TURN_LLM_RESPONSE = json.dumps({
    "test_cases": [
        {
            "id": "tc-001",
            "description": "Agent shows clarification form then completes shopping search",
            "category": "general",
            "goal": "After filling the form the agent returns product results",
            "input_message": "I want to buy something",
            "follow_up_messages": ["I'm looking for a red Nike running shoe under $100"],
        },
        {
            "id": "tc-002",
            "description": "Clear query needs no clarification",
            "category": "general",
            "goal": "Agent returns results without asking for clarification",
            "input_message": "Show me blue Nike running shoes under $80",
            "follow_up_messages": None,
        },
    ]
})

DRIVER_INSTRUCTIONS_LLM_RESPONSE = json.dumps({
    "test_cases": [
        {
            "id": "tc-001",
            "description": "Conditional clarification form test",
            "category": "general",
            "goal": "Agent either asks for clarification or returns results directly",
            "input_message": "I want shoes",
            "driver_instructions": (
                "If the agent shows a clarification form, fill it with "
                "'red Nike running shoes size 10 under $100'. "
                "If the agent returns product results directly, verify they are "
                "relevant and mark done."
            ),
        },
    ]
})


def test_generate_returns_test_cases(config):
    with patch("qa_agent.generator.generator.call_llm", return_value=VALID_LLM_RESPONSE):
        cases = generate_test_cases(agent_context=AGENT_CONTEXT, config=config)
    assert len(cases) == 3
    assert all(isinstance(c, TestCase) for c in cases)
    assert cases[0].id == "tc-001"
    assert cases[1].category == "edge_case"


def test_generate_makes_one_llm_call(config):
    with patch("qa_agent.generator.generator.call_llm", return_value=VALID_LLM_RESPONSE) as mock_llm:
        generate_test_cases(agent_context=AGENT_CONTEXT, config=config)
    assert mock_llm.call_count == 1


def test_generate_passes_context_to_llm(config):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return VALID_LLM_RESPONSE

    with patch("qa_agent.generator.generator.call_llm", side_effect=capture):
        generate_test_cases(agent_context=AGENT_CONTEXT, config=config)

    assert "Shopping Assistant" in captured["user"]


def test_malformed_json_retries_once(config):
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"test_cases": [invalid}'
        return VALID_LLM_RESPONSE

    with patch("qa_agent.generator.generator.call_llm", side_effect=side_effect):
        cases = generate_test_cases(agent_context=AGENT_CONTEXT, config=config)
    assert call_count == 2
    assert len(cases) == 3


def test_malformed_json_raises_after_two_failures(config):
    with patch("qa_agent.generator.generator.call_llm", return_value='{"test_cases": [invalid}'):
        with pytest.raises(ValueError, match="Failed to parse"):
            generate_test_cases(agent_context=AGENT_CONTEXT, config=config)


def test_follow_up_messages_parsed(config):
    with patch("qa_agent.generator.generator.call_llm", return_value=MULTI_TURN_LLM_RESPONSE):
        cases = generate_test_cases(agent_context=AGENT_CONTEXT, config=config)
    assert cases[0].follow_up_messages == ["I'm looking for a red Nike running shoe under $100"]
    assert cases[1].follow_up_messages is None


def test_missing_follow_up_messages_defaults_to_none(config):
    with patch("qa_agent.generator.generator.call_llm", return_value=VALID_LLM_RESPONSE):
        cases = generate_test_cases(agent_context=AGENT_CONTEXT, config=config)
    assert all(c.follow_up_messages is None for c in cases)
    assert all(c.driver_instructions is None for c in cases)


def test_driver_instructions_parsed(config):
    with patch("qa_agent.generator.generator.call_llm", return_value=DRIVER_INSTRUCTIONS_LLM_RESPONSE):
        cases = generate_test_cases(agent_context=AGENT_CONTEXT, config=config)
    assert cases[0].driver_instructions is not None
    assert "clarification form" in cases[0].driver_instructions
    assert cases[0].follow_up_messages is None
