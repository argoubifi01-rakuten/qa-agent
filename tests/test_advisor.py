import pytest
from unittest.mock import patch
from qa_agent.models import TestCase, Turn, RunResult, EvalResult
from qa_agent.advisor.advisor import advise
from qa_agent.config import Config, QALLMConfig, TargetConfig, TestGenerationConfig


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
        test_generation=TestGenerationConfig(num_test_cases=2, max_turns=1),
        qa_llm_api_key="test-key",
        eval_mock_secret="mock-secret",
        mongodb_uri="mongodb://localhost/testdb",
    )


def _make_eval_result(passed, score, rationale, failure_detail=None):
    tc = TestCase(id="tc-001", description="d", category="general", goal="g", input_message="hi")
    turn = Turn(sent="hi", received="hello")
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    return EvalResult(
        run_result=rr, passed=passed, score=score,
        rationale=rationale, failure_detail=failure_detail,
    )


def test_advise_calls_llm_and_returns_text(config):
    expected = "ASSESSMENT\nNeeds improvement.\n\nPROPOSED PROMPT\nBe more specific."
    with patch("qa_agent.advisor.advisor.call_llm", return_value=expected) as mock_llm:
        result = advise(
            current_prompt="You are a bot.",
            eval_results=[_make_eval_result(True, 0.9, "Good.")],
            config=config,
        )
    assert result == expected
    mock_llm.assert_called_once()


def test_advise_includes_prompt_in_call(config):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "advice"

    with patch("qa_agent.advisor.advisor.call_llm", side_effect=lambda **kw: capture(**kw) or "advice"):
        advise(
            current_prompt="You handle returns.",
            eval_results=[_make_eval_result(False, 0.2, "Failed.", "Gave wrong info.")],
            config=config,
        )
    assert "You handle returns." in captured["user"]


def test_advise_includes_agent_context_when_provided(config):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "advice"

    context = "# Agent: Returns Bot\n## Purpose\nHandle return requests."

    with patch("qa_agent.advisor.advisor.call_llm", side_effect=lambda **kw: capture(**kw) or "advice"):
        advise(
            current_prompt="You handle returns.",
            eval_results=[_make_eval_result(True, 0.9, "Good.")],
            config=config,
            agent_context=context,
        )
    assert "Returns Bot" in captured["user"]


def test_advise_without_agent_context_omits_context_section(config):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "advice"

    with patch("qa_agent.advisor.advisor.call_llm", side_effect=lambda **kw: capture(**kw) or "advice"):
        advise(
            current_prompt="You handle returns.",
            eval_results=[_make_eval_result(True, 0.9, "Good.")],
            config=config,
            agent_context=None,
        )
    assert "Agent context document" not in captured["user"]


def test_advise_includes_iteration_history(config):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "advice"

    history = [
        {
            "iteration": 1,
            "timestamp": "2026-08-14T10:00:00+00:00",
            "pass_rate": 0.45,
            "results": [{"passed": True}] * 9 + [{"passed": False}] * 11,
            "summary": "Agent failed on edge cases.",
            "advice": "---\nASSESSMENT\nNeeds work.\n\nPROPOSED PROMPT\nYou are a better bot.",
        }
    ]

    with patch("qa_agent.advisor.advisor.call_llm", side_effect=lambda **kw: capture(**kw) or "advice"):
        advise(
            current_prompt="You are a bot.",
            eval_results=[_make_eval_result(True, 0.8, "Good.")],
            config=config,
            iteration_history=history,
        )

    assert "Iteration 1" in captured["user"]
    assert "2026-08-14" in captured["user"]
    assert "edge cases" in captured["user"]


def test_advise_without_history_omits_history_section(config):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "advice"

    with patch("qa_agent.advisor.advisor.call_llm", side_effect=lambda **kw: capture(**kw) or "advice"):
        advise(
            current_prompt="You are a bot.",
            eval_results=[_make_eval_result(True, 0.9, "Good.")],
            config=config,
            iteration_history=None,
        )

    assert "Iteration history" not in captured["user"]
