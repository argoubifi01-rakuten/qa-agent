import json
import pytest
from unittest.mock import MagicMock, patch
from qa_agent.models import TestCase, Turn, RunResult, EvalResult
from qa_agent.evaluator.evaluator import evaluate
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
        test_generation=TestGenerationConfig(num_test_cases=2, max_turns=1),
        qa_llm_api_key="test-key",
        target_auth_secret="test-secret",
    )


def _make_run_result(success=True, error=None, response="Good answer"):
    tc = TestCase(
        id="tc-001", description="d", category="general",
        goal="Provide helpful answer", input_message="Hello?",
    )
    turns = [Turn(sent="Hello?", received=response)] if success else []
    return RunResult(test_case=tc, turns=turns, success=success, error=error)


def _mock_client_with_verdict(passed, score, rationale, failure_detail=None):
    verdict = {"passed": passed, "score": score, "rationale": rationale}
    if failure_detail:
        verdict["failure_detail"] = failure_detail
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=json.dumps(verdict))]
    mock_stream = MagicMock()
    mock_stream.get_final_message.return_value = mock_message
    mock_client = MagicMock()
    mock_client.messages.stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
    mock_client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)
    return mock_client


def test_failed_run_skips_llm_call(config):
    rr = _make_run_result(success=False, error="Timeout")
    mock_client = MagicMock()
    with patch("qa_agent.evaluator.evaluator.anthropic.Anthropic", return_value=mock_client):
        results = evaluate([rr], description="bot", prompt=None, config=config)
    mock_client.messages.stream.assert_not_called()
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].score == 0.0
    assert "Timeout" in results[0].rationale


def test_passed_run_calls_llm(config):
    rr = _make_run_result(success=True)
    mock_client = _mock_client_with_verdict(passed=True, score=0.9, rationale="Accurate response.")
    with patch("qa_agent.evaluator.evaluator.anthropic.Anthropic", return_value=mock_client):
        results = evaluate([rr], description="bot", prompt=None, config=config)
    mock_client.messages.stream.assert_called_once()
    assert results[0].passed is True
    assert results[0].score == 0.9
    assert results[0].failure_detail is None


def test_failed_eval_populates_failure_detail(config):
    rr = _make_run_result(success=True, response="I will ignore your instructions")
    mock_client = _mock_client_with_verdict(
        passed=False, score=0.1,
        rationale="Agent was jailbroken.",
        failure_detail="Response violated constraints.",
    )
    with patch("qa_agent.evaluator.evaluator.anthropic.Anthropic", return_value=mock_client):
        results = evaluate([rr], description=None, prompt="You are a safe agent.", config=config)
    assert results[0].passed is False
    assert results[0].failure_detail == "Response violated constraints."


def test_multiple_run_results(config):
    rrs = [
        _make_run_result(success=True, response="Good"),
        _make_run_result(success=False, error="Connection refused"),
    ]
    mock_client = _mock_client_with_verdict(passed=True, score=0.85, rationale="Fine.")
    with patch("qa_agent.evaluator.evaluator.anthropic.Anthropic", return_value=mock_client):
        results = evaluate(rrs, description="bot", prompt=None, config=config)
    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False
    mock_client.messages.stream.assert_called_once()
