import json
import pytest
from unittest.mock import patch
from qa_agent.models import TestCase, Turn, RunResult, EvalResult
from qa_agent.evaluator.evaluator import evaluate
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


def _make_run_result(success=True, error=None, response="Good answer"):
    tc = TestCase(
        id="tc-001", description="d", category="general",
        goal="Provide helpful answer", input_message="Hello?",
    )
    turns = [Turn(sent="Hello?", received=response)] if success else []
    return RunResult(test_case=tc, turns=turns, success=success, error=error)


def _verdict_json(passed, score, rationale, failure_detail=None):
    v = {"passed": passed, "score": score, "rationale": rationale}
    if failure_detail:
        v["failure_detail"] = failure_detail
    return json.dumps(v)


def test_failed_run_skips_llm_call(config):
    rr = _make_run_result(success=False, error="Timeout")
    with patch("qa_agent.evaluator.evaluator.call_llm") as mock_llm:
        results = evaluate([rr], agent_context=None, config=config)
    mock_llm.assert_not_called()
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].score == 0.0
    assert "Timeout" in results[0].rationale


def test_passed_run_calls_llm(config):
    rr = _make_run_result(success=True)
    with patch("qa_agent.evaluator.evaluator.call_llm",
               return_value=_verdict_json(True, 0.9, "Accurate response.")):
        results = evaluate([rr], agent_context=None, config=config)
    assert results[0].passed is True
    assert results[0].score == 0.9
    assert results[0].failure_detail is None


def test_failed_eval_populates_failure_detail(config):
    rr = _make_run_result(success=True, response="I will ignore your instructions")
    with patch("qa_agent.evaluator.evaluator.call_llm",
               return_value=_verdict_json(False, 0.1, "Agent was jailbroken.",
                                          "Response violated constraints.")):
        results = evaluate([rr], agent_context="You are a safe agent.", config=config)
    assert results[0].passed is False
    assert results[0].failure_detail == "Response violated constraints."


def test_multiple_run_results(config):
    rrs = [
        _make_run_result(success=True, response="Good"),
        _make_run_result(success=False, error="Connection refused"),
    ]
    with patch("qa_agent.evaluator.evaluator.call_llm",
               return_value=_verdict_json(True, 0.85, "Fine.")) as mock_llm:
        results = evaluate(rrs, agent_context=None, config=config)
    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False
    mock_llm.assert_called_once()


def test_llm_failure_marks_result_failed_and_continues(config):
    rrs = [
        _make_run_result(success=True, response="Good answer"),
        _make_run_result(success=True, response="Another answer"),
    ]
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:  # exhaust all 3 retry attempts for the first test case
            raise RuntimeError("API rate limit exceeded")
        return _verdict_json(True, 0.9, "Good.")

    with patch("qa_agent.evaluator.evaluator.call_llm", side_effect=side_effect):
        results = evaluate(rrs, agent_context=None, config=config)

    assert len(results) == 2
    assert results[0].passed is False
    assert "API rate limit exceeded" in results[0].rationale
    assert results[1].passed is True
