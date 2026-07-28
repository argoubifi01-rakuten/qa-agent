import json
import pytest
from io import StringIO
from unittest.mock import MagicMock, patch
from qa_agent.models import TestCase, Turn, RunResult, EvalResult
from qa_agent.reporter.reporter import report
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


def _make_eval_result(tc_id, category, passed, score, failure_detail=None):
    tc = TestCase(
        id=tc_id, description=f"Test {tc_id}", category=category,
        goal="goal", input_message="hello",
    )
    turn = Turn(sent="hello", received="response")
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    return EvalResult(
        run_result=rr, passed=passed, score=score,
        rationale="Some rationale.", failure_detail=failure_detail,
    )


def _mock_narrative_client(summary_text="Overall quality is good."):
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=summary_text)]
    mock_stream = MagicMock()
    mock_stream.get_final_message.return_value = mock_message
    mock_client = MagicMock()
    mock_client.messages.stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
    mock_client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)
    return mock_client


def test_report_all_pass_returns_exit_0(config, capsys):
    results = [
        _make_eval_result("tc-001", "general", passed=True, score=0.9),
        _make_eval_result("tc-002", "edge_case", passed=True, score=0.8),
    ]
    mock_client = _mock_narrative_client("Agent performed well across all categories.")
    with patch("qa_agent.reporter.reporter.anthropic.Anthropic", return_value=mock_client):
        exit_code = report(results, description="customer bot", config=config, output_path=None)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "tc-001" in captured.out
    assert "tc-002" in captured.out
    assert "PASS" in captured.out


def test_report_any_fail_returns_exit_1(config, capsys):
    results = [
        _make_eval_result("tc-001", "general", passed=True, score=0.9),
        _make_eval_result(
            "tc-002", "adversarial", passed=False, score=0.1,
            failure_detail="Agent was jailbroken.",
        ),
    ]
    mock_client = _mock_narrative_client("One test failed.")
    with patch("qa_agent.reporter.reporter.anthropic.Anthropic", return_value=mock_client):
        exit_code = report(results, description="bot", config=config, output_path=None)
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    assert "Agent was jailbroken." in captured.out


def test_report_passing_cases_not_deep_dived(config, capsys):
    results = [_make_eval_result("tc-001", "general", passed=True, score=0.9)]
    mock_client = _mock_narrative_client()
    with patch("qa_agent.reporter.reporter.anthropic.Anthropic", return_value=mock_client):
        report(results, description="bot", config=config, output_path=None)
    captured = capsys.readouterr()
    assert "FAILURE DEEP-DIVE" not in captured.out


def test_report_writes_json_output(config, tmp_path, capsys):
    output_file = str(tmp_path / "report.json")
    results = [
        _make_eval_result("tc-001", "general", passed=True, score=0.95),
        _make_eval_result(
            "tc-002", "edge_case", passed=False, score=0.2,
            failure_detail="Wrong answer.",
        ),
    ]
    mock_client = _mock_narrative_client("Mixed results.")
    with patch("qa_agent.reporter.reporter.anthropic.Anthropic", return_value=mock_client):
        report(results, description="bot", config=config, output_path=output_file)
    with open(output_file) as f:
        data = json.load(f)
    assert "summary" in data
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == "tc-001"
    assert data["results"][1]["passed"] is False
