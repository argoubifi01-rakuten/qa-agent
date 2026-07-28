import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open
from qa_agent.cli import main


def _make_eval_results(passed=True):
    from qa_agent.models import TestCase, Turn, RunResult, EvalResult
    tc = TestCase(id="tc-001", description="d", category="general", goal="g", input_message="hi")
    turn = Turn(sent="hi", received="hello")
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    return [EvalResult(run_result=rr, passed=passed, score=0.9 if passed else 0.1,
                       rationale="r", failure_detail=None if passed else "detail")]


def test_cli_requires_description_or_prompt(capsys):
    with patch("sys.argv", ["qa-agent", "run", "--env", "dev"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "description" in captured.err.lower() or "prompt" in captured.err.lower()


def test_cli_all_pass_exits_0(tmp_path):
    yaml_content = """\
qa_llm:
  provider: anthropic
  model: claude-opus-5
target:
  websocket_url: wss://x/ws
  auth_url: https://x/auth
  thread_creation_url: https://x/threads
  wss_response_timeout: 30
test_generation:
  num_test_cases: 2
  max_turns: 1
"""
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(yaml_content)
    eval_results = _make_eval_results(passed=True)

    with patch("sys.argv", [
        "qa-agent", "run",
        "--agent-description", "A helpful bot",
        "--config", str(config_file),
    ]), \
    patch.dict("os.environ", {"QA_LLM_API_KEY": "k", "TARGET_AUTH_SECRET": "s"}), \
    patch("qa_agent.cli.generate_test_cases", return_value=[MagicMock()]), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=eval_results), \
    patch("qa_agent.cli.report", return_value=0):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0


def test_cli_any_fail_exits_1(tmp_path):
    yaml_content = """\
qa_llm:
  provider: anthropic
  model: claude-opus-5
target:
  websocket_url: wss://x/ws
  auth_url: https://x/auth
  thread_creation_url: https://x/threads
  wss_response_timeout: 30
test_generation:
  num_test_cases: 2
  max_turns: 1
"""
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(yaml_content)
    eval_results = _make_eval_results(passed=False)

    with patch("sys.argv", [
        "qa-agent", "run",
        "--agent-description", "A bot",
        "--config", str(config_file),
    ]), \
    patch.dict("os.environ", {"QA_LLM_API_KEY": "k", "TARGET_AUTH_SECRET": "s"}), \
    patch("qa_agent.cli.generate_test_cases", return_value=[MagicMock()]), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=eval_results), \
    patch("qa_agent.cli.report", return_value=1):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1


def test_cli_prompt_file_is_read(tmp_path):
    yaml_content = """\
qa_llm:
  provider: anthropic
  model: claude-opus-5
target:
  websocket_url: wss://x/ws
  auth_url: https://x/auth
  thread_creation_url: https://x/threads
  wss_response_timeout: 30
test_generation:
  num_test_cases: 1
  max_turns: 1
"""
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(yaml_content)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a helpful assistant.")
    captured_prompt = []

    def capture_generate(description, prompt, config):
        captured_prompt.append(prompt)
        return [MagicMock()]

    with patch("sys.argv", [
        "qa-agent", "run",
        "--prompt-file", str(prompt_file),
        "--config", str(config_file),
    ]), \
    patch.dict("os.environ", {"QA_LLM_API_KEY": "k", "TARGET_AUTH_SECRET": "s"}), \
    patch("qa_agent.cli.generate_test_cases", side_effect=capture_generate), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=[MagicMock(passed=True)]), \
    patch("qa_agent.cli.report", return_value=0):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        try:
            main()
        except SystemExit:
            pass
    assert captured_prompt[0] == "You are a helpful assistant."
