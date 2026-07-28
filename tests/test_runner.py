import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from qa_agent.runner.base import Runner
from qa_agent.models import TestCase, RunResult


def test_runner_is_abstract():
    with pytest.raises(TypeError):
        Runner()


def test_runner_subclass_must_implement_run():
    class BadRunner(Runner):
        pass
    with pytest.raises(TypeError):
        BadRunner()


def test_runner_concrete_subclass_works():
    class GoodRunner(Runner):
        def run(self, test_case: TestCase) -> RunResult:
            turn_obj = __import__("qa_agent.models", fromlist=["Turn"]).Turn(
                sent=test_case.input_message, received="ok"
            )
            return RunResult(test_case=test_case, turns=[turn_obj], success=True, error=None)

    runner = GoodRunner()
    tc = TestCase(id="x", description="d", category="general", goal="g", input_message="hello")
    result = runner.run(tc)
    assert result.success is True


from qa_agent.runner.websocket_runner import WebSocketRunner
from qa_agent.config import Config, QALLMConfig, TargetConfig, TestGenerationConfig


@pytest.fixture
def single_turn_config():
    return Config(
        qa_llm=QALLMConfig(provider="anthropic", model="claude-opus-5"),
        target=TargetConfig(
            websocket_url="wss://example.com/ws",
            auth_url="https://example.com/api/v2/auth/anonymous",
            thread_creation_url="https://example.com/api/v2/threads",
            wss_response_timeout=5,
        ),
        test_generation=TestGenerationConfig(num_test_cases=2, max_turns=1),
        qa_llm_api_key="test-key",
        target_auth_secret="test-secret",
    )


@pytest.fixture
def multi_turn_config(single_turn_config):
    single_turn_config.test_generation.max_turns = 3
    return single_turn_config


def _make_httpx_mock(access_token="tok-123", thread_id="thread-456"):
    auth_response = MagicMock()
    auth_response.raise_for_status = MagicMock()
    auth_response.json.return_value = {"accessToken": access_token}

    thread_response = MagicMock()
    thread_response.raise_for_status = MagicMock()
    thread_response.json.return_value = {"id": thread_id}

    mock_httpx = MagicMock()
    mock_httpx.__enter__ = MagicMock(return_value=mock_httpx)
    mock_httpx.__exit__ = MagicMock(return_value=False)
    mock_httpx.post.side_effect = [auth_response, thread_response]
    return mock_httpx


def _make_ws_messages(text="The answer is 42."):
    """Returns async generator yielding two WSS messages: content + DONE status."""
    async def _gen():
        yield json.dumps({
            "type": "message",
            "content": text,
            "chatResponseStatus": "STREAMING",
        })
        yield json.dumps({
            "type": "status",
            "chatResponseStatus": "DONE",
        })
    return _gen()


def test_websocket_runner_single_turn_success(single_turn_config):
    tc = TestCase(
        id="tc-001", description="d", category="general",
        goal="Get answer", input_message="What is 6*7?",
    )
    mock_httpx = _make_httpx_mock()
    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=_make_ws_messages("The answer is 42."))
    mock_ws.send = AsyncMock()
    mock_ws.close = AsyncMock()

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description="a bot", prompt=None)
        result = runner.run(tc)

    assert result.success is True
    assert result.error is None
    assert len(result.turns) == 1
    assert result.turns[0].sent == "What is 6*7?"
    assert "42" in result.turns[0].received


def test_websocket_runner_timeout(single_turn_config):
    tc = TestCase(id="tc-002", description="d", category="general", goal="g", input_message="hi")
    mock_httpx = _make_httpx_mock()

    async def slow_ws(*args, **kwargs):
        await asyncio.sleep(999)

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = slow_ws
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description=None, prompt="sys")
        result = runner.run(tc)

    assert result.success is False
    assert result.error is not None
    assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

