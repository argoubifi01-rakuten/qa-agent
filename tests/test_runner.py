import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from qa_agent.runner.base import Runner
from qa_agent.models import TestCase, RunResult
from qa_agent.runner.websocket_runner import WebSocketRunner
from qa_agent.config import Config, QALLMConfig, TargetConfig, TestGenerationConfig

# Patch out HMAC signing for all WebSocketRunner tests — signing is tested separately
_SIGNING_PATCH = patch("qa_agent.runner.websocket_runner.get_secret_key", return_value=None)
_SIGNED_HEADERS_PATCH = patch(
    "qa_agent.runner.websocket_runner.generate_signed_headers",
    return_value={},
)


# ---------------------------------------------------------------------------
# Abstract base class tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# WebSocketRunner fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_signing():
    """Disable HMAC signing for all runner tests."""
    with _SIGNING_PATCH, _SIGNED_HEADERS_PATCH:
        yield


@pytest.fixture
def single_turn_config():
    return Config(
        qa_llm=QALLMConfig(provider="openai", model="gpt-4o"),
        target=TargetConfig(
            websocket_url="wss://example.com/ws",
            auth_url="https://example.com/api/v2/auth/anonymous",
            thread_creation_url="https://example.com/api/v2/threads",
            scenario_id="test-scenario",
            wss_response_timeout=5,
        ),
        test_generation=TestGenerationConfig(num_test_cases=2, max_turns=1),
        qa_llm_api_key="test-key",
        eval_mock_secret="mock-secret",
        mongodb_uri="mongodb://localhost:27017/testdb",
    )


@pytest.fixture
def multi_turn_config():
    return Config(
        qa_llm=QALLMConfig(provider="openai", model="gpt-4o"),
        target=TargetConfig(
            websocket_url="wss://example.com/ws",
            auth_url="https://example.com/api/v2/auth/anonymous",
            thread_creation_url="https://example.com/api/v2/threads",
            scenario_id="test-scenario",
            wss_response_timeout=5,
        ),
        test_generation=TestGenerationConfig(num_test_cases=2, max_turns=3),
        qa_llm_api_key="test-key",
        eval_mock_secret="mock-secret",
        mongodb_uri="mongodb://localhost:27017/testdb",
    )


def _make_httpx_mock(access_token="tok-123", thread_id="6a210a144923b16873045206"):
    """Mock httpx.Client for auth (GET) and thread creation (POST)."""
    auth_response = MagicMock()
    auth_response.raise_for_status = MagicMock()
    auth_response.json.return_value = {"data": {"accessToken": access_token}}

    thread_response = MagicMock()
    thread_response.raise_for_status = MagicMock()
    thread_response.json.return_value = {"data": {"id": thread_id}}

    mock_httpx = MagicMock()
    mock_httpx.__enter__ = MagicMock(return_value=mock_httpx)
    mock_httpx.__exit__ = MagicMock(return_value=False)
    # auth uses .get(), thread creation uses .post()
    mock_httpx.get.return_value = auth_response
    mock_httpx.post.return_value = thread_response
    return mock_httpx


def _make_ws_frames(trace_id=None, trace_url=None):
    """Async generator yielding a chunk frame then DONE with optional trace info."""
    async def _gen():
        yield json.dumps({
            "webSocket": {
                "payload": {
                    "data": {
                        "contents": [{"contentType": "TEXT", "textData": {"text": "The answer is 42."}}],
                        "chatResponseStatus": "IN_PROGRESS",
                    }
                }
            }
        })
        done_data = {"chatResponseStatus": "DONE"}
        if trace_id or trace_url:
            done_data["trace"] = {}
            if trace_id:
                done_data["trace"]["id"] = trace_id
            if trace_url:
                done_data["trace"]["url"] = trace_url
        yield json.dumps({"webSocket": {"payload": {"data": done_data}}})
    return _gen()


def _make_mongo_mock(response_text="The answer is 42."):
    """Mock MongoClient returning a bucketed llm_interactions document."""
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [
        {
            "threadId": "6a210a144923b16873045206",
            "rounds": [
                {
                    "roundId": "round-1",
                    "meta": {"messageId": "msg-1"},
                    "items": [
                        {"type": "message", "role": "user", "content": "What is 6*7?"},
                        {"type": "message", "role": "assistant", "content": response_text},
                    ],
                }
            ],
            "count": 1,
        }
    ]
    mock_collection = MagicMock()
    mock_collection.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_client = MagicMock()
    mock_client.get_default_database.return_value = mock_db
    mock_client.close = MagicMock()
    return mock_client


# ---------------------------------------------------------------------------
# WebSocketRunner tests
# ---------------------------------------------------------------------------

def test_websocket_runner_single_turn_success(single_turn_config):
    tc = TestCase(
        id="tc-001", description="d", category="general",
        goal="Get answer", input_message="What is 6*7?",
    )
    mock_httpx = _make_httpx_mock()
    mock_mongo = _make_mongo_mock("The answer is 42.")

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=_make_ws_frames())
    mock_ws.send = AsyncMock()

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.MongoClient", return_value=mock_mongo), \
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


def test_websocket_runner_uses_get_for_auth(single_turn_config):
    """Auth must use GET not POST, with correct headers."""
    tc = TestCase(id="tc-002", description="d", category="general", goal="g", input_message="hi")
    mock_httpx = _make_httpx_mock()
    mock_mongo = _make_mongo_mock("ok")

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=_make_ws_frames())
    mock_ws.send = AsyncMock()

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.MongoClient", return_value=mock_mongo), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description=None, prompt=None)
        runner.run(tc)

    # auth is GET, thread creation is POST
    mock_httpx.get.assert_called_once()
    mock_httpx.post.assert_called_once()
    get_call = mock_httpx.get.call_args
    assert get_call[1]["headers"]["X-Ninja-Eval-Mock"] == "mock-secret"
    assert get_call[1]["headers"]["x-platform"] == "WEB"


def test_websocket_runner_payload_structure(single_turn_config):
    """Outbound WS payload must match the CONVERSATION envelope."""
    tc = TestCase(id="tc-003", description="d", category="general", goal="g", input_message="hello")
    mock_httpx = _make_httpx_mock()
    mock_mongo = _make_mongo_mock("response")
    sent_payloads: list[dict] = []

    async def capture_send(payload):
        sent_payloads.append(json.loads(payload))

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=_make_ws_frames())
    mock_ws.send = capture_send

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.MongoClient", return_value=mock_mongo), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description=None, prompt=None)
        runner.run(tc)

    assert len(sent_payloads) == 1
    msg = sent_payloads[0]["message"]
    assert msg["type"] == "CONVERSATION"
    data = msg["payload"]["data"]
    assert data["scenarioId"] == "test-scenario"
    assert data["scenarioVersion"] == "main"
    assert data["platform"] == "WEB"
    assert data["contents"][0]["contentType"] == "TEXT"
    assert data["contents"][0]["textData"]["text"] == "hello"


def test_websocket_runner_fetches_response_from_mongo(single_turn_config):
    """Response text comes from MongoDB, not the WebSocket stream."""
    tc = TestCase(id="tc-004", description="d", category="general", goal="g", input_message="hi")
    mock_httpx = _make_httpx_mock()

    # MongoDB returns a specific response the WS stream does NOT contain
    mock_mongo = _make_mongo_mock("authoritative MongoDB response")

    mock_ws = AsyncMock()
    # WS stream returns something different — should be discarded
    async def ws_frames():
        yield json.dumps({"webSocket": {"payload": {"data": {"chatResponseStatus": "DONE"}}}})
    mock_ws.__aiter__ = MagicMock(return_value=ws_frames())
    mock_ws.send = AsyncMock()

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.MongoClient", return_value=mock_mongo), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description=None, prompt=None)
        result = runner.run(tc)

    assert result.success is True
    assert result.turns[0].received == "authoritative MongoDB response"


def test_websocket_runner_captures_trace_from_ws_stream(single_turn_config):
    """trace_id and trace_url from the WS DONE frame are stored on the Turn."""
    tc = TestCase(id="tc-006", description="d", category="general", goal="g", input_message="hi")
    mock_httpx = _make_httpx_mock()
    mock_mongo = _make_mongo_mock("ok")

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=_make_ws_frames(
        trace_id="abc123", trace_url="https://traces.example.com/abc123"
    ))
    mock_ws.send = AsyncMock()

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.MongoClient", return_value=mock_mongo), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description=None, prompt=None)
        result = runner.run(tc)

    assert result.success is True
    assert result.turns[0].trace_id == "abc123"
    assert result.turns[0].trace_url == "https://traces.example.com/abc123"


def test_websocket_runner_no_trace_when_absent(single_turn_config):
    """trace_id and trace_url are None when the WS stream carries no trace info."""
    tc = TestCase(id="tc-007", description="d", category="general", goal="g", input_message="hi")
    mock_httpx = _make_httpx_mock()
    mock_mongo = _make_mongo_mock("ok")

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=_make_ws_frames())
    mock_ws.send = AsyncMock()

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.MongoClient", return_value=mock_mongo), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description=None, prompt=None)
        result = runner.run(tc)

    assert result.turns[0].trace_id is None
    assert result.turns[0].trace_url is None


def test_websocket_runner_mongo_rounds_schema(single_turn_config):
    """Response is extracted from rounds[].items[], not flat top-level fields."""
    tc = TestCase(id="tc-008", description="d", category="general", goal="g", input_message="hi")
    mock_httpx = _make_httpx_mock()

    # Two bucket docs, each with one round
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [
        {
            "rounds": [
                {"items": [
                    {"type": "message", "role": "assistant", "content": "First part."},
                ]}
            ]
        },
        {
            "rounds": [
                {"items": [
                    {"type": "message", "role": "user", "content": "follow-up"},
                    {"type": "message", "role": "assistant", "content": "Second part."},
                ]}
            ]
        },
    ]
    mock_collection = MagicMock()
    mock_collection.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_client = MagicMock()
    mock_client.get_default_database.return_value = mock_db
    mock_client.close = MagicMock()

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=_make_ws_frames())
    mock_ws.send = AsyncMock()

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.MongoClient", return_value=mock_client), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description=None, prompt=None)
        result = runner.run(tc)

    assert "First part." in result.turns[0].received
    assert "Second part." in result.turns[0].received


def test_websocket_runner_json_string_content(single_turn_config):
    """Content stored as a JSON-encoded list of content blocks is decoded cleanly."""
    import json as _json
    tc = TestCase(id="tc-009", description="d", category="general", goal="g", input_message="hi")
    mock_httpx = _make_httpx_mock()

    json_content = _json.dumps([{"contentType": "TEXT", "textData": {"text": "Decoded response."}}])
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [
        {
            "rounds": [
                {"items": [
                    {"type": "message", "role": "assistant", "content": json_content},
                ]}
            ]
        },
    ]
    mock_collection = MagicMock()
    mock_collection.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_client = MagicMock()
    mock_client.get_default_database.return_value = mock_db
    mock_client.close = MagicMock()

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=_make_ws_frames())
    mock_ws.send = AsyncMock()

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch("qa_agent.runner.websocket_runner.MongoClient", return_value=mock_client), \
         patch("qa_agent.runner.websocket_runner.websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)
        runner = WebSocketRunner(config=single_turn_config, description=None, prompt=None)
        result = runner.run(tc)

    received = result.turns[0].received
    assert received == "Decoded response.", f"Got: {received!r}"
    assert "[{" not in received, "Raw JSON artifact leaked into received text"


def test_websocket_runner_timeout(single_turn_config):
    tc = TestCase(id="tc-005", description="d", category="general", goal="g", input_message="hi")
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


def test_websocket_runner_uses_scripted_follow_ups(multi_turn_config):
    """Scripted follow_up_messages drive turns in order; LLM driver not called for those turns."""
    # max_turns=2 so the loop ends exactly when the script is exhausted
    config_2turns = Config(
        qa_llm=multi_turn_config.qa_llm,
        target=multi_turn_config.target,
        test_generation=TestGenerationConfig(num_test_cases=2, max_turns=2),
        qa_llm_api_key=multi_turn_config.qa_llm_api_key,
        eval_mock_secret=multi_turn_config.eval_mock_secret,
        mongodb_uri=multi_turn_config.mongodb_uri,
    )
    tc = TestCase(
        id="tc-010", description="d", category="general",
        goal="Agent completes after form fill",
        input_message="I want to buy something",
        follow_up_messages=["Red Nike running shoes under $100"],
    )
    mock_httpx = _make_httpx_mock()
    exchange_calls: list[str] = []

    async def fake_exchange(self, access_token, thread_id, user_id, device_id, message):
        exchange_calls.append(message)
        responses = [
            "Please fill in the clarification form.",
            "Here are your results!",
        ]
        return responses[len(exchange_calls) - 1], None, None

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch.object(WebSocketRunner, "_exchange", fake_exchange), \
         patch.object(WebSocketRunner, "_decide_next_message", new_callable=AsyncMock) as mock_driver:
        runner = WebSocketRunner(config=config_2turns, description=None, prompt=None)
        result = runner.run(tc)

    # LLM driver must NOT have been called — scripted turn covered the only follow-up slot
    mock_driver.assert_not_called()
    assert len(result.turns) == 2
    assert result.turns[0].sent == "I want to buy something"
    assert result.turns[0].received == "Please fill in the clarification form."
    assert result.turns[1].sent == "Red Nike running shoes under $100"
    assert result.turns[1].received == "Here are your results!"


def test_websocket_runner_llm_driver_used_when_no_script(multi_turn_config):
    """Without follow_up_messages the LLM driver is called to decide next message."""
    tc = TestCase(
        id="tc-011", description="d", category="general",
        goal="Verify LLM driver fires",
        input_message="Hello",
        follow_up_messages=None,
    )
    mock_httpx = _make_httpx_mock()

    async def fake_exchange(self, access_token, thread_id, user_id, device_id, message):
        return "Hi there!", None, None

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch.object(WebSocketRunner, "_exchange", fake_exchange), \
         patch.object(WebSocketRunner, "_decide_next_message", new_callable=AsyncMock, return_value=None) as mock_driver:
        runner = WebSocketRunner(config=multi_turn_config, description=None, prompt=None)
        result = runner.run(tc)

    mock_driver.assert_called_once()
    assert result.success is True


def test_websocket_runner_driver_instructions_passed_to_driver(multi_turn_config):
    """driver_instructions are forwarded to _decide_next_message."""
    instructions = (
        "If the agent shows a form, fill it with 'red Nike shoes size 10'. "
        "If it returns results, mark done."
    )
    tc = TestCase(
        id="tc-012", description="d", category="general",
        goal="Agent responds to conditional instructions",
        input_message="I want shoes",
        driver_instructions=instructions,
    )
    mock_httpx = _make_httpx_mock()

    async def fake_exchange(self, access_token, thread_id, user_id, device_id, message):
        return "Please fill in the form.", None, None

    captured: dict = {}

    async def fake_driver(self, turns, goal, case_description, driver_instructions=None):
        captured["driver_instructions"] = driver_instructions
        return None  # done after first turn

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch.object(WebSocketRunner, "_exchange", fake_exchange), \
         patch.object(WebSocketRunner, "_decide_next_message", fake_driver):
        runner = WebSocketRunner(config=multi_turn_config, description=None, prompt=None)
        runner.run(tc)

    assert captured["driver_instructions"] == instructions


def test_websocket_runner_driver_instructions_in_llm_prompt(multi_turn_config):
    """driver_instructions appear in the user message sent to the LLM."""
    instructions = "If agent shows form, respond with 'blue Adidas size 9'."
    tc = TestCase(
        id="tc-013", description="d", category="general",
        goal="Test conditional branching",
        input_message="I want trainers",
        driver_instructions=instructions,
    )
    mock_httpx = _make_httpx_mock()

    async def fake_exchange(self, access_token, thread_id, user_id, device_id, message):
        return "Please clarify your request.", None, None

    captured_prompt: dict = {}

    def fake_call_llm(**kwargs):
        captured_prompt.update(kwargs)
        return '{"done": true}'

    with patch("qa_agent.runner.websocket_runner.httpx.Client", return_value=mock_httpx), \
         patch.object(WebSocketRunner, "_exchange", fake_exchange), \
         patch("qa_agent.runner.websocket_runner.call_llm", side_effect=fake_call_llm):
        runner = WebSocketRunner(config=multi_turn_config, description=None, prompt=None)
        runner.run(tc)

    assert instructions in captured_prompt.get("user", ""), \
        "driver_instructions must appear in the LLM user message"
