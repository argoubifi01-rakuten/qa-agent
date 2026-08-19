import asyncio
import json
import logging
import time
import uuid
from urllib.parse import urlencode, urlsplit, parse_qsl
import httpx
import websockets
from qa_agent.db import fetch_llm_interaction_docs
from qa_agent.llm import call_llm
from qa_agent.models import TestCase, Turn, RunResult
from qa_agent.runner.base import Runner
from qa_agent.config import Config
from qa_agent.signing import generate_signed_headers, generate_signed_websocket_params, get_secret_key

logger = logging.getLogger(__name__)

_WS_SUBPROTOCOL = "Privileged-key.g7pKt9HZ2Ds4Qz8LvN5jXbR0SaW1eYfCT3mL6U"
_THREAD_TITLE_WAIT = 5.0

_TURN_SYSTEM = """\
You are the QA driver in a multi-turn conversation test. Your goal is to evaluate \
the agent under test by continuing the conversation to achieve the test goal.

Respond with a JSON object:
{
  "done": true/false,
  "next_message": "your next message to the agent (omit if done=true)"
}

When driver instructions are provided, follow them precisely to decide how to respond \
based on what the agent just said. The instructions may describe different branches \
(e.g. "if the agent shows a form, fill it with X; if it responds directly, do Y").
If the goal has been achieved or the agent clearly cannot achieve it, set done=true.
Return ONLY valid JSON."""

import re as _re
_INTERNAL_BLOCK_RE = _re.compile(
    r"```(?:agent_step|end)\s*\n.*?\n```",
    _re.DOTALL,
)


def _strip_internal_blocks(text: str) -> str:
    """Remove agent_step / end code fences that agents embed in their responses."""
    return _INTERNAL_BLOCK_RE.sub("", text).strip()


class WebSocketRunner(Runner):
    def __init__(self, config: Config, description: str | None, prompt: str | None):
        self._config = config
        self._description = description
        self._prompt = prompt

    _MAX_RUN_RETRIES = 5

    def run(self, test_case: TestCase) -> RunResult:
        logger.info("Running test case %s: %s", test_case.id, test_case.description)
        last_error = "unknown error"
        for attempt in range(1, self._MAX_RUN_RETRIES + 1):
            try:
                return asyncio.run(self._run_async(test_case))
            except asyncio.TimeoutError:
                timeout = self._config.target.wss_response_timeout
                last_error = f"timed out waiting for WebSocket response after {timeout}s"
                logger.warning(
                    "Test case %s attempt %d/%d timed out",
                    test_case.id, attempt, self._MAX_RUN_RETRIES,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Test case %s attempt %d/%d failed: %s",
                    test_case.id, attempt, self._MAX_RUN_RETRIES, exc,
                )
        logger.error(
            "Test case %s failed after %d attempts: %s",
            test_case.id, self._MAX_RUN_RETRIES, last_error,
        )
        return RunResult(test_case=test_case, turns=[], success=False, error=last_error)

    async def _run_async(self, test_case: TestCase) -> RunResult:
        timeout = self._config.target.wss_response_timeout
        device_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())

        with httpx.Client() as client:
            access_token = await asyncio.to_thread(
                self._auth, client, device_id, timeout
            )
            thread_id = await asyncio.to_thread(
                self._create_thread, client, access_token, device_id, timeout
            )

        logger.debug("Thread created: thread_id=%s", thread_id)
        turns = await asyncio.wait_for(
            self._ws_session(access_token, thread_id, user_id, device_id, test_case),
            timeout=timeout,
        )
        logger.info(
            "Test case %s complete: %d turns", test_case.id, len(turns)
        )
        return RunResult(test_case=test_case, turns=turns, success=True, error=None)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _auth(self, client: httpx.Client, device_id: str, timeout: int) -> str:
        logger.debug("Authenticating device_id=%s", device_id)
        url = self._config.target.auth_url
        signed = generate_signed_headers("GET", url, None, get_secret_key())
        resp = client.get(
            url,
            headers={
                "x-platform": "WEB",
                "device-id": device_id,
                "X-Ninja-Eval-Mock": self._config.eval_mock_secret,
                **signed,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if "data" not in body:
            raise RuntimeError(f"Auth response missing 'data': {body}")
        token = body["data"]["accessToken"]
        logger.debug("Auth OK, token prefix=%s...", token[:8])
        return token

    def _create_thread(self, client: httpx.Client, access_token: str, device_id: str, timeout: int) -> str:
        url = self._config.target.thread_creation_url
        signed = generate_signed_headers("POST", url, None, get_secret_key())
        resp = client.post(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "device-id": device_id,
                **signed,
            },
            json={"scenarioAgentId": self._config.target.scenario_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if "data" not in body:
            raise RuntimeError(f"Thread creation response missing 'data': {body}")
        return body["data"]["id"]

    # ------------------------------------------------------------------
    # WebSocket session
    # ------------------------------------------------------------------

    async def _ws_session(
        self,
        access_token: str,
        thread_id: str,
        user_id: str,
        device_id: str,
        test_case: TestCase,
    ) -> list[Turn]:
        turns: list[Turn] = []
        current_message = test_case.input_message
        scripted = list(test_case.follow_up_messages or [])

        for turn_num in range(self._config.test_generation.max_turns):
            logger.debug("Turn %d: sending %r", turn_num + 1, current_message[:60])
            response_text, trace_id, trace_url = await self._exchange(
                access_token, thread_id, user_id, device_id, current_message
            )
            logger.debug(
                "Turn %d: received %d chars trace_id=%s",
                turn_num + 1, len(response_text), trace_id,
            )
            turns.append(Turn(
                sent=current_message,
                received=response_text,
                trace_id=trace_id,
                trace_url=trace_url,
            ))

            max_turns = self._config.test_generation.max_turns
            if max_turns == 1 or turn_num == max_turns - 1:
                break

            if scripted:
                current_message = scripted.pop(0)
                logger.debug("Turn %d: using scripted follow-up %r", turn_num + 2, current_message[:60])
                continue

            next_msg = await self._decide_next_message(
                turns, test_case.goal, test_case.description,
                test_case.driver_instructions,
            )
            if next_msg is None:
                logger.debug("QA driver decided done after turn %d", turn_num + 1)
                break
            current_message = next_msg

        return turns

    def _build_ws_url(self, user_id: str, device_id: str, access_token: str) -> str:
        base_ws = self._config.target.websocket_url  # e.g. wss://host
        ws_path = "/ws/v1/chat"
        base_params = {"userId": user_id, "deviceId": device_id, "accessToken": access_token}
        base_url = f"{base_ws}{ws_path}?{urlencode(base_params)}"
        secret = get_secret_key()
        if not secret:
            return base_url
        signed = generate_signed_websocket_params(base_url, secret)
        merged = dict(parse_qsl(urlsplit(base_url).query))
        merged.update(signed)
        return f"{base_ws}{ws_path}?{urlencode(merged)}"

    async def _exchange(
        self,
        access_token: str,
        thread_id: str,
        user_id: str,
        device_id: str,
        message: str,
    ) -> tuple[str, str | None, str | None]:
        message_id = str(uuid.uuid4())
        payload = self._build_payload(
            message=message,
            thread_id=thread_id,
            user_id=user_id,
            access_token=access_token,
            message_id=message_id,
        )
        ws_url = self._build_ws_url(user_id, device_id, access_token)
        logger.debug("Connecting to WS url=%s", ws_url[:80])
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=120,
            open_timeout=120,
            close_timeout=120,
            max_size=10_485_760,
            subprotocols=[_WS_SUBPROTOCOL],
            additional_headers={"X-Ninja-Eval-Mock": self._config.eval_mock_secret},
        ) as ws:
            await ws.send(json.dumps(payload))
            trace_id, trace_url = await self._drain_until_done(ws)

        response_text = self._fetch_response_from_mongo(thread_id, message_id)
        return response_text, trace_id, trace_url

    async def _drain_until_done(self, ws) -> tuple[str | None, str | None]:
        response_done = False
        trace_id: str | None = None
        trace_url: str | None = None

        try:
            async for raw in ws:
                data = json.loads(raw)
                ws_data = data.get("webSocket", {}).get("payload", {}).get("data", {})

                if data.get("webSocket", {}).get("type") == "ERROR":
                    err = data.get("webSocket", {}).get("error", {}).get("message", "unknown error")
                    raise RuntimeError(f"WebSocket error from server: {err}")

                trace = ws_data.get("trace", {})
                if trace.get("id"):
                    trace_id = trace["id"]
                    logger.debug("Captured trace_id=%s", trace_id)
                if trace.get("url"):
                    trace_url = trace["url"]

                status = ws_data.get("chatResponseStatus")
                if status == "DONE":
                    response_done = True

                if response_done:
                    if (
                        data.get("type") == "NOTIFICATION"
                        and data.get("payload", {}).get("action") == "THREAD_DATA"
                    ):
                        break
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=_THREAD_TITLE_WAIT)
                    except (asyncio.TimeoutError, Exception):
                        pass
                    break
        except websockets.exceptions.ConnectionClosedOK:
            pass

        return trace_id, trace_url

    def _build_payload(
        self,
        message: str,
        thread_id: str,
        user_id: str,
        access_token: str,
        message_id: str,
    ) -> dict:
        return {
            "message": {
                "type": "CONVERSATION",
                "metadata": {
                    "messageId": message_id,
                    "timestamp": int(time.time() * 1000),
                },
                "payload": {
                    "action": "USER_INPUT",
                    "data": {
                        "chatRequestType": "USER_INPUT",
                        "role": "user",
                        "userId": user_id,
                        "threadId": thread_id,
                        "accessToken": access_token,
                        "scenarioId": self._config.target.scenario_id,
                        "scenarioVersion": "main",
                        "language": "ja",
                        "platform": "WEB",
                        "messageId": message_id,
                        "contents": [
                            {
                                "contentType": "TEXT",
                                "textData": {"text": message},
                            }
                        ],
                    },
                },
            }
        }

    # ------------------------------------------------------------------
    # DB fetch
    # ------------------------------------------------------------------

    def _fetch_response_from_mongo(self, thread_id: str, message_id: str) -> str:
        logger.debug(
            "Fetching response from DB for thread_id=%s message_id=%s",
            thread_id, message_id,
        )
        docs = fetch_llm_interaction_docs(
            thread_id=thread_id,
            mongodb_uri=self._config.mongodb_uri,
        )

        # Collect only items belonging to the round whose meta.messageId matches.
        # Falls back to all rounds if no round matches (e.g. messageId not stored).
        def _items_for_message(docs, message_id):
            for doc in docs:
                for round_ in doc.get("rounds", []):
                    meta = round_.get("meta", {})
                    if meta.get("messageId") == message_id:
                        yield from round_.get("items", [])

        matched_items = list(_items_for_message(docs, message_id))
        if not matched_items:
            logger.debug("No round matched message_id=%s, falling back to all rounds", message_id)
            matched_items = [
                item
                for doc in docs
                for round_ in doc.get("rounds", [])
                for item in round_.get("items", [])
            ]

        text_parts: list[str] = []
        for item in matched_items:
            item_type = item.get("type")

            if item_type == "message" and item.get("role") == "assistant":
                content = item.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, str):
                            if block:
                                text_parts.append(block)
                            continue
                        if not isinstance(block, dict):
                            continue
                        text = (
                            block.get("textData", {}).get("text", "")
                            or block.get("text", "")
                        )
                        if text:
                            text_parts.append(text)
                elif isinstance(content, str) and content:
                    try:
                        parsed = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        parsed = None
                    if isinstance(parsed, list):
                        for block in parsed:
                            text = (
                                block.get("textData", {}).get("text", "")
                                or block.get("text", "")
                            )
                            if text:
                                text_parts.append(text)
                    else:
                        text_parts.append(content)

            elif item_type == "function_call_output":
                raw_output = item.get("output", "{}")
                if isinstance(raw_output, dict):
                    output = raw_output
                elif isinstance(raw_output, str):
                    try:
                        output = json.loads(raw_output)
                    except (json.JSONDecodeError, TypeError):
                        continue
                else:
                    continue
                if not isinstance(output, dict):
                    continue

                # ask_user_input widget: render title + questions as readable text
                form_data = output.get("form_data")
                if form_data and isinstance(form_data, dict):
                    parts = ["[Clarification form shown to user]"]
                    if form_data.get("title"):
                        parts.append(f"Title: {form_data['title']}")
                    if form_data.get("subtitle"):
                        parts.append(f"Subtitle: {form_data['subtitle']}")
                    questions = form_data.get("questions", [])
                    if isinstance(questions, str):
                        try:
                            questions = json.loads(questions)
                        except (json.JSONDecodeError, TypeError):
                            questions = []
                    for q in questions:
                        if isinstance(q, dict):
                            q_text = q.get("question", "")
                            options = q.get("options", [])
                            if options:
                                parts.append(f"Q: {q_text} (options: {', '.join(str(o) for o in options)})")
                            else:
                                parts.append(f"Q: {q_text}")
                    text_parts.append("\n".join(parts))
                    continue

                for streamed in output.get("streamedContents", []):
                    if streamed.get("contentType") == "TEXT":
                        text = streamed.get("textData", {}).get("text", "")
                        if text:
                            text_parts.append(text)

        result = _strip_internal_blocks("\n".join(text_parts))
        logger.debug(
            "MongoDB response: %d text parts, %d chars total",
            len(text_parts), len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Multi-turn: QA LLM decides next message
    # ------------------------------------------------------------------

    async def _decide_next_message(
        self,
        turns: list[Turn],
        goal: str,
        case_description: str,
        driver_instructions: str | None = None,
    ) -> str | None:
        history = "\n".join(f"QA: {t.sent}\nAgent: {t.received}" for t in turns)
        context_parts = [f"Test goal: {goal}", f"Case description: {case_description}"]
        if driver_instructions:
            context_parts.append(f"Driver instructions:\n{driver_instructions}")
        if self._description:
            context_parts.append(f"Agent description: {self._description}")
        if self._prompt:
            context_parts.append(f"Agent system prompt:\n{self._prompt}")
        context = "\n".join(context_parts)

        user_msg = f"{context}\n\nConversation so far:\n{history}\n\nContinue or mark done?"
        text = call_llm(
            system=_TURN_SYSTEM,
            user=user_msg,
            model=self._config.qa_llm.model,
            api_key=self._config.qa_llm_api_key,
            max_tokens=1024,
        )
        data = json.loads(text)
        if data.get("done"):
            return None
        return data.get("next_message")
