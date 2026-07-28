import asyncio
import json
import anthropic
import httpx
import websockets
from qa_agent.models import TestCase, Turn, RunResult
from qa_agent.runner.base import Runner
from qa_agent.config import Config

_TURN_SYSTEM = """\
You are the QA driver in a multi-turn conversation test. Your goal is to evaluate \
the agent under test by continuing the conversation to achieve the test goal.

Respond with a JSON object:
{
  "done": true/false,
  "next_message": "your next message to the agent (omit if done=true)"
}

If the goal has been achieved or the agent clearly cannot achieve it, set done=true.
Return ONLY valid JSON."""


class WebSocketRunner(Runner):
    def __init__(self, config: Config, description: str | None, prompt: str | None):
        self._config = config
        self._description = description
        self._prompt = prompt

    def run(self, test_case: TestCase) -> RunResult:
        try:
            return asyncio.run(self._run_async(test_case))
        except asyncio.TimeoutError:
            timeout = self._config.target.wss_response_timeout
            return RunResult(
                test_case=test_case,
                turns=[],
                success=False,
                error=f"timed out waiting for WebSocket response after {timeout}s",
            )
        except Exception as exc:
            return RunResult(test_case=test_case, turns=[], success=False, error=str(exc))

    async def _run_async(self, test_case: TestCase) -> RunResult:
        timeout = self._config.target.wss_response_timeout

        with httpx.Client() as client:
            auth_resp = client.post(
                self._config.target.auth_url,
                headers={"x-secret": self._config.target_auth_secret},
            )
            auth_resp.raise_for_status()
            access_token = auth_resp.json()["accessToken"]

            thread_resp = client.post(
                self._config.target.thread_creation_url,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            thread_resp.raise_for_status()
            thread_id = thread_resp.json()["id"]

        ws_url = f"{self._config.target.websocket_url}?threadId={thread_id}&token={access_token}"

        turns = await asyncio.wait_for(
            self._ws_session(ws_url, test_case),
            timeout=timeout,
        )
        return RunResult(test_case=test_case, turns=turns, success=True, error=None)

    async def _ws_session(self, ws_url: str, test_case: TestCase) -> list[Turn]:
        turns: list[Turn] = []
        current_message = test_case.input_message

        async with websockets.connect(ws_url) as ws:
            for _turn_num in range(self._config.test_generation.max_turns):
                response_text = await self._exchange(ws, current_message)
                turns.append(Turn(sent=current_message, received=response_text))

                if self._config.test_generation.max_turns == 1:
                    break

                next_msg = await self._decide_next_message(
                    turns, test_case.goal, test_case.description
                )
                if next_msg is None:
                    break
                current_message = next_msg

        return turns

    async def _exchange(self, ws, message: str) -> str:
        payload = json.dumps({"message": message})
        await ws.send(payload)
        response_parts: list[str] = []
        async for raw in ws:
            data = json.loads(raw)
            if content := data.get("content"):
                response_parts.append(content)
            if data.get("chatResponseStatus") == "DONE":
                break
        return "".join(response_parts)

    async def _decide_next_message(
        self, turns: list[Turn], goal: str, case_description: str
    ) -> str | None:
        client = anthropic.Anthropic(api_key=self._config.qa_llm_api_key)
        history = "\n".join(
            f"QA: {t.sent}\nAgent: {t.received}" for t in turns
        )
        context_parts = [f"Test goal: {goal}", f"Case description: {case_description}"]
        if self._description:
            context_parts.append(f"Agent description: {self._description}")
        if self._prompt:
            context_parts.append(f"Agent system prompt:\n{self._prompt}")
        context = "\n".join(context_parts)

        user_msg = f"{context}\n\nConversation so far:\n{history}\n\nContinue or mark done?"

        with client.messages.stream(
            model=self._config.qa_llm.model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=_TURN_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            message = stream.get_final_message()

        text = next(
            (block.text for block in message.content if hasattr(block, "text")), "{}"
        )
        data = json.loads(text)
        if data.get("done"):
            return None
        return data.get("next_message")
