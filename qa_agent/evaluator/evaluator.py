import json
import anthropic
from qa_agent.models import RunResult, EvalResult
from qa_agent.config import Config

_SYSTEM_PROMPT = """\
You are a QA judge evaluating whether a conversation with an LLM agent met its test goal.

Return a JSON object with this exact schema:
{
  "passed": true/false,
  "score": 0.0 to 1.0,
  "rationale": "one or two sentences explaining the verdict",
  "failure_detail": "specific failure details (omit or null if passed=true)"
}

Be strict but fair. Consider whether the agent's responses were accurate, helpful, \
and consistent with its described purpose.

Return ONLY valid JSON."""


def _build_eval_prompt(
    run_result: RunResult,
    description: str | None,
    prompt: str | None,
) -> str:
    parts = [
        f"Test case: {run_result.test_case.description}",
        f"Category: {run_result.test_case.category}",
        f"Goal: {run_result.test_case.goal}",
    ]
    if description:
        parts.append(f"Agent description: {description}")
    if prompt:
        parts.append(f"Agent system prompt:\n{prompt}")
    parts.append("\nConversation:")
    for turn in run_result.turns:
        parts.append(f"User: {turn.sent}")
        parts.append(f"Agent: {turn.received}")
    return "\n".join(parts)


def evaluate(
    run_results: list[RunResult],
    description: str | None,
    prompt: str | None,
    config: Config,
) -> list[EvalResult]:
    client = anthropic.Anthropic(api_key=config.qa_llm_api_key)
    results: list[EvalResult] = []

    for rr in run_results:
        if not rr.success:
            results.append(EvalResult(
                run_result=rr,
                passed=False,
                score=0.0,
                rationale=f"Run failed: {rr.error}",
                failure_detail=rr.error,
            ))
            continue

        try:
            user_msg = _build_eval_prompt(rr, description, prompt)
            with client.messages.stream(
                model=config.qa_llm.model,
                max_tokens=1024,
                thinking={"type": "adaptive"},
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                message = stream.get_final_message()

            text = next(
                (block.text for block in message.content if hasattr(block, "text")), "{}"
            )
            data = json.loads(text)
            results.append(EvalResult(
                run_result=rr,
                passed=bool(data.get("passed", False)),
                score=float(data.get("score", 0.0)),
                rationale=data.get("rationale", ""),
                failure_detail=data.get("failure_detail"),
            ))
        except Exception as exc:
            results.append(EvalResult(
                run_result=rr,
                passed=False,
                score=0.0,
                rationale=f"Evaluation failed: {exc}",
                failure_detail=str(exc),
            ))

    return results
