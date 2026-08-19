import logging
import openai

logger = logging.getLogger(__name__)


def call_llm(
    system: str,
    user: str,
    model: str,
    api_key: str,
    max_tokens: int = 1024,
) -> str:
    logger.debug("LLM call model=%s max_tokens=%d", model, max_tokens)
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = response.choices[0].message.content or ""
    logger.debug("LLM response length=%d chars", len(text))
    return text


def call_llm_chat(
    system: str,
    messages: list[dict],
    model: str,
    api_key: str,
    max_tokens: int = 1024,
) -> str:
    """Multi-turn variant: pass a full messages list alongside a system prompt."""
    logger.debug("LLM chat call model=%s turns=%d max_tokens=%d", model, len(messages), max_tokens)
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}] + messages,
    )
    text = response.choices[0].message.content or ""
    logger.debug("LLM chat response length=%d chars", len(text))
    return text
