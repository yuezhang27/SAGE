import json
from typing import Any

from sage import config

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY) if anthropic and config.ANTHROPIC_API_KEY else None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    response_format: str = "json",
) -> dict[str, Any] | str:
    if client is None:
        raise RuntimeError("Anthropic client unavailable. Install anthropic package and set ANTHROPIC_API_KEY.")

    model = model or config.DEFAULT_MODEL

    if response_format == "json":
        system_prompt += "\n\nYou MUST respond with valid JSON only. No other text."

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = resp.content[0].text or ""

    if response_format == "json":
        return json.loads(text)
    return text
