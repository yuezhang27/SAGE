import json
from typing import Any

from sage import config

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

client = OpenAI(api_key=config.OPENAI_API_KEY) if OpenAI and config.OPENAI_API_KEY else None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    response_format: str = "json",
) -> dict[str, Any] | str:
    if client is None:
        raise RuntimeError("OpenAI client unavailable. Install openai package and set OPENAI_API_KEY.")

    model = model or config.DEFAULT_MODEL
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: dict[str, Any] = {}
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    text = resp.choices[0].message.content or ""

    if response_format == "json":
        return json.loads(text)
    return text
