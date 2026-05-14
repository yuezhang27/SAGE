import json
from typing import Any

from openai import OpenAI

from sage import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    response_format: str = "json",
) -> dict[str, Any] | str:
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
