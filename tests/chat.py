"""An OpenAI Chat Completions response body, for tests that put lite-model on a recording
transport (the shape every OpenAI-compatible backend answers with)."""

from typing import Any


def chat_completion(
    model: str, content: str | None, finish: str = "stop", usage: tuple[int, int] = (12, 3)
) -> dict[str, Any]:
    prompt, completion = usage
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }
