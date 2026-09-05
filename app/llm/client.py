"""Minimal provider abstraction; demo mode is always available."""

from typing import Protocol

import httpx


class LLMClient(Protocol):
    def explain(self, context: dict[str, object]) -> dict[str, object]: ...


class DemoLLMClient:
    """Use deterministic structured explanations without external credentials."""

    def explain(self, context: dict[str, object]) -> dict[str, object]:
        from app.llm.explanation import build_fallback_explanation
        return build_fallback_explanation(context)


class OpenAICompatibleClient:
    """Call an OpenAI-compatible chat endpoint with structured JSON output."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def explain(self, context: dict[str, object]) -> dict[str, object]:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": _prompt(context)}],
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        import json
        result = json.loads(content)
        required = {"summary", "why_recommended", "matching_skills", "missing_skills", "skill_gap_priority", "confidence_note"}
        if not required.issubset(result):
            raise ValueError("LLM response is missing explanation fields.")
        return result


def _prompt(context: dict[str, object]) -> str:
    return "Generate JSON explanation using only these facts. Do not invent skills: " + str(context)
