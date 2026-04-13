from __future__ import annotations

import json
import logging
import os
from typing import Any

from graph_agent.models import Intent

logger = logging.getLogger(__name__)


class IntentInferrer:
    """Infers business intents from transitions using LLM."""

    def __init__(self):
        self._model = os.getenv("LLM_MODEL", "google/gemini-3-flash-preview")
        self._base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        self._api_key = os.getenv("LLM_API_KEY", "")

    async def infer(self, transition_data: dict[str, Any]) -> Intent:
        """Infer intent from transition context (action, selector, thought)."""
        action = transition_data.get("action", "")
        selector = transition_data.get("selector", "")
        thought = transition_data.get("thought", "")
        from_url = transition_data.get("from_url", "")
        to_url = transition_data.get("to_url", "")
        element_text = transition_data.get("element_text", "")

        prompt = (
            "Given the following UI transition, infer the business intent.\n\n"
            f"Action: {action}\n"
            f"Element: {selector} ({element_text})\n"
            f"Thought: {thought}\n"
            f"From URL: {from_url}\n"
            f"To URL: {to_url}\n\n"
            "Respond in JSON: {\"verb\": \"...\", \"object\": \"...\", \"summary\": \"...\", \"key\": \"verb:object\"}\n"
            "verb = action verb (e.g. create, delete, query, edit)\n"
            "object = business object (e.g. user, order, product)\n"
            "summary = one-line description of the intent\n"
            "key = verb:object format"
        )

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a UI testing expert. Respond only in valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        content = response.choices[0].message.content or "{}"
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for intent: %s", content[:200])
            parsed = {"verb": "unknown", "object": "unknown", "summary": thought or "", "key": "unknown:unknown"}

        intent_id = f"intent:{parsed.get('key', 'unknown:unknown')}"
        return Intent(
            id=intent_id,
            name=f"{parsed.get('verb', '')} {parsed.get('object', '')}".strip(),
            summary=parsed.get("summary", ""),
            verb=parsed.get("verb", ""),
            object=parsed.get("object", ""),
            key=parsed.get("key", ""),
            confidence=0.5,
        )
