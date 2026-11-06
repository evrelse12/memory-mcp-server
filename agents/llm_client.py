# agents/llm_client.py
"""Single shared LLM client: one httpx client, one retry, optional JSON mode."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self._model = model or settings.llm_model
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.llm_api_url,
            timeout=httpx.Timeout(settings.llm_timeout, connect=5.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, messages: Sequence[dict], *, max_tokens: int = 1024,
                   temperature: float = 0.3, json_mode: bool = False) -> str:
        payload = {"model": self._model, "messages": list(messages),
                   "max_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = await self._client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
                last = e
                if attempt == 1:
                    await asyncio.sleep(1.0)
        raise LLMError(f"llm request failed: {last}")

    async def chat_json(self, messages: Sequence[dict], *, max_tokens: int = 1024,
                        temperature: float = 0.2) -> dict | None:
        """JSON mode first, greedy-brace fallback second, None on total failure."""
        try:
            text = await self.chat(messages, max_tokens=max_tokens,
                                   temperature=temperature, json_mode=True)
        except LLMError:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    return None
        return None
