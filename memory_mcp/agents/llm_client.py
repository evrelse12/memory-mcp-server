# memory_mcp/agents/llm_client.py
"""Single shared LLM client: one httpx client, one retry, optional JSON mode."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence

import httpx

from memory_mcp.config.settings import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self._model = model or settings.llm_model
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.llm_api_url,
            timeout=httpx.Timeout(settings.llm_timeout, connect=5.0),
        )
        self.last_finish_reason: str | None = None
        self.last_content_chars: int = 0
        self.last_from_reasoning: bool = False

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = await self._client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                self.last_finish_reason = choice.get("finish_reason")
                self.last_from_reasoning = False
                # Reasoning models may emit a short answer entirely in the
                # reasoning channel, leaving content empty. Use it only when
                # generation completed (never salvage truncated thinking).
                if (
                    not content.strip()
                    and reasoning.strip()
                    and self.last_finish_reason != "length"
                ):
                    content = reasoning
                    self.last_from_reasoning = True
                    logger.debug("LLM content taken from reasoning channel")
                self.last_content_chars = len(content)
                if not content.strip():
                    logger.warning(
                        "LLM empty content (finish_reason=%s, max_tokens=%s, model=%s)",
                        self.last_finish_reason,
                        max_tokens,
                        self._model,
                    )
                return content
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
                last = e
                if attempt == 1:
                    await asyncio.sleep(1.0)
        raise LLMError(f"llm request failed: {last}")

    async def chat_json(
        self, messages: Sequence[dict], *, max_tokens: int = 1024, temperature: float = 0.2
    ) -> dict | None:
        """JSON mode first, greedy-brace fallback second, None on total failure."""
        try:
            text = await self.chat(
                messages, max_tokens=max_tokens, temperature=temperature, json_mode=True
            )
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
