"""Minimal async Claude client over plain httpx.

The official `anthropic` SDK pulls in native DLLs (e.g. `jiter`) that Windows Smart App Control
can block, so we call the Messages API directly. Only two calls are needed:
a forced tool call for structured JSON, and a server-side web search.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

import httpx

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_PARALLEL_LLM, WEB_SEARCH_TOOL


API_URL = 'https://api.anthropic.com/v1/messages'
API_VERSION = '2023-06-01'
RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 4
MIN_MAX_TOKENS = 4000  # reasoning models spend tokens before answering


def blocked_domains_from_error(message: str) -> set[str]:
    """Extract domains from "The following domains are not accessible to our user agent: ['a.com', ...]"."""
    if 'not accessible' not in message or '[' not in message:
        return set()
    listed = message[message.index('[') : message.index(']') + 1] if ']' in message else ''
    return set(re.findall(r"'([^']+)'", listed))


def _json_from_text(text: str) -> dict[str, Any] | None:
    """Parse the outermost JSON object in a text answer (fallback when no tool call came back)."""
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class LLMError(RuntimeError):
    """Raised with the API's own error message so config problems are obvious."""


class LLM:
    """Async Claude wrapper. Must be created inside a running event loop."""

    def __init__(self, model: str = CLAUDE_MODEL, max_parallel: int = MAX_PARALLEL_LLM) -> None:
        if not ANTHROPIC_API_KEY:
            raise LLMError('ANTHROPIC_API_KEY is not set (put it in .env)')
        self.model = model
        self._forced_tool_choice = True  # switched off once if the model rejects it
        self._blocked_domains: set[str] = set()  # domains web search reported as inaccessible
        self._sem = asyncio.Semaphore(max_parallel)
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=15.0),
            headers={'x-api-key': ANTHROPIC_API_KEY, 'anthropic-version': API_VERSION, 'content-type': 'application/json'},
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._sem:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    resp = await self._http.post(API_URL, json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == MAX_ATTEMPTS:
                        raise LLMError(f'network error calling Claude: {exc!r}') from exc
                    await asyncio.sleep(2**attempt + random.random())
                    continue

                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                    retry_after = resp.headers.get('retry-after', '')
                    delay = float(retry_after) if retry_after.replace('.', '', 1).isdigit() else 2**attempt + random.random()
                    await asyncio.sleep(min(delay, 30.0))
                    continue
                try:
                    message = resp.json().get('error', {}).get('message', resp.text)
                except ValueError:
                    message = resp.text
                raise LLMError(f'Claude API {resp.status_code}: {message}')
        raise LLMError('unreachable')

    async def structured(
        self, system: str, prompt: str, schema: dict[str, Any], name: str = 'output', max_tokens: int = 2000
    ) -> dict[str, Any]:
        """Return a JSON object matching `schema` via a tool call.

        Tries a forced tool call first. Some models (e.g. with built-in reasoning) reject forced
        tool_choice; then we switch once to tool_choice=auto with an explicit instruction, and as a
        last resort parse a JSON object from the text answer.
        """
        tool = {'name': name, 'description': 'Return the result. Always call this tool exactly once.', 'input_schema': schema}
        payload: dict[str, Any] = {
            'model': self.model,
            'max_tokens': max(max_tokens, MIN_MAX_TOKENS),
            'system': f'{system}\n\nRespond ONLY by calling the `{name}` tool exactly once with the complete result.',
            'messages': [{'role': 'user', 'content': prompt}],
            'tools': [tool],
        }
        if self._forced_tool_choice:
            try:
                data = await self._post({**payload, 'tool_choice': {'type': 'tool', 'name': name}})
            except LLMError as exc:
                if 'tool_choice' not in str(exc):
                    raise
                self._forced_tool_choice = False
                data = await self._post({**payload, 'tool_choice': {'type': 'auto'}})
        else:
            data = await self._post({**payload, 'tool_choice': {'type': 'auto'}})

        texts: list[str] = []
        for block in data.get('content', []):
            if block.get('type') == 'tool_use':
                return dict(block.get('input') or {})
            if block.get('type') == 'text':
                texts.append(block.get('text', ''))
        parsed = _json_from_text('\n'.join(texts))
        if parsed is not None:
            return parsed
        raise LLMError(f'model returned no structured output (stop_reason={data.get("stop_reason")})')

    async def web_search(self, prompt: str, allowed_domains: list[str], max_uses: int = 3, max_tokens: int = 4000) -> list[dict[str, str]]:
        """Run Claude's server-side web search; return only cited snippets (retrieved text).

        If the API reports allowed domains its search agent cannot access, they are remembered
        and the search is retried once without them.
        """
        data: dict[str, Any] = {}
        for attempt in range(2):
            domains = [d for d in allowed_domains if d not in self._blocked_domains]
            try:
                data = await self._post(
                    {
                        'model': self.model,
                        'max_tokens': max(max_tokens, MIN_MAX_TOKENS),
                        'messages': [{'role': 'user', 'content': prompt}],
                        'tools': [{'type': WEB_SEARCH_TOOL, 'name': 'web_search', 'max_uses': max_uses, 'allowed_domains': domains}],
                    }
                )
                break
            except LLMError as exc:
                blocked = blocked_domains_from_error(str(exc))
                if not blocked or attempt == 1:
                    raise
                self._blocked_domains |= blocked
        hits: list[dict[str, str]] = []
        for block in data.get('content', []):
            if block.get('type') != 'text':
                continue
            for c in block.get('citations') or []:
                if c.get('type') == 'web_search_result_location':
                    hits.append({'url': c.get('url', ''), 'title': c.get('title') or '', 'cited_text': c.get('cited_text') or ''})
        return hits

    async def aclose(self) -> None:
        await self._http.aclose()
