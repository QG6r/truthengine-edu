"""LLM client parsing and retries, with a mocked HTTP transport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from app.services.truthengine import llm as llm_mod


def make_llm(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> llm_mod.LLM:
    monkeypatch.setattr(llm_mod, 'ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.setattr(llm_mod.asyncio, 'sleep', _no_sleep)
    client = llm_mod.LLM(model='test-model')
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=client._http.headers)
    return client


async def _no_sleep(_: float) -> None:
    return None


async def test_structured_returns_tool_input_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {'n': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls['n'] += 1
        assert request.headers['x-api-key'] == 'test-key'
        if calls['n'] == 1:
            return httpx.Response(529, json={'error': {'message': 'overloaded'}})
        return httpx.Response(200, json={'content': [{'type': 'tool_use', 'name': 'out', 'input': {'ok': True}}]})

    client = make_llm(monkeypatch, handler)
    assert await client.structured('sys', 'prompt', {'type': 'object'}, name='out') == {'ok': True}
    assert calls['n'] == 2
    await client.aclose()


async def test_web_search_keeps_only_cited_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        'content': [
            {'type': 'server_tool_use', 'name': 'web_search'},
            {
                'type': 'text',
                'text': 'Einstein was born in Ulm.',
                'citations': [
                    {
                        'type': 'web_search_result_location',
                        'url': 'https://britannica.com/e',
                        'title': 'Einstein',
                        'cited_text': 'born in Ulm',
                    }
                ],
            },
            {'type': 'text', 'text': 'no citations here'},
        ]
    }
    client = make_llm(monkeypatch, lambda request: httpx.Response(200, json=body))
    hits = await client.web_search('q', ['britannica.com'])
    assert hits == [{'url': 'https://britannica.com/e', 'title': 'Einstein', 'cited_text': 'born in Ulm'}]
    await client.aclose()


async def test_falls_back_when_forced_tool_choice_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        choice = json.loads(request.content)['tool_choice']['type']
        seen.append(choice)
        if choice == 'tool':
            return httpx.Response(400, json={'error': {'message': 'tool_choice: type "tool" and "any" are not supported for this model.'}})
        return httpx.Response(200, json={'content': [{'type': 'thinking'}, {'type': 'text', 'text': 'Here: {"a": 1}'}]})

    client = make_llm(monkeypatch, handler)
    assert await client.structured('s', 'p', {'type': 'object'}) == {'a': 1}
    assert await client.structured('s', 'p', {'type': 'object'}) == {'a': 1}
    assert seen == ['tool', 'auto', 'auto']  # forced is tried only once
    await client.aclose()


async def test_client_error_surfaces_api_message(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_llm(monkeypatch, lambda request: httpx.Response(400, json={'error': {'message': 'model: not found'}}))
    with pytest.raises(llm_mod.LLMError, match='model: not found'):
        await client.structured('s', 'p', {'type': 'object'})
    await client.aclose()
