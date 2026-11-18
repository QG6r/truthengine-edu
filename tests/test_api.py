"""POST /analyze: success, every fallback path, validation."""

from __future__ import annotations

import sys
import time
import types

import pytest
from fastapi.testclient import TestClient

import api
from analyzer import Analyzer
from mock_data import mock_report
from schemas import Report


REAL = Report(trust_score=1.0, coverage='1 of 1 claims checked', verdicts=[])


def _client(monkeypatch: pytest.MonkeyPatch, run_pipeline=None, timeout_s: float = 90.0) -> TestClient:
    module_name = 'fake_pipeline_for_tests'
    if run_pipeline is None:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    else:
        module = types.ModuleType(module_name)
        module.run_pipeline = run_pipeline
        monkeypatch.setitem(sys.modules, module_name, module)
    analyzer = Analyzer(timeout_s=timeout_s, module_name=module_name)
    api.app.dependency_overrides[api.get_analyzer] = lambda: analyzer
    return TestClient(api.app)


def test_health() -> None:
    assert TestClient(api.app).get('/health').json() == {'status': 'ok'}


def test_pipeline_result_returned_without_demo_header(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _client(monkeypatch, lambda text: REAL).post('/analyze', json={'text': 'hello'})
    assert r.status_code == 200
    assert r.json() == REAL.model_dump()
    assert 'x-demo-data' not in r.headers


def test_async_pipeline_and_dict_result(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_pipeline(text: str) -> dict:
        return REAL.model_dump()

    r = _client(monkeypatch, run_pipeline).post('/analyze', json={'text': 'hello'})
    assert r.json() == REAL.model_dump()
    assert 'x-demo-data' not in r.headers


def test_missing_pipeline_returns_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _client(monkeypatch, None).post('/analyze', json={'text': 'hello'})
    assert r.status_code == 200
    assert r.headers['x-demo-data'] == 'true'
    assert r.json() == mock_report('hello').model_dump()


def test_pipeline_exception_returns_mock(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(text: str) -> Report:
        raise RuntimeError('pipeline exploded')

    r = _client(monkeypatch, boom).post('/analyze', json={'text': 'hello'})
    assert r.status_code == 200
    assert r.headers['x-demo-data'] == 'true'
    assert 'pipeline exploded' in caplog.text


def test_invalid_pipeline_output_returns_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _client(monkeypatch, lambda text: {'nope': 1}).post('/analyze', json={'text': 'hello'})
    assert r.status_code == 200
    assert r.headers['x-demo-data'] == 'true'


def test_timeout_returns_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(text: str) -> Report:
        time.sleep(0.5)
        return REAL

    r = _client(monkeypatch, slow, timeout_s=0.05).post('/analyze', json={'text': 'hello'})
    assert r.status_code == 200
    assert r.headers['x-demo-data'] == 'true'


@pytest.mark.parametrize(('text', 'fragment'), [('', 'empty'), ('   \n ', 'empty'), ('x' * 20_001, 'too long')])
def test_invalid_text_returns_422_with_message(monkeypatch: pytest.MonkeyPatch, text: str, fragment: str) -> None:
    r = _client(monkeypatch, lambda t: REAL).post('/analyze', json={'text': text})
    assert r.status_code == 422
    assert fragment in r.json()['detail']


def test_max_length_text_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _client(monkeypatch, lambda t: REAL).post('/analyze', json={'text': 'x' * 20_000}).status_code == 200


def test_cors_allows_localhost_and_exposes_demo_header(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _client(monkeypatch, None).post('/analyze', json={'text': 'hi'}, headers={'Origin': 'http://localhost:8501'})
    assert r.headers['access-control-allow-origin'] == 'http://localhost:8501'
    assert 'x-demo-data' in r.headers['access-control-expose-headers'].lower()
