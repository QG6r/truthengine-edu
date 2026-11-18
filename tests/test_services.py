"""Tests for the service layer (no network: the engine is replaced by a fake)."""

from __future__ import annotations

import asyncio

import pytest

from app.services.separator import SeparationError, Separator
from app.services.trusted_sources import CheckTrustedSources
from app.services.truthengine import Claim, ClaimResult


def test_separator_requires_configuration(monkeypatch) -> None:
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(SeparationError, match='ANTHROPIC_API_KEY'):
        Separator().separate('some text')


class FakeEngine:
    """Mimics TruthEngine: async context manager + concurrent run_detailed."""

    max_in_flight = 0

    async def __aenter__(self) -> FakeEngine:
        self._in_flight = 0
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def _one(self, item: str | Claim) -> ClaimResult:
        self._in_flight += 1
        FakeEngine.max_in_flight = max(FakeEngine.max_in_flight, self._in_flight)
        await asyncio.sleep(0.01)
        self._in_flight -= 1
        claim = item if isinstance(item, Claim) else Claim(text=item)
        score = 1.0 if 'true' in claim.text else 0.0
        verdict = 'supported' if score else 'contradicted'
        return ClaimResult(claim=claim, verdict=verdict, score=score, confidence=1.0)

    async def run_detailed(self, items: list[str | Claim]) -> list[ClaimResult]:
        return list(await asyncio.gather(*(self._one(i) for i in items)))


def test_run_keeps_order_and_runs_concurrently() -> None:
    FakeEngine.max_in_flight = 0
    checker = CheckTrustedSources(engine_factory=FakeEngine)
    assert checker.run(['a true claim', 'a false claim', 'another true one']) == [1.0, 0.0, 1.0]
    assert FakeEngine.max_in_flight == 3


def test_check_returns_float() -> None:
    assert CheckTrustedSources(engine_factory=FakeEngine).check('true') == 1.0


def test_empty_batch() -> None:
    assert CheckTrustedSources(engine_factory=FakeEngine).run([]) == []


async def test_async_api() -> None:
    checker = CheckTrustedSources(engine_factory=FakeEngine)
    assert await checker.run_async(['true', 'nope']) == [1.0, 0.0]
    assert await checker.check_async(Claim(text='true')) == 1.0
