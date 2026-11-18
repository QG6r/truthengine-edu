from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from app.services.truthengine import Claim, ClaimResult, TruthEngine


SUPPORTED = 1.0
NOT_ENOUGH_EVIDENCE = 0.5
REFUTED = 0.0

ClaimInput = str | Claim


class CheckTrustedSources:
    """Scores already-extracted claims against trusted open sources.

    Sources: peer-reviewed papers (OpenAlex, Crossref), fact-checkers (Google Fact Check),
    Wikipedia and a whitelisted web search. No course materials are used.

    Score is continuous in 0..1: >= 0.75 supported, <= 0.25 refuted, otherwise
    not enough evidence / conflicting (0.5 = nothing found either way).

    All claims of a batch are checked concurrently, and for each claim all sources are
    queried concurrently. Use the *_async methods from async code (e.g. FastAPI);
    the sync methods call asyncio.run and cannot be used inside a running event loop.
    """

    def __init__(self, engine_factory: Callable[[], TruthEngine] = TruthEngine) -> None:
        self._engine_factory = engine_factory

    # --- sync API (scripts, eval) ---------------------------------------------
    def check(self, claim: ClaimInput) -> float:
        """Return the score for a single claim."""
        return self.run([claim])[0]

    def run(self, claims: Sequence[ClaimInput]) -> list[float]:
        """Return a score for each claim, in input order."""
        return [r.score for r in self.run_detailed(claims)]

    def run_detailed(self, claims: Sequence[ClaimInput]) -> list[ClaimResult]:
        """Full results (verdict, confidence, evidence, gates, explanation), in input order."""
        return asyncio.run(self.run_detailed_async(claims))

    # --- async API (FastAPI) ----------------------------------------------------
    async def check_async(self, claim: ClaimInput) -> float:
        return (await self.run_detailed_async([claim]))[0].score

    async def run_async(self, claims: Sequence[ClaimInput]) -> list[float]:
        return [r.score for r in await self.run_detailed_async(claims)]

    async def run_detailed_async(self, claims: Sequence[ClaimInput]) -> list[ClaimResult]:
        if not claims:
            return []
        async with self._engine_factory() as engine:
            return await engine.run_detailed(list(claims))
