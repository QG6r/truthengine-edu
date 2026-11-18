"""TruthEngine: the async pipeline planner -> sources (in parallel) -> judge -> gates -> verdict."""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import httpx

from .config import CLAIM_TIMEOUT_S, HTTP_TIMEOUT, MAX_EVIDENCE_PER_CLAIM, MAX_PARALLEL_CLAIMS, SOURCE_CONCURRENCY, TIER_BASE, USER_AGENT
from .judge import Judge
from .llm import LLM
from .models import CitationRef, Claim, ClaimResult, Evidence, GateResult, QueryPlan
from .planner import Planner
from .scoring import aggregate, score_evidence
from .sources import CrossrefSource, FactCheckSource, KnowledgeSource, OpenAlexSource, WebSearchSource, WikipediaSource


ACADEMIC_TYPES = {'scientific', 'medical', 'causal', 'statistic', 'citation'}


class TruthEngine:
    """Async engine: planner -> all sources in parallel -> judge -> gates -> weighted verdict.

    async with TruthEngine() as engine:
        results = await engine.run_detailed(['claim 1', 'claim 2'])

    Everything bound to the event loop (HTTP client, Claude client, semaphores) is created
    in __aenter__, so one engine instance must be used inside one event loop.
    """

    def __init__(self, max_parallel_claims: int = MAX_PARALLEL_CLAIMS, claim_timeout_s: float = CLAIM_TIMEOUT_S) -> None:
        self.max_parallel_claims = max_parallel_claims
        self.claim_timeout_s = claim_timeout_s
        self._http: httpx.AsyncClient | None = None

    # --- lifecycle ------------------------------------------------------------
    async def __aenter__(self) -> TruthEngine:
        self._http = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            headers={'User-Agent': USER_AGENT},
            limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
            follow_redirects=True,
        )
        self.llm = LLM()
        self.planner = Planner(self.llm)
        self.judge = Judge(self.llm)
        c = SOURCE_CONCURRENCY
        self.openalex = OpenAlexSource(self._http, c['openalex'])
        self.crossref = CrossrefSource(self._http, c['crossref'])
        self.sources: dict[str, Any] = {
            'knowledge': KnowledgeSource(),
            'openalex': self.openalex,
            'wikipedia': WikipediaSource(self._http, c['wikipedia']),
            'factcheck': FactCheckSource(self._http, c['factcheck']),
            'websearch': WebSearchSource(self.llm),
        }
        self._claim_sem = asyncio.Semaphore(self.max_parallel_claims)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._http:
            await self._http.aclose()
            await self.llm.aclose()
            self._http = None

    def _require_open(self) -> None:
        if self._http is None:
            raise RuntimeError('use `async with TruthEngine() as engine:`')

    # --- public API -------------------------------------------------------------
    async def run_detailed(self, items: list[str | Claim]) -> list[ClaimResult]:
        """Check every claim concurrently (bounded); results keep input order."""
        self._require_open()
        return list(await asyncio.gather(*(self._guarded(i, it) for i, it in enumerate(items))))

    async def check_detailed(self, item: str | Claim, idx: int = 0) -> ClaimResult:
        self._require_open()
        claim = item if isinstance(item, Claim) else Claim(text=item)
        if not claim.id:
            claim = claim.model_copy(update={'id': f'C{idx + 1}'})

        try:
            plan = await self.planner.plan(claim)
        except Exception as exc:
            return self._unverifiable(claim, None, [f'planner: {exc!r}'], 'Could not analyse the claim.')

        # Every source (and the citation check) runs at the same time.
        tasks: dict[str, Any] = {name: self.sources[name].search(plan, claim) for name in self._route(plan)}
        if plan.citation:
            tasks['citation'] = self._verify_citation(plan.citation)
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        errors: list[str] = []
        evidence: list[Evidence] = []
        citation: dict[str, Any] | None = None
        for name, res in zip(tasks, results, strict=True):
            if isinstance(res, BaseException):
                errors.append(f'{name}: {type(res).__name__}: {res}')
            elif name == 'citation':
                citation = res
            else:
                evidence.extend(res)

        claim_gates: list[GateResult] = []
        if citation:
            short = self._citation_verdict(claim, plan, citation, errors)
            if short:
                return short
            claim_gates.append(GateResult(name='G1_citation_exists', status='pass', reason=f'doi:{citation["doi"]}'))
            if citation.get('evidence'):
                evidence.insert(0, citation['evidence'])

        evidence = self._dedupe(evidence)
        if not evidence:
            return self._unverifiable(claim, plan, errors, 'No evidence was found in trusted sources.')

        summary = await self.judge.judge(claim, plan, evidence)
        year = dt.date.today().year
        for ev in evidence:
            score_evidence(ev, plan, year)
        agg = aggregate(evidence)

        return ClaimResult(claim=claim, plan=plan.model_dump(), evidence=evidence, gates=claim_gates, errors=errors, **summary, **agg)

    # --- helpers ----------------------------------------------------------------
    async def _guarded(self, idx: int, item: str | Claim) -> ClaimResult:
        async with self._claim_sem:
            claim = item if isinstance(item, Claim) else Claim(text=item)
            if not claim.id:
                claim = claim.model_copy(update={'id': f'C{idx + 1}'})
            try:
                return await asyncio.wait_for(self.check_detailed(claim, idx), timeout=self.claim_timeout_s)
            except TimeoutError:
                return self._unverifiable(claim, None, [f'timeout after {self.claim_timeout_s:.0f}s'], 'The check took too long.')
            except Exception as exc:  # one broken claim must not break the batch
                return self._unverifiable(claim, None, [f'engine: {exc!r}'], 'Check failed.')

    @staticmethod
    def _route(plan: QueryPlan) -> list[str]:
        names = ['knowledge', 'wikipedia', 'websearch', 'factcheck']
        if plan.claim_type in ACADEMIC_TYPES:
            names.append('openalex')
        return names

    async def _verify_citation(self, cit: CitationRef) -> dict[str, Any]:
        found = await self.crossref.find_citation(cit)
        if not found:
            return {'status': 'not_found', 'query': cit.model_dump()}
        work = await self.openalex.by_doi(found['doi'])
        if work and work.get('is_retracted'):
            return {'status': 'retracted', **found}
        ev = self.openalex.to_evidence(work) if work else None
        if ev:  # the cited work itself is the primary source for "study X found Y"
            ev.tier, ev.base_weight = '1C', TIER_BASE['1C']
        return {'status': 'found', **found, 'evidence': ev}

    @staticmethod
    def _citation_verdict(claim: Claim, plan: QueryPlan, cit: dict[str, Any], errors: list[str]) -> ClaimResult | None:
        if cit['status'] == 'not_found':
            return ClaimResult(
                claim=claim,
                plan=plan.model_dump(),
                verdict='contradicted',
                score=0.15,
                confidence=0.7,
                errors=errors,
                flags=['invented_source'],
                gates=[GateResult(name='G1_citation_exists', status='fail', reason='no matching work in Crossref')],
                explanation=(
                    'The cited work could not be found with these authors, year and journal. '
                    'This is a typical sign of an invented reference.'
                ),
                critical_question='Can you find this study on the journal website or in Google Scholar by its authors and year?',
            )
        if cit['status'] == 'retracted':
            return ClaimResult(
                claim=claim,
                plan=plan.model_dump(),
                verdict='contradicted',
                score=0.1,
                confidence=0.8,
                errors=errors,
                flags=['retracted_source'],
                gates=[GateResult(name='G2_not_retracted', status='fail', reason=f'doi:{cit["doi"]}')],
                explanation='The cited work exists but has been retracted, so it cannot support the claim.',
                critical_question='Before trusting a study, did you check whether it was retracted or corrected?',
            )
        return None

    @staticmethod
    def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
        seen: set[str] = set()
        out: list[Evidence] = []
        for ev in sorted(evidence, key=lambda e: -e.base_weight):
            key = (ev.doi or ev.url).lower()
            if key and key not in seen:
                seen.add(key)
                out.append(ev)
        return out[:MAX_EVIDENCE_PER_CLAIM]

    @staticmethod
    def _unverifiable(claim: Claim, plan: QueryPlan | None, errors: list[str], explanation: str) -> ClaimResult:
        return ClaimResult(
            claim=claim,
            plan=plan.model_dump() if plan else {},
            verdict='unverifiable',
            score=0.5,
            confidence=0.0,
            explanation=explanation,
            errors=errors,
        )
