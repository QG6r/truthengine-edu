"""OpenAlex: scholarly works, study design, retraction status (tiers 2A / 2C / 4B)."""

from __future__ import annotations

from typing import Any

import httpx

from ..config import CONTACT_EMAIL, OPENALEX_API_KEY, TIER_BASE
from ..models import Claim, Evidence, GateResult, QueryPlan
from .base import BaseSource


SELECT = 'id,doi,display_name,publication_year,cited_by_count,is_retracted,type,primary_location,abstract_inverted_index'


def rebuild_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ''
    words: dict[int, str] = {}
    for word, positions in inverted.items():
        for p in positions:
            words[p] = word
    return ' '.join(words[i] for i in sorted(words))


def classify_work(work: dict[str, Any]) -> tuple[str, float, str]:
    title = (work.get('display_name') or '').lower()
    wtype = (work.get('type') or '').lower()
    source = (work.get('primary_location') or {}).get('source') or {}
    if wtype == 'preprint' or source.get('type') == 'repository':
        return '4B', TIER_BASE['4B'], 'preprint'
    if 'systematic review' in title or 'meta-analysis' in title or 'meta analysis' in title:
        return '2A', TIER_BASE['2A'], 'systematic_review'
    if 'randomized' in title or 'randomised' in title:
        return '2C', 0.90, 'rct'
    if 'cohort' in title:
        return '2C', 0.80, 'cohort'
    if wtype == 'review':
        return '2C', 0.80, 'narrative_review'
    return '2C', 0.75, 'unknown_design'


class OpenAlexSource(BaseSource):
    name = 'openalex'
    URL = 'https://api.openalex.org/works'

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = {'select': SELECT, **extra}
        if CONTACT_EMAIL:
            params['mailto'] = CONTACT_EMAIL
        if OPENALEX_API_KEY:
            params['api_key'] = OPENALEX_API_KEY
        return params

    async def search(self, plan: QueryPlan, claim: Claim) -> list[Evidence]:
        query = plan.academic_query or plan.neutral_query
        data = await self.get_json(self.URL, self._params({'search': query, 'per_page': 6}))
        return [ev for w in data.get('results', []) if (ev := self.to_evidence(w))]

    async def by_doi(self, doi: str) -> dict[str, Any] | None:
        try:
            return await self.get_json(f'{self.URL}/doi:{doi}', self._params({}))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    def to_evidence(self, work: dict[str, Any]) -> Evidence | None:
        title = work.get('display_name') or ''
        abstract = rebuild_abstract(work.get('abstract_inverted_index'))
        if not title and not abstract:
            return None
        tier, base, design = classify_work(work)
        source = (work.get('primary_location') or {}).get('source') or {}
        venue = source.get('display_name')
        doi = (work.get('doi') or '').removeprefix('https://doi.org/') or None

        gates = [
            GateResult(name='2C_not_retracted', status='fail', reason='retracted')
            if work.get('is_retracted')
            else GateResult(name='2C_not_retracted', status='pass'),
            GateResult(name='2C_venue_known', status='pass' if venue else 'unknown', reason=venue or 'no venue metadata'),
        ]
        if not abstract:
            gates.append(GateResult(name='2C_abstract', status='unknown', reason='title only'))

        return Evidence(
            source=self.name,
            url=work.get('doi') or work.get('id') or '',
            domain=venue or 'openalex',
            title=title,
            text=f'{title}. {abstract}'.strip(),
            tier=tier,
            base_weight=base,
            year=work.get('publication_year'),
            doi=doi,
            gates=gates,
            meta={'design': design, 'cited_by': work.get('cited_by_count'), 'venue': venue, 'in_doaj': source.get('is_in_doaj')},
        )
