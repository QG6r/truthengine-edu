"""Wikipedia: full text of the articles named by the planner plus lead sections of search hits (tier 3D)."""

from __future__ import annotations

import asyncio

from ..config import TIER_BASE
from ..models import Claim, Evidence, GateResult, QueryPlan
from .base import BaseSource


class WikipediaSource(BaseSource):
    """Full text of the articles the planner named, plus lead sections of search hits."""

    name = 'wikipedia'
    URL = 'https://en.wikipedia.org/w/api.php'

    async def search(self, plan: QueryPlan, claim: Claim) -> list[Evidence]:
        # Exact articles first: dedupe keeps the first copy, and we want the full text.
        tasks = [self._article(title) for title in plan.wiki_titles[:3]]
        tasks.append(self._search_leads(plan.neutral_query))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: list[Evidence] = []
        seen: set[str] = set()
        for res in results:
            if isinstance(res, BaseException):
                continue
            for ev in res:
                if ev.url and ev.url not in seen:
                    seen.add(ev.url)
                    out.append(ev)
        if not out:
            errors = [r for r in results if isinstance(r, BaseException)]
            if errors:
                raise errors[0]
        return out

    async def _article(self, title: str) -> list[Evidence]:
        params = {
            'action': 'query',
            'format': 'json',
            'redirects': 1,
            'titles': title,
            'prop': 'extracts|info',
            'explaintext': 1,
            'inprop': 'url',
        }
        data = await self.get_json(self.URL, params)
        return self._to_evidence(data, full=True)

    async def _search_leads(self, query: str) -> list[Evidence]:
        params = {
            'action': 'query',
            'format': 'json',
            'redirects': 1,
            'generator': 'search',
            'gsrsearch': query,
            'gsrlimit': 3,
            'prop': 'extracts|info',
            'exintro': 1,
            'explaintext': 1,
            'exlimit': 3,
            'inprop': 'url',
        }
        data = await self.get_json(self.URL, params)
        return self._to_evidence(data, full=False)

    def _to_evidence(self, data: dict, full: bool) -> list[Evidence]:
        out: list[Evidence] = []
        for page in (data.get('query', {}).get('pages', {}) or {}).values():
            extract = (page.get('extract') or '').strip()
            if not extract:
                continue
            out.append(
                Evidence(
                    source=self.name,
                    url=page.get('fullurl', ''),
                    domain='wikipedia.org',
                    title=page.get('title', ''),
                    text=extract,
                    tier='3D',
                    base_weight=TIER_BASE['3D'],
                    gates=[
                        GateResult(
                            name='3D_footnotes',
                            status='unknown',
                            reason='full article; footnotes not followed yet' if full else 'lead section of a search hit',
                        )
                    ],
                )
            )
        return out
