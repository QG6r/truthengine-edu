"""Google Fact Check Tools API: existing reviews by fact-checking organisations (tier 3A)."""

from __future__ import annotations

from ..config import GOOGLE_FACTCHECK_KEY, TIER_BASE
from ..models import Claim, Evidence, GateResult, QueryPlan
from ..scoring import classify_domain
from .base import BaseSource


class FactCheckSource(BaseSource):
    name = 'factcheck'
    URL = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'

    async def search(self, plan: QueryPlan, claim: Claim) -> list[Evidence]:
        if not GOOGLE_FACTCHECK_KEY:
            return []
        params = {'query': plan.neutral_query, 'key': GOOGLE_FACTCHECK_KEY, 'languageCode': 'en', 'pageSize': 5}
        data = await self.get_json(self.URL, params)
        out: list[Evidence] = []
        for item in data.get('claims', []):
            for review in item.get('claimReview', []):
                url = review.get('url', '')
                publisher = (review.get('publisher') or {}).get('name', 'unknown')
                tier, host = classify_domain(url)
                date = review.get('reviewDate') or ''
                text = (
                    f'Claim reviewed: "{item.get("text", "")}". '
                    f'Rating by {publisher}: {review.get("textualRating", "")}. '
                    f'{review.get("title", "")}'
                )
                out.append(
                    Evidence(
                        source=self.name,
                        url=url,
                        domain=host,
                        title=review.get('title', ''),
                        text=text,
                        tier='3A',
                        base_weight=TIER_BASE['3A'],
                        year=int(date[:4]) if date[:4].isdigit() else None,
                        gates=[GateResult(name='3A_known_factchecker', status='pass' if tier == '3A' else 'unknown', reason=publisher)],
                    )
                )
        return out
