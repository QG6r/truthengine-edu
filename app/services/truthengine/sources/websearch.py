"""Claude's server-side web search restricted to trusted domains."""

from __future__ import annotations

from ..config import TIER_BASE, TRUSTED_WEB_DOMAINS
from ..llm import LLM
from ..models import Claim, Evidence, QueryPlan
from ..scoring import classify_domain


class WebSearchSource:
    """Whitelisted web search; only retrieved snippets are kept as evidence text."""

    name = 'websearch'

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def search(self, plan: QueryPlan, claim: Claim) -> list[Evidence]:
        prompt = (
            'Find reliable evidence about the claim below, both supporting AND refuting it.\n'
            f'Claim: {claim.text}\n'
            f'Search 1 (neutral): {plan.neutral_query}\n'
            f'Search 2 (refuting): {plan.refute_query}\n'
            'Then summarise briefly what the sources say, citing them.'
        )
        hits = await self.llm.web_search(prompt, TRUSTED_WEB_DOMAINS)

        by_url: dict[str, dict] = {}
        for h in hits:
            entry = by_url.setdefault(h['url'], {'title': h['title'], 'parts': []})
            if h['cited_text'] and h['cited_text'] not in entry['parts']:
                entry['parts'].append(h['cited_text'])

        out: list[Evidence] = []
        for url, entry in by_url.items():
            if not entry['parts']:
                continue
            tier, host = classify_domain(url)
            out.append(
                Evidence(
                    source=self.name,
                    url=url,
                    domain=host,
                    title=entry['title'],
                    text='\n'.join(entry['parts']),
                    tier=tier,
                    base_weight=TIER_BASE[tier],
                    meta={'retrieval': 'claude_web_search'},
                )
            )
        return out
