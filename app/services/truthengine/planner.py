"""Planner: claim type, neutral / supporting / refuting search queries, Wikipedia titles, citation."""

from __future__ import annotations

from .llm import LLM
from .models import Claim, QueryPlan


PLANNER_SYSTEM = """You prepare ONE factual claim for verification. Do not judge whether it is true.
Return:
- claim_type: one of scientific, medical, causal, statistic, historical, biographical,
  geographic, physical, quote, citation, definition, current_event, other.
- neutral_query: search keywords about the SUBJECT that do NOT contain the asserted answer
  (good: "Albert Einstein birthplace"; bad: "Einstein born in Ulm").
- support_query: the claim phrased as a search query.
- refute_query: a query likely to surface corrections, e.g. "<topic> myth", "<topic> misconception",
  "<topic> debunked".
- academic_query: 3-8 English scholarly keywords for a paper database.
- wiki_titles: 1-3 exact English Wikipedia article titles most likely to state the fact
  (e.g. "Albert Einstein", "Eiffel Tower", "Melting point"). Use real article titles only.
- time_sensitive: true if the truth can change over time (statistics, office holders, prices, laws).
- citation: fill ONLY if the claim explicitly cites a specific work (authors, year, journal/venue).
  Copy exactly as written; never correct or complete it. Otherwise omit it."""

PLAN_SCHEMA = {
    'type': 'object',
    'properties': {
        'claim_type': {'type': 'string'},
        'neutral_query': {'type': 'string'},
        'support_query': {'type': 'string'},
        'refute_query': {'type': 'string'},
        'academic_query': {'type': 'string'},
        'wiki_titles': {'type': 'array', 'items': {'type': 'string'}},
        'time_sensitive': {'type': 'boolean'},
        'citation': {
            'type': 'object',
            'properties': {
                'authors': {'type': 'array', 'items': {'type': 'string'}},
                'year': {'type': ['integer', 'null']},
                'venue': {'type': ['string', 'null']},
                'title_hint': {'type': ['string', 'null']},
            },
        },
    },
    'required': ['claim_type', 'neutral_query', 'support_query', 'refute_query', 'academic_query', 'wiki_titles', 'time_sensitive'],
}


def claim_context(claim: Claim) -> str:
    lines = [f'Claim: {claim.text}']
    if claim.type:
        lines.append(f'Type hint: {claim.type}')
    if claim.qualifiers:
        lines.append(f'Qualifiers: {", ".join(claim.qualifiers)}')
    if claim.scope:
        lines.append(f'Scope: {claim.scope}')
    if claim.attributed_to:
        lines.append(f'Attributed to: {claim.attributed_to}')
    return '\n'.join(lines)


class Planner:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def plan(self, claim: Claim) -> QueryPlan:
        data = await self.llm.structured(PLANNER_SYSTEM, claim_context(claim), PLAN_SCHEMA, name='query_plan', max_tokens=800)
        plan = QueryPlan.model_validate(data)
        if plan.citation and not (plan.citation.authors or plan.citation.year):
            plan.citation = None
        if not plan.neutral_query:
            plan.neutral_query = claim.text
        return plan
