"""Pydantic models shared by the whole engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


_SECOND_LEVEL = {'co', 'ac', 'gov', 'org', 'com', 'edu', 'net'}


def base_domain(host: str) -> str:
    """Registrable domain, so en.wikipedia.org and wikipedia.org (or x.nasa.gov and nasa.gov) count as one origin."""
    parts = host.lower().removeprefix('www.').split('.')
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])


Stance = Literal['supports', 'contradicts', 'neutral', 'irrelevant']
Verdict = Literal['supported', 'contradicted', 'disputed', 'unverifiable']


class Claim(BaseModel):
    """An already-extracted, self-contained claim (context kept from extraction)."""

    id: str = ''
    text: str
    type: str | None = None
    qualifiers: list[str] = Field(default_factory=list)
    scope: str | None = None
    attributed_to: str | None = None
    span: str | None = None  # exact substring of the input text the claim comes from (for highlighting)


class CitationRef(BaseModel):
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    title_hint: str | None = None


class QueryPlan(BaseModel):
    claim_type: str = 'other'
    neutral_query: str = ''
    support_query: str = ''
    refute_query: str = ''
    academic_query: str = ''
    wiki_titles: list[str] = Field(default_factory=list)
    time_sensitive: bool = False
    citation: CitationRef | None = None


class GateResult(BaseModel):
    name: str
    status: Literal['pass', 'fail', 'unknown']
    reason: str = ''


class Evidence(BaseModel):
    source: str
    url: str
    domain: str
    title: str = ''
    text: str = ''  # fetched text; quotes are verified against it
    tier: str = '5'
    base_weight: float = 0.25
    year: int | None = None
    doi: str | None = None
    stance: Stance = 'neutral'
    quote: str = ''
    directness: float = 0.0
    scope_match: float = 0.0
    note: str = ''
    gates: list[GateResult] = Field(default_factory=list)
    weight: float = 0.0
    meta: dict = Field(default_factory=dict)

    @property
    def group_key(self) -> str:
        """Independence key: copies from the same origin count once."""
        if self.doi:
            return f'doi:{self.doi.lower()}'
        if self.source == 'knowledge':  # pages of one uploaded document are one origin
            return f'knowledge:{self.meta.get("source_id", self.url)}'
        return self.url if self.source == 'openalex' else base_domain(self.domain)


class ClaimResult(BaseModel):
    claim: Claim
    verdict: Verdict
    score: float  # 0..1, 0.5 = no evidence either way
    confidence: float
    support_mass: float = 0.0
    contra_mass: float = 0.0
    best_tier: str | None = None
    explanation: str = ''
    correction: str | None = None
    critical_question: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)  # claim-level gates
    flags: list[str] = Field(default_factory=list)  # e.g. invented_source
    plan: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @property
    def best_quote(self) -> str | None:
        """Quote of the strongest counted piece of evidence, if any."""
        counted = [ev for ev in self.evidence if ev.weight > 0 and ev.quote]
        return max(counted, key=lambda ev: ev.weight).quote if counted else None
