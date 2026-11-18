"""Glue between the team's components and the API: text -> Separator -> TruthEngine -> Report.

api.py (via analyzer.py) calls ``run_pipeline(text)``. Any exception here makes the API fall back
to mock data with ``X-Demo-Data: true`` and logs the reason.
"""

from __future__ import annotations

import os

from app.services.separator import Claim as SeparatedClaim
from app.services.separator import SeparationReport, Separator
from app.services.trusted_sources import CheckTrustedSources
from app.services.truthengine import Claim as EngineClaim
from app.services.truthengine import ClaimResult
from app.services.truthengine import Evidence as EngineEvidence
from app.services.truthengine.config import CLAUDE_MODEL, MAX_CLAIMS_PER_TEXT
from schemas import CitationCheck, Evidence, Report, Trick, Verdict


# TruthEngine source tier -> schemas.Evidence.source_type
TIER_SOURCE_TYPE = {
    '1A': 'official',
    '1B': 'official',
    '1C': 'peer_reviewed',
    '1D': 'teacher_source',  # uploaded by a teacher (knowledge base)
    '2A': 'review',
    '2B': 'official',
    '2C': 'peer_reviewed',
    '3A': 'news',  # IFCN fact-checkers
    '3B': 'encyclopedia',
    '3C': 'encyclopedia',
    '3D': 'encyclopedia',
    '4A': 'news',
    '4B': 'preprint',
    '4C': 'other',
    '5': 'other',
}
TRICK_KINDS = {'rhetoric': 'Rhetoric'}
MAX_EVIDENCE_SHOWN = 6


async def run_pipeline(text: str) -> Report:
    # One model setting for both components: ANTHROPIC_MODEL if set, else CLAUDE_MODEL (engine default).
    separation = await Separator(model=os.getenv('ANTHROPIC_MODEL') or CLAUDE_MODEL).separate_detailed_async(text)
    claims = [_to_engine_claim(c) for c in select_claims(separation.claims)]
    results = await CheckTrustedSources().run_detailed_async(claims)
    return build_report(separation, results)


def select_claims(claims: list[SeparatedClaim], limit: int = MAX_CLAIMS_PER_TEXT) -> list[SeparatedClaim]:
    """The `limit` most check-worthy claims, kept in text order, so a long text still fits the API timeout."""
    if len(claims) <= limit:
        return claims
    keep = {id(c) for c in sorted(claims, key=lambda c: -c.checkworthiness)[:limit]}
    return [c for c in claims if id(c) in keep]


def build_report(separation: SeparationReport, results: list[ClaimResult]) -> Report:
    by_id = {c.id: c for c in separation.claims}
    checked = [r for r in results if r.verdict != 'unverifiable']
    return Report(
        trust_score=sum(r.score for r in checked) / len(checked) if checked else None,
        coverage=f'{len(checked)} of {len(results)} claims checked',
        verdicts=[_to_verdict(r) for r in results],
        citations=_unique_citations(results, by_id),
        tricks=[
            Trick(name=TRICK_KINDS[n.kind], phrase=n.spans[0] if n.spans else n.text, explanation=n.notes or n.text)
            for n in separation.non_claims
            if n.kind in TRICK_KINDS
        ],
    )


def _unique_citations(results: list[ClaimResult], by_id: dict[str, SeparatedClaim]) -> list[CitationCheck]:
    """One entry per cited work: several claims often cite the same study."""
    seen: dict[str, CitationCheck] = {}
    for result in results:
        citation = _to_citation(result, by_id.get(result.claim.id))
        if citation is not None and citation.raw not in seen:
            seen[citation.raw] = citation
    return list(seen.values())


def _to_engine_claim(claim: SeparatedClaim) -> EngineClaim:
    return EngineClaim(
        id=claim.id,
        text=claim.text,
        type=claim.type,
        qualifiers=claim.qualifiers,
        scope=claim.scope,
        attributed_to=claim.attributed_to,
        span=claim.spans[0] if claim.spans else None,
    )


def _to_verdict(result: ClaimResult) -> Verdict:
    claim = result.claim
    explanation = result.explanation or 'No explanation returned.'
    # The UI highlights claim_text inside the input, so show the exact source phrase when we have it.
    shown = claim.span or claim.text
    if claim.span and claim.span.strip().rstrip('.!?').lower() != claim.text.strip().rstrip('.!?').lower():
        explanation = f'Checked as: "{claim.text}" {explanation}'
    shown_evidence = [ev for ev in result.evidence if _is_retracted(ev) or (ev.weight > 0 and ev.stance != 'irrelevant')]
    return Verdict(
        claim_id=claim.id,
        claim_text=shown,
        verdict=result.verdict,
        confidence=max(0.0, min(1.0, result.confidence)),
        explanation=explanation,
        correction=result.correction,
        evidence=[_to_evidence(ev) for ev in sorted(shown_evidence, key=lambda ev: -ev.weight)[:MAX_EVIDENCE_SHOWN]],
        critical_question=result.critical_question,
    )


def _is_retracted(ev: EngineEvidence) -> bool:
    """The engine records retraction as a failed gate (weight 0); show such sources, clearly marked."""
    return any(g.name == '2C_not_retracted' and g.status == 'fail' for g in ev.gates)


def _to_evidence(ev: EngineEvidence) -> Evidence:
    quote_gate = next((g for g in ev.gates if g.name == 'U2_quote'), None)
    return Evidence(
        url=ev.url,
        domain=ev.domain,
        source_type=TIER_SOURCE_TYPE.get(ev.tier, 'other'),
        title=ev.title or ev.domain,
        quote=ev.quote,
        quote_verified=quote_gate is not None and quote_gate.status == 'pass' and bool(ev.quote),
        published_date=str(ev.year) if ev.year else None,
        retracted=_is_retracted(ev),
        stance=ev.stance if ev.stance in ('supports', 'contradicts') else 'neutral',
    )


def _to_citation(result: ClaimResult, claim: SeparatedClaim | None) -> CitationCheck | None:
    if claim is None or claim.citation is None:
        return None
    c = claim.citation
    raw = ', '.join(p for p in (' & '.join(c.authors), str(c.year) if c.year else '', c.venue or '', c.title_hint or '') if p)
    if 'invented_source' in result.flags:
        status = 'not_found'
    elif 'retracted_source' in result.flags:
        status = 'retracted'
    else:
        status = 'found'
    primary = next((ev for ev in result.evidence if ev.tier == '1C'), None)
    return CitationCheck(
        raw=raw or claim.text,
        status=status,
        closest_match=primary.title if primary else None,
        url=primary.url if primary else None,
    )
