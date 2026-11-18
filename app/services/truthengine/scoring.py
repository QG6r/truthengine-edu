"""Evidence weighting and aggregation of evidence into a verdict, score and confidence."""

from __future__ import annotations

from urllib.parse import urlparse

from .config import CONFIDENCE_MASS_SCALE, CONTRA_RATIO, DOMAIN_TIERS, MIN_EVIDENCE_MASS, SUPPORT_RATIO, TIER_CAP, UNKNOWN_GATE_FACTOR
from .gates import date_gate, quote_gate, relevance_gate, retrieval_gate
from .models import Evidence, QueryPlan


def classify_domain(url: str) -> tuple[str, str]:
    host = urlparse(url).netloc.lower().removeprefix('www.')
    for tier, domains in DOMAIN_TIERS:
        for d in domains:
            if d.startswith('.'):
                if host.endswith(d):
                    return tier, host
            elif host == d or host.endswith('.' + d):
                return tier, host
    return '5', host


def score_evidence(ev: Evidence, plan: QueryPlan, now_year: int) -> Evidence:
    """weight = base(tier) x directness x scope x freshness x gate factors."""
    g_date, freshness = date_gate(ev, plan.time_sensitive, now_year)
    ev.gates = [*ev.gates, retrieval_gate(ev), quote_gate(ev), relevance_gate(ev), g_date]

    if ev.stance not in ('supports', 'contradicts'):
        ev.weight = 0.0
        return ev

    factor = 1.0
    for gate in ev.gates:
        if gate.status == 'fail':
            ev.weight = 0.0
            return ev
        if gate.status == 'unknown':
            factor *= UNKNOWN_GATE_FACTOR

    directness = 0.5 + 0.5 * ev.directness
    scope = 0.3 + 0.7 * ev.scope_match
    ev.weight = round(ev.base_weight * directness * scope * freshness * factor, 3)
    return ev


def aggregate(evidence: list[Evidence]) -> dict:
    # Independence: per origin and stance keep only the strongest piece.
    strongest: dict[tuple[str, str], float] = {}
    for ev in evidence:
        if ev.weight > 0:
            key = (ev.group_key, ev.stance)
            strongest[key] = max(strongest.get(key, 0.0), ev.weight)

    support = sum(w for (_, s), w in strongest.items() if s == 'supports')
    contra = sum(w for (_, s), w in strongest.items() if s == 'contradicts')
    mass = support + contra

    counted = [ev for ev in evidence if ev.weight > 0]
    best = min(counted, key=lambda e: (int(e.tier[0]), -e.weight), default=None)
    level = int(best.tier[0]) if best else None
    base = {'support_mass': round(support, 3), 'contra_mass': round(contra, 3), 'best_tier': best.tier if best else None}

    if level is None or mass < MIN_EVIDENCE_MASS or TIER_CAP[level] == 0:
        return {**base, 'verdict': 'unverifiable', 'score': 0.5, 'confidence': 0.0}

    ratio = support / mass
    if ratio >= SUPPORT_RATIO:
        verdict = 'supported'
    elif ratio <= CONTRA_RATIO:
        verdict = 'contradicted'
    else:
        verdict = 'disputed'

    confidence = min(min(1.0, mass / CONFIDENCE_MASS_SCALE) * max(ratio, 1 - ratio), TIER_CAP[level])
    score = 0.5 + (ratio - 0.5) * confidence
    return {**base, 'verdict': verdict, 'score': round(score, 3), 'confidence': round(confidence, 3)}
