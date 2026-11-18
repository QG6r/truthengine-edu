"""Universal evidence gates U1-U4. A gate passes, fails (weight 0) or is unknown (weight reduced)."""

from __future__ import annotations

import re
import unicodedata

from .config import MIN_QUOTE_CHARS, STALE_AFTER_YEARS, STALE_FACTOR
from .models import Evidence, GateResult


_QUOTE_CHARS = '"\'“”‘’«»'


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip().strip(_QUOTE_CHARS).strip()


def retrieval_gate(ev: Evidence) -> GateResult:
    """U1: evidence came from a real fetch, never from model memory."""
    if ev.text.strip():
        return GateResult(name='U1_retrieved', status='pass', reason=f'fetched via {ev.source}')
    return GateResult(name='U1_retrieved', status='fail', reason='no fetched text')


def quote_gate(ev: Evidence) -> GateResult:
    """U2: the decisive quote must exist verbatim in the fetched text."""
    if ev.stance not in ('supports', 'contradicts'):
        return GateResult(name='U2_quote', status='pass', reason='no quote needed')
    if not ev.quote.strip():
        return GateResult(name='U2_quote', status='fail', reason='no quote returned')
    haystack = normalize(ev.text)
    parts = [p for p in (normalize(x) for x in re.split(r'\.\.\.|…', ev.quote)) if p]
    if sum(len(p) for p in parts) < MIN_QUOTE_CHARS:
        return GateResult(name='U2_quote', status='fail', reason='quote too short')
    if any(p not in haystack for p in parts):
        return GateResult(name='U2_quote', status='fail', reason='quote not found in fetched text')
    return GateResult(name='U2_quote', status='pass', reason='verbatim match')


def date_gate(ev: Evidence, time_sensitive: bool, now_year: int) -> tuple[GateResult, float]:
    """U3: freshness matters only for time-sensitive claims."""
    if not time_sensitive:
        return GateResult(name='U3_date', status='pass', reason='not time-sensitive'), 1.0
    if ev.year is None:
        return GateResult(name='U3_date', status='unknown', reason='no publication date'), 1.0
    if now_year - ev.year > STALE_AFTER_YEARS:
        return GateResult(name='U3_date', status='pass', reason=f'stale ({ev.year})'), STALE_FACTOR
    return GateResult(name='U3_date', status='pass', reason=str(ev.year)), 1.0


def relevance_gate(ev: Evidence) -> GateResult:
    """U4: the source must talk about the same thing."""
    if ev.stance == 'irrelevant':
        return GateResult(name='U4_relevance', status='fail', reason='different subject or scope')
    if ev.stance in ('supports', 'contradicts') and ev.directness < 0.2:
        return GateResult(name='U4_relevance', status='fail', reason='too indirect')
    return GateResult(name='U4_relevance', status='pass')
