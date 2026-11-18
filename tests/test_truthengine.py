"""Unit tests for gates and scoring (pure logic, no network)."""

from __future__ import annotations

from typing import Any

from app.services.truthengine.gates import quote_gate
from app.services.truthengine.llm import blocked_domains_from_error
from app.services.truthengine.models import Evidence, QueryPlan
from app.services.truthengine.scoring import aggregate, classify_domain, score_evidence
from app.services.truthengine.text_utils import best_passages, keywords


def make_ev(**kw: Any) -> Evidence:
    base: dict[str, Any] = {
        'source': 'websearch',
        'url': 'https://www.britannica.com/x',
        'domain': 'britannica.com',
        'text': 'Albert Einstein was born in Ulm, in the Kingdom of Württemberg, in 1879.',
        'tier': '3B',
        'base_weight': 0.75,
        'stance': 'supports',
        'quote': 'Albert Einstein was born in Ulm',
        'directness': 1.0,
        'scope_match': 1.0,
    }
    base.update(kw)
    return Evidence(**base)


def test_classify_domain() -> None:
    assert classify_domain('https://en.wikipedia.org/wiki/X') == ('3D', 'en.wikipedia.org')
    assert classify_domain('https://pubmed.ncbi.nlm.nih.gov/123') == ('2C', 'pubmed.ncbi.nlm.nih.gov')
    assert classify_domain('https://www.cdc.gov/flu') == ('2B', 'cdc.gov')
    assert classify_domain('https://data.worldbank.org/x') == ('1B', 'data.worldbank.org')
    assert classify_domain('https://random-blog.net/post') == ('5', 'random-blog.net')


def test_quote_gate_requires_verbatim_quote() -> None:
    assert quote_gate(make_ev()).status == 'pass'
    assert quote_gate(make_ev(quote='Einstein was born in Munich')).status == 'fail'
    assert quote_gate(make_ev(quote='')).status == 'fail'
    assert quote_gate(make_ev(stance='neutral', quote='')).status == 'pass'


def test_invented_quote_gets_zero_weight() -> None:
    ev = score_evidence(make_ev(quote='Einstein was born in Munich in 1879'), QueryPlan(), 2026)
    assert ev.weight == 0.0


def test_supported_verdict_from_two_independent_sources() -> None:
    plan = QueryPlan()
    evs = [
        score_evidence(make_ev(), plan, 2026),
        score_evidence(make_ev(url='https://plato.stanford.edu/e', domain='plato.stanford.edu'), plan, 2026),
    ]
    result = aggregate(evs)
    assert result['verdict'] == 'supported'
    assert result['score'] >= 0.75


def test_copies_from_one_domain_count_once() -> None:
    plan = QueryPlan()
    evs = [score_evidence(make_ev(url=f'https://www.britannica.com/{i}', base_weight=0.5), plan, 2026) for i in range(5)]
    assert aggregate(evs)['verdict'] == 'unverifiable'  # one origin, mass 0.5 < MIN_EVIDENCE_MASS


def test_single_strong_factcheck_is_enough_to_refute() -> None:
    ev = score_evidence(
        make_ev(stance='contradicts', tier='3A', base_weight=0.8, url='https://www.snopes.com/x', domain='snopes.com'), QueryPlan(), 2026
    )
    result = aggregate([ev])
    assert result['verdict'] == 'contradicted'
    assert result['score'] <= 0.25


def test_best_passages_picks_relevant_sentences() -> None:
    text = ('Filler sentence about something else. ' * 80) + 'Albert Einstein was born in Ulm in 1879. ' + ('More filler here. ' * 80)
    excerpt = best_passages(text, keywords('Albert Einstein born Ulm'), 300)
    assert 'born in Ulm' in excerpt
    assert len(excerpt) <= 300


def test_blocked_domains_parsed_from_api_error() -> None:
    msg = "Claude API 400: The following domains are not accessible to our user agent: ['apnews.com', 'bbc.com']. Read more: x"
    assert blocked_domains_from_error(msg) == {'apnews.com', 'bbc.com'}
    assert blocked_domains_from_error('Claude API 400: other') == set()


def test_low_tier_alone_is_unverifiable() -> None:
    plan = QueryPlan()
    evs = [
        score_evidence(make_ev(url=f'https://blog{i}.net/p', domain=f'blog{i}.net', tier='5', base_weight=0.25), plan, 2026)
        for i in range(6)
    ]
    assert aggregate(evs)['verdict'] == 'unverifiable'


def test_contradiction_from_strong_source_wins() -> None:
    plan = QueryPlan()
    evs = [
        score_evidence(
            make_ev(stance='contradicts', tier='2A', base_weight=0.95, url='https://cochranelibrary.com/a', domain='cochranelibrary.com'),
            plan,
            2026,
        ),
        score_evidence(
            make_ev(stance='contradicts', tier='2B', base_weight=0.9, url='https://www.cdc.gov/a', domain='cdc.gov'), plan, 2026
        ),
        score_evidence(make_ev(tier='5', base_weight=0.25, url='https://blog.net/p', domain='blog.net'), plan, 2026),
    ]
    result = aggregate(evs)
    assert result['verdict'] == 'contradicted'
    assert result['score'] <= 0.25
