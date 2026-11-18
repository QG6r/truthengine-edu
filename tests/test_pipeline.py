"""pipeline.build_report maps Separator + TruthEngine output to the API Report."""

from __future__ import annotations

from app.services.separator import Citation, NonClaim, SeparationReport
from app.services.separator import Claim as SepClaim
from app.services.truthengine import Claim, ClaimResult, Evidence, GateResult
from pipeline import build_report, select_claims


def _sep_claim(cid: str, text: str, citation: Citation | None = None) -> SepClaim:
    return SepClaim(
        id=cid,
        text=text,
        spans=[text],
        type='other',
        stance_of_text='asserted',
        attributed_to=None,
        qualifiers=[],
        scope=None,
        resolved={},
        citation=citation,
        depends_on=[],
        parent=None,
        time_sensitive=False,
        checkworthiness=1.0,
        extraction_confidence=1.0,
        needs_context=False,
        notes='',
    )


def test_build_report() -> None:
    cit = Citation(authors=['Smith', 'Johnson'], year=2019, venue='Nature', title_hint=None)
    separation = SeparationReport(
        claims=[_sep_claim('C1', 'Water boils at 100 C.'), _sep_claim('C2', 'A study found X.', cit), _sep_claim('C3', 'Y.')],
        non_claims=[
            NonClaim(
                id='O1',
                kind='rhetoric',
                text='Think about it!',
                spans=['Think about it!'],
                checkable_core=None,
                depends_on=[],
                notes='Pressure phrase.',
            )
        ],
        skipped=[],
        coverage_ok=True,
    )
    good = Evidence(
        source='wikipedia',
        url='https://en.wikipedia.org/wiki/Water',
        domain='en.wikipedia.org',
        tier='3D',
        stance='supports',
        quote='boils at 100',
        weight=0.6,
        year=2024,
        gates=[GateResult(name='U2_quote', status='pass')],
    )
    ignored = Evidence(source='websearch', url='https://x.org', domain='x.org', weight=0.0)
    results = [
        ClaimResult(
            claim=Claim(id='C1', text='Water boils at 100 C.'),
            verdict='supported',
            score=0.9,
            confidence=0.8,
            explanation='ok',
            evidence=[good, ignored],
        ),
        ClaimResult(
            claim=Claim(id='C2', text='A study found X.'), verdict='contradicted', score=0.15, confidence=0.7, flags=['invented_source']
        ),
        ClaimResult(claim=Claim(id='C3', text='Y.'), verdict='unverifiable', score=0.5, confidence=0.0),
    ]
    report = build_report(separation, results)
    assert report.trust_score == (0.9 + 0.15) / 2
    assert report.coverage == '2 of 3 claims checked'
    assert [v.verdict for v in report.verdicts] == ['supported', 'contradicted', 'unverifiable']
    ev = report.verdicts[0].evidence
    assert len(ev) == 1
    assert (ev[0].source_type, ev[0].quote_verified, ev[0].published_date) == ('encyclopedia', True, '2024')
    assert len(report.citations) == 1
    assert report.citations[0].status == 'not_found'
    assert report.citations[0].raw == 'Smith & Johnson, 2019, Nature'
    assert report.tricks[0].phrase == 'Think about it!'


def test_span_correction_question_and_retracted_source() -> None:
    separation = SeparationReport(
        claims=[_sep_claim('C1', 'Albert Einstein was born in Ulm.')], non_claims=[], skipped=[], coverage_ok=True
    )
    retracted = Evidence(
        source='openalex',
        url='https://doi.org/10.1/x',
        domain='Some Journal',
        tier='2C',
        stance='supports',
        quote='q',
        weight=0.0,
        gates=[GateResult(name='2C_not_retracted', status='fail', reason='retracted')],
    )
    result = ClaimResult(
        claim=Claim(id='C1', text='Albert Einstein was born in Ulm.', span='He was born in Ulm'),
        verdict='supported',
        score=0.9,
        confidence=0.8,
        explanation='Sources agree.',
        correction=None,
        critical_question='Where would you check a birthplace?',
        evidence=[retracted],
    )
    verdict = build_report(separation, [result]).verdicts[0]
    assert verdict.claim_text == 'He was born in Ulm'  # exact phrase from the text, so the UI can highlight it
    assert verdict.explanation == 'Checked as: "Albert Einstein was born in Ulm." Sources agree.'
    assert verdict.critical_question == 'Where would you check a birthplace?'
    assert [e.retracted for e in verdict.evidence] == [True]


def test_select_claims_keeps_most_checkworthy_in_text_order() -> None:
    claims = [_sep_claim(f'C{i}', f'Claim {i}.') for i in range(5)]
    for claim, worth in zip(claims, [0.1, 0.9, 0.5, 0.8, 0.2], strict=True):
        claim.checkworthiness = worth
    assert [c.id for c in select_claims(claims, limit=3)] == ['C1', 'C2', 'C3']
    assert select_claims(claims, limit=10) == claims
