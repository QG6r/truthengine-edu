"""Tests for eval/run_eval.py metric logic (fake services, no PDF needed)."""

from __future__ import annotations

import pytest

from eval import run_eval


def test_overlap_and_matching() -> None:
    assert run_eval.overlap('Water boils at 100 degrees.', 'water boils at 100 degrees') == 1.0
    m = run_eval.match_claims(
        ['Marie Curie was born in Warsaw.', 'Marie Curie won two Nobel Prizes.'],
        ['Curie won two Nobel Prizes.', 'Something unrelated.'],
        0.6,
    )
    assert [(i, j) for i, j, _ in m.matched] == [(1, 0)]
    assert m.unmatched_expected == [0]
    assert m.unmatched_predicted == [1]


@pytest.mark.parametrize(('score', 'label'), [(1.0, 'supported'), (0.0, 'refuted'), (0.5, 'not_enough_evidence'), (0.8, 'supported')])
def test_score_to_label(score: float, label: str) -> None:
    assert run_eval.score_to_label(score) == label


def test_parse_check_result_shapes() -> None:
    assert run_eval.parse_check_result(1.0) == (1.0, None)
    assert run_eval.parse_check_result((0.0, 'q')) == (0.0, 'q')
    assert run_eval.parse_check_result({'score': 0.5, 'quote': 'q'}) == (0.5, 'q')


def test_quote_in_pdf_normalises_and_rejects_empty() -> None:
    pages = [run_eval.normalize('The “mitochondria” is the power-\nhouse of the cell.')]
    assert run_eval.quote_in_pdf('the "mitochondria" is the powerhouse', pages)
    assert not run_eval.quote_in_pdf('', pages)
    assert not run_eval.quote_in_pdf('invented quote', pages)


def test_eval_separator_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSeparator:
        def separate(self, text: str) -> list[str]:
            return [s.strip() + '.' for s in text.split('.') if s.strip()]

    monkeypatch.setattr(run_eval, 'Separator', FakeSeparator)
    rows = [
        {'id': 'a', 'text': 'A is B. C is D.', 'expected_claims': ['A is B.', 'C is D.']},
        {'id': 'b', 'text': 'Hello there.', 'expected_claims': []},
    ]
    result = run_eval.eval_separator(rows, 0.6)
    assert result['metrics']['recall'] == {'value': 1.0, 'n': 2}
    assert result['metrics']['claim_count_mae'] == {'value': 0.5, 'n': 2}
    assert result['manual_review'] == [{'id': 'b', 'extra_output': 'Hello there.'}]


def test_eval_checker_refuted_precision_and_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {'c1': (0.0, 'real quote'), 'c2': (0.0, 'made up quote'), 'c3': (1.0, None)}

    class FakeChecker:
        def check(self, claim: str) -> tuple[float, str | None]:
            return answers[claim]

    monkeypatch.setattr(run_eval, 'CheckTrustedSources', FakeChecker)
    rows = [
        {'id': '1', 'claim': 'c1', 'expected_label': 'refuted', 'gold_quote': 'real quote'},
        {'id': '2', 'claim': 'c2', 'expected_label': 'supported', 'gold_quote': 'real quote'},
        {'id': '3', 'claim': 'c3', 'expected_label': 'supported', 'gold_quote': ''},
    ]
    result = run_eval.eval_checker(rows, [run_eval.normalize('This is a real quote from the lecture.')])
    m = result['metrics']
    assert m['refuted_precision'] == {'value': 0.5, 'n': 2}
    assert m['accuracy_3class'] == {'value': 0.6667, 'n': 3}
    assert m['quotes_not_in_pdf'] == {'value': 2 - 1, 'n': 2}
    assert result['gold_quotes_not_in_pdf'] == []
