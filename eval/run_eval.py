"""Evaluate Separator and CheckTrustedSources against hand-written gold data.

Usage:
    uv run python eval/run_eval.py [--pdf eval/lecture.pdf] [--overlap 0.6]

Datasets (all optional; a missing file skips that section):
    eval/separator.jsonl   {"id", "text", "expected_claims", "lang", "note"}
    eval/checker.jsonl     {"id", "claim", "expected_score", "expected_label", "gold_quote", "page", "note"}
    eval/end_to_end.jsonl  {"id", "text", "expected_claims", "expected_scores", "note"}

Claim matching uses token overlap (Dice coefficient on lowercase word tokens). Unmatched
pairs are printed for manual review. No LLM judge, so gold data is never scored by the
same model the system uses.

CheckTrustedSources.check may return a float, a (score, quote) tuple, a dict with
"score"/"quote", or an object with .score/.quote. Any returned quote is checked
verbatim (whitespace/quote normalised) against the PDF text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.separator import Separator  # noqa: E402
from app.services.trusted_sources import CheckTrustedSources  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
LABELS = ('supported', 'refuted', 'not_enough_evidence')
_TOKEN = re.compile(r'\w+', re.UNICODE)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def overlap(a: str, b: str) -> float:
    """Dice coefficient over word tokens: 2|A∩B| / (|A|+|B|)."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


@dataclass
class ClaimMatch:
    matched: list[tuple[int, int, float]]  # (expected_idx, predicted_idx, score)
    unmatched_expected: list[int]
    unmatched_predicted: list[int]


def match_claims(expected: list[str], predicted: list[str], threshold: float) -> ClaimMatch:
    """Greedy one-to-one matching, best-overlap pairs first."""
    pairs = sorted(((overlap(e, p), i, j) for i, e in enumerate(expected) for j, p in enumerate(predicted)), reverse=True)
    used_e: set[int] = set()
    used_p: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for score, i, j in pairs:
        if score < threshold:
            break
        if i in used_e or j in used_p:
            continue
        used_e.add(i)
        used_p.add(j)
        matched.append((i, j, score))
    return ClaimMatch(
        matched=matched,
        unmatched_expected=[i for i in range(len(expected)) if i not in used_e],
        unmatched_predicted=[j for j in range(len(predicted)) if j not in used_p],
    )


def score_to_label(score: float) -> str:
    if score >= 0.75:
        return 'supported'
    if score <= 0.25:
        return 'refuted'
    return 'not_enough_evidence'


def parse_check_result(result: Any) -> tuple[float, str | None]:
    if isinstance(result, int | float):
        return float(result), None
    if isinstance(result, tuple | list):
        return float(result[0]), (result[1] if len(result) > 1 else None)
    if isinstance(result, dict):
        return float(result['score']), result.get('quote')
    return float(result.score), getattr(result, 'quote', None)


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('­', '')  # soft hyphen
    text = re.sub(r'-\s*\n\s*', '', text)  # hyphenated line breaks
    for src, dst in (('“', '"'), ('”', '"'), ('„', '"'), ('«', '"'), ('»', '"'), ('‘', "'"), ('’', "'"), ('–', '-'), ('—', '-')):
        text = text.replace(src, dst)
    return re.sub(r'\s+', ' ', text).strip().lower()


def load_pdf_pages(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    from pypdf import PdfReader  # noqa: PLC0415

    return [normalize(page.extract_text() or '') for page in PdfReader(path).pages]


def quote_in_pdf(quote: str, pages: list[str]) -> bool:
    q = normalize(quote)
    if not q:
        return False
    # also search the joined text so quotes spanning a page break are found
    return any(q in page for page in pages) or q in ' '.join(pages)


def metric(value: float | None, n: int) -> dict[str, Any]:
    return {'value': None if value is None else round(value, 4), 'n': n}


# ---------------------------------------------------------------------------
# evaluations
# ---------------------------------------------------------------------------


def eval_separator(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    separator = Separator()
    count_errors: list[int] = []
    expected_total = matched_total = 0
    review: list[dict[str, Any]] = []
    for row in rows:
        try:
            predicted = separator.separate(row['text'])
        except NotImplementedError:
            return {'status': 'not_implemented'}
        expected = row['expected_claims']
        count_errors.append(abs(len(predicted) - len(expected)))
        m = match_claims(expected, predicted, threshold)
        expected_total += len(expected)
        matched_total += len(m.matched)
        for i in m.unmatched_expected:
            best = max(((overlap(expected[i], p), p) for p in predicted), default=(0.0, None))
            review.append({'id': row['id'], 'expected': expected[i], 'closest_output': best[1], 'overlap': round(best[0], 2)})
        review.extend({'id': row['id'], 'extra_output': predicted[j]} for j in m.unmatched_predicted)
    n = len(count_errors)
    return {
        'status': 'ok',
        'metrics': {
            'claim_count_mae': metric(sum(count_errors) / n if n else None, n),
            'claim_count_exact': metric(sum(e == 0 for e in count_errors) / n if n else None, n),
            'recall': metric(matched_total / expected_total if expected_total else None, expected_total),
        },
        'manual_review': review,
    }


def eval_checker(rows: list[dict[str, Any]], pages: list[str] | None) -> dict[str, Any]:
    checker = CheckTrustedSources()
    confusion = {g: dict.fromkeys(LABELS, 0) for g in LABELS}
    errors: list[dict[str, Any]] = []
    quotes_returned = 0
    bad_quotes: list[dict[str, Any]] = []
    gold_quotes_missing = (
        [r['id'] for r in rows if r.get('gold_quote') and not quote_in_pdf(r['gold_quote'], pages)] if pages is not None else None
    )
    for row in rows:
        try:
            score, quote = parse_check_result(checker.check(row['claim']))
        except NotImplementedError:
            return {'status': 'not_implemented', 'gold_quotes_not_in_pdf': gold_quotes_missing}
        gold, pred = row['expected_label'], score_to_label(score)
        confusion[gold][pred] += 1
        if gold != pred:
            errors.append({'id': row['id'], 'claim': row['claim'], 'expected': gold, 'predicted': pred, 'score': score})
        if quote:
            quotes_returned += 1
            if pages is not None and not quote_in_pdf(quote, pages):
                bad_quotes.append({'id': row['id'], 'quote': quote})
    n = len(rows)
    correct = sum(confusion[label][label] for label in LABELS)
    predicted_refuted = sum(confusion[g]['refuted'] for g in LABELS)
    return {
        'status': 'ok',
        'metrics': {
            'accuracy_3class': metric(correct / n if n else None, n),
            'refuted_precision': metric(
                confusion['refuted']['refuted'] / predicted_refuted if predicted_refuted else None, predicted_refuted
            ),
            'quotes_not_in_pdf': metric(len(bad_quotes) if pages is not None else None, quotes_returned),
        },
        'confusion_matrix': confusion,
        'errors': errors,
        'bad_quotes': bad_quotes,
        'gold_quotes_not_in_pdf': gold_quotes_missing,
    }


def eval_end_to_end(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    separator, checker = Separator(), CheckTrustedSources()
    expected_total = matched_total = agree = 0
    scenarios_ok = 0
    details: list[dict[str, Any]] = []
    for row in rows:
        try:
            predicted = separator.separate(row['text'])
            m = match_claims(row['expected_claims'], predicted, threshold)
            scored = [(i, j, parse_check_result(checker.check(predicted[j]))[0]) for i, j, _ in m.matched]
        except NotImplementedError:
            return {'status': 'not_implemented'}
        hits = [score_to_label(s) == score_to_label(row['expected_scores'][i]) for i, _, s in scored]
        expected_total += len(row['expected_claims'])
        matched_total += len(m.matched)
        agree += sum(hits)
        ok = not m.unmatched_expected and not m.unmatched_predicted and all(hits)
        scenarios_ok += ok
        details.append(
            {
                'id': row['id'],
                'ok': ok,
                'predicted_claims': predicted,
                'scores': {predicted[j]: s for _, j, s in scored},
                'missed_claims': [row['expected_claims'][i] for i in m.unmatched_expected],
                'extra_claims': [predicted[j] for j in m.unmatched_predicted],
            }
        )
    return {
        'status': 'ok',
        'metrics': {
            'claim_recall': metric(matched_total / expected_total if expected_total else None, expected_total),
            'label_agreement_on_matched': metric(agree / matched_total if matched_total else None, matched_total),
            'scenarios_fully_correct': metric(scenarios_ok / len(rows) if rows else None, len(rows)),
        },
        'details': details,
    }


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def print_table(report: dict[str, Any]) -> None:
    print(f'\n{"section":<12} {"metric":<28} {"value":>8} {"n":>5}')
    print('-' * 56)
    for section, result in report['sections'].items():
        if result['status'] != 'ok':
            print(f'{section:<12} {result["status"]:<28}')
            continue
        for name, m in result['metrics'].items():
            value = '-' if m['value'] is None else f'{m["value"]:.3f}'
            print(f'{section:<12} {name:<28} {value:>8} {m["n"]:>5}')
    checker = report['sections'].get('checker', {})
    if checker.get('status') == 'ok':
        print('\nconfusion (rows = gold, cols = predicted):')
        print(f'{"":<22}' + ''.join(f'{label[:12]:>14}' for label in LABELS))
        for gold in LABELS:
            print(f'{gold:<22}' + ''.join(f'{checker["confusion_matrix"][gold][p]:>14}' for p in LABELS))
    if checker.get('gold_quotes_not_in_pdf'):
        print(f'\n!! dataset error: gold_quote not found in PDF for {checker["gold_quotes_not_in_pdf"]}')
    review = report['sections'].get('separator', {}).get('manual_review', [])
    if review:
        print(f'\nseparator unmatched pairs for manual review ({len(review)}):')
        for item in review:
            print(f'  {json.dumps(item, ensure_ascii=False)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdf', type=Path, default=EVAL_DIR / 'lecture.pdf')
    parser.add_argument('--overlap', type=float, default=0.6, help='token-overlap threshold for claim matching')
    parser.add_argument('--out', type=Path, default=EVAL_DIR / 'report.json')
    args = parser.parse_args()

    pages = load_pdf_pages(args.pdf)
    if pages is None:
        print(f'warning: {args.pdf} not found; quote checks skipped')

    report: dict[str, Any] = {'config': {'overlap_threshold': args.overlap, 'pdf': str(args.pdf)}, 'sections': {}}
    datasets = {
        'separator': lambda rows: eval_separator(rows, args.overlap),
        'checker': lambda rows: eval_checker(rows, pages),
        'end_to_end': lambda rows: eval_end_to_end(rows, args.overlap),
    }
    for name, run in datasets.items():
        rows = load_jsonl(EVAL_DIR / f'{name}.jsonl')
        report['sections'][name] = run(rows) if rows is not None else {'status': 'missing_dataset'}

    print_table(report)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nsaved {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}')


if __name__ == '__main__':
    main()
