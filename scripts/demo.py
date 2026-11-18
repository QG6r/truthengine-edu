"""Smoke test against the live APIs. Needs ANTHROPIC_API_KEY in .env.

Usage:
    uv run python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.trusted_sources import CheckTrustedSources  # noqa: E402
from app.services.truthengine import Claim  # noqa: E402


CLAIMS = [
    Claim(text='Humans use only 10% of their brains.'),
    Claim(text='Albert Einstein was born in Ulm.'),
    Claim(text='Albert Einstein was born in 1879.'),
    Claim(
        text='Albert Einstein said that insanity is doing the same thing over and over and expecting different results.',
        type='quote_attribution',
    ),
    Claim(
        text=(
            'A 2019 study by Smith and Johnson published in Nature found that students '
            'who sleep less than six hours score 40% lower on exams.'
        ),
        qualifiers=['proved'],
    ),
    Claim(
        text='Moderate coffee consumption may reduce the risk of type 2 diabetes in some adults.',
        qualifiers=['may', 'moderate', 'some adults'],
        attributed_to='some researchers',
    ),
    Claim(text='Vitamin C cures the common cold.', attributed_to='a popular wellness blog'),
    Claim(text='Water boils at 100 degrees Celsius at sea level.'),
    Claim(text='Ice melts at 10 degrees Celsius.'),
    Claim(text='The Eiffel Tower is 450 meters tall.'),
]


def main() -> None:
    results = CheckTrustedSources().run_detailed(CLAIMS)
    for r in results:
        print(f'\n[{r.claim.id}] {r.claim.text}')
        print(
            f'  {r.verdict.upper()}  score={r.score:.2f}  conf={r.confidence:.2f}  '
            f'S={r.support_mass:.2f}  C={r.contra_mass:.2f}  best_tier={r.best_tier}'
        )
        print(f'  {r.explanation}')
        for g in r.gates:
            print(f'  gate {g.name}: {g.status} ({g.reason})')
        for ev in r.evidence:
            if ev.weight > 0:
                print(f'   + {ev.stance:<11} w={ev.weight:.2f} tier={ev.tier:<3} {ev.domain} :: "{ev.quote[:100]}"')
        for flag in r.flags:
            print(f'   ! {flag}')
        for err in r.errors:
            print(f'   x {err}')


if __name__ == '__main__':
    main()
