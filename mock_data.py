"""Demo data: a fallback Report for when the pipeline is unavailable, plus example inputs.

The mock report matches DEMO_INPUTS[0] so claim highlighting works on that example.
Sources and quotes are illustrative demo content, not output of the real pipeline.
"""

from __future__ import annotations

from schemas import CitationCheck, Evidence, Report, Trick, Verdict


DEMO_INPUTS: list[dict[str, str]] = [
    {
        'title': 'Chatbot answer with a fake citation',
        'text': (
            'Vitamin C reliably prevents colds in everyone. '
            'According to a study by Ivanov and Smith (2021, Journal of Immunology Research), '
            'taking 1 gram of vitamin C a day reduces the risk of a cold by 80%. '
            'Linus Pauling, a two-time Nobel Prize winner, popularised high doses of vitamin C in the 1970s. '
            'High doses of vitamin C taken at the start of a cold shorten its duration.'
        ),
    },
    {
        'title': 'Lecture notes with an error',
        'text': (
            'Lecture notes: cell biology.\n'
            'Mitochondria are the organelles where photosynthesis takes place.\n'
            'DNA is made of four types of nucleotides: adenine, guanine, cytosine and thymine.\n'
            'Cell theory was formulated by Theodor Schwann and Matthias Schleiden in 1838-1839.'
        ),
    },
    {
        'title': 'Manipulative social media post',
        'text': (
            'ATTENTION!!! Doctors are hiding the truth: plain water with lemon cures high blood pressure better than pills. '
            'Scientists proved this long ago, but pharma companies stay silent because it is not profitable for them. '
            'Everyone I know has already stopped taking their medication. Share before this post gets deleted!'
        ),
    },
    {
        'title': 'Short conversation transcript',
        'text': (
            '- Today we are talking about climate. '
            'The average temperature on Earth has risen by about 1.1 degrees since the late 19th century.\n'
            '- Is that a lot?\n'
            '- Yes. And the main cause, according to most scientists, is greenhouse gas emissions.'
        ),
    },
]

_COCHRANE_URL = 'https://doi.org/10.1002/14651858.CD000980.pub4'
_COCHRANE_TITLE = 'Vitamin C for preventing and treating the common cold (Cochrane Review)'


def mock_report(text: str) -> Report:
    """Return a realistic demo Report. ``text`` is accepted for API symmetry; the content is fixed."""
    del text
    return Report(
        trust_score=0.42,
        coverage='3 of 4 claims checked',
        verdicts=[
            Verdict(
                claim_id='c1',
                claim_text='Linus Pauling, a two-time Nobel Prize winner, popularised high doses of vitamin C in the 1970s',
                verdict='supported',
                confidence=0.9,
                explanation='Pauling received the Nobel Prize in Chemistry (1954) and the Nobel Peace Prize (1962); '
                'his 1970 book on vitamin C made high doses popular.',
                evidence=[
                    Evidence(
                        url='https://www.nobelprize.org/prizes/chemistry/1954/pauling/facts/',
                        domain='nobelprize.org',
                        source_type='official',
                        title='Linus Pauling – Facts – NobelPrize.org',
                        quote='The Nobel Prize in Chemistry 1954 was awarded to Linus Carl Pauling',
                        quote_verified=True,
                        stance='supports',
                    ),
                    Evidence(
                        url='https://en.wikipedia.org/wiki/Linus_Pauling',
                        domain='en.wikipedia.org',
                        source_type='encyclopedia',
                        title='Linus Pauling – Wikipedia',
                        quote='Pauling promoted the use of high doses of vitamin C',
                        quote_verified=False,
                        stance='supports',
                    ),
                ],
                critical_question='Does a Nobel Prize in Chemistry make someone an expert in medicine?',
            ),
            Verdict(
                claim_id='c2',
                claim_text='Vitamin C reliably prevents colds in everyone',
                verdict='contradicted',
                confidence=0.85,
                explanation='In the Cochrane review, regular vitamin C did not reduce how often colds occur in the general population.',
                correction='For most people, regular vitamin C does not reduce how often they catch colds; '
                'an effect was seen mainly in people under heavy physical stress.',
                evidence=[
                    Evidence(
                        url=_COCHRANE_URL,
                        domain='doi.org',
                        source_type='review',
                        title=_COCHRANE_TITLE,
                        quote='The failure of vitamin C supplementation to reduce the incidence of colds '
                        'in the general population indicates that routine vitamin C supplementation is not justified',
                        quote_verified=True,
                        published_date='2013-01-31',
                        stance='contradicts',
                    )
                ],
                critical_question='What exactly does "everyone" mean, and was this tested across different groups?',
            ),
            Verdict(
                claim_id='c3',
                claim_text='High doses of vitamin C taken at the start of a cold shorten its duration',
                verdict='disputed',
                confidence=0.5,
                explanation='Studies disagree: in controlled trials, taking it after symptoms start showed no consistent effect, '
                'while some papers report a benefit.',
                evidence=[
                    Evidence(
                        url=_COCHRANE_URL,
                        domain='doi.org',
                        source_type='review',
                        title=_COCHRANE_TITLE,
                        quote='Trials of high doses of vitamin C administered therapeutically, starting after the onset '
                        'of symptoms, showed no consistent effect on either duration or severity of common cold symptoms',
                        quote_verified=True,
                        published_date='2013-01-31',
                        stance='contradicts',
                    ),
                    Evidence(
                        url='https://doi.org/10.3390/nu9040339',
                        domain='doi.org',
                        source_type='peer_reviewed',
                        title='Vitamin C and Infections',
                        quote='therapeutic trials of 6-8 g/day showed a dose-dependent benefit',
                        quote_verified=False,
                        published_date='2017-03-29',
                        stance='supports',
                    ),
                ],
                critical_question='Which doses, and at what point in the illness, did the studies compare?',
            ),
            Verdict(
                claim_id='c4',
                claim_text='taking 1 gram of vitamin C a day reduces the risk of a cold by 80%',
                verdict='unverifiable',
                confidence=0.2,
                explanation='The cited study could not be found, and no other source supports this figure.',
                critical_question='Can this study be found by DOI or in PubMed?',
            ),
        ],
        citations=[
            CitationCheck(
                raw='Ivanov and Smith (2021, Journal of Immunology Research)',
                status='not_found',
                closest_match='Hemilä H., Chalker E. Vitamin C for preventing and treating the common cold. '
                'Cochrane Database Syst Rev. 2013',
                url=_COCHRANE_URL,
            )
        ],
        tricks=[
            Trick(
                name='Overgeneralisation',
                phrase='reliably prevents colds in everyone',
                explanation='The words "reliably" and "everyone" turn a partial or disputed effect into a universal law.',
            ),
            Trick(
                name='Appeal to authority',
                phrase='a two-time Nobel Prize winner',
                explanation='Achievements in one field are used as proof of being right in another.',
            ),
        ],
    )
