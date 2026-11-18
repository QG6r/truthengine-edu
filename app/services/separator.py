# ruff: noqa: W293
# The user-authored system prompt below is intentionally preserved verbatim.
from __future__ import annotations

import asyncio
import os
from contextlib import nullcontext
from typing import Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class SeparationError(RuntimeError):
    """Claude returned an API error or an invalid extraction."""


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')


class Citation(_Record):
    authors: list[str]
    year: int | None
    venue: str | None
    title_hint: str | None


class Claim(_Record):
    id: str
    text: str
    spans: list[str]
    type: Literal[
        'statistic',
        'numeric',
        'date',
        'biographical',
        'geographic',
        'historical',
        'physical',
        'scientific',
        'medical',
        'causal',
        'inference',
        'quote_attribution',
        'citation',
        'definition_of_term',
        'current_event',
        'belief_report',
        'other',
    ]
    stance_of_text: Literal['asserted', 'attributed', 'reported_belief', 'questioned']
    attributed_to: str | None
    qualifiers: list[str]
    scope: str | None
    resolved: dict[str, str]
    citation: Citation | None
    depends_on: list[str]
    parent: str | None
    time_sensitive: bool
    checkworthiness: float = Field(ge=0, le=1, allow_inf_nan=False)
    extraction_confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    needs_context: bool
    notes: str


class NonClaim(_Record):
    id: str
    kind: Literal[
        'opinion',
        'rhetoric',
        'author_stance',
        'definition',
        'recommendation',
        'question',
        'other',
    ]
    text: str
    spans: list[str]
    checkable_core: str | None
    depends_on: list[str]
    notes: str


class Skipped(_Record):
    span: str
    reason: str


class SeparationReport(_Record):
    claims: list[Claim]
    non_claims: list[NonClaim]
    skipped: list[Skipped]
    coverage_ok: bool


_SYSTEM = """You are the CLAIM SEPARATOR of TruthEngine Edu, a tool that fact-checks learning material for students.
Your job: split the input text into atomic, self-contained, checkable factual claims, and separately
record everything that is NOT a checkable fact. You do NOT judge whether anything is true.
Every claim you output will be searched for and verified on its own, without the original text.
So each claim must carry all the context needed to check it, and must say exactly what the text says:
never stronger, never weaker, never more specific, never less specific.
1. What counts as a claim
A claim is a statement about the world that could in principle be confirmed or refuted with evidence:
facts, numbers, dates, places, names, measurements, causal statements ("X causes / reduces Y"),
statements about what research found, quotes attributed to someone, references to studies or sources,
historical events, definitions of real-world terms (e.g. "Photosynthesis happens in mitochondria").
NOT claims (record them in non_claims, never in claims):
- opinion / evaluation: "flashcards are the best way to study", "this is a fascinating topic"
- rhetoric: "anyone who disagrees is wrong", "think about it"
- the author's stance or framing: "this is the first thing to question"
- in-text definitions that only set the meaning of a word for this text: "In this guide, 'learning' means ..."
- advice / recommendations: "you should take vitamin C before every exam" (record it, and link it to the
  claim it relies on via depends_on)
- questions, greetings, headings, instructions, jokes, pure narrative with nothing checkable
2. Splitting rules
1. ATOMIC: one checkable proposition per claim. Split compound sentences:
   "Einstein was born in Ulm in 1879" -> "Albert Einstein was born in Ulm." + "Albert Einstein was born in 1879."
   "Water boils at 100 °C and ice melts at 10 °C" -> two claims.
   Do not split a single proposition whose parts only make sense together (a value with its unit and condition,
   a range, a comparison).
2. RELATIVE AND APPOSITIVE CLAUSES are claims of their own:
   "The Eiffel Tower, completed in 1889, is 450 m tall" -> "The Eiffel Tower was completed in 1889." +
   "The Eiffel Tower is 450 meters tall."
3. SELF-CONTAINED: resolve every pronoun and reference ("he", "she", "it", "this", "they", "the study",
   "the city", "that year") to the explicit entity named in the text. Record each resolution in resolved.
   If the text does NOT name the referent, do not guess: keep the vague wording, set needs_context: true,
   and explain in notes. Never add facts from your own knowledge (e.g. do not turn "the city" into "Paris"
   unless the text says Paris).
4. KEEP NUMBERS, DATES, UNITS AND NAMES EXACTLY as written (no rounding, no unit conversion, no correcting
   misspellings or wrong values: if the text says 450 meters, the claim says 450 meters).
5. KEEP QUALIFIERS AND HEDGES inside the claim text AND list them in qualifiers:
   may, might, can, suggest, some, many, most, often, about, around, at least, up to, on average,
   in some adults, at sea level, in mice, in 2019, according to estimates...
   Never drop them and never add them. "may reduce the risk in some adults" must stay hedged.
   6. KEEP ABSOLUTE AND STRENGTH WORDS and list them in qualifiers: always, never, all, every, only,
   proved, cures, guarantees, completely, the best, clearly. They are often exactly what is false.
7. ATTRIBUTION: if the text reports what someone says or found ("according to a blog", "researchers suggest",
   "a 2019 Nature study found", "Einstein said"), produce:
  1. the content claim itself (e.g. "Vitamin C cures the common cold."), with attributed_to set
     and stance_of_text = "attributed";
  2. when the attribution itself is checkable (a named person said X, a specific study found X,
     a named organisation states X), ALSO an attribution claim:
     "Albert Einstein said that insanity is doing the same thing over and over and expecting different results."
     "A 2019 study by Smith and Johnson in Nature found that students who sleep less than six hours score
     40% lower on exams."
    Vague sources ("a popular blog", "some researchers", "experts say") are not checkable as attributions:
    keep them only in attributed_to and flag them in notes as an unnamed source.
8. CITATIONS: if the text cites a specific work (authors, year, journal, title, DOI), keep those details
   verbatim inside the claim and fill citation. Strip bare reference markers that carry no information
   ("[12]", "(ref. 3)", "¹") from claim text.
9. CAUSAL AND "THEREFORE" STATEMENTS: a statement presented as a consequence ("therefore", "so",
   "which is why", "because") is its own claim, of type "causal" or "inference". Record what it relies on
   in depends_on. Do not merge it into the premises.
10. DEFINITIONS USED BY CLAIMS: if a claim depends on a definition given in the text, put that definition in
   non_claims (kind "definition"), reference it from the claim's depends_on, and write the claim so it is
   understandable on its own ("Learning, defined as long-term retention of information, requires ...").
11. NEGATIONS: keep them exactly ("is not", "never", "no evidence").
12. NO DUPLICATES: if the text repeats the same proposition, output it once and list all spans in spans.
13. LANGUAGE: write claims in the same language as the source text. Do not translate.
14. COVERAGE: every sentence of the input must be accounted for, either in a claim, in non_claims, or in
   skipped with a reason (heading, greeting, empty). Nothing silently disappears.
3. Claim types (field type)
statistic | numeric | date | biographical | geographic | historical | physical | scientific | medical |
causal | inference | quote_attribution | citation | definition_of_term | current_event | belief_report | other
time_sensitive: true when the truth can change over time (current office holders, prices, populations,
records, laws, "currently", "the latest").
4. Check-worthiness (field checkworthiness, 0-1)
High (0.8-1): specific numbers, dates, medical or health statements, causal claims, cited studies, quotes,
absolute words ("cures", "proved", "all"), anything a student might repeat in an exam.
Medium (0.4-0.7): general background facts.
Low (0-0.3): vague, trivial or descriptive statements ("many students repeat this").
5. Self-check before answering (layer 0), for EVERY claim
1. Self-contained: understandable and searchable without the original text?
2. Faithful: does it follow from its source span, and does the span follow from it?
   (not stronger, not weaker, not more specific, no added facts)
3. Qualifiers preserved: every hedge, scope and strength word from the span is still there?
4. Atomic: exactly one checkable proposition?
5. Numbers, units, names, dates copied exactly?
  Set extraction_confidence (0-1) accordingly and explain any doubt in notes.
  If a check fails and you cannot fix it, keep the claim but lower extraction_confidence and say why.
6. Output
Return ONLY valid JSON in this exact shape (no prose before or after):
{
  "claims": [
    {
      "id": "C1",
      "text": "self-contained claim, same language as the source",
      "spans": ["exact substring(s) of the input this claim comes from"],
      "type": "one of the types above",
      "stance_of_text": "asserted | attributed | reported_belief | questioned",
      "attributed_to": "who says it, or null",
      "qualifiers": ["may", "some adults"],
      "scope": "population / place / time / conditions, or null",
      "resolved": {"He": "Albert Einstein"},
      "citation": {"authors": ["Smith", "Johnson"], "year": 2019, "venue": "Nature", "title_hint": null} or null,
      "depends_on": ["D1", "C5"],
      "parent": "id of the compound sentence group, or null",
      "time_sensitive": false,
      "checkworthiness": 0.9,
      "extraction_confidence": 0.95,
      "needs_context": false,
      "notes": "short note or empty string"
    }
  ],
  "non_claims": [
    {
      "id": "O1",
      "kind": "opinion | rhetoric | author_stance | definition | recommendation | question | other",
      "text": "the statement",
      "spans": ["exact substring(s)"],
      "checkable_core": "a checkable claim hidden inside it, or null",
      "depends_on": ["C13"],
      "notes": "why it is not checked as a fact; name the rhetorical technique if any"
    }
  ],
  "skipped": [{"span": "Study Smarter: A Short Guide", "reason": "heading"}],
  "coverage_ok": true
}
spans must be copied character for character from the input: they are used to highlight the text in the UI.
7. Short example
Input: "Some researchers suggest that coffee may reduce the risk of type 2 diabetes in some adults, but
according to a wellness blog, vitamin C cures the common cold, so you should take it before every exam."
Output (abridged):
- C1 "Moderate... " -> text "Coffee consumption may reduce the risk of type 2 diabetes in some adults.",
  stance_of_text "attributed", attributed_to "some researchers (unnamed)", qualifiers ["suggest","may","some adults"]
- C2 text "Vitamin C cures the common cold.", stance_of_text "attributed", attributed_to "a wellness blog (unnamed)",
  qualifiers ["cures"], type "medical", checkworthiness 0.95
- O1 kind "recommendation", text "Students should take vitamin C before every exam.", depends_on ["C2"]

   
"""


class Separator:
    """Extract claims using the project system prompt.

    separate() retains the list[str] interface, returning claim texts.
    separate_detailed() returns claims, non-claims, and source spans.
    Sync methods use asyncio.run; async callers should use the async methods.
    Injected clients remain caller-owned and must belong to the current loop.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key, self._model, self._client = api_key, model, client

    def separate(self, text: str) -> list[str]:
        return [claim.text for claim in self.separate_detailed(text).claims]

    async def separate_async(self, text: str) -> list[str]:
        return [claim.text for claim in (await self.separate_detailed_async(text)).claims]

    def separate_detailed(self, text: str) -> SeparationReport:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.separate_detailed_async(text))
        raise SeparationError('Use await separate_async() or separate_detailed_async() inside an event loop')

    async def separate_detailed_async(self, text: str) -> SeparationReport:
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        if not text.strip():
            return SeparationReport(claims=[], non_claims=[], skipped=[], coverage_ok=True)
        if len(text) > 24_000:
            raise ValueError('Input exceeds the 24,000-character limit')
        load_dotenv()
        api_key = self._api_key or os.getenv('ANTHROPIC_API_KEY', '')
        model = self._model or os.getenv('ANTHROPIC_MODEL', '')
        if not api_key.strip() or not model.strip():
            raise SeparationError('Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL')
        messages = [{'role': 'user', 'content': text}]
        context = nullcontext(self._client) if self._client is not None else httpx.AsyncClient()
        async with context as client:
            for attempt in range(2):
                try:
                    response = await client.post(
                        'https://api.anthropic.com/v1/messages',
                        headers={
                            'x-api-key': api_key,
                            'anthropic-version': '2023-06-01',
                        },
                        json={
                            'model': model,
                            'max_tokens': 8192,
                            'system': _SYSTEM,
                            'messages': messages,
                        },
                        timeout=90.0,
                    )
                except httpx.RequestError:
                    raise SeparationError('Claude request failed; check network access') from None
                if response.is_error:
                    detail = ''
                    try:
                        message = response.json().get('error', {}).get('message', '')
                        if isinstance(message, str):
                            detail = message.replace(api_key, '[REDACTED]')[:1500]
                    except (ValueError, AttributeError, TypeError):
                        pass
                    raise SeparationError(f'Claude API returned HTTP {response.status_code}: {detail}')
                try:
                    body = response.json()
                    if body['stop_reason'] != 'end_turn':
                        raise ValueError('Incomplete or refused response')
                    output = ''.join(b['text'] for b in body['content'] if b['type'] == 'text')
                    report = SeparationReport.model_validate_json(output)
                    self._validate(report, text)
                    return report
                except (ValueError, KeyError, TypeError):
                    if attempt:
                        raise SeparationError(
                            'Claude returned invalid extraction twice: check JSON, spans, '
                            'references, and source coverage; long outputs may be truncated'
                        ) from None
                    # Never modify the user's system prompt, including on retry.
                    messages = [
                        {'role': 'user', 'content': text},
                        {
                            'role': 'user',
                            'content': (
                                'Return the JSON shape specified in the system prompt. '
                                'The previous attempt failed validation. Use exact nonempty source '
                                'spans, unique IDs, existing dependency IDs, and account for all '
                                'non-whitespace source characters. Do not include markdown fences.'
                            ),
                        },
                    ]
        raise SeparationError('No extraction returned')

    @staticmethod
    def _validate(report: SeparationReport, text: str) -> None:
        records = [*report.claims, *report.non_claims]
        ids = [r.id for r in records]
        if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
            raise ValueError('Empty or duplicate IDs')
        known_ids = set(ids)
        covered = bytearray(len(text))
        spans = []
        for record in records:
            if not record.text.strip() or not record.spans:
                raise ValueError('Missing text or source spans')
            if any(ref not in known_ids or ref == record.id for ref in record.depends_on):
                raise ValueError('Unknown or self dependency')
            spans.extend(record.spans)
        for skipped in report.skipped:
            if not skipped.reason.strip():
                raise ValueError('Skipped text requires a reason')
            spans.append(skipped.span)
        for span in spans:
            if not span.strip() or span not in text:
                raise ValueError('Invented or empty source span')
            # Identical passages may recur: mark all occurrences of an exact span.
            start = 0
            while True:
                found = text.find(span, start)
                if found < 0:
                    break
                covered[found : found + len(span)] = bytes([1]) * len(span)
                start = found + 1
        complete = all(covered[i] or char.isspace() for i, char in enumerate(text))
        if not complete or not report.coverage_ok:
            raise ValueError('Incomplete source coverage')
