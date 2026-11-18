"""Judge: asks Claude how each fetched text relates to the claim, with a verbatim quote."""

from __future__ import annotations

from typing import Any

from .config import EVIDENCE_CHARS_FOR_JUDGE
from .llm import LLM
from .models import Claim, Evidence, QueryPlan
from .planner import claim_context
from .text_utils import best_passages, keywords


JUDGE_SYSTEM = """You assess how each evidence TEXT relates to a claim.
Use ONLY the evidence texts. Ignore your own knowledge, even if you think a text is wrong.

For every evidence item return:
- stance:
  supports    = the text affirms the claim AS WORDED, including its qualifiers;
  contradicts = the text states something incompatible with the claim as worded
                (calls it a myth, gives a different value, or refutes an absolute word
                such as "cures", "always", "proved", "all");
  neutral     = on topic but does not settle the claim;
  irrelevant  = different entity, population, place, period or question.
  If the claim is hedged ("may", "some", "suggest"), judge the hedged version:
  a modest or mixed effect supports a hedged claim.
- quote: the single most decisive sentence or phrase copied EXACTLY, character for character,
  from that item's TEXT. Never paraphrase. Empty string when neutral or irrelevant.
- directness 0-1: 1 = addresses this exact claim, 0.5 = closely related, 0.2 = tangential.
- scope_match 0-1: do population, place, time, units and conditions match the claim?
  (animal vs human study, other country, other year -> lower).
- note: one short sentence.
Then, for the claim as a whole:
- explanation: two plain-language sentences for a student about what the evidence shows,
  mentioning qualifiers when they matter.
- correction: if the evidence shows the claim is wrong or overstated, one sentence stating what the
  evidence actually says (e.g. "The Eiffel Tower is about 330 m tall, not 450 m."). Empty string otherwise.
  Only use facts present in the evidence texts.
- critical_question: one short question that teaches the student how to check this kind of claim
  themselves (e.g. "Who first said this quote, and can you find it in their own writings?")."""

JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'items': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'index': {'type': 'integer'},
                    'stance': {'type': 'string', 'enum': ['supports', 'contradicts', 'neutral', 'irrelevant']},
                    'quote': {'type': 'string'},
                    'directness': {'type': 'number'},
                    'scope_match': {'type': 'number'},
                    'note': {'type': 'string'},
                },
                'required': ['index', 'stance', 'quote', 'directness', 'scope_match'],
            },
        },
        'explanation': {'type': 'string'},
        'correction': {'type': 'string'},
        'critical_question': {'type': 'string'},
    },
    'required': ['items', 'explanation', 'correction', 'critical_question'],
}

_STANCES = ('supports', 'contradicts', 'neutral', 'irrelevant')


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


class Judge:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def judge(self, claim: Claim, plan: QueryPlan, evidence: list[Evidence]) -> dict[str, str | None]:
        """Set stance/quote/directness/scope on each evidence; return explanation, correction, critical_question."""
        terms = keywords(claim.text, plan.neutral_query, plan.support_query)
        blocks = [
            f'[{i}] source={ev.source} tier={ev.tier} year={ev.year or "n/a"} title={ev.title[:150]}\n'
            f'TEXT: {best_passages(ev.text, terms, EVIDENCE_CHARS_FOR_JUDGE)}'
            for i, ev in enumerate(evidence)
        ]
        prompt = f'{claim_context(claim)}\n\nEVIDENCE:\n\n' + '\n\n'.join(blocks)
        data = await self.llm.structured(JUDGE_SYSTEM, prompt, JUDGE_SCHEMA, name='evidence_assessment', max_tokens=3000)
        for item in data.get('items', []):
            i = item.get('index')
            if not isinstance(i, int) or not 0 <= i < len(evidence):
                continue
            ev = evidence[i]
            stance = item.get('stance', 'neutral')
            ev.stance = stance if stance in _STANCES else 'neutral'
            ev.quote = item.get('quote') or ''
            ev.directness = _clamp(item.get('directness'))
            ev.scope_match = _clamp(item.get('scope_match'))
            ev.note = item.get('note') or ''
        return {
            'explanation': data.get('explanation') or '',
            'correction': (data.get('correction') or '').strip() or None,
            'critical_question': (data.get('critical_question') or '').strip() or None,
        }
