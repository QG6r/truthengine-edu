"""Teacher knowledge base: passages retrieved from sources a teacher uploaded (tier 1D).

Retrieval only finds candidate passages. Like every other source they still go through the judge
(stance + verbatim quote) and the gates, so an uploaded file is evidence, not an automatic verdict.
"""

from __future__ import annotations

import asyncio

from app.services.knowledge import KnowledgeStore, get_store

from ..config import KNOWLEDGE_TOP_PAGES, PUBLIC_API_URL, TIER_BASE
from ..models import Claim, Evidence, GateResult, QueryPlan


class KnowledgeSource:
    name = 'knowledge'

    def __init__(self, store: KnowledgeStore | None = None) -> None:
        self.store = store or get_store()

    async def search(self, plan: QueryPlan, claim: Claim) -> list[Evidence]:
        query = ' '.join([claim.text, plan.neutral_query, plan.support_query])
        chunks = await asyncio.to_thread(self.store.search, query, KNOWLEDGE_TOP_PAGES * 3)

        # Merge the chunks of one page into one piece of evidence (they overlap by a sentence).
        pages: dict[tuple[str, int], list[str]] = {}
        titles: dict[str, str] = {}
        for chunk in chunks:
            key = (chunk.source_id, chunk.page)
            if key not in pages and len(pages) >= KNOWLEDGE_TOP_PAGES:
                continue
            pages.setdefault(key, []).append(chunk.text)
            titles[chunk.source_id] = chunk.title

        return [
            Evidence(
                source=self.name,
                url=f'{PUBLIC_API_URL}/sources/{source_id}/file#page={page}',
                domain='uploaded document',
                title=f'{titles[source_id]}, p. {page}',
                text='\n'.join(texts),
                tier='1D',
                base_weight=TIER_BASE['1D'],
                gates=[GateResult(name='1D_uploaded_by_teacher', status='pass', reason=titles[source_id])],
                meta={'source_id': source_id, 'page': page},
            )
            for (source_id, page), texts in pages.items()
        ]
