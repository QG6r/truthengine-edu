"""Crossref: gate G1, does the cited work exist with these authors, year and venue."""

from __future__ import annotations

from typing import Any

from ..config import CONTACT_EMAIL
from ..models import CitationRef
from .base import BaseSource


def _year(item: dict[str, Any]) -> int | None:
    try:
        return int(item['issued']['date-parts'][0][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


class CrossrefSource(BaseSource):
    """Gate G1: does the cited work actually exist with these authors, year and venue?"""

    name = 'crossref'
    URL = 'https://api.crossref.org/works'

    async def find_citation(self, cit: CitationRef) -> dict[str, Any] | None:
        parts = [*cit.authors, cit.title_hint or '', cit.venue or '', str(cit.year or '')]
        query = ' '.join(p for p in parts if p).strip()
        if not query:
            return None
        params: dict[str, Any] = {'query.bibliographic': query, 'rows': 10, 'select': 'DOI,title,author,issued,container-title,type'}
        if CONTACT_EMAIL:
            params['mailto'] = CONTACT_EMAIL
        data = await self.get_json(self.URL, params)

        wanted = {a.split()[-1].lower() for a in cit.authors if a.strip()}
        for item in data.get('message', {}).get('items', []):
            families = {(a.get('family') or '').lower() for a in item.get('author', [])}
            venues = [v.lower() for v in item.get('container-title') or []]
            year_ok = cit.year is None or _year(item) == cit.year
            authors_ok = wanted <= families  # every cited author must be present
            venue_ok = not cit.venue or cit.venue.lower() in venues
            if year_ok and authors_ok and venue_ok:
                return {
                    'doi': item.get('DOI'),
                    'title': (item.get('title') or [''])[0],
                    'year': _year(item),
                    'venue': venues[0] if venues else None,
                }
        return None
