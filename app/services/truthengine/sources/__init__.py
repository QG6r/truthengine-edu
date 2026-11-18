"""Evidence sources. Each one returns list[Evidence] built only from fetched text."""

from .crossref import CrossrefSource
from .factcheck import FactCheckSource
from .knowledge import KnowledgeSource
from .openalex import OpenAlexSource
from .websearch import WebSearchSource
from .wikipedia import WikipediaSource


__all__ = ['CrossrefSource', 'FactCheckSource', 'KnowledgeSource', 'OpenAlexSource', 'WebSearchSource', 'WikipediaSource']
