"""Teacher knowledge base (RAG): uploaded trusted sources, chunked and searched with BM25."""

from .store import Chunk, KnowledgeError, KnowledgeStore, SourceInfo, get_store


__all__ = ['Chunk', 'KnowledgeError', 'KnowledgeStore', 'SourceInfo', 'get_store']
