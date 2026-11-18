"""Storage, chunking and retrieval for sources uploaded by teachers.

Pure Python on purpose (pypdf + BM25, no vector database): native DLLs of ChromaDB / FAISS /
onnxruntime are blocked by Windows Smart App Control on some of our laptops.

Layout of the data directory (KNOWLEDGE_DIR, default data/knowledge, git-ignored):
    sources.json      metadata of every uploaded source
    <id>.<ext>        the original file
    <id>.chunks.json  its text chunks with page numbers
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import threading
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md'}
MAX_FILE_BYTES = 20 * 1024 * 1024
CHUNK_CHARS = 700
CHUNK_OVERLAP_SENTENCES = 1
BM25_K1 = 1.5
BM25_B = 0.75

_WORD = re.compile(r'\w+', re.UNICODE)
_SENTENCE = re.compile(r'(?<=[.!?])\s+|\n{2,}')
_STOP_WORDS = """
the and for that this with from was were are its his her their has have had not but which who what
when where how than then into about some only also can may all any one out over they them been being such more most
"""
_STOP = frozenset(_STOP_WORDS.split())


class KnowledgeError(ValueError):
    """Upload or lookup problem with a short user-facing message."""


class SourceInfo(BaseModel):
    id: str
    title: str
    filename: str
    uploaded_at: str
    pages: int
    chunks: int
    size_bytes: int


class Chunk(BaseModel):
    source_id: str
    title: str
    page: int
    text: str
    score: float = 0.0


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w.isdigit() or (len(w) > 2 and w not in _STOP)]


def split_into_chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Sentence-aware chunks of about `limit` characters; the last sentence is repeated in the next chunk."""
    sentences = [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if current and size + len(sentence) > limit:
            chunks.append(' '.join(current))
            current = current[-CHUNK_OVERLAP_SENTENCES:]
            size = sum(len(s) + 1 for s in current)
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(' '.join(current))
    return chunks


def extract_pages(filename: str, data: bytes) -> list[str]:
    """Text per page. PDF via pypdf; .txt / .md are one page."""
    if Path(filename).suffix.lower() == '.pdf':
        from pypdf import PdfReader  # imported lazily: only needed for PDF uploads

        try:
            reader = PdfReader(io.BytesIO(data))
            return [re.sub(r'[ \t]+', ' ', page.extract_text() or '') for page in reader.pages]
        except Exception as exc:
            raise KnowledgeError('The PDF could not be read. Is the file damaged or password-protected?') from exc
    try:
        return [data.decode('utf-8')]
    except UnicodeDecodeError:
        return [data.decode('utf-8', errors='replace')]


class KnowledgeStore:
    """Thread-safe store. Search is BM25 over all chunks; the index is rebuilt after every change."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.dir = Path(directory or os.getenv('KNOWLEDGE_DIR', 'data/knowledge'))
        self._lock = threading.Lock()
        self._index: _Bm25Index | None = None

    # --- metadata -------------------------------------------------------------
    @property
    def _meta_path(self) -> Path:
        return self.dir / 'sources.json'

    def _load_meta(self) -> list[SourceInfo]:
        if not self._meta_path.exists():
            return []
        return [SourceInfo.model_validate(item) for item in json.loads(self._meta_path.read_text(encoding='utf-8'))]

    def _save_meta(self, sources: list[SourceInfo]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps([s.model_dump() for s in sources], ensure_ascii=False, indent=2), encoding='utf-8')

    def list_sources(self) -> list[SourceInfo]:
        with self._lock:
            return self._load_meta()

    def get(self, source_id: str) -> SourceInfo:
        for source in self.list_sources():
            if source.id == source_id:
                return source
        raise KnowledgeError('Source not found.')

    def file_path(self, source_id: str) -> Path:
        source = self.get(source_id)  # the id is looked up, never used as a raw path
        return self.dir / f'{source.id}{Path(source.filename).suffix.lower()}'

    # --- changes --------------------------------------------------------------
    def add(self, filename: str, data: bytes, title: str | None = None) -> SourceInfo:
        name = Path(filename or '').name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise KnowledgeError(f'Unsupported file type "{ext or name}". Upload a PDF, TXT or MD file.')
        if not data:
            raise KnowledgeError('The file is empty.')
        if len(data) > MAX_FILE_BYTES:
            raise KnowledgeError(f'The file is too large: maximum is {MAX_FILE_BYTES // (1024 * 1024)} MB.')

        pages = extract_pages(name, data)
        chunks = [{'page': number, 'text': chunk} for number, page in enumerate(pages, start=1) for chunk in split_into_chunks(page)]
        if not chunks:
            raise KnowledgeError('No text could be extracted. Scanned PDFs need OCR first.')

        source = SourceInfo(
            id=uuid.uuid4().hex[:12],
            title=(title or '').strip() or Path(name).stem,
            filename=name,
            uploaded_at=datetime.now(UTC).isoformat(timespec='seconds'),
            pages=len(pages),
            chunks=len(chunks),
            size_bytes=len(data),
        )
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f'{source.id}{ext}').write_bytes(data)
            (self.dir / f'{source.id}.chunks.json').write_text(json.dumps(chunks, ensure_ascii=False), encoding='utf-8')
            self._save_meta([*self._load_meta(), source])
            self._index = None
        return source

    def delete(self, source_id: str) -> None:
        with self._lock:
            sources = self._load_meta()
            source = next((s for s in sources if s.id == source_id), None)
            if source is None:
                raise KnowledgeError('Source not found.')
            self._save_meta([s for s in sources if s.id != source_id])
            for path in (self.dir / f'{source.id}{Path(source.filename).suffix.lower()}', self.dir / f'{source.id}.chunks.json'):
                path.unlink(missing_ok=True)
            self._index = None

    # --- retrieval ------------------------------------------------------------
    def search(self, query: str, k: int = 6) -> list[Chunk]:
        """The `k` best chunks for the query (BM25), best first. Empty when nothing shares a term."""
        with self._lock:
            if self._index is None:
                self._index = _Bm25Index(self._load_chunks())
            index = self._index
        return index.search(query, k)

    def _load_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for source in self._load_meta():
            path = self.dir / f'{source.id}.chunks.json'
            if not path.exists():
                continue
            for item in json.loads(path.read_text(encoding='utf-8')):
                chunks.append(Chunk(source_id=source.id, title=source.title, page=item['page'], text=item['text']))
        return chunks


class _Bm25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.term_counts = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(tc.values()) for tc in self.term_counts]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for tc in self.term_counts:
            doc_freq.update(tc.keys())
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def search(self, query: str, k: int) -> list[Chunk]:
        terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for i, tc in enumerate(self.term_counts):
            score = 0.0
            for term in terms:
                tf = tc.get(term, 0)
                if tf:
                    norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[i] / (self.avg_length or 1.0))
                    score += self.idf[term] * tf * (BM25_K1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self.chunks[i].model_copy(update={'score': round(score, 4)}) for score, i in scored[:k]]


_default_store: KnowledgeStore | None = None


def get_store() -> KnowledgeStore:
    """Process-wide store shared by the API (uploads) and the engine (retrieval)."""
    global _default_store
    if _default_store is None:
        _default_store = KnowledgeStore()
    return _default_store
