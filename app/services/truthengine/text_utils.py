"""Keyword extraction and relevant-passage selection for long source texts."""

from __future__ import annotations

import re


_WORD = re.compile(r'\w+', re.UNICODE)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')
_STOP_WORDS = """
the and for that this with from was were are its his her their has have had not but which who what
when where how than then into about some only also can may all any one out over they them been being such more most
"""
_STOP = frozenset(_STOP_WORDS.split())


def keywords(*texts: str) -> set[str]:
    """Lowercase content words and numbers from the given texts."""
    words: set[str] = set()
    for text in texts:
        for w in _WORD.findall(text.lower()):
            if w.isdigit() or (len(w) > 2 and w not in _STOP):
                words.add(w)
    return words


def best_passages(text: str, terms: set[str], limit: int) -> str:
    """Pick the sentences most relevant to `terms`, keep their original order, fit into `limit` chars.

    Sentences are joined with ' … ' so a quote spanning two of them is still verified part by part.
    """
    if len(text) <= limit or not terms:
        return text[:limit]
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    scored = sorted(
        ((len(terms & set(_WORD.findall(s.lower()))), i) for i, s in enumerate(sentences)),
        key=lambda x: (-x[0], x[1]),
    )
    if not scored or scored[0][0] == 0:
        return text[:limit]
    chosen: list[int] = []
    used = 0
    for score, i in scored:
        if score == 0:
            break
        length = len(sentences[i]) + 3
        if used + length > limit:
            continue
        chosen.append(i)
        used += length
    return ' … '.join(sentences[i] for i in sorted(chosen))
