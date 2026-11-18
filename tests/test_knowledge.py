"""Teacher knowledge base: store, BM25 retrieval, engine source and API endpoints (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
from app.services.knowledge import KnowledgeError, KnowledgeStore, get_store
from app.services.knowledge.store import split_into_chunks
from app.services.truthengine.models import Claim, QueryPlan
from app.services.truthengine.scoring import aggregate, score_evidence
from app.services.truthengine.sources import KnowledgeSource


BIOLOGY = (
    b'Photosynthesis takes place in the chloroplasts of plant cells. '
    b'Mitochondria are the site of cellular respiration and produce most of the ATP of the cell. '
    b'The light-dependent reactions occur in the thylakoid membranes.'
)
HISTORY = b'The French Revolution began in 1789. Napoleon Bonaparte became emperor of the French in 1804.'


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / 'knowledge')


def test_add_list_search_delete(store: KnowledgeStore) -> None:
    bio = store.add('biology.txt', BIOLOGY, title='Biology 101')
    store.add('history.md', HISTORY)
    assert [s.title for s in store.list_sources()] == ['Biology 101', 'history']
    assert bio.pages == 1 and bio.chunks >= 1

    hits = store.search('Where does photosynthesis happen in plant cells?')
    assert hits[0].source_id == bio.id and 'chloroplasts' in hits[0].text
    assert store.search('quantum chromodynamics gluons') == []

    store.delete(bio.id)
    assert [s.title for s in store.list_sources()] == ['history']
    assert all(hit.source_id != bio.id for hit in store.search('photosynthesis chloroplasts'))
    assert not list(store.dir.glob(f'{bio.id}*'))


def test_index_survives_restart(store: KnowledgeStore) -> None:
    store.add('history.txt', HISTORY)
    reopened = KnowledgeStore(store.dir)
    assert 'Napoleon' in reopened.search('When did Napoleon become emperor?')[0].text


@pytest.mark.parametrize(
    ('filename', 'data', 'message'),
    [('notes.docx', b'x', 'Unsupported file type'), ('notes.txt', b'', 'empty'), ('scan.txt', b'   \n ', 'No text')],
)
def test_rejects_bad_uploads(store: KnowledgeStore, filename: str, data: bytes, message: str) -> None:
    with pytest.raises(KnowledgeError, match=message):
        store.add(filename, data)
    assert store.list_sources() == []


def test_chunks_are_sentence_aware_and_overlap() -> None:
    text = ' '.join(f'Sentence number {i} talks about topic {i}.' for i in range(40))
    chunks = split_into_chunks(text, limit=200)
    assert len(chunks) > 3 and all(len(c) <= 260 for c in chunks)
    assert chunks[0].split('. ')[-1] in chunks[1]  # last sentence repeated in the next chunk


async def test_engine_source_returns_tier_1d_evidence_that_passes_gates(store: KnowledgeStore) -> None:
    bio = store.add('biology.txt', BIOLOGY, title='Biology 101')
    plan = QueryPlan(neutral_query='photosynthesis location plant cell')
    evidence = await KnowledgeSource(store).search(plan, Claim(text='Photosynthesis happens in mitochondria.'))

    ev = evidence[0]
    assert (ev.tier, ev.base_weight, ev.title) == ('1D', 0.9, 'Biology 101, p. 1')
    assert ev.url.endswith(f'/sources/{bio.id}/file#page=1')

    # What the judge would return: a verbatim quote from the uploaded text contradicting the claim.
    ev.stance, ev.quote, ev.directness, ev.scope_match = 'contradicts', 'Photosynthesis takes place in the chloroplasts', 1.0, 1.0
    result = aggregate([score_evidence(ev, plan, 2026)])
    assert result['verdict'] == 'contradicted' and result['best_tier'] == '1D'


async def test_engine_source_is_empty_without_uploads(store: KnowledgeStore) -> None:
    assert await KnowledgeSource(store).search(QueryPlan(), Claim(text='Anything at all.')) == []


def test_sources_api(store: KnowledgeStore) -> None:
    api.app.dependency_overrides[get_store] = lambda: store
    try:
        client = TestClient(api.app)
        assert client.get('/sources').json() == []

        created = client.post('/sources', files={'file': ('biology.txt', BIOLOGY)}, data={'title': 'Biology 101'})
        assert created.status_code == 201
        source_id = created.json()['id']
        assert [s['title'] for s in client.get('/sources').json()] == ['Biology 101']

        download = client.get(f'/sources/{source_id}/file')
        assert download.status_code == 200 and download.content == BIOLOGY

        bad = client.post('/sources', files={'file': ('virus.exe', b'MZ')})
        assert bad.status_code == 422 and 'Unsupported file type' in bad.json()['detail']

        assert client.delete(f'/sources/{source_id}').status_code == 204
        assert client.delete(f'/sources/{source_id}').status_code == 404
        assert client.get(f'/sources/{source_id}/file').status_code == 404
    finally:
        api.app.dependency_overrides.pop(get_store, None)
