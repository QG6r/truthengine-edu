import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from app.services.separator import _SYSTEM, SeparationError, Separator


def sample():
    return {
        'claims': [
            {
                'id': 'C1',
                'text': 'Sleep may improve recall.',
                'spans': ['Sleep may improve recall.'],
                'type': 'scientific',
                'stance_of_text': 'asserted',
                'attributed_to': None,
                'qualifiers': ['may'],
                'scope': None,
                'resolved': {},
                'citation': None,
                'depends_on': [],
                'parent': None,
                'time_sensitive': False,
                'checkworthiness': 0.8,
                'extraction_confidence': 0.9,
                'needs_context': False,
                'notes': '',
            }
        ],
        'non_claims': [
            {
                'id': 'O1',
                'kind': 'opinion',
                'text': 'Wonderful!',
                'spans': ['Wonderful!'],
                'checkable_core': None,
                'depends_on': ['C1'],
                'notes': 'Evaluation',
            }
        ],
        'skipped': [],
        'coverage_ok': True,
    }


SOURCE = 'Sleep may improve recall. Wonderful!'


def run(outputs, detailed=False):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        value = outputs[min(len(calls) - 1, len(outputs) - 1)]
        if isinstance(value, int):
            return httpx.Response(value, json={'error': {'message': 'Bad test-key'}})
        if isinstance(value, Exception):
            raise value
        body = {
            'stop_reason': 'end_turn',
            'content': [{'type': 'text', 'text': json.dumps(value)}],
        }
        return httpx.Response(200, json=body)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            separator = Separator(api_key='test-key', model='test-model', client=client)
            result = await separator.separate_detailed_async(SOURCE) if detailed else await separator.separate_async(SOURCE)
            return result, calls

    return asyncio.run(exercise())


def test_original_interface_returns_claim_texts():
    result, calls = run([sample()])
    assert result == ['Sleep may improve recall.']
    assert calls[0]['system'] == _SYSTEM
    assert calls[0]['messages'][0]['content'] == SOURCE


def test_detailed_report_retains_non_claims_and_source_spans():
    result, _ = run([sample()], detailed=True)
    assert result.non_claims[0].kind == 'opinion'
    assert result.claims[0].qualifiers == ['may']
    assert result.coverage_ok


@pytest.mark.parametrize(
    'kind',
    [
        'span',
        'coverage',
        'id',
        'dependency',
        'range',
        'field',
        'empty',
        'type',
    ],
)
def test_invalid_outputs_fail_after_retry(kind):
    data = sample()
    claim = data['claims'][0]
    if kind == 'span':
        claim['spans'] = ['Invented text']
    elif kind == 'coverage':
        data['non_claims'] = []
    elif kind == 'id':
        data['non_claims'][0]['id'] = 'C1'
    elif kind == 'dependency':
        claim['depends_on'] = ['C999']
    elif kind == 'range':
        claim['checkworthiness'] = 2.0
    elif kind == 'field':
        claim['extra'] = 'unexpected'
    elif kind == 'empty':
        claim['spans'] = []
    else:
        claim['type'] = 'unknown'
    with pytest.raises(SeparationError, match='twice'):
        run([data])


def test_retry_preserves_system_prompt():
    invalid = deepcopy(sample())
    invalid['coverage_ok'] = False
    _, calls = run([invalid, sample()])
    assert len(calls) == 2
    assert all(call['system'] == _SYSTEM for call in calls)


def test_skipped_source_counts_towards_coverage():
    data = sample()
    data['non_claims'] = []
    data['skipped'] = [{'span': 'Wonderful!', 'reason': 'framing'}]
    report, _ = run([data], detailed=True)
    assert report.skipped[0].span == 'Wonderful!'


def test_non_claims_only_is_valid():
    data = sample()
    data['claims'] = []
    data['non_claims'][0].update(spans=[SOURCE], text=SOURCE, depends_on=[])
    assert run([data])[0] == []


@pytest.mark.parametrize('status', [400, 401, 429, 500])
def test_api_errors_redact_key(status):
    with pytest.raises(SeparationError) as error:
        run([status])
    assert str(status) in str(error.value)
    assert 'test-key' not in str(error.value)


def test_network_error():
    with pytest.raises(SeparationError, match='network'):
        run([httpx.ReadTimeout('timeout')])


def test_empty_input_requires_no_key():
    assert Separator().separate(' \n') == []


def test_missing_configuration(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_MODEL', raising=False)
    with pytest.raises(SeparationError, match='ANTHROPIC_API_KEY'):
        Separator().separate(SOURCE)


def test_limit():
    with pytest.raises(ValueError, match='limit'):
        Separator().separate('x' * 24001)
