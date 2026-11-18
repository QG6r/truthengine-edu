"""Pure helpers of app.py (Streamlit front end); loaded by path because `app/` is also a package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import requests

from mock_data import DEMO_INPUTS, mock_report
from schemas import Report, Verdict


_spec = importlib.util.spec_from_file_location('streamlit_app', Path(__file__).resolve().parent.parent / 'app.py')
frontend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frontend)


def _verdict(text: str, kind: str = 'supported') -> Verdict:
    return Verdict(claim_id='x', claim_text=text, verdict=kind, confidence=1.0, explanation='')


def test_highlight_escapes_html_and_matches_case_insensitively() -> None:
    out = frontend.highlight_html('<b>Water</b> boils at 100 C. Ice is cold.', [_verdict('WATER</B> BOILS at 100 c.')])
    assert '<b>' not in out
    assert out.startswith('&lt;b&gt;<mark')
    assert 'Water&lt;/b&gt; boils at 100 C</mark>' in out


def test_highlight_skips_missing_and_overlapping_claims() -> None:
    out = frontend.highlight_html('Alpha beta gamma.', [_verdict('alpha beta'), _verdict('beta gamma', 'contradicted'), _verdict('delta')])
    assert out.count('<mark') == 1


def test_mock_claims_all_highlight_in_first_demo_input() -> None:
    out = frontend.highlight_html(DEMO_INPUTS[0]['text'], mock_report('').verdicts)
    assert out.count('<mark') == 4


def test_safe_url_blocks_javascript() -> None:
    assert frontend.safe_url('javascript:alert(1)') is None
    assert frontend.safe_url('https://a.org/x') == 'https://a.org/x'


class _Resp:
    def __init__(self, status: int, body: object, headers: dict | None = None) -> None:
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.ok = status < 400

    def json(self) -> object:
        return self._body


def test_analyze_via_api_success_and_demo_header(monkeypatch: pytest.MonkeyPatch) -> None:
    body = mock_report('').model_dump()
    monkeypatch.setattr(frontend.requests, 'post', lambda *a, **k: _Resp(200, body, {'X-Demo-Data': 'true'}))
    report, demo = frontend.analyze_via_api('text')
    assert isinstance(report, Report)
    assert demo is True


@pytest.mark.parametrize(
    ('side_effect', 'fragment'),
    [
        (requests.ConnectionError(), 'unavailable'),
        (requests.Timeout(), '200 seconds'),
        (_Resp(422, {'detail': 'Text is empty'}), 'Text is empty'),
        (_Resp(503, {}), 'code 503'),
        (_Resp(200, {'bad': 1}), 'unexpected format'),
    ],
)
def test_analyze_via_api_errors_are_short_messages(monkeypatch: pytest.MonkeyPatch, side_effect: object, fragment: str) -> None:
    def fake_post(*args: object, **kwargs: object) -> object:
        if isinstance(side_effect, Exception):
            raise side_effect
        return side_effect

    monkeypatch.setattr(frontend.requests, 'post', fake_post)
    with pytest.raises(frontend.ApiError, match=fragment):
        frontend.analyze_via_api('text')


def test_md_escapes_markdown_links_and_formatting() -> None:
    out = frontend.md('[click](https://evil.example) **bold** :red[x] $x$')
    assert '[click](' not in out
    assert '**' not in out.replace('\\*', '')
    assert out.replace('\\', '') == '[click](https://evil.example) **bold** :red[x] $x$'


def test_safe_url_encodes_link_breaking_chars() -> None:
    url = frontend.safe_url('https://a.org/x) [y](https://evil')
    assert url is not None
    assert ')' not in url
    assert ' ' not in url
    assert '[' not in url


def test_long_inputs_sizes() -> None:
    from long_inputs import LONG_INPUTS, MAX_TEXT_LENGTH

    sizes = [len(d['text']) for d in LONG_INPUTS]
    assert all(s > 1500 for s in sizes[:3])
    assert sizes[3] == MAX_TEXT_LENGTH - 500
    assert sizes[4] == MAX_TEXT_LENGTH + 1
