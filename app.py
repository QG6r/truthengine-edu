"""TruthEngine Edu front end. Run: streamlit run app.py

Talks to the backend only over HTTP (POST {API_URL}/analyze).
"""

from __future__ import annotations

import html
import os
import re
from urllib.parse import quote

import requests
import streamlit as st

from long_inputs import LONG_INPUTS
from mock_data import DEMO_INPUTS
from schemas import Evidence, Report, Verdict


API_URL = os.environ.get('API_URL', 'http://localhost:8000').rstrip('/')
REQUEST_TIMEOUT_S = 200  # must exceed the backend pipeline timeout (180 s) so the mock fallback arrives
SOURCES_TIMEOUT_S = 60
EXAMPLES = DEMO_INPUTS + LONG_INPUTS

# verdict -> (icon, label, highlight color)
VERDICTS = {
    'supported': ('✅', 'Supported', '#b7eb8f'),
    'contradicted': ('❌', 'Contradicted', '#ffa39e'),
    'disputed': ('⚠️', 'Disputed', '#ffe58f'),
    'unverifiable': ('❔', 'Not enough data', '#d9d9d9'),
}
CITATION_STATUS = {
    'found': '✅ found',
    'not_found': '❌ not found in Crossref or OpenAlex',
    'metadata_mismatch': '⚠️ metadata mismatch',
    'retracted': '❌ article retracted',
}


class ApiError(Exception):
    """Error with a short user-facing message."""


def analyze_via_api(text: str) -> tuple[Report, bool]:
    """Returns (report, is_demo_data). Raises ApiError with a short message."""
    try:
        response = requests.post(f'{API_URL}/analyze', json={'text': text}, timeout=REQUEST_TIMEOUT_S)
    except requests.Timeout as exc:
        raise ApiError(f'The server did not respond within {REQUEST_TIMEOUT_S} seconds.') from exc
    except requests.RequestException as exc:
        raise ApiError(f'The analysis server is unavailable ({API_URL}).') from exc
    if response.status_code == 422:
        try:
            detail = response.json().get('detail')
        except ValueError:
            detail = None
        raise ApiError(detail if isinstance(detail, str) else 'The text did not pass validation.')
    if not response.ok:
        raise ApiError(f'The server returned an error (code {response.status_code}).')
    try:
        report = Report.model_validate(response.json())
    except ValueError as exc:  # bad JSON or schema mismatch
        raise ApiError('The server returned a response in an unexpected format.') from exc
    return report, 'X-Demo-Data' in response.headers


def _api_error(response: requests.Response, fallback: str) -> ApiError:
    try:
        detail = response.json().get('detail')
    except ValueError:
        detail = None
    return ApiError(detail if isinstance(detail, str) else fallback)


def _sources_request(method: str, path: str, **kwargs: object) -> requests.Response:
    try:
        return requests.request(method, f'{API_URL}{path}', timeout=SOURCES_TIMEOUT_S, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f'The analysis server is unavailable ({API_URL}).') from exc


def list_sources_via_api() -> list[dict]:
    """Trusted sources uploaded by teachers: [{id, title, filename, pages, chunks, uploaded_at, size_bytes}]."""
    response = _sources_request('GET', '/sources')
    if not response.ok:
        raise _api_error(response, f'Could not load the sources (code {response.status_code}).')
    return response.json()


def upload_source_via_api(filename: str, data: bytes, title: str | None = None) -> dict:
    response = _sources_request('POST', '/sources', files={'file': (filename, data)}, data={'title': title} if title else None)
    if not response.ok:
        raise _api_error(response, f'The upload failed (code {response.status_code}).')
    return response.json()


def delete_source_via_api(source_id: str) -> None:
    response = _sources_request('DELETE', f'/sources/{source_id}')
    if not response.ok:
        raise _api_error(response, f'Could not delete the source (code {response.status_code}).')


_MD_SPECIAL = re.compile(r'([\\`*_{}\[\]()<>#+\-.!|:$~])')


def md(text: str | None) -> str:
    """Escape Markdown control chars in untrusted text (pipeline output, user input, web pages)."""
    return _MD_SPECIAL.sub(r'\\\1', text or '')


def safe_url(url: str | None) -> str | None:
    """Only http(s) links; chars that could break out of a Markdown link are percent-encoded."""
    if not url or not re.match(r'https?://', url, re.IGNORECASE):
        return None
    return quote(url, safe=":/?#@!$&'*+,;=%~-._")  # encodes ( ) < > [ ] spaces and quotes


def highlight_html(text: str, verdicts: list[Verdict]) -> str:
    """Escape text, then wrap each claim (case-insensitive) in <mark> colored by verdict; skip missing/overlapping."""
    escaped = html.escape(text)
    spans = []
    for verdict in verdicts:
        words = html.escape(verdict.claim_text.strip().rstrip('.!?…')).split()
        match = re.search(r'\s+'.join(map(re.escape, words)), escaped, re.IGNORECASE) if words else None
        if match:
            spans.append((match.start(), match.end(), verdict.verdict))
    parts, pos = [], 0
    for start, end, kind in sorted(spans):
        if start < pos:
            continue
        parts += [escaped[pos:start], f'<mark style="background:{VERDICTS[kind][2]};color:#000">{escaped[start:end]}</mark>']
        pos = end
    parts.append(escaped[pos:])
    return ''.join(parts)


def render_summary(report: Report, demo_data: bool) -> None:
    trust = 'no data' if report.trust_score is None else f'{report.trust_score:.0%}'
    col1, col2, col3 = st.columns(3)
    col1.metric('Trust score', trust)
    col2.metric('Coverage', report.coverage)
    col3.metric('Tricks', len(report.tricks))
    if demo_data:
        st.caption('⚠️ demo data')


def render_highlighted_text(text: str, report: Report) -> None:
    st.markdown(f'<div style="white-space:pre-wrap;line-height:1.7">{highlight_html(text, report.verdicts)}</div>', unsafe_allow_html=True)


def render_evidence(evidence: Evidence) -> None:
    url = safe_url(evidence.url)
    title = f'[{md(evidence.title)}]({url})' if url else md(evidence.title)
    checked = '✅ quote verified' if evidence.quote_verified else '❌ quote not found in source'
    retracted = ' · ❌ **article retracted**' if evidence.retracted else ''
    meta = ' · '.join(md(p) for p in (evidence.domain, evidence.source_type, evidence.published_date) if p)
    st.markdown(f'{title}  \n{meta} · {checked}{retracted}')
    st.caption(f'"{md(evidence.quote)}"')


def render_verdict_card(verdict: Verdict) -> None:
    icon, label, _ = VERDICTS[verdict.verdict]
    with st.expander(f'{icon} {label}: {md(verdict.claim_text)}'):
        st.markdown(md(verdict.explanation))
        if verdict.correction:
            st.success(f'Correction: {md(verdict.correction)}')
        st.progress(max(0.0, min(1.0, verdict.confidence)), text=f'Confidence: {verdict.confidence:.0%}')
        for evidence in verdict.evidence:
            render_evidence(evidence)
        if not verdict.evidence:
            st.warning('Not enough data: no reliable sources found.')
        if verdict.critical_question:
            st.info(md(verdict.critical_question))


def render_citations(report: Report) -> None:
    if report.citations:
        st.subheader('Citations')
        for c in report.citations:
            closest = f' · closest match: {md(c.closest_match)}' if c.closest_match else ''
            st.markdown(f'**{md(c.raw)}**: {CITATION_STATUS[c.status]}{closest}')


def render_tricks(report: Report) -> None:
    if report.tricks:
        st.subheader('Manipulation tricks')
        for t in report.tricks:
            st.markdown(f'**{md(t.name)}**: "{md(t.phrase)}"  \n{md(t.explanation)}')


def render_sources_panel() -> None:
    """Teacher knowledge base: upload trusted documents; claims are then checked against them too."""
    with st.expander('📚 Trusted sources (for teachers)'):
        st.caption(
            'Upload textbooks, lecture notes or papers you trust (PDF, TXT, MD). Claims are checked against them '
            'together with the open sources, and matching passages are cited with the page number.'
        )
        uploaded = st.file_uploader('Document', type=['pdf', 'txt', 'md'], key='source_file')
        title = st.text_input('Title (optional)', key='source_title', placeholder='e.g. Biology 101, chapter 3')
        if st.button('Add to trusted sources', disabled=uploaded is None):
            try:
                with st.spinner('Reading and indexing the document…'):
                    info = upload_source_via_api(uploaded.name, uploaded.getvalue(), title.strip() or None)
                st.success(f'Added "{info["title"]}": {info["pages"]} pages, {info["chunks"]} passages indexed.')
            except ApiError as exc:
                st.error(str(exc))

        try:
            sources = list_sources_via_api()
        except ApiError as exc:
            st.warning(str(exc))
            return
        if not sources:
            st.info('No trusted sources yet. Checks use open sources only.')
            return
        st.markdown(f'**{len(sources)} trusted source(s)**')
        for source in sources:
            col_info, col_delete = st.columns([5, 1])
            link = safe_url(f'{API_URL}/sources/{source["id"]}/file')
            col_info.markdown(
                f'[{md(source["title"])}]({link})  \n'
                f'{md(source["filename"])} · {source["pages"]} pages · {source["chunks"]} passages · {md(source["uploaded_at"][:10])}'
            )
            if col_delete.button('Delete', key=f'delete_{source["id"]}'):
                try:
                    delete_source_via_api(source['id'])
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))


def _fill_example() -> None:
    titles = [d['title'] for d in EXAMPLES]
    if st.session_state.example in titles:
        st.session_state.input_text = EXAMPLES[titles.index(st.session_state.example)]['text']


def main() -> None:
    st.set_page_config(page_title='TruthEngine Edu', page_icon='🔎')
    st.title('TruthEngine Edu')
    st.caption('Evidence instead of a verdict')
    render_sources_panel()

    st.selectbox('Examples', ['—'] + [d['title'] for d in EXAMPLES], key='example', on_change=_fill_example)
    st.text_area('Text to check', key='input_text', height=160)
    st.caption(f'{len(st.session_state.get("input_text", "")):,} / 20,000 characters')

    if st.button('Check', type='primary'):
        text = st.session_state.input_text
        try:
            with st.spinner('Checking claims and sources…'):
                report, demo_data = analyze_via_api(text)
            st.session_state.result = (text, report, demo_data)
        except ApiError as exc:
            st.error(str(exc))

    if 'result' in st.session_state:
        text, report, demo_data = st.session_state.result
        st.divider()
        render_summary(report, demo_data)
        render_highlighted_text(text, report)
        for verdict in report.verdicts:
            render_verdict_card(verdict)
        render_citations(report)
        render_tricks(report)

    st.divider()
    st.caption('The system shows signals and sources, not a verdict. Authorship of the text is not checked.')


if __name__ == '__main__':
    main()
