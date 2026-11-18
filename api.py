"""TruthEngine Edu backend. Run: uvicorn api:app --port 8000"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from analyzer import Analyzer
from app.services.knowledge import KnowledgeError, KnowledgeStore, SourceInfo, get_store
from app.services.knowledge.store import MAX_FILE_BYTES
from schemas import Report


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

MAX_TEXT_LENGTH = 20_000
DEMO_HEADER = 'X-Demo-Data'


class AnalyzeRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def _check_length(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Text is empty: paste a text to check.')
        if len(value) > MAX_TEXT_LENGTH:
            raise ValueError(f'Text is too long: {len(value)} characters, maximum is {MAX_TEXT_LENGTH}.')
        return value


app = FastAPI(title='TruthEngine Edu')
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r'https?://(localhost|127\.0\.0\.1)(:\d+)?',
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=[DEMO_HEADER],
)

_analyzer = Analyzer()


def get_analyzer() -> Analyzer:
    return _analyzer


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with a single readable message instead of pydantic's error list."""
    messages = [str(err['msg']).removeprefix('Value error, ') for err in exc.errors()]
    return JSONResponse(status_code=422, content={'detail': ' '.join(messages)})


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/analyze', response_model=Report)
async def analyze(body: AnalyzeRequest, response: Response, analyzer: Analyzer = Depends(get_analyzer)) -> Report:
    result = await analyzer.analyze(body.text)
    if result.demo_data:
        response.headers[DEMO_HEADER] = 'true'
    return result.report


# --- teacher knowledge base (RAG) ------------------------------------------------------


@app.get('/sources', response_model=list[SourceInfo])
async def list_sources(store: KnowledgeStore = Depends(get_store)) -> list[SourceInfo]:
    """Trusted sources uploaded by teachers."""
    return store.list_sources()


@app.post('/sources', response_model=SourceInfo, status_code=201)
async def upload_source(
    file: UploadFile, title: Annotated[str | None, Form()] = None, store: KnowledgeStore = Depends(get_store)
) -> SourceInfo:
    """Upload a PDF / TXT / MD file; it is chunked and becomes searchable evidence (tier 1D) at once."""
    data = await file.read(MAX_FILE_BYTES + 1)
    try:
        return await asyncio.to_thread(store.add, file.filename or '', data, title)
    except KnowledgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete('/sources/{source_id}', status_code=204)
async def delete_source(source_id: str, store: KnowledgeStore = Depends(get_store)) -> Response:
    try:
        store.delete(source_id)
    except KnowledgeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get('/sources/{source_id}/file')
async def download_source(source_id: str, store: KnowledgeStore = Depends(get_store)) -> FileResponse:
    """The original file, so evidence links open the document (PDF links carry #page=N)."""
    try:
        source, path = store.get(source_id), store.file_path(source_id)
    except KnowledgeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media = 'application/pdf' if path.suffix == '.pdf' else 'text/plain; charset=utf-8'
    return FileResponse(path, media_type=media, filename=source.filename, content_disposition_type='inline')
