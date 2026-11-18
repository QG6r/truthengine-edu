# TruthEngine Edu

Hackathon project for **AI for Good Vol2** (challenge: *TruthEngine for EdTech: Multi-Modal Cognitive Defense*).

Students learn from AI chatbots, YouTube and TikTok, and much of that content sounds confident but is wrong:
invented facts, fake references, edited images, deepfake "professors". TruthEngine Edu takes a text (later also
images, audio and video), splits it into atomic factual claims and checks each claim against **trusted open
sources**. For every claim it returns a verdict, an explainable confidence score, the sources with verbatim
quotes, and the kind of trick that was used, so students learn to spot it themselves.

Teachers can also upload documents they trust (textbooks, lecture notes, papers); claims are then checked against them
too. It does **not** claim to detect deepfakes: media checks are reported as risk signals.

## How a claim is checked

```
text -> Separator (atomic, self-contained claims, context kept)
     -> Planner        claim type, neutral / supporting / refuting queries, explicit citation
     -> Sources        all queried concurrently (asyncio)
     -> Judge          stance of each source + verbatim quote (LLM reads ONLY fetched text)
     -> Gates          evidence is real, quote exists, relevant, fresh, not retracted
     -> Scoring        weight = tier x directness x scope x freshness x gates
     -> Verdict        supported / contradicted / disputed / unverifiable + confidence + explanation
```

### Source tiers (most to least reliable)

| Tier | Source | Base weight | Why it is trusted |
|---|---|---|---|
| 1A | Computation (planned) | 1.0 | Reproducible, no trust in a source needed |
| 1B | Official statistics (World Bank, Data Commons, national statistics) | 0.95 | Mandated collection, published methodology |
| 1C | Primary document / the cited work itself | 0.95 | The original, not a retelling |
| 1D | Document uploaded by a teacher (knowledge base) | 0.90 | Chosen by the person responsible for the course |
| 2A | Systematic reviews, meta-analyses | 0.95 | Synthesise many studies with explicit criteria |
| 2B | Official scientific / health bodies (WHO, CDC, NIH, academies) | 0.90 | Expert consensus with a public process |
| 2C | Peer-reviewed research (RCT 0.9, cohort 0.8, other 0.75) | 0.75-0.9 | Independent methodological review |
| 3A | IFCN fact-checkers | 0.80 | Public methodology, sources, corrections policy |
| 3B | Academic encyclopedias, open textbooks | 0.75 | Expert authors, editorial review |
| 3C / 3D | Wikidata / Wikipedia | 0.70 / 0.65 | Referenced, but quality varies per article |
| 4A / 4B / 4C | Quality news / preprints / organisation sites | 0.6 / 0.5 / 0.5 | Editorial control, or not yet peer reviewed |
| 5 | Blogs, social media, unknown sites | 0.25 | Shows a claim circulates, never that it is true |
| - | The LLM's own knowledge | **0** | The model reasons, it is never a source |

The best tier found caps the confidence (tiers 1-2: 1.0, tier 3: 0.85, tier 4: 0.6, tier 5 only: unverifiable).
All weights are tunable heuristics in `app/services/truthengine/config.py`.

### Gates (pass / fail / unknown)

Universal: **U1** evidence was really fetched, **U2** the quote exists verbatim in the fetched text,
**U3** the date fits time-sensitive claims, **U4** the source is about the same subject and scope.
Source-specific: **G1** a cited work exists (Crossref: authors, year, venue), **G2** it is not retracted (OpenAlex),
venue known, known fact-checker, Wikipedia footnotes not yet followed. `fail` sets the weight to 0, `unknown` multiplies it by 0.85.

### Aggregation

`S` = summed weight of supporting evidence, `C` = contradicting; copies from the same origin count once.
`S + C < 0.6` -> unverifiable. Otherwise `ratio = S / (S + C)`: >= 0.75 supported, <= 0.25 contradicted, else disputed.
`confidence = min(1, (S + C) / 1.2) * max(ratio, 1 - ratio)`, capped by tier. `score = 0.5 + (ratio - 0.5) * confidence`.

## Code

```
app/services/
├── separator.py          Separator.separate(text) -> list[str]
├── trusted_sources.py    CheckTrustedSources: check / run (sync), *_async, run_detailed
└── truthengine/
    ├── config.py         keys, tiers, trusted domains, thresholds
    ├── models.py         Claim, QueryPlan, Evidence, GateResult, ClaimResult
    ├── llm.py            async Claude over httpx (no SDK): forced-JSON tool call + whitelisted web search, retries
    ├── planner.py        claim type, queries, citation extraction
    ├── judge.py          stance + verbatim quote per source
    ├── gates.py          U1-U4
    ├── scoring.py        evidence weight + aggregation
    ├── engine.py         TruthEngine (async pipeline)
    └── sources/          openalex, crossref, wikipedia, factcheck, websearch
scripts/demo.py           live smoke test on 10 sample claims
```

`CheckTrustedSources.check(claim) -> float` (0..1). `run(claims)` checks all claims concurrently and keeps input order.
`run_detailed` returns verdict, confidence, evidence with quotes and gates, explanation and flags (e.g. `invented_source`).
From async code (FastAPI) use `check_async` / `run_async` / `run_detailed_async`.

## Run the demo (backend + front end)

```bash
cp .env.example .env               # set ANTHROPIC_API_KEY (Google Fact Check / OpenAlex keys optional)
uv sync
uv run python scripts/demo.py           # live check of 10 sample claims (engine only)
uv run uvicorn api:app --port 8000      # backend:   http://localhost:8000/docs
uv run streamlit run app.py             # front end: http://localhost:8501
```

The front end calls `POST {API_URL}/analyze` over HTTP only (`API_URL` env var, default `http://localhost:8000`).

| file | role |
|---|---|
| `api.py` | `POST /analyze {"text"}` -> `Report`, `GET /health` |
| `analyzer.py` | runs `pipeline.run_pipeline` with fallback to mock data |
| `schemas.py` | `Report` contract shared by backend, front end and pipeline (don't change field names) |
| `mock_data.py` | `mock_report(text)` fallback and `DEMO_INPUTS` (4 examples) |
| `app.py` | Streamlit UI |

**Fallback:** if `pipeline.py` is missing, `run_pipeline` raises, returns something that isn't a valid `Report`, or takes longer than 180 s, the API returns `mock_report(text)` with header `X-Demo-Data: true` (the UI shows a "demo data" badge) and logs the reason. The API never returns 500. Empty text or more than 20 000 chars returns 422 with a readable `detail`.


### Plugging in the pipeline

`pipeline.py` (repo root) wires the components together:

```
text -> Separator.separate_detailed_async -> CheckTrustedSources.run_detailed_async -> build_report -> Report
```

- Needs `ANTHROPIC_API_KEY` and `CLAUDE_MODEL` in `.env` (see `.env.example`). The same model is passed to the Separator; set `ANTHROPIC_MODEL` to override it there.
- `trust_score` is the mean score of checked (non-unverifiable) claims, and `coverage` is "N of M claims checked".
- Evidence includes only items the engine counted (weight > 0). `quote_verified` comes from the engine's `U2_quote` gate, and `source_type` from the source tier.
- Citations come from separator claims that carry a citation, one per cited work. `invented_source` maps to `not_found`, and `retracted_source` to `retracted`.
- Tricks are separator non-claims of kind `rhetoric`.
- Typical latency is 20-60 s per text, so the API timeout is 180 s and the UI's is 200 s.

To replace it, keep the contract:

```python
from schemas import Report


async def run_pipeline(text: str) -> Report:   # plain `def` also works
    ...
```

- Return a `Report`, or a dict with the same shape. It's validated; invalid output falls back to the mock.
- `Verdict.claim_text` should be copied from the input text so the UI can highlight it (case-insensitive match).
- `Trick.phrase` should be the exact phrase from the input text.
- A sync `run_pipeline` runs in a worker thread. After the 180 s timeout the API stops waiting, but the thread isn't killed.
- Restart uvicorn (or use `--reload`) after adding `pipeline.py`. Check the uvicorn log for "returning mock report" lines to see why a fallback happened.

## Tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Teacher knowledge base (RAG)

Teachers upload trusted documents in the UI ("📚 Trusted sources (for teachers)") or through the API:

| endpoint | what it does |
|---|---|
| `POST /sources` (multipart `file`, optional `title`) | store a PDF / TXT / MD (max 20 MB), split it into ~700-character passages with page numbers |
| `GET /sources` | list uploaded sources |
| `DELETE /sources/{id}` | remove a source and its passages |
| `GET /sources/{id}/file` | the original file (evidence links open it at `#page=N`) |

For every claim the engine retrieves the best passages (BM25, up to 3 pages) at the same time as it queries the open
sources. A passage is only a candidate: the judge must find a verbatim quote in it, the quote is verified in code, and the
result is weighted as tier **1D** (0.90). So a teacher's document outweighs Wikipedia, while strong open evidence against it
yields "disputed" instead of blind trust. Files live in `data/knowledge/` (git-ignored; set `KNOWLEDGE_DIR` to move it).
Set `PUBLIC_API_URL` if the API is not on `http://localhost:8000`, so evidence links point to the right host.

Retrieval is keyword-based (pure Python, no vector database: native DLLs are blocked on some of our Windows laptops), so
it does not match synonyms or other languages. Uploads are not authenticated: fine for the demo, not for production.

## Evaluation

```bash
uv run python eval/run_eval.py     # prints a table, writes eval/report.json
```

| file | rows | content |
|---|---|---|
| `eval/separator.jsonl` | 15 | text -> expected claims (EN/RU; compound, pronouns, opinions, citations, dates, non-factual) |
| `eval/checker.jsonl` | to do | claim -> expected score / label, checked against public sources (e.g. SciFact, FEVER samples) |
| `eval/end_to_end.jsonl` | to do | demo scenarios: text -> claims -> scores |

Metrics (each printed with its `n`):
- **Separator:** claim-count MAE, exact-count rate, recall (token-overlap Dice >= 0.6). Unmatched pairs are printed for manual review.
- **Checker:** 3-class accuracy (score >= 0.75 supported, <= 0.25 refuted), confusion matrix, **refuted precision** (main metric).
  The optional `--pdf` quote check is legacy from the earlier lecture-only design and is skipped when no PDF is present.
- **End-to-end:** claim recall, label agreement on matched claims, scenarios fully correct.

Gold data is written by hand. Don't generate labels with the model the system uses.

## Roadmap

1. Phase 1 (done): async evidence engine, tiers, gates, aggregation.
2. Phase 2: Separator (LLM, keeps qualifiers / attribution / scope, layer-0 extraction check), FastAPI `/check`, UI with highlighted claims.
3. Phase 3: computation (1A), official statistics (1B), Wikidata, manipulation-trick classifier.
4. Phase 4: images (vision, EXIF, C2PA, AI-generation signals), audio/video via transcripts.


## Claude claim separator

`Separator().separate(text)` now returns atomic claim texts using the preserved
TruthEngine Edu claim-separator prompt. Set `ANTHROPIC_API_KEY` and either
`ANTHROPIC_MODEL` or the existing `CLAUDE_MODEL` in your environment or `.env`.
`ANTHROPIC_MODEL` takes precedence. No credentials are committed.

`separate_detailed(text)` returns a typed report containing claims, non-claims,
skipped passages, verbatim source spans, and coverage. Async callers should use
`await separate_async(text)` or `await separate_detailed_async(text)`; synchronous
wrappers must be called outside a running event loop. Network I/O uses AsyncClient.

The original input is sent to Claude as source text. JSON output is validated
with Pydantic; invented spans, missing source coverage, invalid dependencies,
and duplicate IDs are rejected. Invalid or incomplete output is retried once.
The system prompt is unchanged on retry. HTTP errors are explicit and are not
retried. Empty input needs no credentials. Limits: 24,000 input characters,
8,192 response tokens, and 90 seconds per HTTP request. Long responses can fail
if truncated. No silent fallback or long-document chunking is included.

Span presence and character coverage do not prove factual truth, faithful
paraphrasing, or semantic completeness. Live review of the user-authored prompt
is still needed. `separate()` omits non-claims from its list; use the detailed
report for full source accounting. Tests use mocked HTTP responses, not paid API calls.
