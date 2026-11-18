from typing import Literal
from pydantic import BaseModel

class Evidence(BaseModel):
    url: str
    domain: str
    source_type: str          # peer_reviewed, review, preprint, official, encyclopedia, news, other
    title: str
    quote: str                # verbatim quote from the source
    quote_verified: bool      # code found the quote in the page text
    published_date: str | None = None
    retracted: bool = False
    stance: Literal["supports", "contradicts", "neutral"]

class Trick(BaseModel):
    name: str
    phrase: str               # exact phrase from the input text
    explanation: str

class Verdict(BaseModel):
    claim_id: str
    claim_text: str
    verdict: Literal["supported", "contradicted", "disputed", "unverifiable"]
    confidence: float         # 0..1
    explanation: str
    correction: str | None = None
    evidence: list[Evidence] = []
    critical_question: str | None = None

class CitationCheck(BaseModel):
    raw: str
    status: Literal["found", "not_found", "metadata_mismatch", "retracted"]
    closest_match: str | None = None
    url: str | None = None

class Report(BaseModel):
    trust_score: float | None # 0..1, None if nothing verifiable
    coverage: str             # e.g. "7 из 10 утверждений проверено"
    verdicts: list[Verdict]
    citations: list[CitationCheck] = []
    tricks: list[Trick] = []
