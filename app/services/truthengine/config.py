"""Settings: API keys, concurrency limits, source tiers and verdict thresholds (tune weights here only)."""

from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()

# --- keys -----------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-sonnet-5-5')
GOOGLE_FACTCHECK_KEY = os.getenv('GOOGLE_FACTCHECK_KEY', '')
OPENALEX_API_KEY = os.getenv('OPENALEX_API_KEY', '')
CONTACT_EMAIL = os.getenv('CONTACT_EMAIL', '')

# --- network --------------------------------------------------------------
HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '20'))
USER_AGENT = f'TruthEngineEdu/0.1 (hackathon; mailto:{CONTACT_EMAIL or "unknown"})'
WEB_SEARCH_TOOL = os.getenv('WEB_SEARCH_TOOL', 'web_search_20250305')

# --- concurrency ----------------------------------------------------------
MAX_PARALLEL_CLAIMS = int(os.getenv('MAX_PARALLEL_CLAIMS', '8'))
MAX_PARALLEL_LLM = int(os.getenv('MAX_PARALLEL_LLM', '8'))
CLAIM_TIMEOUT_S = float(os.getenv('CLAIM_TIMEOUT_S', '65'))  # the API waits 90 s for a whole text
MAX_CLAIMS_PER_TEXT = int(os.getenv('MAX_CLAIMS_PER_TEXT', '10'))
SOURCE_CONCURRENCY = {'openalex': 5, 'crossref': 3, 'wikipedia': 5, 'factcheck': 5}

MAX_EVIDENCE_PER_CLAIM = 12
KNOWLEDGE_TOP_PAGES = 3  # pages of teacher sources retrieved per claim
PUBLIC_API_URL = os.getenv('PUBLIC_API_URL', 'http://localhost:8000').rstrip('/')  # for links to uploaded files
EVIDENCE_CHARS_FOR_JUDGE = 2000

# --- source tiers (base weights are tunable heuristics) --------------------
TIER_BASE = {
    '1A': 1.0,  # computation
    '1B': 0.95,  # official statistics
    '1C': 0.95,  # primary document / the cited work itself
    '1D': 0.90,  # source uploaded by a teacher (knowledge base)
    '2A': 0.95,  # systematic review, meta-analysis
    '2B': 0.90,  # official scientific / health bodies
    '2C': 0.85,  # peer-reviewed primary research (adjusted by study design)
    '3A': 0.80,  # IFCN fact-checkers
    '3B': 0.75,  # academic encyclopedias, open textbooks
    '3C': 0.70,  # Wikidata
    '3D': 0.65,  # Wikipedia
    '4A': 0.60,  # quality news
    '4B': 0.50,  # preprints
    '4C': 0.50,  # organisation / university sites
    '5': 0.25,  # everything else
}
# Max confidence allowed by the best tier level that was found.
TIER_CAP = {1: 1.0, 2: 1.0, 3: 0.85, 4: 0.6, 5: 0.0}

# Ordered: first match wins. Entries starting with "." are suffix rules.
DOMAIN_TIERS: list[tuple[str, tuple[str, ...]]] = [
    ('1B', ('data.worldbank.org', 'datacommons.org', 'data.un.org', 'census.gov', 'armstat.am')),
    ('2A', ('cochranelibrary.com',)),
    (
        '2C',
        (
            'pubmed.ncbi.nlm.nih.gov',
            'ncbi.nlm.nih.gov',
            'nature.com',
            'science.org',
            'thelancet.com',
            'nejm.org',
            'bmj.com',
            'jamanetwork.com',
            'cell.com',
            'plos.org',
            'pnas.org',
            'sciencedirect.com',
            'springer.com',
            'wiley.com',
        ),
    ),
    (
        '2B',
        (
            'who.int',
            'un.org',
            'worldbank.org',
            'oecd.org',
            'europa.eu',
            'cdc.gov',
            'nih.gov',
            'nasa.gov',
            'noaa.gov',
            'nationalacademies.org',
            '.gov',
        ),
    ),
    ('3A', ('snopes.com', 'politifact.com', 'factcheck.org', 'fullfact.org', 'factcheck.afp.com', 'healthfeedback.org', 'leadstories.com')),
    ('3B', ('britannica.com', 'openstax.org', 'plato.stanford.edu', 'khanacademy.org')),
    ('3D', ('wikipedia.org',)),
    ('4A', ('reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'npr.org')),
    ('4B', ('arxiv.org', 'biorxiv.org', 'medrxiv.org')),
    ('4C', ('.edu',)),
]
# Sites that block Claude's search agent: still classified above, but never sent to web search.
WEB_SEARCH_EXCLUDE = {'apnews.com', 'bbc.co.uk', 'bbc.com', 'factcheck.afp.com', 'reuters.com'}
TRUSTED_WEB_DOMAINS = [d for _, ds in DOMAIN_TIERS for d in ds if not d.startswith('.') and d not in WEB_SEARCH_EXCLUDE]

# --- verdict thresholds ---------------------------------------------------
MIN_EVIDENCE_MASS = 0.6  # S + C below this -> unverifiable
CONFIDENCE_MASS_SCALE = 1.2  # mass at which evidence volume stops limiting confidence
SUPPORT_RATIO = 0.75
CONTRA_RATIO = 0.25
MIN_QUOTE_CHARS = 20  # shorter 'quotes' are fragments, not evidence
UNKNOWN_GATE_FACTOR = 0.85  # each "unknown" gate multiplies the weight
STALE_AFTER_YEARS = 3  # only for time-sensitive claims
STALE_FACTOR = 0.6
