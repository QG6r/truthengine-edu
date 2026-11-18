"""TruthEngine: checks already-extracted claims against trusted open sources."""

from .engine import TruthEngine
from .models import Claim, ClaimResult, Evidence, GateResult


__all__ = ['Claim', 'ClaimResult', 'Evidence', 'GateResult', 'TruthEngine']
