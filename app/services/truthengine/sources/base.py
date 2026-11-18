"""Base class for HTTP JSON sources with a per-source concurrency limit."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..models import Claim, Evidence, QueryPlan


class BaseSource:
    name = 'base'

    def __init__(self, http: httpx.AsyncClient, concurrency: int) -> None:
        self.http = http
        self._sem = asyncio.Semaphore(concurrency)

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        async with self._sem:
            resp = await self.http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    async def search(self, plan: QueryPlan, claim: Claim) -> list[Evidence]:
        raise NotImplementedError
