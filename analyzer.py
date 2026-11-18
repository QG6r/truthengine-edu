"""Runs the team's pipeline (pipeline.run_pipeline) with a mock fallback. Used by api.py."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from mock_data import mock_report
from schemas import Report


logger = logging.getLogger(__name__)

PIPELINE_TIMEOUT_S = 180.0


@dataclass(frozen=True)
class AnalysisResult:
    report: Report
    demo_data: bool


class Analyzer:
    def __init__(self, timeout_s: float = PIPELINE_TIMEOUT_S, module_name: str = 'pipeline') -> None:
        self.timeout_s = timeout_s
        self.module_name = module_name

    async def analyze(self, text: str) -> AnalysisResult:
        """Run the pipeline; on missing module, error, invalid output or timeout return the mock report."""
        try:
            report = await asyncio.wait_for(self._run_pipeline(text), timeout=self.timeout_s)
        except TimeoutError:
            # A sync pipeline keeps running in its worker thread; we just stop waiting for it.
            logger.warning('pipeline timed out after %.0fs, returning mock report', self.timeout_s)
        except ModuleNotFoundError as exc:
            if exc.name != self.module_name:
                logger.exception('pipeline import failed, returning mock report')
            else:
                logger.warning('%s.py not found, returning mock report', self.module_name)
        except Exception:
            logger.exception('pipeline failed, returning mock report')
        else:
            return AnalysisResult(report=report, demo_data=False)
        return AnalysisResult(report=mock_report(text), demo_data=True)

    async def _run_pipeline(self, text: str) -> Report:
        run_pipeline = importlib.import_module(self.module_name).run_pipeline
        if inspect.iscoroutinefunction(run_pipeline):
            result = await run_pipeline(text)
        else:
            result = await asyncio.to_thread(run_pipeline, text)
        # Accept a Report, any pydantic model with the same shape, or a plain dict.
        return Report.model_validate(result.model_dump() if isinstance(result, BaseModel) else result)
