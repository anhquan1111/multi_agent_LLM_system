"""LangSmith tracing, enabled by environment variable and off by default.

LangSmith hooks itself into LangGraph implicitly — there is no client object to pass around.
It reads its configuration straight from `os.environ`, which `pydantic-settings` never
populates (AGENTS.md pattern #6), so the values have to be copied across explicitly.

Tracing is optional on purpose: the app must run identically without a LangSmith key, since
the deployed demo has none.
"""

import logging
import os

from src.config import settings

logger = logging.getLogger(__name__)


def enable_tracing() -> bool:
    """Copy tracing settings into os.environ. Returns whether tracing is actually on."""
    if not settings.langchain_tracing_v2:
        return False

    if not settings.langchain_api_key.strip():
        logger.warning("LANGCHAIN_TRACING_V2 is on but LANGCHAIN_API_KEY is empty; tracing off")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    logger.info("LangSmith tracing enabled, project=%s", settings.langchain_project)
    return True
