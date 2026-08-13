"""LLM client with retry on transient failures.

Retries rate limits and timeouts, never validation errors — retrying a malformed request
produces the same malformed request and costs money twice. The key is passed explicitly
because pydantic-settings does not populate `os.environ` (AGENTS.md pattern #6).
"""

from functools import lru_cache

from google.genai.errors import ClientError, ServerError
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def is_transient(exc: BaseException) -> bool:
    """Rate limits, 5xx and timeouts are worth retrying. Nothing else is.

    A 400 means the request itself is malformed — sending the identical request again
    produces the identical error and bills you twice.
    """
    if isinstance(exc, ServerError | TimeoutError):
        return True
    return isinstance(exc, ClientError) and getattr(exc, "code", None) == 429


with_retry = retry(
    retry=retry_if_exception(is_transient),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_attempt(4),
    reraise=True,
)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """One shared client. Cached so each node does not rebuild it per request."""
    from src.config import settings

    if not settings.has_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set — check .env or the host's env vars.")
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        timeout=settings.request_timeout_seconds,
        max_retries=0,  # retries are handled by `with_retry`, not silently by the SDK
    )
