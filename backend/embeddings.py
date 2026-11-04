"""
embeddings.py — Gemini embedding generation for CropSense AI.

Provides helpers to embed search queries and documents with the
gemini-embedding-2 model. The module loads environment variables from
backend/.env and initialises the Gemini client lazily, so importing it makes
no network request and does not require a valid API key. The client is created
once and reused for subsequent requests.

Error messages are deliberately generic: raw SDK exception text (which can
contain request details or key fragments) is never copied into the exceptions
raised here. Safe, non-sensitive metadata (HTTP status code, failure category)
is exposed through the ``status_code`` and ``reason`` attributes instead.

The SDK exception object itself is also dropped, not just hidden: the handlers
record only safe metadata and the sanitized error is raised *after* the handler
exits, so neither ``__cause__`` nor ``__context__`` points back at the original
exception (``raise ... from None`` alone would still keep it in ``__context__``).
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ── Load .env before reading os.environ ──────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")


# ── Configuration ───────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIM = 768

# gemini-embedding-2 only accepts output dimensionalities in this range.
MIN_EMBEDDING_DIM = 128
MAX_EMBEDDING_DIM = 3072


# ── Public exceptions ─────────────────────────────────────────────────────────

class EmbeddingError(Exception):
    """Raised when an embedding request cannot be completed."""


class MissingApiKeyError(EmbeddingError):
    """Raised when the Gemini API key is not configured."""


class EmptyInputError(EmbeddingError):
    """Raised when the input text is empty or whitespace only."""


class EmbeddingApiError(EmbeddingError):
    """
    Raised when the Gemini embedding request cannot be completed.

    Attributes
    ----------
    reason:
        A short, generic description of the failure category.
    status_code:
        The HTTP status code reported by the SDK, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "unknown-error",
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


class EmptyEmbeddingError(EmbeddingError):
    """Raised when the API returns no embeddings."""


class InvalidEmbeddingError(EmbeddingError):
    """Raised when the configuration or the returned vector is invalid."""


# ── Internal helpers: client ─────────────────────────────────────────────────

# Lazily created and reused across requests. Never populated at import time.
_client = None


def _get_client():
    """
    Return the shared google-genai client, creating it on first use.

    Raises MissingApiKeyError if the key is absent, EmbeddingApiError if the
    SDK cannot be initialised.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. Add it to backend/.env to enable embeddings."
        )

    init_failed = False
    try:
        from google import genai  # type: ignore
        _client = genai.Client(api_key=api_key)
    except Exception:
        # Nothing about the original exception is recorded — SDK init errors can
        # echo configuration details, and the object itself would otherwise stay
        # reachable through the sanitized error's __context__.
        init_failed = True

    if init_failed:
        # Raised outside the except block, so __context__ remains None.
        raise EmbeddingApiError(
            "Could not initialise the Gemini client. Check that the google-genai "
            "package is installed and that GEMINI_API_KEY is set correctly in "
            "backend/.env.",
            reason="client-initialisation-failed",
        ) from None

    return _client


def _reset_client() -> None:
    """
    Discard the cached client.

    Used when the Gemini configuration changes at runtime (mostly by tests);
    the next embedding call builds a fresh client.
    """
    global _client
    _client = None


# ── Internal helpers: configuration ─────────────────────────────────────────

def _model_name() -> str:
    """Return the configured embedding model name, or raise InvalidEmbeddingError."""
    name = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    if not name:
        raise InvalidEmbeddingError(
            "GEMINI_EMBEDDING_MODEL is set but empty. Provide a model name, or "
            "unset it to use the default 'gemini-embedding-2'."
        )
    return name


def _embedding_dim() -> int:
    """Return the configured embedding dimension, or raise InvalidEmbeddingError."""
    raw = os.getenv("GEMINI_EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)).strip()
    try:
        dim = int(raw)
    except ValueError:
        raise InvalidEmbeddingError(
            f"GEMINI_EMBEDDING_DIM must be an integer, got {raw!r}."
        ) from None

    if not MIN_EMBEDDING_DIM <= dim <= MAX_EMBEDDING_DIM:
        raise InvalidEmbeddingError(
            f"GEMINI_EMBEDDING_DIM must be between {MIN_EMBEDDING_DIM} and "
            f"{MAX_EMBEDDING_DIM}, got {dim}."
        )
    return dim


# ── Internal helpers: error sanitisation ─────────────────────────────────────

_STATUS_REASONS = {
    400: "the request was rejected as invalid",
    401: "the API key is not valid for this request",
    403: "access to the embedding model is forbidden for this API key",
    404: "the configured embedding model was not found",
    408: "the Gemini service timed out",
    429: "too many requests; the rate limit was exceeded",
}

_API_FAILURE_ADVICE = (
    "Check GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL in backend/.env, confirm the "
    "embedding service is reachable, and try again."
)


def _safe_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from SDK metadata, ignoring the message text."""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _classify_api_error(exc: BaseException, status_code: int | None) -> str:
    """Return a generic failure description derived only from safe metadata."""
    if status_code is not None:
        if status_code in _STATUS_REASONS:
            return _STATUS_REASONS[status_code]
        if 500 <= status_code < 600:
            return "the Gemini service returned a server error"
        return f"the Gemini service rejected the request (HTTP {status_code})"

    # No status code: fall back to the exception class name, which carries no
    # request data, instead of the message text.
    name = type(exc).__name__.lower()
    if "timeout" in name or "deadline" in name:
        return "the request timed out"
    if any(token in name for token in ("connection", "connect", "network", "dns", "unreachable")):
        return "the Gemini service could not be reached"
    if any(token in name for token in ("valueerror", "typeerror", "keyerror", "attributeerror")):
        return "the embedding response could not be parsed"
    return "an unexpected error occurred while calling the embedding service"


def _api_failure_fields(exc: BaseException) -> tuple[str, int | None]:
    """
    Extract safe classification metadata from a raw SDK exception.

    Only the HTTP status code and the exception *class name* are read — never
    the message text, which can echo request payloads or configuration. The
    result is plain str/int, so nothing keeps the SDK exception object alive.
    """
    status_code = _safe_status_code(exc)
    return _classify_api_error(exc, status_code), status_code


def _api_error(reason: str, status_code: int | None) -> EmbeddingApiError:
    """Build the sanitized EmbeddingApiError that callers actually see."""
    detail = f"{reason} (HTTP {status_code})" if status_code is not None else reason
    return EmbeddingApiError(
        f"Gemini embedding request failed: {detail}. {_API_FAILURE_ADVICE}",
        reason=reason,
        status_code=status_code,
    )


# ── Internal helpers: validation ─────────────────────────────────────────────

def _validate_input(text: str) -> str:
    """Return stripped input or raise EmptyInputError."""
    if text is None:
        raise EmptyInputError("Input text cannot be None.")
    if not isinstance(text, str):
        raise EmptyInputError("Input text must be a string.")
    stripped = text.strip()
    if not stripped:
        raise EmptyInputError("Input text is empty.")
    return stripped


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return the L2-normalized vector, or raise InvalidEmbeddingError."""
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm):
        raise InvalidEmbeddingError(
            "Embedding vector has a non-finite norm and cannot be normalized."
        )
    if norm <= 0.0:
        raise InvalidEmbeddingError(
            "Embedding vector has a zero norm and cannot be normalized."
        )

    normalized = np.asarray(vector / norm, dtype=np.float32)

    if not np.all(np.isfinite(normalized)):
        raise InvalidEmbeddingError(
            "Normalizing the embedding vector produced non-finite values."
        )

    return normalized


def _call_embedding(text: str) -> np.ndarray:
    """
    Send a single text to the Gemini embedding API and return a validated,
    L2-normalised float32 NumPy vector of the configured dimension.
    """
    # Configuration is validated before any client work or network request.
    model = _model_name()
    dim = _embedding_dim()
    client = _get_client()

    failure: tuple[str, int | None] | None = None

    try:
        from google.genai import types  # type: ignore

        response = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=dim),
        )
    except Exception as exc:
        # Only safe metadata is captured. The handler does not raise, so the SDK
        # exception object is released when the block exits and cannot be
        # reached through the sanitized error's __context__.
        failure = _api_failure_fields(exc)

    if failure is not None:
        # Raised outside the except block, so __context__ remains None.
        reason, status_code = failure
        raise _api_error(reason, status_code) from None

    # Validate that exactly one embedding was returned.
    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise EmptyEmbeddingError(
            "Gemini returned no embeddings for the provided input."
        )
    if len(embeddings) != 1:
        raise InvalidEmbeddingError(
            f"Expected exactly one embedding, got {len(embeddings)}."
        )

    values = getattr(embeddings[0], "values", None)
    if values is None or len(values) == 0:
        raise EmptyEmbeddingError("Gemini returned an empty embedding vector.")

    if len(values) != dim:
        raise InvalidEmbeddingError(
            f"Expected embedding dimension {dim}, got {len(values)}."
        )

    vector = np.asarray(values, dtype=np.float32)

    if vector.shape != (dim,):
        raise InvalidEmbeddingError(
            f"Expected embedding shape ({dim},), got {vector.shape}."
        )

    if not np.all(np.isfinite(vector)):
        raise InvalidEmbeddingError("Embedding contains non-finite values.")

    return _l2_normalize(vector)


# ── Public API ───────────────────────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    """
    Embed a search query using the Gemini Embedding 2 query format.

    Parameters
    ----------
    query:
        The user's search query.

    Returns
    -------
    np.ndarray
        A 1-D L2-normalised float32 vector with shape ``(embedding_dim,)``.

    Raises
    ------
    EmptyInputError
        If the query is empty or whitespace only.
    InvalidEmbeddingError
        If GEMINI_EMBEDDING_MODEL / GEMINI_EMBEDDING_DIM are invalid, or the
        returned vector is unusable.
    MissingApiKeyError
        If GEMINI_API_KEY is not set.
    EmbeddingApiError
        If the Gemini request fails.
    EmptyEmbeddingError
        If the response contains no embedding.
    """
    validated = _validate_input(query)
    prepared = f"task: search result | query: {validated}"
    return _call_embedding(prepared)


def embed_document(text: str, title: str | None = None) -> np.ndarray:
    """
    Embed a document using the Gemini Embedding 2 document format.

    Parameters
    ----------
    text:
        The document text to embed.
    title:
        Optional document title. If not provided, ``"none"`` is used.

    Returns
    -------
    np.ndarray
        A 1-D L2-normalised float32 vector with shape ``(embedding_dim,)``.

    Raises
    ------
    EmptyInputError
        If the text is empty or whitespace only.
    InvalidEmbeddingError
        If GEMINI_EMBEDDING_MODEL / GEMINI_EMBEDDING_DIM are invalid, or the
        returned vector is unusable.
    MissingApiKeyError
        If GEMINI_API_KEY is not set.
    EmbeddingApiError
        If the Gemini request fails.
    EmptyEmbeddingError
        If the response contains no embedding.
    """
    validated = _validate_input(text)
    prepared = f"title: {title or 'none'} | text: {validated}"
    return _call_embedding(prepared)
