"""
vision.py — Gemini vision analysis for CropSense AI.

Reads GEMINI_API_KEY and GEMINI_MODEL from the environment (set via .env).
The file must NOT be imported at module level if the key is absent; callers
should catch VisionUnavailableError and fall back to text-only mode.
"""

import os
from pathlib import Path

# ── Load .env before reading os.environ ──────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Single source of truth for the fallback model: vision and text generation
# deliberately share one GEMINI_MODEL, so the default is defined once.
from generator import DEFAULT_GENERATION_MODEL

# ── Public exceptions ─────────────────────────────────────────────────────────

class VisionUnavailableError(Exception):
    """
    Raised when vision analysis cannot run (missing key, bad image, SDK failure).

    The message is always safe to surface: it never contains raw SDK text,
    request headers, URLs or credentials. ``reason`` carries a short failure
    category and ``status_code`` the HTTP status reported by the SDK, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "unavailable",
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


# ── Internal helpers ──────────────────────────────────────────────────────────

# Safe, fixed descriptions keyed by HTTP status. Derived only from the status
# code, never from the SDK's message text.
_SAFE_STATUS_REASONS = {
    400: "the image request was rejected as invalid",
    401: "the API key is not valid for this request",
    403: "access to the vision model is forbidden for this API key",
    404: "the configured vision model was not found",
    408: "the Gemini service timed out",
    429: "too many requests; the rate limit was exceeded",
}


def _safe_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from SDK metadata, ignoring the message text."""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _safe_reason(exc: BaseException, status_code: int | None) -> str:
    """Return a generic failure description derived only from safe metadata."""
    if status_code is not None:
        if status_code in _SAFE_STATUS_REASONS:
            return _SAFE_STATUS_REASONS[status_code]
        if 500 <= status_code < 600:
            return "the Gemini service returned a server error"
        return f"the Gemini service rejected the request (HTTP {status_code})"

    name = type(exc).__name__.lower()
    if "timeout" in name or "deadline" in name:
        return "the request timed out"
    if any(token in name for token in ("connection", "connect", "network", "dns", "unreachable")):
        return "the Gemini service could not be reached"
    if any(token in name for token in ("valueerror", "typeerror", "keyerror", "attributeerror")):
        return "the vision response could not be parsed"
    return "an unexpected error occurred while calling the vision service"


def _vision_failure(exc: BaseException, *, stage: str) -> VisionUnavailableError:
    """
    Build a sanitized VisionUnavailableError from a raw SDK exception.

    Only the status code and the exception class name are consulted. The SDK's
    message is deliberately discarded: it can echo request URLs, headers and
    partial credentials, so it must never reach a caller or the frontend.
    """
    status_code = _safe_status_code(exc)
    reason = _safe_reason(exc, status_code)
    detail = f"{reason} (HTTP {status_code})" if status_code is not None else reason
    return VisionUnavailableError(
        f"Gemini vision {stage} failed: {detail}.",
        reason=reason,
        status_code=status_code,
    )


def _get_client():
    """Return a configured google-genai client or raise VisionUnavailableError."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise VisionUnavailableError(
            "GEMINI_API_KEY is not set. "
            "Add it to backend/.env to enable image analysis.",
            reason="missing-api-key",
        )
    # google-genai >= 0.5 uses google.genai.Client
    try:
        from google import genai  # type: ignore
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise _vision_failure(exc, stage="client initialisation") from None


_SUPPORTED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif",
    "image/heic", "image/heif",
}

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_plant_image(
    image_bytes: bytes,
    mime_type: str,
    user_text: str,
    context_documents: list[dict],
) -> str:
    """
    Send an image + optional text question to Gemini and return the reply.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the uploaded image.
    mime_type:
        MIME type reported by the upload (e.g. ``"image/jpeg"``).
    user_text:
        The text message the user sent alongside the image.
        May be ``"(Image attached)"`` when the user sent only the image.
    context_documents:
        Passages retrieved from the local agricultural guide.
        Included in the prompt as supporting context.

    Returns
    -------
    str
        The model's reply text.

    Raises
    ------
    VisionUnavailableError
        If the API key is missing, the image is invalid, or the Gemini
        call fails.
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not image_bytes:
        raise VisionUnavailableError("No image data received.")

    if len(image_bytes) > _MAX_BYTES:
        raise VisionUnavailableError(
            f"Image is too large ({len(image_bytes) // 1024} KB). "
            "Maximum allowed size is 10 MB."
        )

    mime_type_clean = (mime_type or "").split(";")[0].strip().lower()
    if mime_type_clean not in _SUPPORTED_MIME_TYPES:
        raise VisionUnavailableError(
            f"Unsupported image type '{mime_type_clean}'. "
            f"Supported types: {', '.join(sorted(_SUPPORTED_MIME_TYPES))}."
        )

    # ── Build prompt ──────────────────────────────────────────────────────────
    context_text = ""
    if context_documents:
        context_text = (
            "\n\n--- Agricultural guide context (for reference) ---\n"
            + "\n\n".join(doc["text"] for doc in context_documents)
            + "\n--- End of context ---"
        )

    prompt = (
        "You are CropSense AI, an agricultural assistant that helps farmers "
        "identify potential crop diseases and conditions from plant images.\n\n"
        "The user has uploaded an image of a plant or leaf. "
        "Please describe what you can observe in the image: leaf colour, "
        "visible spots, lesions, wilting, discolouration, or other abnormalities.\n\n"
        "Based on your visual observations, suggest what conditions or diseases "
        "might match the symptoms. Be specific about what you see, but do NOT "
        "claim a confirmed diagnosis from a single photo — always recommend "
        "physical inspection and professional guidance for important decisions."
        + context_text
        + (
            f"\n\nUser's question: {user_text}"
            if user_text and user_text.strip() not in ("", "(Image attached)")
            else ""
        )
        + "\n\nPlease provide a clear, helpful response for a farmer."
    )

    # ── Call the Gemini API ───────────────────────────────────────────────────
    client = _get_client()
    # Same resolution rule as generator.py: an explicit GEMINI_MODEL always wins,
    # and a missing or blank value falls back to the current working model rather
    # than the retired gemini-2.5-flash-lite.
    model_name = (os.getenv("GEMINI_MODEL") or "").strip() or DEFAULT_GENERATION_MODEL

    try:
        from google.genai import types as genai_types  # type: ignore

        response = client.models.generate_content(
            model=model_name,
            contents=[
                genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type_clean,
                ),
                prompt,
            ],
        )
        text = (response.text or "").strip()
        if not text:
            # An empty result is a failure, not a silent success: the caller must
            # show its fallback instead of presenting nothing as an assessment.
            raise VisionUnavailableError(
                "Gemini returned an empty vision response.",
                reason="empty-response",
            )
        return text

    except VisionUnavailableError:
        raise
    except Exception as exc:
        raise _vision_failure(exc, stage="request") from None
