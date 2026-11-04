"""
main.py â€” FastAPI application for CropSense AI.

``POST /chat`` runs a hybrid-RAG pipeline for text questions:

    validate request
      -> crop / unsupported-crop guard (derived from validated chunk metadata)
      -> named-disease guard     (inventory derived from the same chunk metadata)
      -> hybrid_search()        (FAISS semantic + BM25 lexical, fused by RRF)
      -> crop/disease filter    (single-crop, single-disease context only)
      -> generate_grounded_answer()  (citations + evidence levels)
      -> frontend source mapping

The image path is preserved: multipart handling, MIME/size validation and the
Gemini Vision call are unchanged. Vision output is kept visibly separate from
source-grounded reference information.

Persistence is additive and opt-in. Without an ``X-Session-Token`` header the
request is answered statelessly exactly as before. With one, the user turn is
committed before Gemini is called and the assistant turn is committed after the
reply is composed; the pipeline above is otherwise untouched.

The legacy keyword corpus (``retrieval.py`` / ``data/tomato_potato_guide.txt``)
is deliberately not used anywhere in this request path: it is unattributed sample
text, and a grounded answer must cite real provenance or say it has none.
"""

from pathlib import Path
import logging
import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

import repository
import storage
from db import get_session
from models import Conversation, Message, Session as AnonymousSession
from sessions import create_session, get_current_session, validate_session_token
from knowledge_graph import search_knowledge_graph
from crop_registry import (
    SUPPORTED_CROPS,
    UNSUPPORTED_CROPS,
    matches_crop,
    normalize_text,
)
from conversation_context import (
    crop_from_history,
    disease_from_history,
    history_record,
    is_followup,
    mentions_disease_term,
)
from disease_registry import build_disease_inventory, detect_disease_intent
from hybrid_retrieval import (
    CorpusUnavailableError,
    HybridRetrievalError,
    hybrid_search,
)
from generator import (
    GeneratorError,
    MissingCitationError,
    generate_grounded_answer,
)
from vector_store import VectorStoreUnavailableError, load_vector_store

app = FastAPI(
    title="CropSense AI API",
    description="Backend API for the CropSense AI crop assistance chatbot",
    version="0.4.0",
)

logger = logging.getLogger(__name__)

# Allow the Next.js frontend to communicate with this backend.
# ALLOWED_ORIGINS accepts a comma-separated list of URLs.
# Example: ALLOWED_ORIGINS=https://cropsense.vercel.app,http://localhost:3000
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
_cors_origins: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€ Retrieval configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#: How many fused chunks to request before crop filtering. The current index has
#: seven chunks, so this is the whole corpus; the value is set above that so a
#: growing knowledge base does not silently lose a crop's only evidence.
HYBRID_TOP_K = 10

#: ``UNSUPPORTED_CROPS`` is imported from crop_registry.py: a name listed there is
#: only blocked while the validated index does not actually contain it.

#: Names this application understands in the negative sense only: if one is named
#: but absent from the validated index, the request is reported as unsupported
#: instead of being called "ambiguous". Used solely to classify a named crop the
#: index cannot serve â€” never to claim support the index does not back.
KNOWN_CROP_NAMES = SUPPORTED_CROPS


EDUCATIONAL_DISCLAIMER = (
    "\nThis is educational information, not a confirmed diagnosis. "
    "Check the crop and symptoms carefully before taking action."
)

INSUFFICIENT_EVIDENCE_REPLY = (
    "I couldn't find enough relevant information in my agricultural guide to "
    "answer your question reliably.\n\n"
    "Please tell me which crop you are growing, what symptoms you can see, "
    "which leaves are affected, and when the symptoms started. "
    "You can also upload a clear photo of the affected plant."
)

_IMAGE_MAX_BYTES = 10 * 1024 * 1024


# â”€â”€ Crop guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _supported_crops() -> tuple[set[str], str | None]:
    """
    Derive the crops this deployment can actually answer for.

    The list comes from the validated chunk metadata, not a hardcoded claim, so
    the endpoint cannot advertise coverage the index does not have. Returns
    ``(crops, None)`` or ``(set(), reason)`` when the corpus cannot be loaded.
    """
    try:
        store = load_vector_store()
    except VectorStoreUnavailableError as exc:
        return set(), exc.reason

    crops = {
        str(chunk.get("crop")).strip().lower()
        for chunk in store.chunks
        if isinstance(chunk.get("crop"), str) and chunk.get("crop").strip()
    }
    return crops, None


def _supported_diseases() -> tuple[list[dict[str, Any]], str | None]:
    """
    Derive the disease inventory from the validated chunk metadata.

    Like :func:`_supported_crops`, this makes no coverage claim that the index
    does not back: every record comes from a chunk's own ``disease_name`` and
    ``crop`` fields. Returns ``(records, None)`` or ``([], reason)``.
    """
    try:
        store = load_vector_store()
    except VectorStoreUnavailableError as exc:
        return [], exc.reason

    return build_disease_inventory(store.chunks), None


def _resolve_crop_scope(message: str, supported: set[str]) -> dict[str, Any]:
    """
    Decide which crop a question is about, before any retrieval happens.

    ``supported`` is the set derived from validated chunk metadata, so recognition
    is dynamic: any crop the index actually contains is understood, with no
    hardcoded name list to drift out of sync. Matching is case- and
    whitespace-insensitive and accepts the aliases declared in crop_registry.py
    (for example ``corn`` for ``maize``). A crop in ``UNSUPPORTED_CROPS`` is
    treated as blocked only while the index does not contain it.

    Outcomes
    --------
    ``{"status": "ok", "crop": "tomato"}``
        Exactly one supported crop is named. Retrieval is filtered to it.
    ``{"status": "unsupported", "crops": [...]}``
        A crop absent from the guide is named. No retrieval, no generation.
    ``{"status": "ambiguous", "crops": [...], "reason": "multiple"|"none"}``
        Several supported crops are named, or none at all. Crop-specific
        treatment advice differs between crops, so the user is asked to say which
        one rather than being given blended guidance.
    """
    text = normalize_text(message)

    supported_named = sorted(
        crop for crop in supported if matches_crop(text, crop)
    )

    unsupported_named = sorted(
        crop
        for crop in set(UNSUPPORTED_CROPS) | set(KNOWN_CROP_NAMES)
        if crop not in supported and matches_crop(text, crop)
    )

    if unsupported_named:
        # An explicitly out-of-scope crop wins even if a supported crop is also
        # named: its passages must not lend credibility to another crop's case.
        return {"status": "unsupported", "crops": unsupported_named}

    if len(supported_named) > 1:
        return {"status": "ambiguous", "crops": supported_named, "reason": "multiple"}
    if len(supported_named) == 1:
        return {"status": "ok", "crop": supported_named[0]}
    return {"status": "ambiguous", "crops": [], "reason": "none"}


def _resolve_turn_context(
    message: str,
    supported: set[str],
    conversation_crop: str | None,
    history: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Resolve a turn's crop and disease, reusing this conversation's own context.

    Explicit text always wins. The stored context is consulted only when the
    current message names no crop and reads as a follow-up ("What conditions
    favour it?", "Tell me the symptoms."). The same applies to the disease:
    a follow-up that names none inherits the last disease discussed for the
    resolved crop, but only when the current inventory still backs it.

    ``query`` is what retrieval and generation receive. It is the raw message
    plus the inherited name (if any), so lexical ranking still has a term to
    match. The user's stored message is never rewritten.
    """
    crop_scope = _resolve_crop_scope(message, supported)
    followup = is_followup(message)
    inherited_crop = False

    if (
        message
        and followup
        and crop_scope.get("status") == "ambiguous"
        and crop_scope.get("reason") == "none"
    ):
        candidate = conversation_crop or crop_from_history(history, supported)
        if candidate in supported:
            crop_scope = {"status": "ok", "crop": candidate, "inherited": True}
            inherited_crop = True

    disease_scope: dict[str, Any] = {"status": "none"}
    query = message

    if crop_scope.get("status") == "ok" and message:
        crop = crop_scope["crop"]
        disease_scope = detect_disease_intent(message, crop, inventory)

        if (
            disease_scope.get("status") == "none"
            and followup
            and history
            and not mentions_disease_term(message)
        ):
            previous = disease_from_history(history, crop, inventory)
            if previous:
                disease_scope = {
                    "status": "supported",
                    "crop": crop,
                    "disease_names": [previous],
                    "named_phrase": previous,
                    "known_crops": [],
                    "inherited": True,
                }

        if disease_scope.get("inherited"):
            query = f"{message} {disease_scope['disease_names'][0]}"
        elif inherited_crop:
            query = f"{message} {crop}"

    return {
        "crop_scope": crop_scope,
        "disease_scope": disease_scope,
        "followup": followup,
        "query": query,
    }


def _chunks_for_crop(chunks: list[dict[str, Any]], crop: str) -> list[dict[str, Any]]:
    """
    Keep only chunks whose metadata says they belong to ``crop``.

    Chunks with a missing or non-string crop are dropped: without that field the
    crop cannot be proven, and unverifiable context must not reach the generator.
    """
    selected: list[dict[str, Any]] = []
    for chunk in chunks:
        value = chunk.get("crop")
        if isinstance(value, str) and value.strip().lower() == crop:
            selected.append(chunk)
    return selected


def _unsupported_message(crops: list[str], supported: set[str]) -> str:
    named = ", ".join(crops)
    supported_text = (
        "Currently supported crops: " + ", ".join(sorted(supported)) + "."
        if supported
        else "No crop reference material is currently available."
    )
    return (
        f"I don't have verified agricultural reference material for {named}, so I "
        f"can't give a source-grounded answer for it.\n\n{supported_text}\n\n"
        "I'd rather say that than apply guidance written for a different crop."
    )


def _ambiguous_message(scope: dict[str, Any], supported: set[str]) -> str:
    supported_text = (
        "Currently supported crops: " + ", ".join(sorted(supported)) + "."
        if supported
        else ""
    )
    if scope.get("reason") == "multiple":
        named = ", ".join(scope.get("crops") or [])
        opening = (
            f"You mentioned {named}. Treatment advice differs between crops, so "
            "which one should I look up?"
        )
    else:
        opening = (
            "Which crop is this about? Symptom and treatment guidance differs "
            "between crops, so I need to know before I look it up."
        )
    return f"{opening}\n\n{supported_text}".strip()


def _unsupported_disease_message(
    disease_scope: dict[str, Any], inventory: list[dict[str, Any]]
) -> str:
    """
    Explain that a named disease is not in the guide, without inventing advice.

    The supported-disease list is derived from the same inventory the detector
    used, so it cannot promise coverage the index lacks.
    """
    named = str(disease_scope.get("named_phrase") or "that disease").strip()
    named = named[:1].upper() + named[1:]
    crop = disease_scope.get("crop")

    supported_for_crop = sorted(
        record["disease_name"]
        for record in inventory
        if record.get("crop") == crop
    )

    if supported_for_crop:
        coverage = (
            f"For {crop}, my verified guide covers: "
            + ", ".join(supported_for_crop)
            + "."
        )
    else:
        coverage = "My verified guide does not cover that disease for this crop."

    return (
        f"I don't have a verified reference record for {named} in my agricultural "
        "guide, so I can't give a source-grounded answer about it. "
        "I won't apply guidance written for a different disease.\n\n"
        f"{coverage}"
    )


# â”€â”€ Retrieval + generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _retrieve(
    message: str,
    crop: str,
    disease_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Run hybrid retrieval and keep only the requested crop's chunks.

    When ``disease_names`` is supplied, hybrid retrieval filters to those
    diseases before applying its top-k cut, and the crop/disease filters are
    re-applied here as a second guarantee. Returns ``(chunks, failure_reason)``.
    A retrieval failure is reported as a category, never raised: the endpoint
    must answer, not 500.
    """
    try:
        if disease_names:
            fused = hybrid_search(
                message, top_k=HYBRID_TOP_K, disease_names=disease_names
            )
        else:
            fused = hybrid_search(message, top_k=HYBRID_TOP_K)
    except CorpusUnavailableError as exc:
        return [], exc.reason
    except HybridRetrievalError as exc:
        return [], exc.reason
    except Exception:
        return [], "retrieval-failed"

    guarded = _chunks_for_crop(fused, crop)

    if disease_names:
        guarded = [
            chunk
            for chunk in guarded
            if str(chunk.get("disease_name")) in disease_names
        ]

    return guarded, None


def _retrieve_for_scope(
    message: str,
    crop: str,
    disease_scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Apply the disease scope before anything reaches the generator.

    An unsupported named disease returns no chunks and no retrieval error, so
    unrelated diseases can never become evidence for it. A supported named
    disease restricts retrieval to that disease; ``status == "none"`` keeps the
    normal general-crop behaviour.
    """
    status = disease_scope.get("status", "none")

    if status == "unsupported":
        return [], None

    if status == "supported":
        names = {str(name) for name in (disease_scope.get("disease_names") or [])}
        return _retrieve(message, crop, names or None)

    return _retrieve(message, crop)


def _resolve_knowledge_graph(
    query: str,
    scope: dict[str, Any],
    disease_scope: dict[str, Any],
    store_error: str | None,
) -> list[dict[str, Any]]:
    """
    Return graph edges only for a turn that has a validated crop scope.

    Follow-ups pass the resolved query, so inherited crop/disease context is
    available to the symptom matching. A refused disease, an ambiguous or
    unsupported crop, and a missing index all return ``[]``: unrelated edges
    must never accompany a declined answer.
    """
    if store_error is not None:
        return []
    if scope.get("status") != "ok":
        return []
    if disease_scope.get("status") == "unsupported":
        return []
    return search_knowledge_graph(query)


def _semantic_participated(chunks: list[dict[str, Any]]) -> bool:
    """
    Whether the semantic signal contributed to the chunks being answered from.

    Derived from the retrieval metadata on this request's own crop-filtered
    results â€” ``hybrid_search`` stamps ``semantic_rank`` on every chunk the
    FAISS signal ranked. This deliberately avoids the process-global
    ``retrieval_status()["last_search"]`` record, which a concurrent request could
    overwrite between retrieval and response assembly.
    """
    return any(chunk.get("semantic_rank") is not None for chunk in chunks)


def _generate(
    message: str, chunks: list[dict[str, Any]]
) -> tuple[str | None, list[dict[str, Any]], str | None]:
    """
    Generate a grounded answer. Returns ``(reply, sources, failure_reason)``.

    Every generator failure is a category, not an exception: the endpoint falls
    back to a clear message instead of presenting unverified text as grounded.
    """
    try:
        result = generate_grounded_answer(message, chunks)
    except MissingCitationError:
        # The model replied but grounded nothing in the supplied passages. With a
        # thin or abstract-only corpus that is how it reports the evidence cannot
        # answer the question, so it is shown as insufficient evidence rather
        # than as a service failure. Nothing is invented to replace the citation.
        return None, [], "insufficient-context"
    except GeneratorError as exc:
        return None, [], getattr(exc, "reason", None) or type(exc).__name__
    except Exception:
        return None, [], "generation-failed"

    if result.get("status") != "success":
        return None, [], "insufficient-context"

    reply = (result.get("reply") or "").strip()
    if not reply:
        return None, [], "empty-generation"

    return reply, list(result.get("sources") or []), None


def _frontend_sources(
    generator_sources: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Map generator citations onto the frontend's source shape.

    ``filename`` is required by the frontend and is rendered verbatim, so it is
    filled from real provenance (title, else URL). ``match_score`` is intentionally
    absent: the only numeric values available are an RRF rank-fusion score and a
    raw cosine similarity, and presenting either as a "match score" would imply a
    confidence the pipeline does not compute. The frontend declares
    ``match_score: number`` but never reads it, so omitting the key is a type-level
    mismatch only; making it optional in the frontend type is the correct fix.

    Citation numbers are resolved back to the guarded chunk list, so the number in
    the reply and the source entry always refer to the same original chunk. One
    entry is emitted per cited number: two chunks taken from the same document
    share a title and URL but are still separate references, so de-duplicating on
    (title, url) would silently drop the later citation from the list.
    """
    mapped: list[dict[str, Any]] = []
    seen: set[Any] = set()

    for source in generator_sources:
        citation = source.get("citation")
        chunk = None
        if isinstance(citation, int) and 1 <= citation <= len(chunks):
            chunk = chunks[citation - 1]

        title = source.get("title") or ""
        url = source.get("url") or ""

        base_name = title.strip() or url.strip() or (
            f"Reference [{citation}]" if citation is not None else "Reference"
        )

        # The citation number identifies the chunk, so it is the dedupe key. The
        # (title, url) fallback only applies to a source the generator supplied
        # without a number.
        key: Any = citation if isinstance(citation, int) else (base_name, url.strip())
        if key in seen:
            continue
        seen.add(key)

        entry: dict[str, Any] = {
            # Required by the frontend; it de-duplicates and renders this value.
            "filename": base_name,
            # Provenance, preserved as additional information.
            "citation": citation,
            "title": source.get("title"),
            "url": source.get("url"),
            "institution": source.get("institution"),
            "evidence_level": source.get("evidence_level"),
        }
        if chunk is not None:
            entry["chunk_id"] = chunk.get("chunk_id")
        mapped.append(entry)

    return mapped


def _persistable_sources(
    sources: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Map the response's source entries onto the repository's source shape.

    Only citations that resolve back to a guarded chunk are stored: the chunk id
    is required by the table, and the chunk's own metadata is what makes the
    history queryable without re-parsing the reply. A citation that cannot be
    resolved is dropped rather than failing the whole turn, and each citation
    number is kept once so the per-message uniqueness rule always holds.
    """
    persisted: list[dict[str, Any]] = []
    seen: set[int] = set()

    for source in sources:
        citation = source.get("citation")
        chunk_id = source.get("chunk_id")
        if not isinstance(citation, int) or citation in seen or not chunk_id:
            continue
        if not 1 <= citation <= len(chunks):
            continue

        chunk = chunks[citation - 1]
        seen.add(citation)
        persisted.append(
            {
                "citation_number": citation,
                "chunk_id": str(chunk_id),
                "filename": source.get("filename"),
                "title": source.get("title"),
                "url": source.get("url"),
                "institution": source.get("institution"),
                "evidence_level": source.get("evidence_level"),
                "disease_name": chunk.get("disease_name"),
                "crop": chunk.get("crop"),
            }
        )

    return persisted


# â”€â”€ Request validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def _read_image(image: Optional[UploadFile]) -> tuple[bytes | None, str | None]:
    """Validate and read an uploaded image, keeping the existing contract."""
    if not image or not image.filename:
        return None, None

    content_type = (image.content_type or "").split(";")[0].strip().lower()
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded file '{image.filename}' does not appear to be an "
                "image. Please upload a JPEG, PNG, or WebP file."
            ),
        )

    data = await image.read()
    if len(data) > _IMAGE_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image exceeds the 10 MB limit.",
        )

    return data, content_type


async def _discard_user_turn(
    db: AsyncSession,
    owner_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    image_id: uuid.UUID | None,
    storage_key: str | None,
    created_conversation: bool,
) -> None:
    """
    Best-effort undo of a partially persisted user turn.

    Called only when the image or the user message could not be saved, so
    nothing from this request may outlive it: the image row, the conversation
    this request created, and the file on disk. Each step is independent, so one
    failure (or a session left unusable by a failed commit) cannot stop the rest.

    Identifiers are passed as plain values because the rollback below expires
    every ORM instance the session holds.
    """
    try:
        await db.rollback()
    except Exception:
        logger.warning("Could not reset the database session during cleanup.")

    if image_id is not None:
        try:
            await repository.delete_image(db, owner_id, image_id)
        except Exception:
            logger.warning("Could not remove image metadata after a failed turn.")

    if created_conversation and conversation_id is not None:
        try:
            await repository.delete_conversation(db, owner_id, conversation_id)
        except Exception:
            logger.warning("Could not remove the conversation after a failed turn.")

    if storage_key is not None:
        try:
            storage.delete_image(storage_key)
        except Exception:
            logger.warning("Could not remove the stored image after a failed turn.")


# â”€â”€ Conversation and image serialisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_CONVERSATION_NOT_FOUND = "Conversation not found."
_IMAGE_NOT_FOUND = "Image not found."
_TITLE_INVALID = "Conversation title must be between 1 and 200 characters."


class RenameConversationRequest(BaseModel):
    """Body of ``PATCH /conversations/{conversation_id}``."""

    title: str


def _conversation_header(conversation: Conversation) -> dict[str, Any]:
    """The fields every conversation response starts with."""
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "crop": conversation.crop,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def _conversation_summary(
    conversation: Conversation, stats: tuple[int, bool]
) -> dict[str, Any]:
    """One row of ``GET /conversations``: the header plus derived history facts."""
    message_count, has_diagnosis = stats
    return {
        **_conversation_header(conversation),
        "message_count": message_count,
        "has_diagnosis": has_diagnosis,
    }


def _message_payload(message: Message) -> dict[str, Any]:
    """
    Serialize one stored message for the history view.

    Only what the view renders is exposed: a user turn may carry its image's
    public id, size and MIME type, and an assistant turn carries its outcome, the
    derived split of the reply, its citations and its graph edges. Storage keys,
    filesystem paths, session ids and token hashes never leave the database.

    Graph edges are ordered so two reads of the same message always agree; the
    table itself has no defined row order.
    """
    payload: dict[str, Any] = {
        "id": str(message.id),
        "role": message.role,
        "sequence": message.sequence,
        "content": message.content,
        "created_at": message.created_at,
    }

    if message.role == "user" and message.image is not None:
        payload["image"] = {
            "id": str(message.image.id),
            "url": f"/images/{message.image.id}",
            "mime_type": message.image.mime_type,
            "byte_size": message.image.byte_size,
        }

    if message.role == "assistant":
        payload["mode"] = message.mode
        payload["vision_assessment"] = message.vision_assessment
        payload["grounded_answer"] = message.grounded_answer
        payload["sources"] = [
            {
                "citation": source.citation_number,
                "filename": source.filename,
                "title": source.title,
                "url": source.url,
                "institution": source.institution,
                "evidence_level": source.evidence_level,
                "chunk_id": source.chunk_id,
            }
            for source in message.sources
        ]
        payload["knowledge_graph"] = sorted(
            (
                {
                    "subject": edge.subject,
                    "relation": edge.relation,
                    "object": edge.object,
                }
                for edge in message.kg_edges
            ),
            key=lambda edge: (edge["subject"], edge["relation"], edge["object"]),
        )

    return payload


async def _conversation_or_404(
    db: AsyncSession, session_id: uuid.UUID, conversation_id: str
) -> Conversation:
    """
    Session-scoped conversation lookup.

    A malformed identifier cannot belong to anyone, so it is reported exactly
    like an unknown one: 404, never 400 or 422. That keeps ids unenumerable and
    keeps the ownership rule in one place.
    """
    try:
        conversation = await repository.get_conversation(
            db, session_id, conversation_id
        )
    except ValueError:
        conversation = None

    if conversation is None:
        raise HTTPException(status_code=404, detail=_CONVERSATION_NOT_FOUND)
    return conversation


# â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.get("/")
def home():
    return {
        "message": "Welcome to CropSense AI API!",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CropSense AI backend",
    }


@app.post("/sessions", status_code=201)
async def create_anonymous_session(
    db: AsyncSession = Depends(get_session),
):
    session, token = await create_session(db)
    return {
        "token": token,
        "session_id": str(session.id),
        "created_at": session.created_at,
        "expires_at": session.expires_at,
    }


@app.get("/sessions/me")
async def read_anonymous_session(
    session: AnonymousSession = Depends(get_current_session),
):
    return {
        "session_id": str(session.id),
        "created_at": session.created_at,
        "expires_at": session.expires_at,
    }


@app.get("/conversations")
async def list_conversations(
    session: AnonymousSession = Depends(get_current_session),
    db: AsyncSession = Depends(get_session),
):
    """
    List the caller's conversations, most recently updated first.

    Returns a JSON array of ``id``, ``title``, ``crop``, ``created_at``,
    ``updated_at``, ``message_count`` and ``has_diagnosis``. The counts come from
    one grouped query rather than one per conversation.
    """
    session_id = session.id
    conversations = await repository.list_conversations(db, session_id)
    stats = await repository.conversation_stats(db, session_id)
    return [
        _conversation_summary(conversation, stats.get(conversation.id, (0, False)))
        for conversation in conversations
    ]


@app.get("/conversations/{conversation_id}")
async def read_conversation(
    conversation_id: str,
    session: AnonymousSession = Depends(get_current_session),
    db: AsyncSession = Depends(get_session),
):
    """
    Return one conversation with its messages in sequence order.

    User turns carry their image metadata when one was stored; assistant turns
    carry ``mode``, ``vision_assessment``, ``grounded_answer``, the citations and
    the knowledge-graph edges recorded with that reply.
    """
    conversation = await _conversation_or_404(db, session.id, conversation_id)
    return {
        **_conversation_header(conversation),
        "messages": [_message_payload(message) for message in conversation.messages],
    }


@app.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    payload: RenameConversationRequest,
    session: AnonymousSession = Depends(get_current_session),
    db: AsyncSession = Depends(get_session),
):
    """
    Rename a conversation.

    Returns the conversation header (``id``, ``title``, ``crop``, ``created_at``,
    ``updated_at``) rather than the full history, which the caller can fetch with
    ``GET /conversations/{id}``.
    """
    title = (payload.title or "").strip()
    # Checked here so the caller gets a clear 400; the repository enforces the
    # same rule, which is why a ValueError below can only be a malformed id.
    if not 1 <= len(title) <= 200:
        raise HTTPException(status_code=400, detail=_TITLE_INVALID)

    try:
        conversation = await repository.rename_conversation(
            db, session.id, conversation_id, title
        )
    except ValueError:
        conversation = None

    if conversation is None:
        raise HTTPException(status_code=404, detail=_CONVERSATION_NOT_FOUND)
    return _conversation_header(conversation)


@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    session: AnonymousSession = Depends(get_current_session),
    db: AsyncSession = Depends(get_session),
):
    """
    Permanently delete a conversation.

    Messages, citations and graph edges are removed by the cascade; image rows
    and files that no longer have a referencing message are removed as well.
    """
    try:
        deleted = await repository.delete_conversation(
            db, session.id, conversation_id
        )
    except ValueError:
        deleted = False

    if not deleted:
        raise HTTPException(status_code=404, detail=_CONVERSATION_NOT_FOUND)
    return Response(status_code=204)


@app.get("/images/{image_id}")
async def read_image(
    image_id: str,
    session: AnonymousSession = Depends(get_current_session),
    db: AsyncSession = Depends(get_session),
):
    """
    Stream one stored image to its owning session.

    The storage key never leaves the server: the lookup is session-scoped, and a
    row whose file is missing is reported as 404 rather than as a server error.
    """
    try:
        image = await repository.get_image(db, session.id, image_id)
    except ValueError:
        image = None

    if image is None:
        raise HTTPException(status_code=404, detail=_IMAGE_NOT_FOUND)

    try:
        data = storage.read_image(image.storage_key)
    except storage.StorageError:
        raise HTTPException(status_code=404, detail=_IMAGE_NOT_FOUND) from None

    return Response(content=data, media_type=image.mime_type)


@app.post("/chat")
async def chat(
    # An empty default makes image-only uploads reach this handler. A blank
    # message with no image is rejected just below, so the 400 contract is kept.
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),
    # Persistence is opt-in: both of these stay absent for the original,
    # stateless behaviour.
    conversation_id: Optional[str] = Form(None),
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
    db: AsyncSession = Depends(get_session),
):
    """
    Answer a question, optionally with an uploaded plant image.

    Text questions go through hybrid retrieval and grounded generation, filtered
    to a single crop. Images are still analysed by Gemini Vision; the visual
    assessment is presented separately from reference-grounded guidance.

    Persistence is additive. Without ``X-Session-Token`` the request is
    stateless: nothing is written and the response keeps its original five keys.
    With a valid token the user turn is committed before Gemini is called and
    the assistant turn is committed once the reply has been composed.

    Returns
    -------
    JSON with keys: user_message, reply, mode, sources, knowledge_graph â€” plus
    conversation_id, user_message_id and assistant_message_id when the exchange
    was persisted.
    """
    # â”€â”€ Step 1: validate the request â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    image_bytes, image_mime = await _read_image(image)

    message_text = (message or "").strip()
    if not message_text and image_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="Please include a question, an image, or both.",
        )

    # â”€â”€ Step 2: resolve the optional persistence context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # A supplied token that fails validation is an error, never a silent
    # downgrade to stateless mode: the caller asked for persistence and has to
    # know it did not happen.
    #
    # The identifiers the rest of the request needs are read once, here, as plain
    # values: a later rollback expires every ORM instance in the session, and
    # reading ``owner.id`` after that would trigger a lazy reload inside async
    # code (MissingGreenlet) instead of returning the id.
    owner: AnonymousSession | None = None
    owner_id: uuid.UUID | None = None
    conversation_uuid: uuid.UUID | None = None
    # Multi-turn context is snapshotted as plain values before anything is
    # written, so a later rollback cannot expire the objects it came from.
    conversation_crop: str | None = None
    history: list[dict[str, Any]] = []

    if x_session_token:
        owner = await validate_session_token(db, x_session_token)
        if owner is None:
            raise HTTPException(
                status_code=401,
                detail="The X-Session-Token header is invalid or has expired.",
            )
        owner_id = owner.id

    if conversation_id:
        if owner_id is None:
            raise HTTPException(
                status_code=400,
                detail="conversation_id requires an X-Session-Token header.",
            )
        try:
            existing = await repository.get_conversation(
                db, owner_id, conversation_id
            )
        except ValueError:
            # A malformed id cannot belong to anyone, and reporting it exactly
            # like an unknown one keeps conversation ids unenumerable.
            existing = None
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        conversation_uuid = existing.id
        # ``get_conversation`` already eager-loads messages and their citations;
        # keep only plain values, because this request may roll the session back.
        conversation_crop = existing.crop
        history = [history_record(m) for m in existing.messages]

    # â”€â”€ Step 3: persist the user turn before any generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # The user message is committed on its own, so a failure during image
    # analysis or generation can never lose what the user actually sent.
    user_message_id: uuid.UUID | None = None
    image_id: uuid.UUID | None = None
    storage_key: str | None = None
    created_conversation = False

    if owner_id is not None:
        try:
            if image_bytes is not None:
                saved_image = storage.save_image(
                    owner_id,
                    image_bytes,
                    image_mime or "image/jpeg",
                    image.filename,
                )
                storage_key = str(saved_image["storage_key"])
                image_row = await repository.create_image(
                    db,
                    owner_id,
                    storage_key,
                    image.filename,
                    str(saved_image["mime_type"]),
                    int(saved_image["byte_size"]),
                    str(saved_image["sha256"]),
                )
                image_id = image_row.id

            if conversation_uuid is None:
                created = await repository.create_conversation(
                    db, owner_id, message_text or "Image attached"
                )
                conversation_uuid = created.id
                created_conversation = True

            appended = await repository.append_user_message(
                db, owner_id, conversation_uuid, message, image_id=image_id
            )
            if appended is not None:
                user_message_id = appended.id
        except storage.StorageError:
            await _discard_user_turn(
                db,
                owner_id,
                conversation_uuid,
                image_id,
                storage_key,
                created_conversation,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "Images saved to a conversation must be PNG, JPEG or WebP. "
                    "Send the question without an X-Session-Token header to "
                    "analyse this file."
                ),
            ) from None
        except Exception:
            logger.exception("Could not persist the user turn.")
            await _discard_user_turn(
                db,
                owner_id,
                conversation_uuid,
                image_id,
                storage_key,
                created_conversation,
            )
            raise HTTPException(
                status_code=500,
                detail="Your message could not be saved. Please try again.",
            ) from None

        if user_message_id is None:
            await _discard_user_turn(
                db,
                owner_id,
                conversation_uuid,
                image_id,
                storage_key,
                created_conversation,
            )
            raise HTTPException(status_code=404, detail="Conversation not found.")

    supported, store_error = _supported_crops()

    guarded_chunks: list[dict[str, Any]] = []
    retrieval_error: str | None = None
    disease_inventory: list[dict[str, Any]] = []
    disease_scope: dict[str, Any] = {"status": "none"}
    effective_query = message_text
    graph_results: list[dict[str, Any]] = []

    if store_error is not None:
        # Without the index there is no supported-crop set to reason about, so
        # scope resolution is skipped rather than guessed at from a hardcoded list.
        scope: dict[str, Any] = {"status": "unavailable"}
    else:
        if message_text:
            disease_inventory, _disease_error = _supported_diseases()

        turn_context = _resolve_turn_context(
            message_text,
            supported,
            conversation_crop,
            history,
            disease_inventory,
        )
        scope = turn_context["crop_scope"]
        disease_scope = turn_context["disease_scope"]
        effective_query = turn_context["query"]

        if scope["status"] == "ok" and message_text:
            # Unsupported disease -> no retrieval at all, so no unrelated
            # disease chunk can reach the generator or the vision context.
            guarded_chunks, retrieval_error = _retrieve_for_scope(
                effective_query, scope["crop"], disease_scope
            )

    # The graph is derived from the same resolved scope as retrieval: a follow-up
    # inherits the conversation's crop, and a refused disease gets no edges.
    graph_results = _resolve_knowledge_graph(
        effective_query, scope, disease_scope, store_error
    )

    # The resolved crop is conversation metadata, so it is recorded once the
    # scope is known. It is never worth failing a committed turn over, so a
    # failure here is logged and the reply continues.
    if conversation_uuid is not None and scope.get("status") == "ok":
        try:
            await repository.set_conversation_crop(
                db, owner_id, conversation_uuid, scope["crop"]
            )
        except Exception:
            logger.warning("Could not record the conversation crop.", exc_info=True)
            # A failed statement leaves the session unusable â€” the assistant turn
            # below would abort with "current transaction is aborted". Rolling
            # back is what makes the rest of this request possible, so it is
            # attempted even though the crop itself is already lost.
            try:
                await db.rollback()
            except Exception:
                logger.warning(
                    "Could not roll back after the failed crop update.",
                    exc_info=True,
                )

    # â”€â”€ Step 4: image analysis (unchanged pipeline, guarded context) â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vision_text: str | None = None
    vision_failed = False

    if image_bytes is not None:
        try:
            from vision import analyze_plant_image

            vision_text = analyze_plant_image(
                image_bytes=image_bytes,
                mime_type=image_mime or "image/jpeg",
                user_text=message_text or "(Image attached)",
                # Only crop-guarded chunks: unrelated crop guidance must never be
                # fed to the vision model as reference context.
                context_documents=guarded_chunks,
            )
        except Exception:
            # Known or unknown, the reply must not leak SDK text. The existing
            # generic warning is preserved.
            vision_failed = True

    # â”€â”€ Step 5: grounded generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    grounded_reply: str | None = None
    generator_sources: list[dict[str, Any]] = []
    generation_error: str | None = None

    if scope["status"] == "ok" and message_text and guarded_chunks:
        grounded_reply, generator_sources, generation_error = _generate(
            effective_query, guarded_chunks
        )

    # â”€â”€ Step 6: compose the reply â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    reply_parts: list[str] = []

    if vision_text:
        reply_parts.append("Image assessment:\n" + vision_text.strip())
    elif vision_failed:
        reply_parts.append(
            "âš ï¸ Image analysis is not available right now. Please ensure the "
            "GEMINI_API_KEY is configured in backend/.env."
        )

    if store_error is not None:
        outcome = "vector_store_unavailable"
        reply_parts.append(
            "My agricultural reference library is unavailable right now, so I "
            "can't give a verified, source-grounded answer. Please try again once "
            "the index is rebuilt."
        )
    elif scope["status"] == "unsupported":
        outcome = "unsupported_crop"
        reply_parts.append(_unsupported_message(scope["crops"], supported))
    elif scope["status"] == "ambiguous":
        outcome = "ambiguous_crop"
        reply_parts.append(_ambiguous_message(scope, supported))
    elif disease_scope["status"] == "unsupported":
        outcome = "unsupported_disease"
        reply_parts.append(
            _unsupported_disease_message(disease_scope, disease_inventory)
        )
    elif retrieval_error is not None:
        outcome = "retrieval_failed"
        reply_parts.append(
            "I couldn't search the agricultural reference library just now, so I "
            "can't give a verified answer to this question."
        )
    elif not guarded_chunks:
        outcome = "insufficient_evidence"
        reply_parts.append(INSUFFICIENT_EVIDENCE_REPLY)
    elif grounded_reply is not None:
        # Derived from this request's own chunks, never from process-global state.
        semantic_ok = _semantic_participated(guarded_chunks)
        outcome = "grounded_rag" if semantic_ok else "grounded_rag_lexical_only"
        label = "Verified reference information:\n" if vision_text else ""
        reply_parts.append(label + grounded_reply)
    elif generation_error == "insufficient-context":
        outcome = "insufficient_evidence"
        reply_parts.append(INSUFFICIENT_EVIDENCE_REPLY)
    else:
        # Generation failed (API error, empty response, invalid citation...).
        # Never present unverified text as a grounded answer.
        outcome = "generation_failed"
        reply_parts.append(
            "I found relevant reference material, but I couldn't produce a "
            "verified answer from it just now. Please try again in a moment."
        )

    reply = "\n\n".join(part for part in reply_parts if part)

    # The generator's system instruction already ends a grounded answer with an
    # educational note, so only the non-generated replies get this one.
    if outcome not in ("grounded_rag", "grounded_rag_lexical_only"):
        reply = reply.rstrip() + EDUCATIONAL_DISCLAIMER

    # â”€â”€ Step 7: sources â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if grounded_reply is not None:
        sources = _frontend_sources(generator_sources, guarded_chunks)
    else:
        # No citations in the reply means no sources: an empty list is honest,
        # whereas listing un-cited chunks would imply a grounding that failed.
        sources = []

    # â”€â”€ Step 8: response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mode = f"vision_and_{outcome}" if image_bytes is not None else outcome

    response: dict[str, Any] = {
        "user_message": message,
        "reply": reply,
        "mode": mode,
        "sources": sources,
        "knowledge_graph": graph_results,
    }

    if owner_id is None or conversation_uuid is None or user_message_id is None:
        # Stateless request: the original five keys, and nothing was written.
        return response

    # â”€â”€ Step 9: persist the assistant turn â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Every user-visible outcome is stored, fallbacks included. Only sanitized
    # categories are recorded â€” ``mode`` and the failure reasons are already
    # categories, never raw SDK text.
    error_category: str | None
    if store_error is not None:
        error_category = "vector-store-unavailable"
    elif retrieval_error is not None:
        error_category = retrieval_error
    elif generation_error is not None:
        error_category = generation_error
    elif vision_failed:
        error_category = "vision-unavailable"
    else:
        error_category = None

    try:
        assistant_message = await repository.append_assistant_message(
            db,
            owner_id,
            conversation_uuid,
            content=reply,
            mode=mode,
            vision_assessment=vision_text or None,
            grounded_answer=grounded_reply,
            error_category=error_category,
            sources=_persistable_sources(sources, guarded_chunks),
            kg_edges=graph_results,
        )
    except Exception:
        logger.exception("Could not persist the assistant turn.")
        raise HTTPException(
            status_code=500,
            detail="Your reply could not be saved. Please try again.",
        ) from None

    if assistant_message is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    response.update(
        conversation_id=str(conversation_uuid),
        user_message_id=str(user_message_id),
        assistant_message_id=str(assistant_message.id),
    )
    return response
