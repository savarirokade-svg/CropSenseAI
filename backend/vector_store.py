"""
vector_store.py — Safe, cached loader for the CropSense AI FAISS vector store.

This module loads and validates the artifacts produced by ``index_builder.py``
(``index.faiss``, ``chunks.json``, ``manifest.json`` in
``backend/data/vector_store/``) so that retrieval code can trust what it gets.

Guarantees
----------
* Paths are resolved relative to the backend directory, never the process
  working directory, so the loader behaves identically under
  ``uvicorn main:app``, a REPL or a test runner.
* The index and metadata are loaded once and reused. A store that fails
  validation is remembered as unavailable, so a broken artifact set does not
  cause disk reads on every request.
* Nothing is returned unless every validation below has passed. A failed load
  raises :class:`VectorStoreUnavailableError`; a partially validated store is
  never handed to a caller.
* Error messages carry a category and a file *name* only. Raw OS/SDK exception
  text is dropped, exactly as in ``embeddings.py``.

This module performs no network access: it never imports the Gemini SDK, never
embeds anything and never rebuilds an index.

Vector-to-chunk alignment
-------------------------
``index_builder.build_index()`` embeds ``chunks`` in list order, stacks the
vectors and calls ``index.add(matrix)`` once, which fixes the invariant
``index row i`` == ``chunks[i]``. That ordering is *inherited* from the builder;
it is not something a reader can prove from the files alone. Matching counts is
therefore necessary but NOT sufficient evidence of alignment.

What this module can verify offline, and does:

1. ``len(chunks) == index.ntotal == manifest["num_chunks"]`` (row exists at all).
2. Three integrity anchors match the values pinned to a known-good build:

   ``EXPECTED_CHUNK_INVENTORY_SHA256``
       the ordered chunk ID sequence — detects reordering, additions, removals.
   ``EXPECTED_CHUNK_CONTENT_SHA256``
       *every* field of every chunk, deterministically re-serialised — detects an
       edit to chunk text, a source URL, a title, an institution or an evidence
       level even when the chunk IDs are unchanged and the counts still agree.
   ``EXPECTED_VECTOR_FINGERPRINT_SHA256``
       the index row bytes in order — detects either artifact being swapped for a
       different build.

   Together they pin the (``chunks.json``, ``index.faiss``) pair to one specific
   build. Leaving an anchor as ``None`` is never a silent pass: the loader reports
   an "integrity pin is not configured" warning on the returned store.
3. Every stored row is finite and unit-norm, which is the precondition for
   treating ``IndexFlatIP`` inner products as cosine similarity.

It still cannot prove that row *i* is the embedding of chunk *i*'s text: that
would require re-embedding the corpus with Gemini, which is out of scope here.

After a legitimate ``index_builder.py --build`` the pinned anchors no longer
match, and the strict path fails closed by design. Run
``python vector_store.py --print-actual-fingerprints`` to see the actual values
and how they differ, then update the constants by hand. That command *reports*
fingerprints: it does not certify semantic row-to-chunk alignment and it does not
approve a rebuilt index, and it never writes to any file.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import faiss

# ── Paths (resolved from this file, not from the working directory) ──────────

BACKEND_DIR = Path(__file__).resolve().parent

# Mutable so tests can point the loader at a fixture directory. Always read
# through _paths() at call time; never captured at import.
VECTOR_STORE_DIR = BACKEND_DIR / "data" / "vector_store"

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"

KNOWLEDGE_BASE_PATH = BACKEND_DIR / "data" / "agricultural_guide.md"

# ── Expected configuration (mirrors index_builder.py / embeddings.py) ───────

EXPECTED_INDEX_TYPE = "IndexFlatIP"
EXPECTED_VECTOR_DTYPE = "float32"
EXPECTED_SIMILARITY = "inner product on L2-normalized vectors (cosine)"

# Unit-norm tolerance. embeddings.py normalizes in float32, so the residual is
# float32 epsilon scale; 1e-4 is far above that and far below "not normalized".
NORM_TOLERANCE = 1e-4

# Fingerprints pinned to the build recorded in manifest.json
# (built_at 2026-11-03T17:16:12+00:00, 82 chunks, dim 768). These are integrity
# anchors, not approval of a rebuild: regenerate the values with
# `python vector_store.py --print-actual-fingerprints` and paste them here only
# after you have decided the new build is the one you intend to serve.
#
# Leaving one as None is never done implicitly for a release build: the loader
# then records an "integrity pins not configured" warning on the returned store
# instead of silently skipping the check.
EXPECTED_CHUNK_INVENTORY_SHA256 = (
    "0b7973591c755c29173bbb7b0b172ba935a519aed38894280523907b41404051"
)
EXPECTED_CHUNK_CONTENT_SHA256 = (
    "62fba045267d6a51f30175cda5287057ac1911b0b15dcbb028170a3e93105467"
)
EXPECTED_VECTOR_FINGERPRINT_SHA256 = (
    "25dc55b4342f4eede8f45938f8878c2aea52b58c8c3f010952c9ebcad0364f93"
)

# ── Exceptions ───────────────────────────────────────────────────────────────


class VectorStoreUnavailableError(Exception):
    """
    Raised when the vector store is missing, corrupt or incompatible.

    Attributes
    ----------
    reason:
        A short, generic failure category (see ``_REASON_HINTS``).
    detail:
        Safe, non-sensitive remediation hint. Never contains API keys or raw
        exception text from FAISS or the JSON decoder.
    """

    def __init__(self, message: str, *, reason: str = "unavailable", detail: str = ""):
        # The complete text is assembled once, so clone() cannot duplicate the hint.
        self.message = message
        self.detail = detail
        super().__init__(f"{message} {detail}".strip() if detail else message)
        self.reason = reason

    def clone(self) -> "VectorStoreUnavailableError":
        """
        An equivalent, independently raisable instance.

        The cached verdict is reused across requests, but a shared exception
        object accumulates traceback state and identity surprises callers, so
        each raise gets a fresh copy of the same verdict.
        """
        return VectorStoreUnavailableError(
            self.message, reason=self.reason, detail=self.detail
        )


# ── Public result type ───────────────────────────────────────────────────────


class VectorStore:
    """
    A fully validated vector store: index, chunk metadata and manifest.

    Instances are produced only by :func:`load_vector_store` (all validations,
    including the pinned integrity anchors) or :func:`diagnose_vector_store`
    (all validations *except* the pinned anchors, for reporting). Callers use
    ``index`` / ``chunks`` / ``manifest`` directly; the remaining attributes
    record what was verified.
    """

    __slots__ = (
        "index",
        "chunks",
        "manifest",
        "dimension",
        "vector_count",
        "metric",
        "embedding_model",
        "chunk_ids",
        "chunk_inventory_sha256",
        "chunk_content_sha256",
        "vector_fingerprint_sha256",
        "integrity_pins_enforced",
        "stale_knowledge_base",
        "warnings",
    )

    def __init__(
        self,
        *,
        index: "faiss.Index",
        chunks: tuple[dict[str, Any], ...],
        manifest: dict[str, Any],
        dimension: int,
        metric: str,
        embedding_model: str,
        chunk_ids: tuple[str, ...],
        chunk_inventory_sha256: str,
        chunk_content_sha256: str,
        vector_fingerprint_sha256: str,
        integrity_pins_enforced: bool,
        stale_knowledge_base: bool,
        warnings: tuple[str, ...],
    ):
        self.index = index
        self.chunks = chunks
        self.manifest = manifest
        self.dimension = dimension
        self.metric = metric
        self.embedding_model = embedding_model
        self.chunk_ids = chunk_ids
        self.chunk_inventory_sha256 = chunk_inventory_sha256
        self.chunk_content_sha256 = chunk_content_sha256
        self.vector_fingerprint_sha256 = vector_fingerprint_sha256
        self.integrity_pins_enforced = integrity_pins_enforced
        self.stale_knowledge_base = stale_knowledge_base
        self.warnings = warnings
        self.vector_count = len(chunks)

    def __len__(self) -> int:
        """Number of indexed chunks (== index rows == valid vector positions)."""
        return self.vector_count

    def chunk_at(self, row: int) -> dict[str, Any]:
        """
        Return the chunk metadata for FAISS row ``row``.

        Row order is the alignment contract: row i is chunks[i], exactly as
        ``index_builder.build_index()`` wrote them.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError("row must be an int")
        if row < 0 or row >= self.vector_count:
            raise IndexError(
                f"row {row} is outside the indexed range 0..{self.vector_count - 1}."
            )
        return self.chunks[row]

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"<VectorStore metric={self.metric} dim={self.dimension} "
            f"vectors={self.vector_count} model={self.embedding_model} "
            f"stale_kb={self.stale_knowledge_base}>"
        )


# ── Failure categories ───────────────────────────────────────────────────────

_REASON_HINTS = {
    "missing-file": "Rebuild the index with 'python index_builder.py --build'.",
    "unreadable-file": "Check that the vector store files are readable.",
    "invalid-manifest": "The manifest is not valid JSON or is missing fields.",
    "invalid-chunks": "Chunk metadata is not valid JSON or has unusable entries.",
    "duplicate-chunk-id": "Chunk IDs must be unique.",
    "index-load-failed": "The FAISS index could not be read; it may be truncated.",
    "index-type-mismatch": "The loaded index is not the expected flat index type.",
    "metric-mismatch": "The index does not use inner-product similarity.",
    "dimension-mismatch": "The index dimension does not match the embedding config.",
    "count-mismatch": "Vector count and chunk count disagree.",
    "config-mismatch": "The manifest disagrees with the configured embedding model.",
    "vectors-not-normalized": "Stored vectors are not unit-norm, so cosine "
                              "similarity is not valid for this index.",
    "chunk-inventory-mismatch": "chunks.json does not match the index build. "
                                "Rebuild the index or update "
                                "EXPECTED_CHUNK_INVENTORY_SHA256.",
    "chunk-content-mismatch": "Chunk content or metadata changed without a "
                              "matching index rebuild. Recover the real values "
                              "with 'python vector_store.py "
                              "--print-actual-fingerprints'.",
    "vector-fingerprint-mismatch": "index.faiss does not match the recorded build. "
                                   "Rebuild the index or update "
                                   "EXPECTED_VECTOR_FINGERPRINT_SHA256.",
    "dependency-missing": "Install the backend requirements (faiss-cpu, numpy).",
    "embedding-config-invalid": "Check GEMINI_EMBEDDING_MODEL / GEMINI_EMBEDDING_DIM "
                                "in backend/.env.",
}


def _failure(reason: str, what: str) -> VectorStoreUnavailableError:
    hint = _REASON_HINTS.get(reason, "")
    return VectorStoreUnavailableError(
        f"Vector store is unavailable: {what}.",
        reason=reason,
        detail=hint,
    )


# ── Path helpers ─────────────────────────────────────────────────────────────


def _paths() -> dict[str, Path]:
    """Resolve artifact paths at call time, independent of the CWD."""
    base = Path(VECTOR_STORE_DIR)
    return {
        "index": base / INDEX_FILENAME,
        "chunks": base / CHUNKS_FILENAME,
        "manifest": base / MANIFEST_FILENAME,
    }


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_text_file(path: Path, reason: str, label: str) -> str:
    if not path.exists():
        raise _failure(reason, f"{label} '{path.name}' was not found in '{path.parent.name}'.")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Only the file name is reported; decoding errors can quote bytes content.
        raise _failure("unreadable-file", f"{label} '{path.name}' could not be read.") from None


# ── Embedding configuration (single source of truth) ────────────────────────


def _configured_embedding() -> tuple[str, int]:
    """
    Return the (model, dimension) the project is configured to embed with.

    Reuses index_builder.embedding_config(), which reads the environment the
    same way embeddings.py does and makes no network request.
    """
    from index_builder import IndexBuildError, embedding_config

    try:
        return embedding_config()
    except IndexBuildError:
        raise _failure(
            "embedding-config-invalid",
            "The configured embedding model or dimension is unusable.",
        ) from None


# ── Manifest validation ─────────────────────────────────────────────────────


_REQUIRED_MANIFEST_FIELDS = (
    "embedding_model",
    "embedding_dimension",
    "num_chunks",
    "index_type",
    "vector_dtype",
)


def _parse_manifest(raw: str) -> dict[str, Any]:
    try:
        manifest = json.loads(raw)
    except (ValueError, RecursionError):
        raise _failure("invalid-manifest", f"'{MANIFEST_FILENAME}' is not valid JSON.") from None

    if not isinstance(manifest, dict):
        raise _failure("invalid-manifest", f"'{MANIFEST_FILENAME}' is not a JSON object.")

    missing = [f for f in _REQUIRED_MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise _failure(
            "invalid-manifest",
            f"'{MANIFEST_FILENAME}' is missing field(s): {', '.join(sorted(missing))}.",
        )

    if not isinstance(manifest["embedding_model"], str) or not manifest["embedding_model"].strip():
        raise _failure("invalid-manifest", "'embedding_model' is not a non-empty string.")
    if not isinstance(manifest["embedding_dimension"], int) or isinstance(
        manifest["embedding_dimension"], bool
    ):
        raise _failure("invalid-manifest", "'embedding_dimension' is not an integer.")
    if not isinstance(manifest["num_chunks"], int) or isinstance(manifest["num_chunks"], bool):
        raise _failure("invalid-manifest", "'num_chunks' is not an integer.")

    return manifest


# ── Chunk metadata validation ───────────────────────────────────────────────


def _parse_chunks(raw: str) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    try:
        chunks = json.loads(raw)
    except (ValueError, RecursionError):
        raise _failure("invalid-chunks", f"'{CHUNKS_FILENAME}' is not valid JSON.") from None

    if not isinstance(chunks, list) or not chunks:
        raise _failure("invalid-chunks", f"'{CHUNKS_FILENAME}' is not a non-empty list.")

    validated: list[dict[str, Any]] = []

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise _failure(
                "invalid-chunks",
                f"Chunk #{position} in '{CHUNKS_FILENAME}' is not an object.",
            )

        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise _failure(
                "invalid-chunks",
                f"Chunk #{position} in '{CHUNKS_FILENAME}' has no usable 'chunk_id'.",
            )

        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _failure(
                "invalid-chunks",
                f"Chunk '{chunk_id.strip()}' has no usable text.",
            )

        validated.append(chunk)

    ids = tuple(str(c["chunk_id"]).strip() for c in validated)

    # Single pass: see each ID once, record only the ones seen again.
    seen: set[str] = set()
    repeated: set[str] = set()
    for chunk_id in ids:
        if chunk_id in seen:
            repeated.add(chunk_id)
        else:
            seen.add(chunk_id)

    if repeated:
        duplicates = sorted(repeated)
        raise _failure(
            "duplicate-chunk-id",
            f"{len(duplicates)} duplicated chunk ID(s) found: {', '.join(duplicates[:3])}"
            f"{' ...' if len(duplicates) > 3 else ''}.",
        )

    return tuple(validated), ids


def _chunk_inventory_sha256(chunk_ids: tuple[str, ...]) -> str:
    """
    Deterministic fingerprint of the chunk ID sequence, in file order.

    Cheap identity check: it detects reordering, additions and removals. It
    cannot see edits to the content of a chunk whose ID is unchanged, which is
    what :func:`_chunk_content_sha256` is for.
    """
    return _sha256_hex("\n".join(chunk_ids).encode("utf-8"))


def _chunk_content_sha256(chunks: tuple[dict[str, Any], ...]) -> str:
    """
    Deterministic fingerprint of the complete ordered chunk records.

    Serialises every field of every chunk with ``sort_keys=True`` and compact,
    fixed separators, then hashes the UTF-8 bytes. Consequences:

    * Object key order in the file does not matter (keys are sorted), so a
      re-serialised but semantically identical file fingerprints the same.
    * List order DOES matter, so reordering chunks changes the digest.
    * Any change to text, source URLs, titles, institutions, evidence level or
      any other field changes the digest, even when the chunk IDs are untouched.

    Note that this hashes the values as parsed from JSON, so it is invariant to
    incidental whitespace and formatting of ``chunks.json`` itself — it protects
    content, not byte-for-byte file identity.
    """
    payload = json.dumps(
        list(chunks), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _sha256_hex(payload.encode("utf-8"))


# ── Index validation ────────────────────────────────────────────────────────


def _load_index(path: Path) -> "faiss.Index":
    try:
        import faiss
    except ImportError:
        raise _failure("dependency-missing", "The FAISS package is not installed.") from None

    if not path.exists():
        raise _failure(
            "missing-file", f"Index '{path.name}' was not found in '{path.parent.name}'."
        )

    try:
        index = faiss.read_index(str(path))
    except Exception:
        # FAISS error text can include absolute paths; keep only the file name.
        raise _failure("index-load-failed", f"'{path.name}' could not be read.") from None

    if index is None:
        raise _failure("index-load-failed", f"'{path.name}' could not be read.")

    return index


def _check_index_identity(index: "faiss.Index") -> str:
    """Return the metric name, or raise. Guards the inner-product assumption."""
    import faiss

    if type(index).__name__ != EXPECTED_INDEX_TYPE:
        raise _failure(
            "index-type-mismatch",
            f"Expected '{EXPECTED_INDEX_TYPE}' but loaded '{type(index).__name__}'.",
        )

    metric = getattr(index, "metric_type", None)
    if metric != faiss.METRIC_INNER_PRODUCT:
        raise _failure(
            "metric-mismatch",
            "The index is not inner-product, so similarity scores are not cosine.",
        )

    return "METRIC_INNER_PRODUCT"


def _check_vectors(
    index: "faiss.Index",
    *,
    dimension: int,
    chunk_count: int,
    manifest: dict[str, Any],
) -> str:
    """Validate shape, count, finiteness and unit norms; return row fingerprint."""
    import numpy as np

    if index.d != dimension:
        raise _failure(
            "dimension-mismatch",
            f"Index dimension is {index.d} but the embedding configuration is {dimension}.",
        )

    if manifest["embedding_dimension"] != dimension:
        raise _failure(
            "config-mismatch",
            f"The manifest declares dimension {manifest['embedding_dimension']} but the "
            f"configured embedding dimension is {dimension}.",
        )

    if index.ntotal != chunk_count:
        raise _failure(
            "count-mismatch",
            f"The index holds {index.ntotal} vectors but '{CHUNKS_FILENAME}' has "
            f"{chunk_count} chunks.",
        )

    if manifest["num_chunks"] != chunk_count:
        raise _failure(
            "count-mismatch",
            f"The manifest declares {manifest['num_chunks']} chunks but "
            f"'{CHUNKS_FILENAME}' has {chunk_count}.",
        )

    try:
        vectors = index.reconstruct_n(0, index.ntotal)
    except Exception:
        # reconstruct_n is refused for some index types; that is a hard failure,
        # because without it the unit-norm precondition cannot be verified.
        raise _failure(
            "vectors-not-normalized",
            "The stored vectors could not be inspected to confirm normalization.",
        ) from None

    array = np.asarray(vectors, dtype=np.float32)

    if array.shape != (chunk_count, dimension):
        raise _failure(
            "count-mismatch",
            f"Recovered vector matrix has shape {array.shape}, expected "
            f"({chunk_count}, {dimension}).",
        )

    if not np.all(np.isfinite(array)):
        raise _failure("vectors-not-normalized", "Some stored vectors contain non-finite values.")

    norms = np.linalg.norm(array, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise _failure("vectors-not-normalized", "Some stored vectors have a zero norm.")

    deviation = float(np.max(np.abs(norms - 1.0)))
    if deviation > NORM_TOLERANCE:
        raise _failure(
            "vectors-not-normalized",
            f"Vectors are not unit-norm (max deviation {deviation:.6g}); inner product "
            "would not be cosine similarity.",
        )

    canonical = np.ascontiguousarray(array, dtype="<f4")
    return _sha256_hex(canonical.tobytes())


# ── Public API ───────────────────────────────────────────────────────────────

_cached: VectorStore | None = None
_cached_error: VectorStoreUnavailableError | None = None
_lock = threading.Lock()


def _remember_failure(exc: VectorStoreUnavailableError) -> VectorStoreUnavailableError:
    """Store the verdict and return a fresh, equivalent exception to raise."""
    global _cached, _cached_error
    _cached = None
    _cached_error = exc
    return exc.clone()


def load_vector_store(*, force_reload: bool = False) -> VectorStore:
    """
    Return the validated vector store, loading it at most once per process.

    Both outcomes are cached: a validated store is reused, and a validation
    failure is remembered so retrieval code can fall back without re-reading the
    artifacts on every request. ``force_reload=True`` re-derives the verdict from
    disk and replaces the cached state entirely, so a store that was repaired or
    swapped is picked up immediately.

    Raises
    ------
    VectorStoreUnavailableError
        If any required file is missing or corrupt, or the index is
        incompatible with the current embedding configuration.
    """
    global _cached, _cached_error

    if _cached is not None and not force_reload:
        return _cached

    with _lock:
        # Another thread may have settled the state while we waited.
        if _cached is not None and not force_reload:
            return _cached

        if _cached_error is not None and not force_reload:
            raise _cached_error.clone()

        try:
            store = _load_once()
        except VectorStoreUnavailableError as exc:
            raise _remember_failure(exc) from None

        _cached = store
        _cached_error = None
        return store


def reset_cache() -> None:
    """Discard the cached store and any remembered failure. For tests."""
    global _cached, _cached_error
    with _lock:
        _cached = None
        _cached_error = None


def is_available() -> bool:
    """
    Non-raising availability probe for retrieval code.

    Returns False (and caches that verdict) when the artifacts are missing,
    corrupt or incompatible, so callers can fall back without re-reading disk
    on every request.
    """
    try:
        load_vector_store()
        return True
    except VectorStoreUnavailableError:
        return False


def diagnose_vector_store() -> VectorStore:
    """
    Run every validation EXCEPT the pinned-fingerprint comparisons, and return
    the store with its actual fingerprints.

    This exists to break the circular dependency that otherwise makes a
    legitimate rebuild hard to recover from: :func:`load_vector_store` refuses a
    store whose fingerprints no longer match the pinned constants, so it cannot
    be used to report what the new values should be.

    Everything else is still enforced here — file presence, JSON validity, chunk
    schema and usable text, chunk-ID uniqueness, index loadability, index type,
    inner-product metric, dimension, vector count, finiteness and unit-norm
    normalization. A store that fails any of those raises exactly as it would
    from :func:`load_vector_store`.

    Two deliberate limits:

    * The result is NOT cached. The pinned anchors were skipped, so the object is
      unverified against the known-good build and must never be handed to the
      serving path by accident.
    * Reporting a fingerprint is not an approval. This function writes no files,
      changes no constants, and says nothing about whether the vectors
      semantically correspond to the chunks they are paired with. Deciding that a
      rebuilt index is the one you want to serve remains a human step.

    Raises
    ------
    VectorStoreUnavailableError
        For any validation failure other than a pinned-fingerprint mismatch.
    """
    return _load_once(enforce_pins=False)


def _load_once(*, enforce_pins: bool = True) -> VectorStore:
    paths = _paths()
    warnings: list[str] = []

    model, dimension = _configured_embedding()

    manifest = _parse_manifest(_load_text_file(paths["manifest"], "missing-file", "Manifest"))
    chunks, chunk_ids = _parse_chunks(_load_text_file(paths["chunks"], "missing-file", "Chunks"))

    if manifest["index_type"] != EXPECTED_INDEX_TYPE:
        raise _failure(
            "index-type-mismatch",
            f"The manifest declares index_type {manifest['index_type']!r}, expected "
            f"'{EXPECTED_INDEX_TYPE}'.",
        )
    if manifest["vector_dtype"] != EXPECTED_VECTOR_DTYPE:
        raise _failure(
            "config-mismatch",
            f"The manifest declares vector_dtype {manifest['vector_dtype']!r}, expected "
            f"'{EXPECTED_VECTOR_DTYPE}'.",
        )
    if manifest["embedding_model"] != model:
        raise _failure(
            "config-mismatch",
            f"The index was built with embedding model {manifest['embedding_model']!r} "
            f"but the project is configured for {model!r}.",
        )
    similarity = manifest.get("similarity")
    if isinstance(similarity, str) and similarity != EXPECTED_SIMILARITY:
        warnings.append(
            "manifest 'similarity' text differs from the expected inner-product description"
        )

    index = _load_index(paths["index"])
    metric = _check_index_identity(index)

    vector_fingerprint = _check_vectors(
        index,
        dimension=dimension,
        chunk_count=len(chunks),
        manifest=manifest,
    )

    inventory = _chunk_inventory_sha256(chunk_ids)
    content = _chunk_content_sha256(chunks)

    warnings.extend(
        _compare_integrity_pins(
            actual=_actual_fingerprints(
                chunk_inventory_sha256=inventory,
                chunk_content_sha256=content,
                vector_fingerprint_sha256=vector_fingerprint,
            ),
            enforce=enforce_pins,
        )
    )

    stale = False
    kb_hash = _knowledge_base_sha256()
    if kb_hash is None:
        stale = True
        warnings.append("knowledge base could not be read; freshness is unverified")
    elif manifest.get("source_knowledge_base_sha256") != kb_hash:
        stale = True
        warnings.append(
            "knowledge base has changed since the index was built; retrieval reflects "
            "the older corpus"
        )

    return VectorStore(
        index=index,
        chunks=chunks,
        manifest=manifest,
        dimension=dimension,
        metric=metric,
        embedding_model=model,
        chunk_ids=chunk_ids,
        chunk_inventory_sha256=inventory,
        chunk_content_sha256=content,
        vector_fingerprint_sha256=vector_fingerprint,
        integrity_pins_enforced=enforce_pins,
        stale_knowledge_base=stale,
        warnings=tuple(warnings),
    )


# ── Integrity pins ────────────────────────────────────────────────────────────

# (constant name, expected value, human label, failure reason)
def _pinned_integrity() -> tuple[tuple[str, str | None, str, str], ...]:
    """The pinned anchors, read at call time so tests can patch them."""
    return (
        (
            "EXPECTED_CHUNK_INVENTORY_SHA256",
            EXPECTED_CHUNK_INVENTORY_SHA256,
            "chunk ID inventory",
            "chunk-inventory-mismatch",
        ),
        (
            "EXPECTED_CHUNK_CONTENT_SHA256",
            EXPECTED_CHUNK_CONTENT_SHA256,
            "chunk content",
            "chunk-content-mismatch",
        ),
        (
            "EXPECTED_VECTOR_FINGERPRINT_SHA256",
            EXPECTED_VECTOR_FINGERPRINT_SHA256,
            "index vector bytes",
            "vector-fingerprint-mismatch",
        ),
    )


def _actual_fingerprints(
    *,
    chunk_inventory_sha256: str,
    chunk_content_sha256: str,
    vector_fingerprint_sha256: str,
) -> dict[str, str]:
    """Map each pinned constant name to the value computed from the artifacts."""
    return {
        "EXPECTED_CHUNK_INVENTORY_SHA256": chunk_inventory_sha256,
        "EXPECTED_CHUNK_CONTENT_SHA256": chunk_content_sha256,
        "EXPECTED_VECTOR_FINGERPRINT_SHA256": vector_fingerprint_sha256,
    }


def _compare_integrity_pins(
    actual: dict[str, str], *, enforce: bool
) -> list[str]:
    """
    Compare computed fingerprints with the pinned anchors.

    Returns warning strings. Raises :class:`VectorStoreUnavailableError` on the
    first mismatch when ``enforce`` is true. A pin left as ``None`` never passes
    silently: it is reported as unconfigured on every load.
    """
    notes: list[str] = []

    for constant, expected, label, reason in _pinned_integrity():
        value = actual[constant]

        if expected is None:
            notes.append(
                f"{label} integrity pin is not configured ({constant}=None); integrity "
                "comparison was skipped"
            )
            continue

        if value == expected:
            continue

        message = (
            f"{label} fingerprint {value[:12]}… does not match the pinned build "
            f"{expected[:12]}…"
        )
        if enforce:
            raise _failure(reason, message + ".")

        notes.append(message + " (diagnostic run: comparison reported, not enforced)")

    return notes


def _knowledge_base_sha256() -> str | None:
    """
    Hash the knowledge base using index_builder's own recipe.

    index_builder.manifest_record() hashes the UTF-8 *decoded then re-encoded*
    text, so this must do the same to be comparable. Returns None if unreadable.
    """
    try:
        text = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return _sha256_hex(text.encode("utf-8"))


# ── Diagnostics CLI ──────────────────────────────────────────────────────────

_USAGE = """\
usage: vector_store.py [-h] [--print-actual-fingerprints]

  (no arguments)                 Validate and report the vector store using the
                                 serving-path rules, including the pinned
                                 integrity anchors. Exits 2 if unavailable.

  --print-actual-fingerprints    Diagnostic mode. Runs every validation except
                                 the pinned-anchor comparison, then prints the
                                 ACTUAL fingerprints for the current artifacts
                                 and how they differ from what is pinned. Use it
                                 after an intentional index rebuild to obtain the
                                 values to paste into the constants by hand.

This command never writes files, never edits this module and never contacts
Gemini or generates embeddings.
"""

_RECIPE_NOTE = (
    "Fingerprint recipes (all SHA-256, hex digest):\n"
    "  inventory : sha256('\\n'.join(chunk_id) in chunks.json order, utf-8)\n"
    "  content   : sha256(json.dumps(chunks, sort_keys=True,\n"
    "              separators=(',',':'), ensure_ascii=False).encode('utf-8'))\n"
    "              - every field of every chunk, list order preserved\n"
    "  vectors   : sha256(index.reconstruct_n(0, ntotal) as little-endian float32 bytes)"
)


def _pin_table(actual: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """(constant name, actual, pinned, verdict) for each integrity anchor."""
    rows: list[tuple[str, str, str, str]] = []
    for constant, expected, _label, _reason in _pinned_integrity():
        value = actual[constant]
        if expected is None:
            verdict = "UNCONFIGURED"
        elif value == expected:
            verdict = "match"
        else:
            verdict = "MISMATCH"
        rows.append((constant, value, expected or "-", verdict))
    return rows


def main(argv: list[str] | None = None) -> int:
    """
    Report vector store status, or print the actual fingerprints.

        python vector_store.py                          # strict serving check
        python vector_store.py --print-actual-fingerprints   # diagnostic
    """
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)

    if "-h" in argv or "--help" in argv:
        print(_USAGE)
        return 0

    unknown = [a for a in argv if a not in ("--print-actual-fingerprints",)]
    if unknown:
        print(f"unrecognised argument(s): {', '.join(unknown)}\n", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 64

    diagnostic = "--print-actual-fingerprints" in argv

    try:
        if diagnostic:
            # Pins are reported, not enforced, so this works after a rebuild
            # that legitimately changed every artifact.
            store = diagnose_vector_store()
        else:
            store = load_vector_store(force_reload=True)
    except VectorStoreUnavailableError as exc:
        print(f"Vector store unavailable: reason={exc.reason}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 2

    actual = _actual_fingerprints(
        chunk_inventory_sha256=store.chunk_inventory_sha256,
        chunk_content_sha256=store.chunk_content_sha256,
        vector_fingerprint_sha256=store.vector_fingerprint_sha256,
    )

    if diagnostic:
        print("VECTOR STORE FINGERPRINT DIAGNOSTIC")
        print("  Validation ran: manifest, chunk schema and text, chunk-ID uniqueness,")
        print("                  index load/type/metric/dimension/count, finiteness,")
        print("                  unit-norm normalization.")
        print("  Validation SKIPPED: comparison against the pinned EXPECTED_* constants.")
        print()
        print(f"index_type      : {type(store.index).__name__} ({store.metric})")
        print(f"dimension       : {store.dimension}")
        print(f"vectors/chunks  : {store.vector_count}")
        print(f"embedding model : {store.embedding_model}")
        print(f"built_at        : {store.manifest.get('built_at')}")
        print(f"stale KB        : {store.stale_knowledge_base}")
        print()
        print("ACTUAL fingerprints (these describe the files as they are now):")
        for constant, value, pinned, verdict in _pin_table(actual):
            print(f"  {constant} = {value!r}")
            print(f"      pinned: {pinned}")
            print(f"      status: {verdict}")
        print()
        print(_RECIPE_NOTE)
        print()
        print("IMPORTANT: printing a fingerprint does not certify it.")
        print("  * It proves nothing about semantic row-to-chunk alignment; only")
        print("    index_builder.build_index()'s append order does, and that cannot be")
        print("    re-derived here without calling Gemini to re-embed the corpus.")
        print("  * It does not approve a rebuild. Paste these values into the")
        print("    EXPECTED_* constants only after you have confirmed that this index")
        print("    is the build you intend to serve.")
        print("  * Ordinary loading (no flag) still rejects any mismatch.")
        return 0

    print(f"index_type      : {type(store.index).__name__} ({store.metric})")
    print(f"dimension       : {store.dimension}")
    print(f"vectors/chunks  : {store.vector_count}")
    print(f"embedding model : {store.embedding_model}")
    print(f"built_at        : {store.manifest.get('built_at')}")
    print(f"stale KB        : {store.stale_knowledge_base}")
    print(f"inventory       : {store.chunk_inventory_sha256}")
    print(f"chunk content   : {store.chunk_content_sha256}")
    print(f"vector fp       : {store.vector_fingerprint_sha256}")
    for warning in store.warnings:
        print(f"  warning: {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
