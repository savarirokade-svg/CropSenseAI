"""
hybrid_retrieval.py — Hybrid (semantic + lexical) retrieval for CropSense AI.

Combines two ranking signals over ONE validated corpus and fuses them with
Reciprocal Rank Fusion:

1. Semantic search — the query is embedded with ``embeddings.embed_query()`` and
   searched against the FAISS index held by ``vector_store.load_vector_store()``.
2. Lexical search — a deterministic BM25 ranking computed in-process over the
   same ``store.chunks`` sequence, using only the Python standard library.
3. Reciprocal Rank Fusion — ``RRF(d) = sum(1 / (k + rank(d)))`` over the two
   rankings, default ``k = 60``, one-based ranks.

Both signals address the identical chunk objects, so a fused chunk always carries
full source provenance (title, URL, institution, evidence level). The legacy
keyword corpus in ``retrieval.py`` / ``data/tomato_potato_guide.txt`` is
deliberately NOT used here: it holds sample text with no provenance, which would
surface as an unattributable citation in a grounded answer.

Nothing in this module decides what to *do* with the chunks; it only ranks and
returns them. Answer generation is ``generator.py``'s job, and wiring this into
``/chat`` is Task 11f.4.

Design notes
------------
* **No similarity threshold, anywhere.** FAISS inner-product scores are returned
  for diagnostics only and never gate a result, because ``IndexFlatIP`` always
  returns its top-k rows: with a seven-chunk corpus every query gets candidates
  whether or not they are relevant. Sufficiency must be decided by the caller
  (the crop guard planned for 11f.4, plus the generator's own
  insufficient-evidence rule), not by a cutoff invented here.
* **RRF uses ranks only.** Raw similarities are never added into
  ``rrf_score``. The two scales are unrelated — with two rankings the maximum
  fused score is ``2 / (k + 1)`` (about 0.033 at k=60), while cosine similarity
  spans ``[-1, 1]`` — so a fused score must never be compared to a similarity or
  thresholded using similarity intuition.
* ``rrf_score`` is **not** a confidence, a probability, or a measure of
  diagnostic certainty. Its magnitude depends on ``k`` and on how many rankings
  a chunk appears in; only the ordering it produces is meaningful.
* Errors from the embedding and vector-store layers are re-raised as this
  module's own documented exception types, which carry a failure *category*
  only. Raw exception text from those layers, or from the Gemini SDK beneath
  them, never crosses this boundary.
"""

from __future__ import annotations

import copy
import math
import os
import re
import threading
import unicodedata
from collections import Counter
from typing import Any

# ── Configuration ────────────────────────────────────────────────────────────

#: Standard RRF smoothing constant. Not a threshold; it only damps rank 1.
RRF_K = 60

#: BM25 term-saturation and length-normalisation parameters. Fixed for
#: determinism. BM25 scores are scale-free, so these are not similarity cutoffs.
BM25_K1 = 1.2
BM25_B = 0.75

#: How many candidates each signal contributes to the fusion pool, as a multiple
#: of top_k with a floor, capped by the corpus size. A breadth choice, not a
#: relevance cutoff: it decides how far each ranking is consulted.
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATE_DEPTH = 20

#: Field weights for the lexical document model. Disease and section names match
#: a farmer's phrasing ("early blight", "symptoms") far more sharply than the
#: chunk body, so their tokens are repeated into the term counts.
FIELD_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("text", 1),
    ("disease_name", 3),
    ("section_name", 2),
    ("subsections", 2),
    ("crop", 2),
    ("source_title", 1),
)

DEFAULT_TOP_K = 5
MAX_TOP_K = 50

#: Query vocabulary -> the chunk section labels it should promote, with the
#: additive bonus applied when ordering the fused results. RRF scores for this
#: corpus sit around 0.016-0.033, so these bonuses break ties and overtake
#: near-ties without letting intent outrank a clearly more relevant chunk. They
#: never remove a result: secondary evidence stays available below the preferred
#: section.
SECTION_INTENT_RULES: tuple[tuple[frozenset[str], tuple[str, ...], float], ...] = (
    (
        frozenset({
            "symptom", "symptoms", "sign", "signs", "lesion", "lesions",
            "spot", "spots", "blotch", "blotches", "yellowing", "wilt",
            "wilting", "wilted", "discoloration", "discoloured", "discolored",
        }),
        ("symptoms",),
        0.0020,
    ),
    (
        frozenset({
            "cause", "causes", "caused", "causal", "organism", "pathogen",
            "pathogens", "why", "spread", "transmission", "transmitted",
            "infection", "infect", "source",
        }),
        ("disease spread and causes", "favourable environmental conditions"),
        0.0014,
    ),
    (
        frozenset({
            "prevent", "prevention", "preventive", "avoid", "avoiding",
            "prophylaxis", "sanitation", "rotation", "cultural", "cultivation",
            "spacing", "irrigation", "field",
        }),
        ("prevention and cultural management",),
        0.0016,
    ),
    (
        frozenset({"biological", "bio", "biocontrol", "biofungicide", "antagonist", "antagonists"}),
        ("biological management",),
        0.0016,
    ),
    (
        frozenset({
            "chemical", "chemicals", "fungicide", "fungicides", "pesticide",
            "pesticides", "spray", "sprays", "dosage", "dose", "application",
        }),
        ("chemical management",),
        0.0020,
    ),
    (
        frozenset({"evidence", "limitation", "limitations", "warning", "warnings", "safety", "verified", "verification"}),
        ("evidence limitations and safety warnings",),
        0.0014,
    ),
    (
        frozenset({"management", "manage", "control", "treat", "treatment"}),
        (
            "prevention and cultural management",
            "biological management",
            "chemical management",
        ),
        0.0008,
    ),
)

#: Cap for the summed section bonus, so several intent words cannot stack into a
#: stronger signal than a genuinely better-matching chunk.
MAX_SECTION_INTENT_BONUS = 0.0040

#: Small English stopword list, owned by this module. It cannot reuse
#: retrieval.py's list without importing the legacy .txt corpus this module is
#: designed to avoid, and the corpus is seven chunks, so recall is not at risk.
STOP_WORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    he her him his how i if in into is it its me my no nor of on or our out she
    should so than that the their them then there these they this to us was we
    were what when where which who whom why will with would you your
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_COMBINING_MARKS = re.compile(r"[\u0300-\u036f]")

# ── Exceptions ───────────────────────────────────────────────────────────────


class HybridRetrievalError(Exception):
    """
    Base class for every failure raised by this module.

    Attributes
    ----------
    reason:
        A short, stable failure category. Safe to log or branch on.
    """

    def __init__(self, message: str, *, reason: str = "retrieval-failed"):
        super().__init__(message)
        self.reason = reason


class InvalidQueryError(HybridRetrievalError):
    """Raised when the query is missing, not a string, or has no usable terms."""

    def __init__(self, message: str):
        super().__init__(message, reason="invalid-query")


class InvalidTopKError(HybridRetrievalError):
    """Raised when top_k is not an integer within the allowed range."""

    def __init__(self, message: str):
        super().__init__(message, reason="invalid-top-k")


class CorpusUnavailableError(HybridRetrievalError):
    """
    Raised when the validated chunk corpus cannot be loaded.

    Both signals read the same corpus from ``vector_store``, so a missing,
    corrupt or incompatible vector store stops hybrid retrieval entirely. The
    vector store's failure category is carried through in ``reason`` (never its
    raw message). There is deliberately no fallback to the legacy sample corpus.
    """

    def __init__(self, message: str, *, vector_store_reason: str = "unavailable"):
        super().__init__(message, reason=f"vector-store-{vector_store_reason}")
        self.vector_store_reason = vector_store_reason


class SemanticSearchUnavailableError(HybridRetrievalError):
    """
    Raised when the semantic signal cannot run.

    ``hybrid_search`` only raises this if the caller passes
    ``require_semantic=True``; by default it degrades to lexical-only ranking
    and reports the reason through :func:`retrieval_status` and the
    ``semantic_rank = None`` entries in the results.
    """

    def __init__(self, message: str, *, reason: str):
        super().__init__(message, reason=f"semantic-{reason}")


# ── Internal: corpus access ──────────────────────────────────────────────────

_corpus_lock = threading.Lock()
_corpus_state: dict[str, Any] = {"store": None, "chunks": (), "ids": (), "rows": {}}
_tables_state: dict[str, Any] = {"store": None, "tables": None}
_diagnostics_lock = threading.Lock()
_last_search: dict[str, Any] | None = None


def _corpus():
    """
    Return ``(store, chunks, chunk_ids, row_by_chunk_id)`` for the validated corpus.

    The vector store is already cached inside ``vector_store``, so this module
    only adds an id->row map and a BM25 table, both keyed on the identity of that
    cached object. Rebuilding the index or calling ``vector_store.reset_cache()``
    yields a different store object, so the derived state rebuilds itself and
    never pairs fresh tables with a swapped corpus.
    """
    import vector_store

    try:
        store = vector_store.load_vector_store()
    except vector_store.VectorStoreUnavailableError as exc:
        raise CorpusUnavailableError(
            "The validated chunk corpus is unavailable, so hybrid retrieval cannot "
            "run. Build the index with 'python index_builder.py --build' and confirm "
            "the pinned fingerprints in vector_store.py.",
            vector_store_reason=exc.reason,
        ) from None

    with _corpus_lock:
        if _corpus_state["store"] is not store:
            _corpus_state.update(
                store=store,
                chunks=store.chunks,
                ids=store.chunk_ids,
                rows={cid: row for row, cid in enumerate(store.chunk_ids)},
            )
        chunks = _corpus_state["chunks"]
        ids = _corpus_state["ids"]
        rows = _corpus_state["rows"]

    return store, chunks, ids, rows


def _tables(store, chunks) -> dict[str, Any]:
    """Cached BM25 statistics for this exact store object."""
    with _corpus_lock:
        if _tables_state["store"] is not store:
            _tables_state.update(store=store, tables=_build_lexical_tables(chunks))
        return _tables_state["tables"]


def reset_cache() -> None:
    """
    Drop this module's derived lookup tables.

    Tests should call this after replacing artifacts or after
    ``vector_store.reset_cache()``. Retrieval also self-invalidates when the
    cached store object changes, so this is the explicit hook rather than a
    requirement.
    """
    with _corpus_lock:
        _corpus_state.update(store=None, chunks=(), ids=(), rows={})
        _tables_state.update(store=None, tables=None)


def _record_diagnostics(**fields: Any) -> None:
    global _last_search
    with _diagnostics_lock:
        _last_search = dict(fields)


def _last_search_snapshot() -> dict[str, Any] | None:
    with _diagnostics_lock:
        return dict(_last_search) if _last_search is not None else None


# ── Internal: tokenisation ───────────────────────────────────────────────────


def _as_text(value: Any) -> str:
    """Coerce a chunk field to text. Lists (e.g. 'subsections') are joined."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if isinstance(item, (str, int, float)))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _tokenize(text: str) -> list[str]:
    """
    Normalise and split text into lowercase alphanumeric terms.

    Accents are folded (``café`` -> ``cafe``) with the standard library, and both
    query and document sides call this one function, so the two normalisations
    cannot drift apart.
    """
    if not text:
        return []
    decomposed = unicodedata.normalize("NFKD", text)
    folded = _COMBINING_MARKS.sub("", decomposed).lower()
    return _TOKEN_RE.findall(folded)


def _query_terms(query: str) -> list[str]:
    """Content terms of the query: stopword-filtered, deduplicated, sorted."""
    return sorted({term for term in _tokenize(query) if term not in STOP_WORDS})


# ── Internal: section intent ─────────────────────────────────────────────────


def detect_section_intents(query: str) -> dict[str, float]:
    """
    Map a query's wording onto the section labels it asks about.

    Returns ``{section_label: bonus}``. Matching is vocabulary-driven and
    deterministic; the highest weight wins when several rules name the same
    label. An empty dict means the query expresses no section preference, in
    which case ranking is left exactly as the fused RRF order produced it.
    """
    terms = set(_tokenize(query))
    bonuses: dict[str, float] = {}

    for vocabulary, labels, weight in SECTION_INTENT_RULES:
        if not terms & vocabulary:
            continue
        for label in labels:
            bonuses[label] = max(bonuses.get(label, 0.0), weight)

    return bonuses


def _section_intent_bonus(chunk: dict[str, Any], bonuses: dict[str, float]) -> float:
    """Additive RRF bonus for a chunk whose sections match the query intent."""
    if not bonuses:
        return 0.0

    names = [str(name).lower() for name in (chunk.get("subsections") or [])]
    names.append(str(chunk.get("section_name") or "").lower())
    haystack = " | ".join(names)

    total = sum(weight for label, weight in bonuses.items() if label in haystack)
    return min(total, MAX_SECTION_INTENT_BONUS)


# ── Internal: semantic signal ────────────────────────────────────────────────


def _embedding_reason(exc: BaseException) -> str:
    """Map an embeddings exception to a stable category name."""
    import embeddings

    if isinstance(exc, embeddings.MissingApiKeyError):
        return "missing-api-key"
    if isinstance(exc, embeddings.EmptyInputError):
        return "empty-input"
    if isinstance(exc, embeddings.EmptyEmbeddingError):
        return "empty-embedding"
    if isinstance(exc, embeddings.InvalidEmbeddingError):
        return "invalid-embedding"
    if isinstance(exc, embeddings.EmbeddingApiError):
        return "api-error"
    return "embedding-failed"


def _validated_embedding(raw: Any, *, expected_dim: int):
    """
    Convert and validate an embedding into a contiguous float32 row matrix.

    Checks, in order: non-empty, exactly one-dimensional, the expected
    dimensionality, and all values finite. ``embeddings.embed_query`` already
    returns a 1-D L2-normalised float32 ``(dim,)`` array, so this guards against a
    changed contract or an unexpected return value rather than re-implementing it.

    A dimension mismatch is the signature of an index built with a different
    embedding model or width, so it is rejected here instead of being handed to
    FAISS, which would fail with an opaque internal error.

    Returns a ``(1, expected_dim)`` array, which is the shape FAISS expects.

    Raises
    ------
    SemanticSearchUnavailableError
        With reason ``empty-embedding``, ``invalid-shape``, ``dimension-mismatch``
        or ``non-finite-embedding``.
    """
    import numpy as np

    if raw is None:
        raise SemanticSearchUnavailableError(
            "The embedding service returned no vector.", reason="empty-embedding"
        )

    try:
        array = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError):
        raise SemanticSearchUnavailableError(
            "The embedding response is not numeric and cannot be searched.",
            reason="invalid-shape",
        ) from None

    if array.ndim != 1:
        raise SemanticSearchUnavailableError(
            f"Expected a 1-D embedding vector, got a {array.ndim}-D array with shape "
            f"{array.shape}.",
            reason="invalid-shape",
        )

    if array.size == 0:
        raise SemanticSearchUnavailableError(
            "The embedding service returned an empty vector.", reason="empty-embedding"
        )

    if array.shape[0] != expected_dim:
        raise SemanticSearchUnavailableError(
            f"The embedding dimension is {array.shape[0]} but the validated index "
            f"expects {expected_dim}. The index was probably built with a different "
            "embedding configuration.",
            reason="dimension-mismatch",
        )

    if not bool(np.all(np.isfinite(array))):
        raise SemanticSearchUnavailableError(
            "The embedding contains non-finite values and was rejected.",
            reason="non-finite-embedding",
        )

    # FAISS requires a C-contiguous 2-D float32 block of queries: (n, d).
    return np.ascontiguousarray(array.reshape(1, expected_dim), dtype="<f4")


def _embed_and_search(store, query: str, depth: int) -> tuple[list[str], dict[str, float]]:
    """
    Embed ``query`` and run the FAISS search.

    Returns ``(ranked_chunk_ids, chunk_id_to_similarity)``. The similarities are
    diagnostic only and never enter the fusion score.

    FAISS row ``i`` maps to ``store.chunks[i]`` through ``store.chunk_at``, the
    ordering contract established by ``index_builder.build_index()`` and validated
    by ``vector_store``. Any position outside that range, or repeating an
    already-seen chunk, is dropped rather than trusted.

    Raises
    ------
    SemanticSearchUnavailableError
        For embedding-layer failures (category carried through, message not),
        invalid embeddings, or a FAISS search error.
    """
    import embeddings

    try:
        raw = embeddings.embed_query(query)
    except embeddings.EmbeddingError as exc:
        raise SemanticSearchUnavailableError(
            "The query could not be embedded, so the semantic signal was skipped.",
            reason=_embedding_reason(exc),
        ) from None
    except Exception:
        raise SemanticSearchUnavailableError(
            "The query could not be embedded, so the semantic signal was skipped.",
            reason="embedding-failed",
        ) from None

    matrix = _validated_embedding(raw, expected_dim=store.dimension)

    if store.vector_count == 0:
        return [], {}

    k = min(depth, store.vector_count)

    try:
        distances, indices = store.index.search(matrix, k)
    except Exception:
        raise SemanticSearchUnavailableError(
            "The FAISS search could not be executed.", reason="search-failed"
        ) from None

    ranked: list[str] = []
    similarities: dict[str, float] = {}
    seen: set[str] = set()

    for position in range(int(indices.shape[1])):
        row = int(indices[0, position])

        # FAISS pads with -1 once k exceeds the number of stored vectors.
        if row < 0:
            continue

        try:
            chunk = store.chunk_at(row)
        except (IndexError, TypeError):
            continue

        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            continue

        seen.add(chunk_id)
        ranked.append(chunk_id)
        similarities[chunk_id] = float(distances[0, position])

    return ranked, similarities


# ── Internal: lexical signal ─────────────────────────────────────────────────


def _document_tokens(chunk: dict[str, Any]) -> list[str]:
    """Tokenise the weighted lexical fields of one chunk."""
    tokens: list[str] = []
    for field, weight in FIELD_WEIGHTS:
        field_tokens = _tokenize(_as_text(chunk.get(field)))
        tokens.extend(field_tokens if weight == 1 else field_tokens * weight)
    return tokens


def _build_lexical_tables(chunks: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """
    Compute BM25 corpus statistics. A pure function of the corpus, so it is
    cached per store object.
    """
    term_counts = [Counter(_document_tokens(chunk)) for chunk in chunks]
    doc_lengths = [sum(counts.values()) for counts in term_counts]

    document_frequency: Counter[str] = Counter()
    for counts in term_counts:
        for term in counts:
            document_frequency[term] += 1

    total = sum(doc_lengths)
    return {
        "term_counts": term_counts,
        "doc_lengths": doc_lengths,
        "average_length": (total / len(doc_lengths)) if doc_lengths else 0.0,
        "document_frequency": document_frequency,
        "document_count": len(chunks),
    }


def _bm25_term_score(tf: int, dl: int, avgdl: float, n_docs: int, df: int) -> float:
    """Okapi BM25 contribution of one query term in one document."""
    if avgdl <= 0.0 or tf <= 0:
        return 0.0
    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / avgdl))
    if denominator <= 0.0:
        return 0.0
    return idf * (tf * (BM25_K1 + 1.0)) / denominator


def _lexical_ranking(terms: list[str], tables, chunks, depth: int) -> list[str]:
    """
    Rank chunk IDs by BM25 score, dropping chunks with no matching query term.

    Ties break on chunk ID, so the ordering is reproducible across runs and
    independent of dict insertion order or floating-point summation order.
    """
    if not terms or not chunks:
        return []

    n_docs = tables["document_count"]
    document_frequency = tables["document_frequency"]
    avgdl = tables["average_length"]

    scored: list[tuple[float, str]] = []

    for index, chunk in enumerate(chunks):
        counts = tables["term_counts"][index]
        dl = tables["doc_lengths"][index]
        score = 0.0
        matched = False

        for term in terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            matched = True
            score += _bm25_term_score(tf, dl, avgdl, n_docs, document_frequency[term])

        if matched and score > 0.0:
            scored.append((score, str(chunk.get("chunk_id"))))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _score, chunk_id in scored[:depth]]


# ── Internal: fusion ─────────────────────────────────────────────────────────


def _dedupe(chunk_ids) -> list[str]:
    """Keep first-occurrence order, dropping repeated IDs."""
    seen: set[str] = set()
    result: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            result.append(chunk_id)
    return result


def _rank_map(ranked_ids) -> dict[str, int]:
    """One-based rank per chunk ID; position in the ranking defines the rank."""
    return {chunk_id: rank for rank, chunk_id in enumerate(_dedupe(ranked_ids), start=1)}


def fuse_rankings(
    semantic_ids,
    lexical_ids,
    *,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float, int | None, int | None]]:
    """
    Reciprocal Rank Fusion over two chunk-ID rankings.

    ``RRF(d) = sum over the lists containing d of 1 / (rrf_k + rank(d))``, using
    one-based ranks only. Raw FAISS similarities are never added in.

    Returns ``(chunk_id, rrf_score, semantic_rank, lexical_rank)`` sorted by
    descending score, then chunk ID. A chunk present in only one ranking
    contributes just that term, and its missing rank is reported as ``None``
    rather than being treated as a last place. Inputs are deduplicated by chunk
    ID, so a repeated ID cannot produce a repeated result.

    Pure with respect to the corpus: it never touches the vector store, so the
    arithmetic can be tested on its own.
    """
    semantic_ranks = _rank_map(semantic_ids)
    lexical_ranks = _rank_map(lexical_ids)

    fused: list[tuple[str, float, int | None, int | None]] = []

    for chunk_id in set(semantic_ranks) | set(lexical_ranks):
        score = 0.0
        if chunk_id in semantic_ranks:
            score += 1.0 / (rrf_k + semantic_ranks[chunk_id])
        if chunk_id in lexical_ranks:
            score += 1.0 / (rrf_k + lexical_ranks[chunk_id])
        fused.append(
            (chunk_id, score, semantic_ranks.get(chunk_id), lexical_ranks.get(chunk_id))
        )

    fused.sort(key=lambda item: (-item[1], item[0]))
    return fused


# ── Internal: input validation ───────────────────────────────────────────────


def _validated_query(query: Any) -> str:
    if query is None:
        raise InvalidQueryError("query is required.")
    if not isinstance(query, str):
        raise InvalidQueryError(f"query must be a string, got {type(query).__name__}.")
    if not query.strip():
        raise InvalidQueryError("query must not be empty or whitespace only.")
    return query.strip()


def _validated_top_k(top_k: Any, *, name: str = "top_k") -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise InvalidTopKError(f"{name} must be an integer, got {type(top_k).__name__}.")
    if top_k < 1 or top_k > MAX_TOP_K:
        raise InvalidTopKError(f"{name} must be between 1 and {MAX_TOP_K}, got {top_k}.")
    return top_k


def _validated_k(rrf_k: Any) -> int:
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
        raise InvalidQueryError(f"rrf_k must be an integer, got {type(rrf_k).__name__}.")
    if rrf_k < 1:
        raise InvalidQueryError(f"rrf_k must be a positive integer, got {rrf_k}.")
    return rrf_k


def _annotated(chunk: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """A deep copy of a corpus chunk plus retrieval metadata. Never mutates the store."""
    result = copy.deepcopy(chunk)
    result.update(fields)
    return result


# ── Public API ───────────────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    rrf_k: int = RRF_K,
    require_semantic: bool = False,
    disease_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Rank the validated chunk corpus for ``query`` using hybrid retrieval.

    Parameters
    ----------
    query:
        The user's question. Must be a non-empty string. It is used only as
        retrieval text; nothing here treats it as an instruction.
    top_k:
        Maximum number of chunks to return; an integer in ``1..MAX_TOP_K``.
        Fewer are returned when the corpus or the fused pool is smaller.
    rrf_k:
        The RRF smoothing constant. Integer >= 1, default 60.
    require_semantic:
        When True, a failed semantic signal raises
        :class:`SemanticSearchUnavailableError` instead of degrading to
        lexical-only ranking. Default False, because lexical results are still
        grounded in the same validated corpus.
    disease_names:
        Optional set of canonical ``disease_name`` values. When supplied, only
        chunks of those diseases are returned, and every chunk is considered
        (the top-k cut is applied after filtering). This is how an explicitly
        named disease receives priority over related diseases without letting
        another disease's evidence into the answer. Section-intent ranking still
        orders the surviving chunks. ``None`` keeps the normal whole-corpus
        behaviour.

    Returns
    -------
    list[dict]
        Deep copies of the highest-ranked chunks, best first. Every original
        field (``chunk_id``, ``text``, ``source_title``, ``source_url``,
        ``source_institution``, ``source_access_date``, ``source_access_status``,
        ``evidence_level``, ``section_name``, ``crop``, ``disease_name`` ...) is
        preserved unchanged, with these retrieval fields added:

        ``rrf_score``
            The fused RRF score, derived from ranks only. NOT a confidence, a
            probability, or a measure of diagnostic certainty, and not comparable
            to a similarity score.
        ``section_intent_bonus``
            The additive bonus applied for matching the query's section intent
            (symptoms, causes, chemical management, ...). It orders results but is
            not part of ``rrf_score`` and is not a confidence either.
        ``semantic_rank``
            One-based rank in the semantic ranking, or ``None`` if the chunk came
            only from the lexical ranking — including when semantic search was
            skipped entirely.
        ``lexical_rank``
            One-based rank in the lexical ranking, or ``None``.
        ``faiss_row``
            The index row the semantic signal matched, or ``None`` when the chunk
            arrived only through the lexical ranking.
        ``semantic_similarity``
            Raw FAISS inner product (== cosine, the stored vectors are unit-norm),
            for offline calibration only. ``None`` when there was no semantic hit.
            Never used for ranking or gating.
        ``retrieval_source``
            ``"hybrid"``, ``"semantic_only"`` or ``"lexical_only"``: which signals
            placed this chunk in the result.

    Raises
    ------
    InvalidQueryError
        If the query is missing, not a string, or blank.
    InvalidTopKError
        If top_k is not an integer in range.
    CorpusUnavailableError
        If the validated vector store cannot be loaded. There is no fallback to
        the legacy sample corpus.
    SemanticSearchUnavailableError
        Only when ``require_semantic=True`` and the semantic signal failed.
    """
    text = _validated_query(query)
    k = _validated_top_k(top_k)
    smoothing = _validated_k(rrf_k)
    requested_diseases = {str(name) for name in disease_names} if disease_names else set()

    store, chunks, _ids, rows = _corpus()

    if not chunks:
        _record_diagnostics(
            semantic_used=False,
            semantic_reason="empty-corpus",
            lexical_used=False,
            stale_knowledge_base=store.stale_knowledge_base,
            candidate_count=0,
            returned=0,
            query_terms=0,
        )
        return []

    terms = _query_terms(text)

    # When an explicit disease is requested, consult every chunk: the top-k cut
    # is applied after the disease filter, so the named disease's evidence can
    # never be displaced by unrelated chunks that happen to score higher.
    depth = (
        len(chunks)
        if requested_diseases
        else min(max(k * CANDIDATE_MULTIPLIER, MIN_CANDIDATE_DEPTH), len(chunks))
    )

    semantic_ids: list[str] = []
    similarities: dict[str, float] = {}
    semantic_reason: str | None = None

    try:
        semantic_ids, similarities = _embed_and_search(store, text, depth)
    except SemanticSearchUnavailableError as exc:
        if require_semantic:
            raise
        semantic_reason = exc.reason

    lexical_ids = _lexical_ranking(terms, _tables(store, chunks), chunks, depth)
    fused = fuse_rankings(semantic_ids, lexical_ids, rrf_k=smoothing)

    section_bonuses = detect_section_intents(text)
    ranked: list[tuple[str, float, float, int | None, int | None]] = []

    for chunk_id, score, semantic_rank, lexical_rank in fused:
        row = rows.get(chunk_id)
        if row is None:
            # A ranked ID with no row in the validated corpus is a contradiction.
            # Drop it rather than fabricate a chunk.
            continue

        chunk = chunks[row]

        if requested_diseases and str(chunk.get("disease_name")) not in requested_diseases:
            continue

        # Section intent reorders the surviving candidates; it never filters.
        bonus = _section_intent_bonus(chunk, section_bonuses)
        ranked.append((chunk_id, score, bonus, semantic_rank, lexical_rank))

    ranked.sort(key=lambda item: (-(item[1] + item[2]), item[0]))

    results: list[dict[str, Any]] = []

    for chunk_id, score, bonus, semantic_rank, lexical_rank in ranked[:k]:
        row = rows[chunk_id]

        if semantic_rank is not None and lexical_rank is not None:
            source = "hybrid"
        elif semantic_rank is not None:
            source = "semantic_only"
        else:
            source = "lexical_only"

        results.append(
            _annotated(
                chunks[row],
                rrf_score=score,
                section_intent_bonus=bonus,
                semantic_rank=semantic_rank,
                lexical_rank=lexical_rank,
                faiss_row=row if semantic_rank is not None else None,
                semantic_similarity=similarities.get(chunk_id),
                retrieval_source=source,
            )
        )

    _record_diagnostics(
        semantic_used=bool(semantic_ids),
        semantic_reason=semantic_reason,
        lexical_used=bool(lexical_ids),
        stale_knowledge_base=store.stale_knowledge_base,
        candidate_count=len(fused),
        returned=len(results),
        query_terms=len(terms),
    )

    return results


def semantic_search(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """
    The FAISS-only ranking, as annotated chunk copies with ``semantic_rank`` and
    ``semantic_similarity``.

    Exposed separately for retrieval evaluation: any future similarity threshold
    must be derived from measured data like this, not assumed inside
    :func:`hybrid_search`.

    Raises :class:`SemanticSearchUnavailableError` when the semantic signal cannot
    run, and :class:`CorpusUnavailableError` when the corpus cannot load.
    """
    text = _validated_query(query)
    store, chunks, _ids, rows = _corpus()

    if chunks:
        wanted = len(chunks) if k is None else _validated_top_k(k, name="k")
        ranked, similarities = _embed_and_search(store, text, max(1, min(wanted, len(chunks))))
    else:
        ranked, similarities = [], {}

    results: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ranked, start=1):
        row = rows.get(chunk_id)
        if row is None:
            continue
        results.append(
            _annotated(
                chunks[row],
                semantic_rank=rank,
                semantic_similarity=similarities.get(chunk_id),
                faiss_row=row,
            )
        )
    return results


def lexical_search(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """
    The BM25-only ranking over the same corpus, as annotated chunk copies with
    ``lexical_rank``.

    Standard library only: no extra dependencies, no index build, no network.
    """
    text = _validated_query(query)
    store, chunks, _ids, rows = _corpus()

    wanted = len(chunks) if k is None else _validated_top_k(k, name="k")
    ranked = _lexical_ranking(
        _query_terms(text), _tables(store, chunks), chunks, max(1, min(wanted, len(chunks)))
    )

    results: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ranked, start=1):
        row = rows.get(chunk_id)
        if row is None:
            continue
        results.append(_annotated(chunks[row], lexical_rank=rank, faiss_row=None))
    return results


def retrieval_status() -> dict[str, Any]:
    """
    Report corpus state, for callers that must not claim freshness they lack.

    Keys
    ----
    available:
        False when the vector store cannot be loaded.
    reason:
        The vector store's failure category when unavailable, else ``None``.
    chunk_count, dimension, embedding_model, index_type, metric:
        Descriptions of the validated corpus.
    stale_knowledge_base:
        Copied directly from ``VectorStore.stale_knowledge_base``. True means the
        knowledge base changed since this index was built, or could not be
        hashed. It is surfaced, never suppressed.
    freshness_note:
        A plain-language statement of the above, so a caller cannot imply the
        index is current by omission.
    warnings:
        The vector store's own warning strings.
    integrity_pins_enforced:
        Whether the pinned fingerprint anchors were compared on load.
    semantic_configured:
        Whether ``GEMINI_API_KEY`` is present. A configuration check, not a
        connectivity test, and not a record of the last search.
    last_search:
        Diagnostics from the most recent :func:`hybrid_search` in this process, or
        ``None`` before the first. ``semantic_used`` is False whenever the
        semantic signal was skipped or returned nothing, and ``semantic_reason``
        carries the category when it was skipped.

    This function never raises for an unavailable store, so it is safe to call on
    a request path that must not fail.
    """
    import vector_store

    try:
        store = vector_store.load_vector_store()
    except vector_store.VectorStoreUnavailableError as exc:
        return {
            "available": False,
            "reason": exc.reason,
            "chunk_count": 0,
            "dimension": None,
            "embedding_model": None,
            "index_type": None,
            "metric": None,
            "stale_knowledge_base": True,
            "freshness_note": (
                "The corpus could not be loaded, so no freshness claim can be made."
            ),
            "warnings": (str(exc),),
            "integrity_pins_enforced": None,
            "semantic_configured": _api_key_present(),
            "last_search": _last_search_snapshot(),
        }

    if store.stale_knowledge_base:
        note = (
            "The knowledge base has changed since this index was built, or could not "
            "be hashed, so retrieval may reflect an older corpus. Rebuild with "
            "'python index_builder.py --build'."
        )
    else:
        note = (
            "The knowledge base hash matches the index build recorded in "
            "manifest.json."
        )

    return {
        "available": True,
        "reason": None,
        "chunk_count": store.vector_count,
        "dimension": store.dimension,
        "embedding_model": store.embedding_model,
        "index_type": type(store.index).__name__,
        "metric": store.metric,
        "stale_knowledge_base": store.stale_knowledge_base,
        "freshness_note": note,
        "warnings": tuple(store.warnings),
        "integrity_pins_enforced": store.integrity_pins_enforced,
        "semantic_configured": _api_key_present(),
        "last_search": _last_search_snapshot(),
    }


def is_available() -> bool:
    """True when the validated corpus can be loaded, which is all lexical search needs."""
    import vector_store

    return vector_store.is_available()


def _api_key_present() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())
