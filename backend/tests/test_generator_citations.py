"""
Offline regression tests for citation parsing and the insufficient-evidence mapping.

Two confirmed submission issues are covered here:

1. ``generator.py`` only understood single-number citations, so a reply citing
   "[1, 4]" produced no parsable citation and listed no sources.
2. A citation-less reply (how the model reports that an abstract-only corpus
   cannot answer) reached the generic "couldn't produce a verified answer"
   branch instead of the designed insufficient-evidence reply.

Every Gemini call is replaced by the autouse fixtures in ``conftest.py``, so
nothing here touches the network.
"""

from __future__ import annotations

import pytest

import generator
import main
from conftest import TOMATO_QUESTION


@pytest.fixture(autouse=True)
def clean_database():
    """These tests use mocks and never touch PostgreSQL."""
    yield


CHUNKS = [
    {
        "chunk_id": f"chunk-{index}",
        "text": f"Passage {index}.",
        "source_title": f"Source {index}",
        "source_url": f"https://example.test/source-{index}",
        "source_institution": "Example University",
        "evidence_level": "full_page",
    }
    for index in range(1, 5)
]


# ── 1. Citation extraction ──────────────────────────────────────────────────


def test_individual_citations_are_parsed():
    assert generator._extract_citation_numbers("Spots [1]. Wilting [3].") == {1, 3}


def test_combined_citation_group_is_expanded():
    assert generator._extract_citation_numbers("Neither lists yellowing [1, 4].") == {1, 4}


def test_combined_group_with_three_numbers():
    assert generator._extract_citation_numbers("All three agree [1, 2, 4].") == {1, 2, 4}


def test_combined_group_without_spaces():
    assert generator._extract_citation_numbers("All three agree [1,2,4].") == {1, 2, 4}


def test_repeated_citations_are_deduplicated():
    assert generator._extract_citation_numbers("[1] again [2, 1] and [1].") == {1, 2}


def test_mixed_single_and_combined_citations():
    reply = "First [1]. Then [2, 3]. Finally [4]."
    assert generator._extract_citation_numbers(reply) == {1, 2, 3, 4}


# ── 2. Citation validation ──────────────────────────────────────────────────


def test_validate_returns_every_number_of_a_combined_group():
    assert generator._validate_citations("Answer [1, 4].", CHUNKS) == {1, 4}


def test_zero_citation_is_rejected():
    with pytest.raises(generator.InvalidCitationError):
        generator._validate_citations("Answer [0].", CHUNKS)


def test_out_of_range_single_citation_is_rejected():
    with pytest.raises(generator.InvalidCitationError):
        generator._validate_citations("Answer [5].", CHUNKS)


def test_out_of_range_number_inside_combined_group_is_rejected():
    """A valid number must not let an invalid neighbour through."""
    with pytest.raises(generator.InvalidCitationError) as excinfo:
        generator._validate_citations("Answer [1, 9].", CHUNKS)

    assert "9" in str(excinfo.value)


def test_reply_without_any_citation_raises_missing():
    with pytest.raises(generator.MissingCitationError):
        generator._validate_citations("The passages do not say.", CHUNKS)


def test_combined_citations_map_to_every_source():
    sources = generator._build_sources({1, 4}, CHUNKS)

    assert [source["citation"] for source in sources] == [1, 4]
    assert [source["title"] for source in sources] == ["Source 1", "Source 4"]


# ── 3. Full generation path (Gemini client replaced) ────────────────────────


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate_content(self, **kwargs):
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.models = _FakeModels(text)


def test_generate_grounded_answer_expands_combined_citations(monkeypatch):
    """The real generation path returns a source per cited number, not one."""
    monkeypatch.setattr(
        generator,
        "_get_client",
        lambda: _FakeClient("Neither passage lists leaf yellowing [1, 4]."),
    )

    result = generator.generate_grounded_answer("question", CHUNKS)

    assert result["status"] == "success"
    assert [source["citation"] for source in result["sources"]] == [1, 4]


def test_generate_grounded_answer_rejects_out_of_range_combined_group(monkeypatch):
    monkeypatch.setattr(
        generator,
        "_get_client",
        lambda: _FakeClient("Neither passage lists leaf yellowing [1, 9]."),
    )

    with pytest.raises(generator.InvalidCitationError):
        generator.generate_grounded_answer("question", CHUNKS)


# ── 4. Exception mapping in main._generate ──────────────────────────────────


def _raiser(exception: Exception):
    def _raise(query, context_chunks):
        raise exception

    return _raise


def test_missing_citation_maps_to_insufficient_context(monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        _raiser(generator.MissingCitationError("the reply cited nothing")),
    )

    assert main._generate("question", CHUNKS) == (None, [], "insufficient-context")


def test_invalid_citation_is_not_treated_as_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        _raiser(generator.InvalidCitationError("out-of-range citation")),
    )

    assert main._generate("question", CHUNKS) == (None, [], "InvalidCitationError")


def test_empty_response_is_not_treated_as_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        _raiser(generator.EmptyResponseError("empty response")),
    )

    assert main._generate("question", CHUNKS) == (None, [], "EmptyResponseError")


def test_api_failure_keeps_its_own_reason(monkeypatch):
    reason = "too many requests; the rate limit was exceeded"
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        _raiser(generator.GenerationApiError("failed", reason=reason)),
    )

    assert main._generate("question", CHUNKS) == (None, [], reason)


def test_successful_generation_is_unaffected(monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        lambda query, context_chunks: {
            "status": "success",
            "reply": "Grounded answer [1].",
            "sources": [{"citation": 1, "title": "Source 1", "url": "u"}],
        },
    )

    reply, sources, error = main._generate("question", CHUNKS)

    assert reply == "Grounded answer [1]."
    assert sources == [{"citation": 1, "title": "Source 1", "url": "u"}]
    assert error is None


# ── 5. End-to-end /chat fallbacks ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_reports_insufficient_evidence_when_the_model_cannot_cite(
    client, monkeypatch
):
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        _raiser(generator.MissingCitationError("the reply cited nothing")),
    )

    response = await client.post("/chat", data={"message": TOMATO_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "insufficient_evidence"
    assert main.INSUFFICIENT_EVIDENCE_REPLY in body["reply"]
    assert body["sources"] == []


@pytest.mark.asyncio
async def test_chat_maps_a_real_citationless_generation_to_insufficient_evidence(
    client, monkeypatch
):
    """
    The potato path, reproduced offline: the real generator runs, the model
    answers without citing anything, and the user still receives the clear
    insufficient-evidence reply rather than a generic error.
    """
    monkeypatch.setattr(
        main, "generate_grounded_answer", generator.generate_grounded_answer
    )
    monkeypatch.setattr(
        generator,
        "_get_client",
        lambda: _FakeClient(
            "The supplied passages do not describe symptoms for this crop, "
            "so I cannot answer from them."
        ),
    )

    response = await client.post("/chat", data={"message": TOMATO_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "insufficient_evidence"
    assert main.INSUFFICIENT_EVIDENCE_REPLY in body["reply"]
    assert body["sources"] == []


@pytest.mark.asyncio
async def test_chat_still_reports_a_generic_failure_for_api_errors(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        _raiser(
            generator.GenerationApiError(
                "failed", reason="the Gemini service returned a server error"
            )
        ),
    )

    response = await client.post("/chat", data={"message": TOMATO_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "generation_failed"
    assert "couldn't produce a verified answer" in body["reply"]
    assert body["sources"] == []


# ── 6. Source mapping must keep one entry per cited number ──────────────────

# Mirrors the potato corpus, where the Common Scab and Black Scurf chunks come
# from two sections of the SAME article: identical title and URL, different
# chunk_id. De-duplicating on (title, url) silently dropped the later citation.
SHARED_DOC_CHUNKS = [
    {
        "chunk_id": "late-blight",
        "text": "Late blight is caused by Phytophthora infestans.",
        "source_title": "Potato late blight and its management",
        "source_url": "https://epubs.icar.org.in/article/view/102049",
        "source_institution": "ICAR",
        "evidence_level": "abstract_only",
    },
    {
        "chunk_id": "common-scab",
        "text": "Common scab is listed among tuber-disfiguring diseases.",
        "source_title": "Management of soil and tuber borne diseases of potato",
        "source_url": "https://epubs.icar.org.in/article/view/102050",
        "source_institution": "ICAR",
        "evidence_level": "abstract_only",
    },
    {
        "chunk_id": "black-scurf",
        "text": "Black scurf is listed alongside common scab.",
        "source_title": "Management of soil and tuber borne diseases of potato",
        "source_url": "https://epubs.icar.org.in/article/view/102050",
        "source_institution": "ICAR",
        "evidence_level": "abstract_only",
    },
]


def _response_sources(reply: str, chunks=None) -> list[dict]:
    """Run the reply through validation, source building and the /chat mapping."""
    chunks = SHARED_DOC_CHUNKS if chunks is None else chunks
    used = generator._validate_citations(reply, chunks)
    return main._frontend_sources(generator._build_sources(used, chunks), chunks)


def test_combined_pair_maps_to_both_sources():
    sources = _response_sources("Both are tuber-borne diseases [1, 3].")

    assert [source["citation"] for source in sources] == [1, 3]


def test_combined_triple_maps_to_all_three_sources():
    sources = _response_sources("All three are described [1, 2, 3].")

    assert [source["citation"] for source in sources] == [1, 2, 3]


def test_sources_from_one_document_keep_their_own_numbers():
    """The reported bug: citations 2 and 3 share a title and URL."""
    sources = _response_sources("Each is listed [1], [2] and [3].")

    assert [source["citation"] for source in sources] == [1, 2, 3]
    assert [source["chunk_id"] for source in sources] == [
        "late-blight",
        "common-scab",
        "black-scurf",
    ]
    assert sources[1]["title"] == sources[2]["title"]
    assert sources[1]["url"] == sources[2]["url"]


def test_individual_citations_still_map_one_to_one():
    sources = _response_sources("First [1]. Second [2].")

    assert [source["citation"] for source in sources] == [1, 2]


def test_source_provenance_is_preserved():
    sources = _response_sources("All three [1, 2, 3].")

    for source in sources:
        assert source["title"]
        assert source["url"]
        assert source["institution"] == "ICAR"
        assert source["evidence_level"] == "abstract_only"
        assert source["filename"] == source["title"]


def test_no_source_is_fabricated():
    sources = _response_sources("Only the first is discussed [1].")

    assert [source["citation"] for source in sources] == [1]


@pytest.mark.asyncio
async def test_chat_response_preserves_every_cited_number(client, monkeypatch):
    """
    End to end through /chat: the response carries one source per cited number,
    with distinct chunk ids. The frontend de-duplicates on ``chunk_id``, so three
    distinct ids render as three sources.
    """
    shared_chunks = [dict(chunk, crop="tomato") for chunk in SHARED_DOC_CHUNKS]

    monkeypatch.setattr(
        main,
        "hybrid_search",
        lambda query, top_k=10: [dict(chunk, semantic_rank=i + 1) for i, chunk in enumerate(shared_chunks)],
    )
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        lambda query, context_chunks: {
            "status": "success",
            "reply": "Each disease is described [1], [2] and [3].",
            "sources": [
                {
                    "citation": index,
                    "title": chunk["source_title"],
                    "url": chunk["source_url"],
                    "institution": chunk["source_institution"],
                    "evidence_level": chunk["evidence_level"],
                }
                for index, chunk in enumerate(shared_chunks, start=1)
            ],
        },
    )

    response = await client.post("/chat", data={"message": TOMATO_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "grounded_rag"
    assert [source["citation"] for source in body["sources"]] == [1, 2, 3]

    chunk_ids = [source["chunk_id"] for source in body["sources"]]
    assert chunk_ids == ["late-blight", "common-scab", "black-scurf"]
    assert len(set(chunk_ids)) == 3
