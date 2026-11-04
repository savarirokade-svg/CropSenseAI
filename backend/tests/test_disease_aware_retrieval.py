"""
Disease-aware retrieval and section-intent tests (Phase 2C.1).

These are offline unit/integration tests over the real validated vector store:
the embeddings call is replaced, so ranking is deterministic and no Gemini
request is made. The autouse ``clean_database`` fixture from ``conftest.py`` is
overridden because nothing here touches PostgreSQL.
"""

from __future__ import annotations

import pytest

import generator
import hybrid_retrieval
import main
from disease_registry import build_disease_inventory, detect_disease_intent
from vector_store import load_vector_store


@pytest.fixture(autouse=True)
def clean_database():
    """These tests never touch the database; skip the conftest cleanup."""
    yield


@pytest.fixture(autouse=True)
def offline_pipeline():
    """
    Override conftest's fake pipeline.

    These tests exercise the real hybrid retrieval against the real chunk
    metadata; only the embedding network call is replaced below.
    """
    yield


@pytest.fixture(scope="module")
def inventory() -> list[dict]:
    """The disease inventory derived from the real chunk metadata."""
    return build_disease_inventory(load_vector_store().chunks)


@pytest.fixture(autouse=True)
def offline_semantic(monkeypatch):
    """Skip the embedding network call; lexical ranking stays deterministic."""
    monkeypatch.setattr(
        hybrid_retrieval,
        "_embed_and_search",
        lambda store, query, depth: ([], {}),
    )


# ── Unsupported vs supported named diseases ──────────────────────────────────


def test_potato_early_blight_is_unsupported(inventory):
    scope = detect_disease_intent(
        "Potato Early Blight symptoms", "potato", inventory
    )

    assert scope["status"] == "unsupported"
    assert scope["named_phrase"] == "early blight"
    assert scope["known_crops"] == ["tomato"]
    assert scope["disease_names"] == []


@pytest.mark.parametrize(
    "query",
    [
        "Potato Late Blight",
        "potato late blight management",
        "Late blight on potato",
    ],
)
def test_potato_late_blight_is_supported(query, inventory):
    scope = detect_disease_intent(query, "potato", inventory)

    assert scope["status"] == "supported"
    assert scope["disease_names"] == ["Potato Late Blight"]


def test_unsupported_disease_returns_no_evidence(monkeypatch, inventory):
    message = "Potato Early Blight symptoms"
    scope = detect_disease_intent(message, "potato", inventory)

    def _fail(*args, **kwargs):
        raise AssertionError(
            "retrieval must not run for an unsupported named disease"
        )

    monkeypatch.setattr(main, "hybrid_search", _fail)

    chunks, error = main._retrieve_for_scope(message, "potato", scope)

    assert chunks == []
    assert error is None


def test_unsupported_disease_message_lists_real_coverage(inventory):
    scope = detect_disease_intent(
        "Potato Early Blight symptoms", "potato", inventory
    )
    message = main._unsupported_disease_message(scope, inventory)

    assert "Early blight" in message
    assert "Potato Late Blight" in message
    assert "Potato Common Scab" in message
    assert "source-grounded answer" in message
    # No fabricated management advice for the missing disease.
    assert "Mancozeb" not in message
    assert "chlorothalonil" not in message.lower()


@pytest.mark.asyncio
async def test_chat_returns_unsupported_disease_without_retrieval_or_generation(
    client, monkeypatch
):
    """End to end: the five-key contract, no sources, no evidence used."""

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "retrieval/generation must not run for an unsupported disease"
        )

    monkeypatch.setattr(main, "hybrid_search", _forbidden)
    monkeypatch.setattr(main, "generate_grounded_answer", _forbidden)

    response = await client.post(
        "/chat", data={"message": "Potato Early Blight symptoms"}
    )

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "user_message",
        "reply",
        "mode",
        "sources",
        "knowledge_graph",
    }
    assert body["mode"] == "unsupported_disease"
    assert body["sources"] == []
    assert "Early blight" in body["reply"]
    assert "won't apply guidance written for a different disease" in body["reply"]


# ── Maize downy mildew discrimination ────────────────────────────────────────


@pytest.mark.parametrize(
    "query,disease,organism,other_organism",
    [
        (
            "Maize sorghum downy mildew",
            "Maize Sorghum Downy Mildew",
            "Peronosclerospora sorghi",
            "Sclerophthora macrospora",
        ),
        (
            "sorghum downy mildew of maize",
            "Maize Sorghum Downy Mildew",
            "Peronosclerospora sorghi",
            "Sclerophthora macrospora",
        ),
        (
            "Maize crazy top downy mildew",
            "Maize Crazy Top Downy Mildew",
            "Sclerophthora macrospora",
            "Peronosclerospora sorghi",
        ),
        (
            "crazy top downy mildew in maize",
            "Maize Crazy Top Downy Mildew",
            "Sclerophthora macrospora",
            "Peronosclerospora sorghi",
        ),
    ],
)
def test_maize_downy_mildews_remain_distinct(
    query, disease, organism, other_organism, inventory
):
    scope = detect_disease_intent(query, "maize", inventory)

    assert scope["status"] == "supported"
    assert scope["disease_names"] == [disease]

    chunks, error = main._retrieve_for_scope(query, "maize", scope)

    assert error is None
    assert chunks
    assert {chunk["disease_name"] for chunk in chunks} == {disease}

    combined = "\n".join(chunk["text"] for chunk in chunks)
    assert organism in combined
    assert other_organism not in combined


# ── Section-intent ranking ───────────────────────────────────────────────────


def test_symptoms_query_ranks_symptoms_section_first(inventory):
    query = "Tomato late blight symptoms"
    scope = detect_disease_intent(query, "tomato", inventory)

    chunks, error = main._retrieve_for_scope(query, "tomato", scope)

    assert error is None
    assert chunks
    assert all(chunk["disease_name"] == "Tomato Late Blight" for chunk in chunks)
    assert "symptoms" in chunks[0]["section_name"].lower()
    # Secondary management evidence is retained below the preferred section.
    assert any(
        "chemical management" in chunk["section_name"].lower() for chunk in chunks
    )


def test_chemical_query_ranks_chemical_section_first(inventory):
    query = "Tomato late blight chemical management"
    scope = detect_disease_intent(query, "tomato", inventory)

    chunks, error = main._retrieve_for_scope(query, "tomato", scope)

    assert error is None
    assert chunks
    assert all(chunk["disease_name"] == "Tomato Late Blight" for chunk in chunks)
    assert "chemical management" in chunks[0]["section_name"].lower()


def test_section_intent_detection_is_vocabulary_driven():
    assert "symptoms" in hybrid_retrieval.detect_section_intents(
        "tomato late blight symptoms"
    )
    assert "chemical management" in hybrid_retrieval.detect_section_intents(
        "tomato late blight chemical management"
    )
    assert hybrid_retrieval.detect_section_intents("tomato late blight") == {}


# ── General questions and existing behaviour ─────────────────────────────────


def test_general_tomato_disease_question_still_retrieves(inventory):
    query = "What diseases affect tomato plants?"
    scope = detect_disease_intent(query, "tomato", inventory)

    assert scope["status"] == "none"

    chunks, error = main._retrieve_for_scope(query, "tomato", scope)

    assert error is None
    assert chunks
    assert all(chunk["crop"] == "tomato" for chunk in chunks)


def test_general_crop_question_is_not_rejected_by_disease_guard(inventory):
    for query in (
        "My tomato leaves are turning yellow",
        "What diseases affect tomato?",
        "How do I manage potato problems?",
    ):
        crop = "tomato" if "tomato" in query else "potato"
        scope = detect_disease_intent(query, crop, inventory)
        assert scope["status"] == "none", query


def test_unknown_crop_behaviour_is_preserved():
    scope = main._resolve_crop_scope(
        "Mango leaves are yellow", {"tomato", "potato"}
    )

    assert scope == {"status": "unsupported", "crops": ["mango"]}


# ── Provenance preservation ──────────────────────────────────────────────────


def test_source_ids_and_evidence_survive_disease_aware_retrieval(inventory):
    query = "Soybean yellow mosaic"
    scope = detect_disease_intent(query, "soybean", inventory)

    chunks, error = main._retrieve_for_scope(query, "soybean", scope)

    assert error is None
    assert chunks

    for chunk in chunks:
        assert chunk["disease_name"] == "Soybean Yellow Mosaic"
        assert chunk["source_ids"] == ["S16", "S46", "S67"]
        assert chunk["primary_source_id"] == "S46"
        assert chunk["evidence_level"] == "full_page"

    sources = generator._build_sources({1}, chunks)
    assert sources[0]["source_ids"] == ["S16", "S46", "S67"]
    assert sources[0]["primary_source_id"] == "S46"
    assert sources[0]["evidence_level"] == "full_page"
