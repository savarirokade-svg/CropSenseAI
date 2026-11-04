"""
Retrieval / knowledge-graph context consistency — unit tests (Phase 2E.3).

Offline only: the real chunk metadata is used for the inventory, the graph is
the local hardcoded educational graph, and no database or Gemini call is made.
"""

from __future__ import annotations

import pytest

import main
from conversation_context import is_followup, mentions_disease_term
from disease_registry import DISEASE_HEAD_WORDS, build_disease_inventory
from vector_store import load_vector_store

EARLY = "Tomato Early Blight"


@pytest.fixture(autouse=True)
def clean_database():
    yield


@pytest.fixture(autouse=True)
def offline_pipeline():
    yield


@pytest.fixture(scope="module")
def inventory() -> list[dict]:
    return build_disease_inventory(load_vector_store().chunks)


@pytest.fixture(scope="module")
def supported(inventory) -> set[str]:
    return {record["crop"] for record in inventory}


def turn(role: str, content: str, sources=()) -> dict:
    return {
        "role": role,
        "content": content,
        "sources": [{"disease_name": name, "crop": crop} for name, crop in sources],
    }


def tomato_history() -> list[dict]:
    return [
        turn("user", "What are the symptoms of tomato early blight?"),
        turn("assistant", "Early blight causes concentric spots [1].", [(EARLY, "tomato")]),
    ]


# ── Knowledge graph gets the resolved query ──────────────────────────────────


def test_graph_returns_edges_for_an_explicit_tomato_question():
    edges = main._resolve_knowledge_graph(
        "Why are my tomato leaves turning yellow?",
        {"status": "ok", "crop": "tomato"},
        {"status": "none"},
        None,
    )

    assert edges
    assert any(edge["subject"] == "Tomato" for edge in edges)


def test_graph_accepts_a_resolved_followup_query():
    edges = main._resolve_knowledge_graph(
        "Are the leaves still yellow? tomato",
        {"status": "ok", "crop": "tomato", "inherited": True},
        {"status": "none"},
        None,
    )

    assert edges
    assert any(edge["object"] == "Yellow leaves" for edge in edges)


def test_graph_is_empty_for_an_unsupported_disease():
    edges = main._resolve_knowledge_graph(
        "Tomato bacterial wilt with yellow leaves",
        {"status": "ok", "crop": "tomato"},
        {"status": "unsupported", "named_phrase": "bacterial wilt"},
        None,
    )

    assert edges == []


@pytest.mark.parametrize(
    "scope",
    [
        {"status": "ambiguous", "crops": [], "reason": "none"},
        {"status": "unsupported", "crops": ["mango"]},
        {"status": "unavailable"},
    ],
)
def test_graph_is_empty_without_a_validated_crop(scope):
    assert (
        main._resolve_knowledge_graph(
            "tomato yellow leaves", scope, {"status": "none"}, None
        )
        == []
    )


def test_graph_is_empty_when_the_index_is_unavailable():
    assert (
        main._resolve_knowledge_graph(
            "tomato yellow leaves",
            {"status": "ok", "crop": "tomato"},
            {"status": "none"},
            "vector-store-unavailable",
        )
        == []
    )


# ── Context resolution still feeds a supported disease follow-up ─────────────


def test_supported_disease_followup_still_resolves(inventory, supported):
    context = main._resolve_turn_context(
        "What conditions favour it?",
        supported,
        "tomato",
        tomato_history(),
        inventory,
    )

    assert context["crop_scope"]["crop"] == "tomato"
    assert context["disease_scope"]["status"] == "supported"
    assert context["disease_scope"]["disease_names"] == [EARLY]


def test_unknown_disease_followup_is_unsupported_not_inherited(inventory, supported):
    context = main._resolve_turn_context(
        "What about bacterial wilt?",
        supported,
        "tomato",
        tomato_history(),
        inventory,
    )

    assert context["disease_scope"]["status"] == "unsupported"
    assert context["disease_scope"]["disease_names"] == []


# ── Shared vocabulary between resolution and detection ──────────────────────


def test_followup_cues_are_consistent_with_the_generic_detector():
    assert is_followup("Are the leaves still yellow?")
    assert is_followup("What about bacterial wilt?")
    assert mentions_disease_term("bacterial wilt")
    assert not mentions_disease_term("dark spots on leaves")


def test_conversation_context_uses_the_shared_head_words():
    import conversation_context
    import disease_registry

    assert conversation_context.DISEASE_HEAD_WORDS is disease_registry.DISEASE_HEAD_WORDS
