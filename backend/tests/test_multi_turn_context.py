"""
Multi-turn context resolution tests (Phase 2E.2) — pure unit level.

No PostgreSQL, no Gemini, no network: the inventory comes from the validated
chunk metadata and the conversation history is a plain snapshot. The autouse
fixtures from ``conftest.py`` are overridden so this module runs without a
database and without the fake pipeline.
"""

from __future__ import annotations

import pytest

import main
from conversation_context import (
    crop_from_history,
    disease_from_history,
    is_followup,
    mentions_disease_term,
)
from disease_registry import build_disease_inventory
from vector_store import load_vector_store

EARLY = "Tomato Early Blight"
LATE = "Tomato Late Blight"
POTATO_LATE = "Potato Late Blight"
POTATO_EARLY = "Potato Early Blight"


@pytest.fixture(autouse=True)
def clean_database():
    """These tests never touch the database; skip the conftest cleanup."""
    yield


@pytest.fixture(autouse=True)
def offline_pipeline():
    """Keep the real metadata; nothing here calls retrieval or generation."""
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
        "sources": [
            {"disease_name": name, "crop": crop} for name, crop in sources
        ],
    }


def tomato_history() -> list[dict]:
    return [
        turn("user", "What are the symptoms of tomato early blight?"),
        turn("assistant", "Early blight causes concentric leaf spots [1].", [(EARLY, "tomato")]),
    ]


def resolve(message, inventory, supported, history=(), conversation_crop="tomato"):
    return main._resolve_turn_context(
        message, supported, conversation_crop, list(history), inventory
    )


# ── Follow-ups that inherit the conversation's context ───────────────────────


def test_symptoms_followup_inherits_crop_and_disease(inventory, supported):
    context = resolve("Tell me the symptoms.", inventory, supported, tomato_history())

    assert context["crop_scope"]["status"] == "ok"
    assert context["crop_scope"]["crop"] == "tomato"
    assert context["crop_scope"].get("inherited") is True
    assert context["disease_scope"]["status"] == "supported"
    assert context["disease_scope"]["disease_names"] == [EARLY]
    assert context["disease_scope"].get("inherited") is True
    assert EARLY in context["query"]


def test_favourable_conditions_followup_inherits_context(inventory, supported):
    context = resolve("What conditions favour it?", inventory, supported, tomato_history())

    assert context["crop_scope"]["crop"] == "tomato"
    assert context["disease_scope"]["disease_names"] == [EARLY]


def test_management_followup_inherits_context(inventory, supported):
    context = resolve("How do I manage it?", inventory, supported, tomato_history())

    assert context["crop_scope"]["crop"] == "tomato"
    assert context["disease_scope"]["disease_names"] == [EARLY]


@pytest.mark.parametrize(
    "message",
    ["How does it spread?", "Is that infection serious?", "What about its yield impact?"],
)
def test_pronoun_followups_resolve_to_the_same_disease(message, inventory, supported):
    context = resolve(message, inventory, supported, tomato_history())

    assert context["crop_scope"]["crop"] == "tomato"
    assert context["disease_scope"]["status"] == "supported"
    assert context["disease_scope"]["disease_names"] == [EARLY]


# ── Explicit text always wins ────────────────────────────────────────────────


def test_disease_specific_followup_switches_disease(inventory, supported):
    context = resolve("What about late blight?", inventory, supported, tomato_history())

    assert context["crop_scope"]["crop"] == "tomato"
    assert context["disease_scope"]["status"] == "supported"
    assert context["disease_scope"]["disease_names"] == [LATE]
    assert context["disease_scope"].get("inherited") is not True


def test_explicit_disease_in_followup_switches_disease(inventory, supported):
    context = resolve(
        "What are the symptoms of tomato late blight?", inventory, supported, tomato_history()
    )

    assert context["crop_scope"]["crop"] == "tomato"
    assert context["disease_scope"]["disease_names"] == [LATE]


def test_switching_crop_drops_the_previous_crop_disease(inventory, supported):
    context = resolve(
        "What about potato late blight?", inventory, supported, tomato_history()
    )

    assert context["crop_scope"]["status"] == "ok"
    assert context["crop_scope"]["crop"] == "potato"
    assert context["disease_scope"]["disease_names"] == [POTATO_LATE]
    assert EARLY not in (context["disease_scope"]["disease_names"] or [])


def test_explicit_crop_followup_does_not_carry_another_crops_disease(inventory, supported):
    context = resolve(
        "What are the symptoms on potato?", inventory, supported, tomato_history()
    )

    assert context["crop_scope"]["crop"] == "potato"
    assert context["disease_scope"]["status"] == "none"


def test_unsupported_disease_followup_is_declined_without_inheritance(inventory, supported):
    context = resolve(
        "What about potato early blight?", inventory, supported, tomato_history()
    )

    assert context["crop_scope"]["crop"] == "potato"
    assert context["disease_scope"]["status"] == "unsupported"
    assert context["disease_scope"]["named_phrase"] == "early blight"
    assert context["disease_scope"]["disease_names"] == []
    assert context["disease_scope"].get("inherited") is not True


def test_unknown_disease_term_does_not_inherit_a_different_disease(inventory, supported):
    context = resolve("What about bacterial wilt?", inventory, supported, tomato_history())

    assert context["crop_scope"]["crop"] == "tomato"
    assert context["disease_scope"]["status"] == "unsupported"
    assert context["disease_scope"]["disease_names"] == []
    assert context["disease_scope"].get("inherited") is not True


# ── No context to inherit ────────────────────────────────────────────────────


def test_stateless_followup_stays_ambiguous(inventory, supported):
    context = resolve(
        "What conditions favour it?", inventory, supported, history=(), conversation_crop=None
    )

    assert context["crop_scope"]["status"] == "ambiguous"
    assert context["crop_scope"]["reason"] == "none"
    assert context["disease_scope"]["status"] == "none"


def test_non_followup_chatter_is_not_pulled_into_old_context(inventory, supported):
    context = resolve("Thanks!", inventory, supported, tomato_history())

    assert context["crop_scope"]["status"] == "ambiguous"


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message,expected",
    [
        ("What conditions favour it?", True),
        ("Tell me the symptoms.", True),
        ("What about late blight?", True),
        ("How does it spread?", True),
        ("Thanks!", False),
        ("", False),
    ],
)
def test_is_followup(message, expected):
    assert is_followup(message) is expected


@pytest.mark.parametrize(
    "message,expected",
    [
        ("bacterial wilt on tomato", True),
        ("what about the spots", False),
        ("what conditions favour", False),
    ],
)
def test_mentions_disease_term(message, expected):
    assert mentions_disease_term(message) is expected


def test_crop_from_history_uses_newest_user_turn():
    history = [
        turn("user", "Tell me about tomato late blight"),
        turn("assistant", "..." ),
        turn("user", "Now what about potato?"),
    ]

    assert crop_from_history(history, {"tomato", "potato"}) == "potato"


def test_crop_from_history_is_none_when_two_crops_are_named():
    history = [turn("user", "Compare tomato and potato")]

    assert crop_from_history(history, {"tomato", "potato"}) is None


def test_disease_from_history_ignores_cross_crop_sources(inventory):
    history = [
        turn("user", "Tomato question"),
        turn("assistant", "...", [(POTATO_LATE, "potato")]),
    ]

    assert disease_from_history(history, "tomato", inventory) is None


def test_disease_from_history_prefers_the_latest_assistant_citation(inventory):
    history = [
        turn("user", "Tomato early blight?"),
        turn("assistant", "...", [(EARLY, "tomato")]),
        turn("user", "And late blight?"),
        turn("assistant", "...", [(LATE, "tomato")]),
    ]

    assert disease_from_history(history, "tomato", inventory) == LATE


def test_disease_from_history_skips_an_ambiguous_multi_disease_turn(inventory):
    history = [
        turn("user", "Why are my tomato leaves yellow?"),
        turn("assistant", "...", [(EARLY, "tomato"), (LATE, "tomato")]),
    ]

    assert disease_from_history(history, "tomato", inventory) is None
