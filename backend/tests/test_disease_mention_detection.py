"""
Generic unsupported-disease detection tests (Phase 2E.3).

The detector is morphological: it recognises disease-shaped names the guide
does not contain without enumerating diseases. It must never reject a general
crop or symptom question.

Offline only: the inventory comes from the validated chunk metadata; no database
and no Gemini call.
"""

from __future__ import annotations

import pytest

import main
from disease_registry import (
    _generic_disease_phrase,
    build_disease_inventory,
    detect_disease_intent,
)
from vector_store import load_vector_store


@pytest.fixture(autouse=True)
def clean_database():
    """These tests never touch the database; skip the conftest cleanup."""
    yield


@pytest.fixture(autouse=True)
def offline_pipeline():
    """Keep the real metadata; nothing here calls the model pipeline."""
    yield


@pytest.fixture(scope="module")
def inventory() -> list[dict]:
    return build_disease_inventory(load_vector_store().chunks)


# ── Existing supported disease detection is preserved ────────────────────────


@pytest.mark.parametrize(
    "message,crop,disease",
    [
        ("Tomato late blight", "tomato", "Tomato Late Blight"),
        ("tomato early blight symptoms", "tomato", "Tomato Early Blight"),
        ("Potato late blight management", "potato", "Potato Late Blight"),
        ("wheat powdery mildew", "wheat", "Wheat Powdery Mildew"),
        ("Soybean yellow mosaic", "soybean", "Soybean Yellow Mosaic"),
    ],
)
def test_supported_diseases_are_unchanged(message, crop, disease, inventory):
    scope = detect_disease_intent(message, crop, inventory)

    assert scope["status"] == "supported"
    assert scope["disease_names"] == [disease]


def test_other_crop_disease_is_still_unsupported(inventory):
    scope = detect_disease_intent("Potato early blight symptoms", "potato", inventory)

    assert scope["status"] == "unsupported"
    assert scope["named_phrase"] == "early blight"
    assert scope["known_crops"] == ["tomato"]


# ── Unknown disease names are declined, not answered from unrelated chunks ───


@pytest.mark.parametrize(
    "message,phrase",
    [
        ("Tomato bacterial wilt symptoms?", "bacterial wilt"),
        ("What about bacterial wilt?", "bacterial wilt"),
        ("Tomato leaf curl virus?", "leaf curl virus"),
        ("root rot on tomato", "root rot"),
        ("Tomato bacterial spot with yellow leaves", "bacterial spot"),
    ],
)
def test_unknown_disease_is_declined(message, phrase, inventory):
    scope = detect_disease_intent(message, "tomato", inventory)

    assert scope["status"] == "unsupported"
    assert scope["named_phrase"] == phrase
    assert scope["disease_names"] == []
    assert scope["known_crops"] == []


def test_declined_disease_runs_no_retrieval(monkeypatch, inventory):
    message = "Tomato bacterial wilt symptoms"
    scope = detect_disease_intent(message, "tomato", inventory)

    def _fail(*args, **kwargs):
        raise AssertionError("retrieval must not run for an unsupported disease")

    monkeypatch.setattr(main, "hybrid_search", _fail)

    chunks, error = main._retrieve_for_scope(message, "tomato", scope)

    assert chunks == []
    assert error is None


# ── General and symptom questions are not rejected ───────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "Why are my tomato leaves turning yellow?",
        "My tomato plant has dark spots on the leaves.",
        "My tomato leaves have yellow spots.",
        "What diseases affect tomato plants?",
        "How do I manage tomato problems?",
        "Tomato leaf spot",
        "Are the leaves still yellow?",
        "How do I stop tomatoes rotting?",
    ],
)
def test_general_questions_are_not_rejected(message, inventory):
    scope = detect_disease_intent(message, "tomato", inventory)

    assert scope["status"] == "none", message


def test_a_crop_name_alone_is_not_a_disease_name(inventory):
    assert _generic_disease_phrase("tomato virus") is None


@pytest.mark.parametrize(
    "message,expected",
    [
        ("bacterial wilt", "bacterial wilt"),
        ("brown rot", None),
        ("root rot", "root rot"),
        ("leaf blight", "leaf blight"),
        ("bacterial spot", "bacterial spot"),
        ("dark spots", None),
        ("leaf spots", None),
        ("blight", None),
    ],
)
def test_generic_phrase_extraction(message, expected):
    assert _generic_disease_phrase(message) == expected
