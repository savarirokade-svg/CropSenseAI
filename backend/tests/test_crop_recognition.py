"""
Crop recognition and guide-heading parsing tests (Phase 1).

These are pure unit tests: no PostgreSQL, no network, no Gemini. The autouse
``clean_database`` fixture from ``conftest.py`` is overridden below so this
module can run without a running database.
"""

from __future__ import annotations

import pytest

import index_builder
import main
from knowledge_graph import search_knowledge_graph
from crop_registry import (
    CROP_ALIASES,
    SUPPORTED_CROPS,
    UNSUPPORTED_CROPS,
    find_crops,
    matches_crop,
    normalize_text,
)

EXPECTED_CROPS = (
    "tomato",
    "potato",
    "rice",
    "wheat",
    "maize",
    "cotton",
    "soybean",
    "groundnut",
    "chickpea",
    "mustard",
    "onion",
    "chilli",
)

ALIASES = (
    ("corn", "maize"),
    ("chili", "chilli"),
    ("paddy", "rice"),
    ("peanut", "groundnut"),
    ("chana", "chickpea"),
    ("chick pea", "chickpea"),
    ("soya", "soybean"),
    ("soy", "soybean"),
)


@pytest.fixture(autouse=True)
def clean_database():
    """These tests never touch the database; skip the conftest cleanup."""
    yield


# ── Registry invariants ──────────────────────────────────────────────────────


def test_canonical_crops_are_the_twelve_planned():
    assert SUPPORTED_CROPS == EXPECTED_CROPS
    assert len(SUPPORTED_CROPS) == 12
    assert len(set(SUPPORTED_CROPS)) == 12
    assert all(crop == crop.strip().lower() for crop in SUPPORTED_CROPS)


def test_supported_and_unsupported_crops_are_disjoint():
    assert not (set(SUPPORTED_CROPS) & set(UNSUPPORTED_CROPS))


def test_alias_keys_are_known_crops_and_aliases_do_not_collide():
    known = set(SUPPORTED_CROPS) | set(UNSUPPORTED_CROPS)
    assert set(CROP_ALIASES).issubset(known)

    canonical = set(SUPPORTED_CROPS)
    for crop, aliases in CROP_ALIASES.items():
        for alias in aliases:
            assert alias != crop
            assert alias not in canonical


def test_index_builder_and_main_share_the_registry():
    assert index_builder.CROPS == SUPPORTED_CROPS
    assert main.KNOWN_CROP_NAMES == SUPPORTED_CROPS
    assert main.UNSUPPORTED_CROPS == UNSUPPORTED_CROPS


# ── Heading parsing ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("crop", EXPECTED_CROPS)
def test_every_planned_crop_heading_parses_to_its_crop(crop):
    parsed = index_builder.parse_section_heading(
        f"3.1 Test Disease of {crop.title()}"
    )

    assert parsed["section_number"] == "3.1"
    assert parsed["crop"] == crop
    assert parsed["disease_names"] == [f"{crop.title()} Test Disease"]


def test_existing_tomato_and_potato_headings_still_parse():
    tomato = index_builder.parse_section_heading("1.1 Early Blight of Tomato")
    potato = index_builder.parse_section_heading("2.1 Late Blight of Potato")

    assert tomato["crop"] == "tomato"
    assert tomato["disease_names"] == ["Tomato Early Blight"]
    assert potato["crop"] == "potato"
    assert potato["disease_names"] == ["Potato Late Blight"]


# ── Question recognition ─────────────────────────────────────────────────────


@pytest.mark.parametrize("crop", EXPECTED_CROPS)
def test_every_planned_crop_is_recognised_in_a_question(crop):
    text = normalize_text(f"  What affects my {crop.upper()} crop?  ")

    assert matches_crop(text, crop)
    assert find_crops(text, SUPPORTED_CROPS) == [crop]


@pytest.mark.parametrize("alias,canonical", ALIASES)
def test_common_aliases_resolve_to_one_canonical_crop(alias, canonical):
    text = normalize_text(f"My {alias} leaves are yellow")

    assert find_crops(text, SUPPORTED_CROPS) == [canonical]


@pytest.mark.parametrize("crop", EXPECTED_CROPS)
def test_crop_scope_accepts_every_supported_crop(crop):
    scope = main._resolve_crop_scope(
        f"My {crop.title()} has spots", supported={crop}
    )

    assert scope == {"status": "ok", "crop": crop}


@pytest.mark.parametrize("alias,canonical", ALIASES)
def test_crop_scope_accepts_aliases(alias, canonical):
    scope = main._resolve_crop_scope(
        f"My {alias} has spots", supported={canonical}
    )

    assert scope == {"status": "ok", "crop": canonical}


def test_case_and_whitespace_are_normalised():
    assert main._resolve_crop_scope("   TOMATO   leaves", {"tomato"}) == {
        "status": "ok",
        "crop": "tomato",
    }
    assert main._resolve_crop_scope("  CHICK   PEA  ", {"chickpea"}) == {
        "status": "ok",
        "crop": "chickpea",
    }


def test_existing_tomato_and_potato_behaviour_is_unchanged():
    supported = {"tomato", "potato"}

    assert main._resolve_crop_scope("tomato leaves", supported) == {
        "status": "ok",
        "crop": "tomato",
    }
    assert main._resolve_crop_scope("potato leaves", supported) == {
        "status": "ok",
        "crop": "potato",
    }


# ── Unsupported and ambiguous crops ──────────────────────────────────────────


@pytest.mark.parametrize("name", ["mango", "banana", "apple", "grape", "grapes"])
def test_unsupported_crops_still_decline(name):
    scope = main._resolve_crop_scope(
        f"My {name} leaves are yellow", supported={"tomato", "potato"}
    )

    assert scope["status"] == "unsupported"
    assert len(scope["crops"]) == 1


def test_unsupported_crop_wins_over_a_supported_one():
    scope = main._resolve_crop_scope(
        "tomato and mango blight", supported={"tomato", "potato"}
    )

    assert scope == {"status": "unsupported", "crops": ["mango"]}


def test_crop_scope_asks_when_no_crop_is_named():
    scope = main._resolve_crop_scope(
        "My leaves are yellow", supported={"tomato", "potato"}
    )

    assert scope == {"status": "ambiguous", "crops": [], "reason": "none"}


def test_crop_scope_asks_when_multiple_supported_crops_are_named():
    scope = main._resolve_crop_scope(
        "tomato and potato blight", supported={"tomato", "potato"}
    )

    assert scope == {
        "status": "ambiguous",
        "crops": ["potato", "tomato"],
        "reason": "multiple",
    }


# ── Knowledge graph compatibility ────────────────────────────────────────────


def test_knowledge_graph_keeps_existing_tomato_edges():
    edges = search_knowledge_graph("Why are my tomato leaves turning yellow?")

    assert [(e["subject"], e["relation"], e["object"]) for e in edges] == [
        ("Tomato", "may_show_symptom", "Yellow leaves"),
        ("Yellow leaves", "may_indicate", "Nutrient deficiency"),
        ("Yellow leaves", "may_be_associated_with", "Overwatering"),
        (
            "Nutrient deficiency",
            "suggests_checking",
            "Soil nutrients and growing conditions",
        ),
    ]


def test_knowledge_graph_is_empty_for_unsupported_crops():
    assert search_knowledge_graph("My mango leaves are yellow") == []


@pytest.mark.parametrize("crop", ["rice", "wheat", "maize", "cotton", "corn"])
def test_knowledge_graph_has_no_edges_for_new_crops(crop):
    assert search_knowledge_graph(f"My {crop} leaves are yellow") == []
