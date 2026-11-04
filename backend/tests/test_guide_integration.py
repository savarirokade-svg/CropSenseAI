"""
Integration tests for the real agricultural guide (Phase 2B.2).

These tests exercise index_builder against backend/data/agricultural_guide.md
and validate the research source register. They use no database, network,
embedding or FAISS call; the conftest database fixture is overridden.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

import generator
import index_builder
import main

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
GUIDE_PATH = BACKEND_DIR / "data" / "agricultural_guide.md"
SOURCES_PATH = (
    PROJECT_ROOT / "research" / "agricultural_knowledge" / "sources.md"
)

EXPECTED_CROPS = {
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
}

EXPECTED_DISEASE_COUNTS = {
    "tomato": 4,
    "potato": 3,
    "rice": 3,
    "wheat": 3,
    "maize": 4,
    "cotton": 3,
    "soybean": 3,
    "groundnut": 3,
    "chickpea": 3,
    "mustard": 3,
    "onion": 3,
    "chilli": 4,
}


@pytest.fixture(autouse=True)
def clean_database():
    """These tests never touch the database; skip the conftest cleanup."""
    yield


@pytest.fixture(scope="module")
def built_guide():
    markdown = index_builder.read_knowledge_base(GUIDE_PATH)
    chunks, warnings = index_builder.build_chunks(markdown, GUIDE_PATH.name)
    return chunks, warnings


def _read_sources() -> tuple[dict[str, str], set[str]]:
    text = SOURCES_PATH.read_text(encoding="utf-8")
    known_ids = set(re.findall(r"^\| (S\d{2}) \|", text, re.M))
    url_to_id: dict[str, str] = {}

    for line in text.splitlines():
        if not line.lstrip().startswith("| S"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) < 4:
            continue
        source_id, url = columns[0], columns[3]
        if re.fullmatch(r"S\d{2}", source_id) and url:
            url_to_id[index_builder._normalize_url(url)] = source_id

    return url_to_id, known_ids


def _chunks_for(chunks, disease_name):
    return [chunk for chunk in chunks if chunk["disease_name"] == disease_name]


def test_guide_has_twelve_crops_and_thirty_nine_diseases(built_guide):
    chunks, _warnings = built_guide

    crops = {chunk["crop"] for chunk in chunks}
    diseases = {chunk["disease_name"] for chunk in chunks}

    assert crops == EXPECTED_CROPS
    assert len(diseases) == 39

    for crop, expected in EXPECTED_DISEASE_COUNTS.items():
        actual = len({chunk["disease_name"] for chunk in chunks if chunk["crop"] == crop})
        assert actual == expected, f"{crop}: expected {expected}, got {actual}"


def test_maize_downy_mildews_remain_separate(built_guide):
    chunks, _warnings = built_guide

    sorghum = _chunks_for(chunks, "Maize Sorghum Downy Mildew")
    crazy = _chunks_for(chunks, "Maize Crazy Top Downy Mildew")

    assert sorghum
    assert crazy
    assert "Peronosclerospora sorghi" in sorghum[0]["text"]
    assert "Sclerophthora macrospora" in crazy[0]["text"]
    assert all(chunk["disease_name"] != "Maize Crazy Top Downy Mildew" for chunk in sorghum)
    assert all(chunk["disease_name"] != "Maize Sorghum Downy Mildew" for chunk in crazy)


def test_soybean_yellow_mosaic_is_full_page(built_guide):
    chunks, _warnings = built_guide
    soybean = _chunks_for(chunks, "Soybean Yellow Mosaic")

    assert soybean
    assert all(chunk["evidence_level"] == "full_page" for chunk in soybean)
    assert all(
        chunk["source_access_status"].startswith("Full PDF read")
        for chunk in soybean
    )


def test_every_chunk_is_at_most_500_tokens(built_guide):
    chunks, _warnings = built_guide

    maximum = max(chunk["approx_tokens"] for chunk in chunks)
    assert maximum <= index_builder.MAX_CHUNK_TOKENS, (
        f"maximum observed chunk size is {maximum}"
    )


def test_build_has_no_warnings(built_guide):
    _chunks, warnings = built_guide
    assert warnings == []


def test_every_chunk_has_valid_source_ids(built_guide):
    chunks, _warnings = built_guide
    _url_to_id, known_ids = _read_sources()

    for chunk in chunks:
        assert chunk["source_ids"], f"{chunk['disease_name']} has no source IDs"
        unknown = set(chunk["source_ids"]) - known_ids
        assert not unknown, f"{chunk['disease_name']} has unknown IDs: {unknown}"


def test_primary_source_id_matches_source_url(built_guide):
    chunks, _warnings = built_guide
    url_to_id, _known_ids = _read_sources()

    for chunk in chunks:
        normalized_url = index_builder._normalize_url(chunk["source_url"])
        assert normalized_url in url_to_id, (
            f"{chunk['disease_name']} source URL is absent from sources.md"
        )
        assert chunk["primary_source_id"] == url_to_id[normalized_url], (
            f"{chunk['disease_name']} primary ID {chunk['primary_source_id']} "
            f"does not match URL {chunk['source_url']}"
        )


@pytest.mark.parametrize(
    "disease,expected_ids,expected_primary",
    [
        ("Wheat Powdery Mildew", {"S07", "S44", "S51", "S57"}, "S07"),
        ("Groundnut Rust", {"S19", "S20", "S47", "S59"}, "S20"),
        ("Soybean Yellow Mosaic", {"S16", "S46", "S67"}, "S46"),
        ("Maize Turcicum Leaf Blight", {"S09", "S65"}, "S09"),
        ("Chilli Bacterial Leaf Spot", {"S34", "S56", "S64"}, "S34"),
    ],
)
def test_multi_source_attribution_is_preserved(
    built_guide, disease, expected_ids, expected_primary
):
    chunks, _warnings = built_guide
    items = _chunks_for(chunks, disease)

    assert items
    for chunk in items:
        assert set(chunk["source_ids"]) == expected_ids
        assert chunk["primary_source_id"] == expected_primary


def test_chemical_warnings_stay_with_chemical_chunks(built_guide):
    chunks, _warnings = built_guide
    chemical_chunks = [
        chunk
        for chunk in chunks
        if any("chemical management" in name.lower() for name in chunk["subsections"])
    ]

    assert chemical_chunks
    for chunk in chemical_chunks:
        assert any(
            "evidence limitations and safety warnings" in name.lower()
            for name in chunk["subsections"]
        ), f"{chunk['disease_name']} chemical chunk lacks its evidence warning"


@pytest.mark.parametrize(
    "disease,required_phrases",
    [
        (
            "Groundnut Rust",
            ("Tridemorph is listed as banned", "location-specific"),
        ),
        (
            "Maize Crazy Top Downy Mildew",
            ("Seed-borne transmission is unverified",),
        ),
        (
            "Soybean Yellow Mosaic",
            ("Transmission mode is unresolved",),
        ),
        (
            "Chickpea Ascochyta Blight",
            ("No biological management is stated",),
        ),
    ],
)
def test_named_safety_records_keep_warning_text(
    built_guide, disease, required_phrases
):
    chunks, _warnings = built_guide
    items = [
        chunk
        for chunk in _chunks_for(chunks, disease)
        if any("chemical management" in name.lower() for name in chunk["subsections"])
    ]

    assert items
    combined = "\n".join(chunk["text"] for chunk in items)
    for phrase in required_phrases:
        assert phrase in combined, f"{disease}: missing {phrase!r} in chemical chunks"


def test_no_source_block_is_omitted(built_guide):
    chunks, _warnings = built_guide
    markdown = index_builder.read_knowledge_base(GUIDE_PATH)
    part, _part_warnings = index_builder.extract_included_part(markdown)

    text_by_disease: dict[str, str] = defaultdict(str)
    for chunk in chunks:
        text_by_disease[chunk["disease_name"]] += "\n" + index_builder._normalize_text(
            chunk["text"]
        )

    missing: list[tuple[str, str, str]] = []

    for section in index_builder.split_sections(part):
        info = index_builder.parse_section_heading(section["heading"])
        _metadata, _causal, body = index_builder.parse_source_metadata(
            section["lines"], [], "test"
        )
        preamble, subsections = index_builder.split_subsections(body)
        disease = info["disease_names"][0]
        haystack = text_by_disease[disease]

        blocks = list(index_builder._blocks_from_body(preamble))
        for _name, lines in subsections:
            blocks.extend(index_builder._blocks_from_body(lines))

        for block in blocks:
            normalized = index_builder._normalize_text(block)
            if normalized and normalized not in haystack:
                missing.append((info["section_number"], disease, normalized[:80]))

    assert not missing, f"source blocks missing from chunks: {missing[:5]}"


def test_section_source_ids_are_preserved_on_every_chunk(built_guide):
    chunks, _warnings = built_guide
    markdown = index_builder.read_knowledge_base(GUIDE_PATH)
    part, _part_warnings = index_builder.extract_included_part(markdown)
    chunks_by_section = defaultdict(list)
    for chunk in chunks:
        chunks_by_section[chunk["section_number"]].append(chunk)

    for section in index_builder.split_sections(part):
        info = index_builder.parse_section_heading(section["heading"])
        metadata, _causal, _body = index_builder.parse_source_metadata(
            section["lines"], [], "test"
        )
        guide_ids = {
            source_id.strip()
            for source_id in (metadata.get("source_ids") or "").split(",")
            if source_id.strip()
        }
        assert guide_ids, f"{info['section_number']} has no Source IDs"

        for chunk in chunks_by_section[info["section_number"]]:
            assert set(chunk["source_ids"]) == guide_ids


def test_chunk_ids_are_unique(built_guide):
    chunks, _warnings = built_guide
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_generator_prompt_and_internal_sources_keep_source_ids(built_guide):
    chunks, _warnings = built_guide
    chunk = next(
        item
        for item in chunks
        if item["disease_name"] == "Wheat Powdery Mildew"
    )

    prompt = generator._build_user_content("wheat powdery mildew", [chunk])
    assert "[source IDs: S07, S44, S51, S57]" in prompt

    sources = generator._build_sources({1}, [chunk])
    assert sources[0]["source_ids"] == ["S07", "S44", "S51", "S57"]
    assert sources[0]["primary_source_id"] == "S07"


def test_frontend_source_shape_is_unchanged(built_guide):
    """source_ids stay internal; the API-facing source keys are unchanged."""
    chunks, _warnings = built_guide
    chunk = next(
        item
        for item in chunks
        if item["disease_name"] == "Wheat Powdery Mildew"
    )
    sources = generator._build_sources({1}, [chunk])
    mapped = main._frontend_sources(sources, [chunk])

    assert set(mapped[0]) == {
        "filename",
        "citation",
        "title",
        "url",
        "institution",
        "evidence_level",
        "chunk_id",
    }
    assert "source_ids" not in mapped[0]


@pytest.mark.parametrize(
    "status,expected",
    [
        ("Full PDF read; relevant disease chapter verified", "full_page"),
        ("Full PDF read; relevant chapter verified", "full_page"),
        ("Full page read; verified", "full_page"),
        ("Full page read; partial (conditions not stated)", "full_page"),
        ("Full PDF read; partial", "unknown"),
        ("Full PDF read; not verified", "unknown"),
        ("Full PDF read; unverified", "unknown"),
        ("Full page read; unverified", "unknown"),
        ("Abstract page read", "abstract_only"),
        ("Abstract only; full PDF not accessed", "abstract_only"),
    ],
)
def test_evidence_level_status_variants(status, expected):
    warnings: list[str] = []
    assert (
        index_builder.evidence_level_from_status(status, "", warnings) == expected
    )
