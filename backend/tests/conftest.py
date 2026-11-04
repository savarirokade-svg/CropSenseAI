"""
Shared fixtures for the ``POST /chat`` persistence integration tests.

These tests run against a real PostgreSQL instance, but only ever against the
dedicated ``cropsense_test`` database: the URL is resolved here, before any
application module is imported, and the process refuses to start if the resolved
database is not the test one. ``backend/.env`` is only read, never written.

Every Gemini-backed call is replaced, and a real SDK client is made to raise, so
no live API call can happen even if a code path changes.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pytest
import pytest_asyncio
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

TEST_DATABASE_NAME = "cropsense_test"


def _test_database_url() -> str:
    """Resolve the test database URL, defaulting to the documented local setup."""
    explicit = os.getenv("TEST_DATABASE_URL", "").strip()
    if explicit:
        return explicit

    user = quote_plus(os.environ["POSTGRES_USER"])
    password = quote_plus(os.environ["POSTGRES_PASSWORD"])
    host = os.getenv("POSTGRES_HOST", "").strip() or "127.0.0.1"
    port = os.getenv("POSTGRES_PORT", "").strip() or "5433"
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{TEST_DATABASE_NAME}"


TEST_DATABASE_URL = _test_database_url()
if TEST_DATABASE_URL.rsplit("/", 1)[-1].split("?")[0] != TEST_DATABASE_NAME:
    raise RuntimeError(
        "Refusing to run the integration tests: they must use the "
        f"{TEST_DATABASE_NAME} database, got {TEST_DATABASE_URL!r}."
    )

# ``settings`` builds the engine's URL at import time, so the override has to be
# in place before the application modules are imported below.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

import httpx  # noqa: E402
from sqlalchemy import text  # noqa: E402

import db  # noqa: E402
import generator  # noqa: E402
import main  # noqa: E402
import storage  # noqa: E402
import vision  # noqa: E402
from settings import settings as app_settings  # noqa: E402

ALL_TABLES = (
    "sessions",
    "conversations",
    "images",
    "messages",
    "message_sources",
    "message_kg_edges",
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24

TOMATO_QUESTION = "Why are my tomato leaves turning yellow?"

FAKE_CHUNKS: list[dict] = [
    {
        "chunk_id": "chunk-tomato-nutrient",
        "crop": "tomato",
        "disease_name": "Tomato Nutrient Deficiency",
        "source_title": "Tomato nutrient deficiency — TNAU Agritech Portal",
        "source_url": "https://example.test/tomato-nutrient",
        "source_institution": "Tamil Nadu Agricultural University",
        "evidence_level": "full_page",
        "text": "Yellowing of lower tomato leaves is linked to nitrogen deficiency.",
    },
    {
        "chunk_id": "chunk-tomato-waterlogging",
        "crop": "tomato",
        "disease_name": "Tomato Waterlogging",
        "source_title": "Tomato water management — TNAU Agritech Portal",
        "source_url": "https://example.test/tomato-water",
        "source_institution": "Tamil Nadu Agricultural University",
        "evidence_level": "full_page",
        "text": "Waterlogged roots also cause yellowing of tomato leaves.",
    },
]

GROUNDED_REPLY = (
    "Yellowing of lower tomato leaves is most often linked to a nitrogen "
    "deficiency or to waterlogged roots [1].\n\n"
    "Check drainage first, then soil nitrogen before adding fertiliser."
)

VISION_TEXT = "The leaf shows interveinal yellowing with no visible lesions."


class FakeVectorStore:
    """Minimal stand-in for the loaded FAISS store (only ``chunks`` is read)."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks


def grounded_result() -> dict:
    """The shape ``generate_grounded_answer`` returns for a verified answer."""
    return {
        "status": "success",
        "reply": GROUNDED_REPLY,
        "sources": [
            {
                "citation": 1,
                "title": FAKE_CHUNKS[0]["source_title"],
                "url": FAKE_CHUNKS[0]["source_url"],
                "institution": FAKE_CHUNKS[0]["source_institution"],
                "evidence_level": FAKE_CHUNKS[0]["evidence_level"],
            }
        ],
    }


def fake_hybrid_search(
    query: str, top_k: int = 10, **kwargs
) -> list[dict]:
    """Return the stand-in corpus with the semantic metadata the guard reads."""
    return [dict(chunk, semantic_rank=index + 1) for index, chunk in enumerate(FAKE_CHUNKS)]


def fake_generate(query: str, context_chunks: list[dict]) -> dict:
    return grounded_result()


def fake_vision(
    image_bytes: bytes, mime_type: str, user_text: str, context_documents: list[dict]
) -> str:
    return VISION_TEXT


@pytest.fixture(autouse=True)
def offline_pipeline(monkeypatch):
    """Replace every Gemini-backed call and forbid creating a real client."""

    def _no_live_client(*args, **kwargs):
        raise AssertionError("A live Gemini client was created during the tests.")

    monkeypatch.setattr(generator, "_get_client", _no_live_client)
    monkeypatch.setattr(vision, "_get_client", _no_live_client)

    monkeypatch.setattr(main, "load_vector_store", lambda: FakeVectorStore(FAKE_CHUNKS))
    monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main, "generate_grounded_answer", fake_generate)
    monkeypatch.setattr(vision, "analyze_plant_image", fake_vision)


@pytest.fixture(autouse=True)
def uploads_root(tmp_path, monkeypatch):
    """Point image storage at a throwaway directory for each test."""
    patched = dataclasses.replace(
        app_settings, uploads_dir=tmp_path / "uploads"
    )
    monkeypatch.setattr(storage, "settings", patched)
    return patched.uploads_dir


@pytest_asyncio.fixture(autouse=True)
async def clean_database():
    """Leave no rows behind and never let a pooled connection cross event loops."""
    yield
    async with db.SessionLocal() as session:
        await session.execute(text("delete from sessions"))
        await session.commit()
    await db.engine.dispose()


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as async_client:
        yield async_client


@pytest_asyncio.fixture
async def session(client):
    """A real anonymous session created through the public endpoint."""
    response = await client.post("/sessions")
    assert response.status_code == 201, response.text
    body = response.json()
    return {
        "session_id": body["session_id"],
        "token": body["token"],
        "headers": {"X-Session-Token": body["token"]},
    }


async def count_rows(table: str) -> int:
    """Count rows in one of the six tables (name is a fixture constant)."""
    assert table in ALL_TABLES
    async with db.SessionLocal() as db_session:
        result = await db_session.execute(text(f"select count(*) from {table}"))
        return result.scalar_one()


async def fetch_all(sql: str, **params):
    async with db.SessionLocal() as db_session:
        result = await db_session.execute(text(sql), params)
        return result.mappings().all()


async def execute(sql: str, **params) -> None:
    async with db.SessionLocal() as db_session:
        await db_session.execute(text(sql), params)
        await db_session.commit()
