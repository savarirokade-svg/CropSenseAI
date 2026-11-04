"""
Integration tests for Task 8: optional persistence in ``POST /chat``.

Every test runs against ``cropsense_test`` only (see ``conftest.py``), with the
Gemini-backed calls replaced. The stateless contract — five response keys and no
writes — is asserted alongside the persisted flow, so an accidental behaviour
change in either path fails the suite.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

import main
import repository
from conftest import (
    FAKE_CHUNKS,
    GROUNDED_REPLY,
    PNG_BYTES,
    TOMATO_QUESTION,
    VISION_TEXT,
    count_rows,
    execute,
    fetch_all,
)

pytestmark = pytest.mark.asyncio

FIVE_KEYS = {"user_message", "reply", "mode", "sources", "knowledge_graph"}
PERSISTENCE_KEYS = {"conversation_id", "user_message_id", "assistant_message_id"}
ALL_TABLES = (
    "sessions",
    "conversations",
    "images",
    "messages",
    "message_sources",
    "message_kg_edges",
)


def stored_files(root) -> list:
    return [path for path in root.rglob("*") if path.is_file()] if root.exists() else []


async def assert_no_writes() -> None:
    for table in ALL_TABLES:
        assert await count_rows(table) == 0, f"{table} should be empty"


# ── 1. Stateless text-only requests ─────────────────────────────────────────


async def test_stateless_text_chat_returns_the_five_original_keys(client):
    response = await client.post("/chat", data={"message": TOMATO_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == FIVE_KEYS
    assert body["user_message"] == TOMATO_QUESTION
    assert body["mode"] == "grounded_rag"
    assert body["reply"]
    assert "[1]" in body["reply"]
    assert body["sources"][0]["chunk_id"] == "chunk-tomato-nutrient"
    assert body["knowledge_graph"]
    await assert_no_writes()


# ── 2. Stateless image requests ─────────────────────────────────────────────


async def test_stateless_image_chat_saves_no_file_and_no_row(client, uploads_root):
    response = await client.post(
        "/chat",
        data={"message": TOMATO_QUESTION},
        files={"image": ("leaf.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == FIVE_KEYS
    assert body["mode"] == "vision_and_grounded_rag"
    assert VISION_TEXT in body["reply"]
    assert stored_files(uploads_root) == []
    await assert_no_writes()


# ── 3. Missing token ────────────────────────────────────────────────────────


async def test_missing_token_creates_no_records(client):
    await client.post("/chat", data={"message": TOMATO_QUESTION})
    await client.post(
        "/chat",
        data={"message": TOMATO_QUESTION},
        files={"image": ("leaf.png", PNG_BYTES, "image/png")},
    )

    await assert_no_writes()


# ── 4. Valid token creates the conversation and both messages ───────────────


async def test_valid_token_creates_conversation_and_messages(client, session):
    response = await client.post(
        "/chat", data={"message": TOMATO_QUESTION}, headers=session["headers"]
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == FIVE_KEYS | PERSISTENCE_KEYS

    conversations = await fetch_all(
        "select id, session_id, title, crop from conversations"
    )
    assert len(conversations) == 1
    conversation = conversations[0]
    assert str(conversation["id"]) == body["conversation_id"]
    assert str(conversation["session_id"]) == session["session_id"]
    assert conversation["title"] == TOMATO_QUESTION
    assert conversation["crop"] == "tomato"

    messages = await fetch_all(
        "select id, role, sequence, content, mode, grounded_answer "
        "from messages order by sequence"
    )
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert [message["sequence"] for message in messages] == [1, 2]

    assert str(messages[0]["id"]) == body["user_message_id"]
    assert messages[0]["content"] == TOMATO_QUESTION
    assert messages[0]["mode"] is None

    assert str(messages[1]["id"]) == body["assistant_message_id"]
    assert messages[1]["content"] == body["reply"]
    assert messages[1]["mode"] == "grounded_rag"
    assert messages[1]["grounded_answer"] == GROUNDED_REPLY


# ── 5. A second turn appends to the same conversation ───────────────────────


async def test_second_request_appends_to_the_same_conversation(client, session):
    first = await client.post(
        "/chat", data={"message": TOMATO_QUESTION}, headers=session["headers"]
    )
    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]

    second = await client.post(
        "/chat",
        data={
            "message": "Does drainage matter for tomato yellowing?",
            "conversation_id": conversation_id,
        },
        headers=session["headers"],
    )

    assert second.status_code == 200
    body = second.json()
    assert body["conversation_id"] == conversation_id
    assert body["user_message_id"] != first.json()["user_message_id"]

    assert await count_rows("conversations") == 1
    messages = await fetch_all("select role, sequence from messages order by sequence")
    assert [message["sequence"] for message in messages] == [1, 2, 3, 4]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


# ── 6. Invalid token ────────────────────────────────────────────────────────


async def test_invalid_token_returns_401(client):
    response = await client.post(
        "/chat",
        data={"message": TOMATO_QUESTION},
        headers={"X-Session-Token": "definitely-not-a-valid-token"},
    )

    assert response.status_code == 401
    assert "X-Session-Token" in response.json()["detail"]
    for table in ("conversations", "images", "messages"):
        assert await count_rows(table) == 0


# ── 7. Expired token ────────────────────────────────────────────────────────


async def test_expired_token_returns_401(client, session):
    await execute(
        "update sessions set expires_at = now() - interval '1 day' where id = :session_id",
        session_id=uuid.UUID(session["session_id"]),
    )

    response = await client.post(
        "/chat", data={"message": TOMATO_QUESTION}, headers=session["headers"]
    )

    assert response.status_code == 401
    assert await count_rows("conversations") == 0
    assert await count_rows("messages") == 0


# ── 8. Foreign conversation ─────────────────────────────────────────────────


async def test_foreign_conversation_returns_404(client, session):
    first = await client.post(
        "/chat", data={"message": TOMATO_QUESTION}, headers=session["headers"]
    )
    conversation_id = first.json()["conversation_id"]

    other = await client.post("/sessions")
    assert other.status_code == 201
    other_headers = {"X-Session-Token": other.json()["token"]}

    response = await client.post(
        "/chat",
        data={
            "message": "Do tomato leaves yellow from overwatering?",
            "conversation_id": conversation_id,
        },
        headers=other_headers,
    )

    assert response.status_code == 404
    assert await count_rows("conversations") == 1
    assert await count_rows("messages") == 2


# ── 9. conversation_id without a token ──────────────────────────────────────


async def test_conversation_id_without_token_returns_400(client):
    response = await client.post(
        "/chat",
        data={
            "message": TOMATO_QUESTION,
            "conversation_id": str(uuid.uuid4()),
        },
    )

    assert response.status_code == 400
    await assert_no_writes()


# ── 10. Image metadata for a persisted turn ─────────────────────────────────


async def test_image_upload_saves_file_and_metadata(client, session, uploads_root):
    response = await client.post(
        "/chat",
        data={"message": TOMATO_QUESTION},
        files={"image": ("leaf.png", PNG_BYTES, "image/png")},
        headers=session["headers"],
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "vision_and_grounded_rag"
    assert "storage_key" not in response.text

    images = await fetch_all(
        "select storage_key, original_filename, mime_type, byte_size, sha256 "
        "from images"
    )
    assert len(images) == 1
    image = images[0]
    assert image["storage_key"].startswith(f"sessions/{session['session_id']}/")
    assert image["storage_key"].endswith(".png")
    assert image["original_filename"] == "leaf.png"
    assert image["mime_type"] == "image/png"
    assert image["byte_size"] == len(PNG_BYTES)
    assert len(image["sha256"]) == 64

    stored = uploads_root / image["storage_key"]
    assert stored.is_file()
    assert stored.read_bytes() == PNG_BYTES

    user_message = (
        await fetch_all("select image_id from messages where role = 'user'")
    )[0]
    assert str(user_message["image_id"]) == str(
        (await fetch_all("select id from images"))[0]["id"]
    )


# ── 11. Image-only uploads ──────────────────────────────────────────────────


async def test_image_only_upload_is_persisted(client, session, uploads_root):
    response = await client.post(
        "/chat",
        files={"image": ("leaf.png", PNG_BYTES, "image/png")},
        headers=session["headers"],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_message"] == ""
    assert VISION_TEXT in body["reply"]
    # An image on its own names no crop, so the existing crop guard still asks
    # which crop it is: persistence must not change that outcome.
    assert body["mode"] == "vision_and_ambiguous_crop"

    messages = await fetch_all(
        "select role, content, image_id from messages order by sequence"
    )
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == ""
    assert messages[0]["image_id"] is not None
    assert await count_rows("images") == 1
    assert len(stored_files(uploads_root)) == 1


# ── 12. Image validation failures leave nothing behind ──────────────────────


async def test_image_validation_failure_leaves_no_rows_or_files(
    client, session, uploads_root
):
    not_an_image = await client.post(
        "/chat",
        data={"message": TOMATO_QUESTION},
        files={"image": ("notes.txt", b"just some text", "text/plain")},
        headers=session["headers"],
    )
    assert not_an_image.status_code == 400

    spoofed_png = await client.post(
        "/chat",
        data={"message": TOMATO_QUESTION},
        files={"image": ("leaf.png", b"certainly not a png", "image/png")},
        headers=session["headers"],
    )
    assert spoofed_png.status_code == 400

    assert await count_rows("conversations") == 0
    assert await count_rows("messages") == 0
    assert await count_rows("images") == 0
    assert stored_files(uploads_root) == []


# ── 13. Fallback outcomes are persisted, with a sanitized category ─────────


async def test_generation_fallback_persists_the_user_message(client, session, monkeypatch):
    def failing_generate(query, context_chunks):
        return {"status": "insufficient_context", "reply": "", "sources": []}

    monkeypatch.setattr(main, "generate_grounded_answer", failing_generate)

    response = await client.post(
        "/chat", data={"message": TOMATO_QUESTION}, headers=session["headers"]
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "insufficient_evidence"

    messages = await fetch_all(
        "select role, content, mode, error_category, grounded_answer "
        "from messages order by sequence"
    )
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == TOMATO_QUESTION
    assert messages[1]["content"] == body["reply"]
    assert messages[1]["mode"] == "insufficient_evidence"
    assert messages[1]["error_category"] == "insufficient-context"
    assert messages[1]["grounded_answer"] is None
    assert await count_rows("message_sources") == 0


# ── 14. Citations and knowledge graph round-trip ────────────────────────────


async def test_sources_and_knowledge_graph_are_persisted(client, session):
    response = await client.post(
        "/chat", data={"message": TOMATO_QUESTION}, headers=session["headers"]
    )

    assert response.status_code == 200
    body = response.json()
    assert body["knowledge_graph"]

    sources = await fetch_all(
        "select message_id, citation_number, chunk_id, filename, title, url, "
        "institution, evidence_level, disease_name, crop from message_sources"
    )
    assert len(sources) == 1
    source = sources[0]
    assert str(source["message_id"]) == body["assistant_message_id"]
    assert source["citation_number"] == 1
    assert source["chunk_id"] == FAKE_CHUNKS[0]["chunk_id"]
    assert source["filename"] == FAKE_CHUNKS[0]["source_title"]
    assert source["title"] == FAKE_CHUNKS[0]["source_title"]
    assert source["url"] == FAKE_CHUNKS[0]["source_url"]
    assert source["institution"] == FAKE_CHUNKS[0]["source_institution"]
    assert source["evidence_level"] == FAKE_CHUNKS[0]["evidence_level"]
    assert source["disease_name"] == FAKE_CHUNKS[0]["disease_name"]
    assert source["crop"] == FAKE_CHUNKS[0]["crop"]

    stored_edges = {
        (edge["subject"], edge["relation"], edge["object"])
        for edge in await fetch_all("select subject, relation, object from message_kg_edges")
    }
    expected_edges = {
        (edge["subject"], edge["relation"], edge["object"])
        for edge in body["knowledge_graph"]
    }
    assert stored_edges == expected_edges


# ── 15. Response fields and ids ─────────────────────────────────────────────


async def test_persisted_response_fields_and_ids_are_correct(client, session):
    response = await client.post(
        "/chat", data={"message": TOMATO_QUESTION}, headers=session["headers"]
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == FIVE_KEYS | PERSISTENCE_KEYS

    for key in PERSISTENCE_KEYS:
        assert uuid.UUID(body[key]), f"{key} is not a UUID"
    assert len({body[key] for key in PERSISTENCE_KEYS}) == 3

    for secret in ("storage_key", "token_hash", session["token"]):
        assert secret not in response.text

    conversation = (await fetch_all("select id from conversations"))[0]
    assert str(conversation["id"]) == body["conversation_id"]
    stored_ids = {str(row["id"]) for row in await fetch_all("select id from messages")}
    assert stored_ids == {body["user_message_id"], body["assistant_message_id"]}


# ── 16. Stateless shape is byte-for-byte the original contract ─────────────


async def test_stateless_response_shape_is_unchanged(client):
    text_only = await client.post("/chat", data={"message": TOMATO_QUESTION})
    assert text_only.status_code == 200
    assert set(text_only.json()) == FIVE_KEYS

    with_image = await client.post(
        "/chat",
        data={"message": TOMATO_QUESTION},
        files={"image": ("leaf.png", PNG_BYTES, "image/png")},
    )
    assert with_image.status_code == 200
    assert set(with_image.json()) == FIVE_KEYS

    await assert_no_writes()


# ── 17. A stateless request must not need a reachable database ─────────────


async def test_stateless_chat_works_without_a_database(client):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import db

    unreachable = create_async_engine(
        "postgresql+asyncpg://postgres:postgres@127.0.0.1:1/cropsense_test"
    )
    factory = async_sessionmaker(bind=unreachable, expire_on_commit=False)

    async def unreachable_session():
        async with factory() as db_session:
            yield db_session

    main.app.dependency_overrides[db.get_session] = unreachable_session
    try:
        response = await client.post("/chat", data={"message": TOMATO_QUESTION})
    finally:
        main.app.dependency_overrides.clear()
        await unreachable.dispose()

    assert response.status_code == 200
    assert set(response.json()) == FIVE_KEYS


# ── 18. A failed crop update must not poison the request's session ─────────


async def test_failed_crop_update_still_saves_the_turn(client, session, monkeypatch):
    """
    A crop update that fails in the database must be rolled back, not left to
    abort the assistant-message write that follows it.
    """

    async def broken_set_conversation_crop(
        db_session, session_id, conversation_id, crop
    ):
        # Invalid on purpose: this aborts the transaction on the request's own
        # session, exactly as a failing crop UPDATE would.
        await db_session.execute(text("select * from table_that_does_not_exist"))

    monkeypatch.setattr(
        repository, "set_conversation_crop", broken_set_conversation_crop
    )

    response = await client.post(
        "/chat", data={"message": TOMATO_QUESTION}, headers=session["headers"]
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == FIVE_KEYS | PERSISTENCE_KEYS
    assert body["reply"]

    messages = await fetch_all(
        "select id, role, sequence, content from messages order by sequence"
    )
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert [message["sequence"] for message in messages] == [1, 2]
    assert messages[0]["content"] == TOMATO_QUESTION
    assert messages[1]["content"] == body["reply"]
    assert str(messages[1]["id"]) == body["assistant_message_id"]

    # The crop was never recorded, so the conversation keeps it unset.
    conversation = (await fetch_all("select id, crop from conversations"))[0]
    assert str(conversation["id"]) == body["conversation_id"]
    assert conversation["crop"] is None

    # The rolled-back session recovered: the same conversation takes a new turn.
    follow_up = await client.post(
        "/chat",
        data={
            "message": "Does drainage matter for tomato yellowing?",
            "conversation_id": body["conversation_id"],
        },
        headers=session["headers"],
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["conversation_id"] == body["conversation_id"]
    assert await count_rows("messages") == 4
