"""
Integration tests for the Task 9 history endpoints.

Covers the five authenticated routes (``GET /conversations``,
``GET/PATCH/DELETE /conversations/{id}``, ``GET /images/{id}``): ownership
scoping, response shape, ordering, validation and deletion side effects.

Everything runs against ``cropsense_test`` only (see ``conftest.py``) and the
Gemini pipeline is mocked by the shared fixtures, so no live API call is made.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

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
from knowledge_graph import search_knowledge_graph

pytestmark = pytest.mark.asyncio

SUMMARY_KEYS = {
    "id",
    "title",
    "crop",
    "created_at",
    "updated_at",
    "message_count",
    "has_diagnosis",
}
HEADER_KEYS = {"id", "title", "crop", "created_at", "updated_at"}


async def new_session(client) -> dict:
    """A second anonymous session, for ownership tests."""
    response = await client.post("/sessions")
    assert response.status_code == 201, response.text
    body = response.json()
    return {
        "session_id": body["session_id"],
        "token": body["token"],
        "headers": {"X-Session-Token": body["token"]},
    }


async def start_conversation(client, headers, message: str = TOMATO_QUESTION) -> dict:
    """Create a conversation through the real /chat endpoint."""
    response = await client.post("/chat", data={"message": message}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


async def start_conversation_with_image(client, headers) -> dict:
    """Create a conversation whose user turn carries a stored image."""
    response = await client.post(
        "/chat",
        data={"message": TOMATO_QUESTION},
        files={"image": ("leaf.png", PNG_BYTES, "image/png")},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


# ── 1. Missing token ────────────────────────────────────────────────────────


async def test_all_endpoints_require_a_session_token(client):
    conversation_id = str(uuid.uuid4())
    image_id = str(uuid.uuid4())

    requests = [
        ("get", "/conversations", None),
        ("get", f"/conversations/{conversation_id}", None),
        ("patch", f"/conversations/{conversation_id}", {"title": "Renamed"}),
        ("delete", f"/conversations/{conversation_id}", None),
        ("get", f"/images/{image_id}", None),
    ]

    for method, path, body in requests:
        response = await client.request(method.upper(), path, json=body)
        assert response.status_code == 401, (method, path, response.text)
        assert "X-Session-Token" in response.json()["detail"]

    assert await count_rows("conversations") == 0


# ── 2. Invalid and expired tokens ───────────────────────────────────────────


async def test_invalid_and_expired_tokens_return_401(client, session):
    bad_headers = {"X-Session-Token": "definitely-not-a-session-token"}
    assert (await client.get("/conversations", headers=bad_headers)).status_code == 401
    assert (
        await client.get(
            f"/conversations/{uuid.uuid4()}", headers=bad_headers
        )
    ).status_code == 401
    assert (
        await client.patch(
            f"/conversations/{uuid.uuid4()}", json={"title": "x"}, headers=bad_headers
        )
    ).status_code == 401
    assert (
        await client.delete(f"/conversations/{uuid.uuid4()}", headers=bad_headers)
    ).status_code == 401
    assert (
        await client.get(f"/images/{uuid.uuid4()}", headers=bad_headers)
    ).status_code == 401

    await execute(
        "update sessions set expires_at = now() - interval '1 day' where id = :session_id",
        session_id=uuid.UUID(session["session_id"]),
    )

    expired = await client.get("/conversations", headers=session["headers"])
    assert expired.status_code == 401


# ── 3. Only the current session's conversations ─────────────────────────────


async def test_list_contains_only_the_current_sessions_conversations(client, session):
    first = await start_conversation(client, session["headers"])
    second = await start_conversation(
        client, session["headers"], "Why does my tomato plant have dark spots?"
    )

    other = await new_session(client)
    foreign = await start_conversation(client, other["headers"])

    response = await client.get("/conversations", headers=session["headers"])
    assert response.status_code == 200
    items = response.json()

    assert {item["id"] for item in items} == {
        first["conversation_id"],
        second["conversation_id"],
    }
    assert foreign["conversation_id"] not in {item["id"] for item in items}

    for item in items:
        assert set(item) == SUMMARY_KEYS
        assert item["crop"] == "tomato"
        assert item["message_count"] == 2
        assert item["has_diagnosis"] is False

    other_items = (await client.get("/conversations", headers=other["headers"])).json()
    assert [item["id"] for item in other_items] == [foreign["conversation_id"]]

    empty = await new_session(client)
    assert (await client.get("/conversations", headers=empty["headers"])).json() == []


# ── 4. Newest updated first ─────────────────────────────────────────────────


async def test_list_orders_newest_updated_conversation_first(client, session):
    first = await start_conversation(client, session["headers"])
    second = await start_conversation(
        client, session["headers"], "Why does my tomato plant have dark spots?"
    )
    third = await start_conversation(
        client, session["headers"], "How often should I water tomato plants?"
    )

    # Set known timestamps instead of relying on how fast the requests run.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for result, minutes in ((first, 1), (second, 3), (third, 2)):
        await execute(
            "update conversations set updated_at = :ts where id = :id",
            ts=base + timedelta(minutes=minutes),
            id=uuid.UUID(result["conversation_id"]),
        )

    items = (await client.get("/conversations", headers=session["headers"])).json()
    assert [item["id"] for item in items] == [
        second["conversation_id"],
        third["conversation_id"],
        first["conversation_id"],
    ]

    # A new turn moves its conversation back to the top.
    follow_up = await client.post(
        "/chat",
        data={
            "message": "Still yellow after fixing drainage?",
            "conversation_id": first["conversation_id"],
        },
        headers=session["headers"],
    )
    assert follow_up.status_code == 200

    items = (await client.get("/conversations", headers=session["headers"])).json()
    assert items[0]["id"] == first["conversation_id"]
    assert items[0]["message_count"] == 4


# ── 5. Detail: ordered messages, citations, knowledge graph ─────────────────


async def test_conversation_detail_returns_messages_citations_and_graph(
    client, session
):
    created = await start_conversation(client, session["headers"])
    conversation_id = created["conversation_id"]

    response = await client.get(
        f"/conversations/{conversation_id}", headers=session["headers"]
    )
    assert response.status_code == 200
    body = response.json()

    assert set(body) == HEADER_KEYS | {"messages"}
    assert body["id"] == conversation_id
    assert body["title"] == TOMATO_QUESTION
    assert body["crop"] == "tomato"
    assert body["created_at"] and body["updated_at"]

    messages = body["messages"]
    assert [message["sequence"] for message in messages] == [1, 2]
    assert [message["role"] for message in messages] == ["user", "assistant"]

    user_message, assistant_message = messages
    assert set(user_message) == {"id", "role", "sequence", "content", "created_at"}
    assert user_message["id"] == created["user_message_id"]
    assert user_message["content"] == TOMATO_QUESTION

    assert set(assistant_message) == {
        "id",
        "role",
        "sequence",
        "content",
        "created_at",
        "mode",
        "vision_assessment",
        "grounded_answer",
        "sources",
        "knowledge_graph",
    }
    assert assistant_message["id"] == created["assistant_message_id"]
    assert assistant_message["content"] == created["reply"]
    assert assistant_message["mode"] == "grounded_rag"
    assert assistant_message["vision_assessment"] is None
    assert assistant_message["grounded_answer"] == GROUNDED_REPLY

    source = assistant_message["sources"][0]
    assert set(source) == {
        "citation",
        "filename",
        "title",
        "url",
        "institution",
        "evidence_level",
        "chunk_id",
    }
    assert source["citation"] == 1
    assert source["chunk_id"] == FAKE_CHUNKS[0]["chunk_id"]
    assert source["filename"] == FAKE_CHUNKS[0]["source_title"]
    assert source["title"] == FAKE_CHUNKS[0]["source_title"]
    assert source["url"] == FAKE_CHUNKS[0]["source_url"]
    assert source["institution"] == FAKE_CHUNKS[0]["source_institution"]
    assert source["evidence_level"] == "full_page"

    stored_edges = sorted(
        (edge["subject"], edge["relation"], edge["object"])
        for edge in assistant_message["knowledge_graph"]
    )
    expected_edges = sorted(
        (edge["subject"], edge["relation"], edge["object"])
        for edge in search_knowledge_graph(TOMATO_QUESTION)
    )
    assert stored_edges
    assert stored_edges == expected_edges

    for internal in ("storage_key", "token_hash", "session_id", "error_category"):
        assert internal not in response.text


# ── 6. User image metadata ──────────────────────────────────────────────────


async def test_user_image_metadata_is_returned(client, session):
    created = await start_conversation_with_image(client, session["headers"])

    body = (
        await client.get(
            f"/conversations/{created['conversation_id']}", headers=session["headers"]
        )
    ).json()
    user_message, assistant_message = body["messages"]

    image = user_message["image"]
    assert set(image) == {"id", "url", "mime_type", "byte_size"}
    assert image["url"] == f"/images/{image['id']}"
    assert image["mime_type"] == "image/png"
    assert image["byte_size"] == len(PNG_BYTES)

    stored = (await fetch_all("select id, storage_key, mime_type, byte_size from images"))[0]
    assert image["id"] == str(stored["id"])
    assert image["mime_type"] == stored["mime_type"]
    assert image["byte_size"] == stored["byte_size"]
    assert stored["storage_key"] not in str(body)

    assert "image" not in assistant_message
    assert assistant_message["vision_assessment"] == VISION_TEXT
    assert assistant_message["mode"] == "vision_and_grounded_rag"

    list_item = (await client.get("/conversations", headers=session["headers"])).json()[0]
    assert list_item["message_count"] == 2
    assert list_item["has_diagnosis"] is True


# ── 7. Rename ───────────────────────────────────────────────────────────────


async def test_rename_conversation_succeeds(client, session):
    created = await start_conversation(client, session["headers"])
    conversation_id = created["conversation_id"]

    response = await client.patch(
        f"/conversations/{conversation_id}",
        json={"title": "Tomato yellowing"},
        headers=session["headers"],
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == HEADER_KEYS
    assert body["id"] == conversation_id
    assert body["title"] == "Tomato yellowing"

    detail = (
        await client.get(
            f"/conversations/{conversation_id}", headers=session["headers"]
        )
    ).json()
    assert detail["title"] == "Tomato yellowing"

    stored = (await fetch_all("select title from conversations"))[0]
    assert stored["title"] == "Tomato yellowing"


# ── 8. Title validation ─────────────────────────────────────────────────────


async def test_invalid_titles_are_rejected(client, session):
    created = await start_conversation(client, session["headers"])
    url = f"/conversations/{created['conversation_id']}"

    for bad_title in ("", "   ", "x" * 201):
        response = await client.patch(
            url, json={"title": bad_title}, headers=session["headers"]
        )
        assert response.status_code == 400, repr(bad_title)
        assert "1 and 200" in response.json()["detail"]

    detail = (await client.get(url, headers=session["headers"])).json()
    assert detail["title"] == TOMATO_QUESTION

    boundary = await client.patch(url, json={"title": "y" * 200}, headers=session["headers"])
    assert boundary.status_code == 200
    assert len(boundary.json()["title"]) == 200

    trimmed = await client.patch(
        url, json={"title": "  Trimmed title  "}, headers=session["headers"]
    )
    assert trimmed.status_code == 200
    assert trimmed.json()["title"] == "Trimmed title"

    missing = await client.patch(url, json={}, headers=session["headers"])
    assert missing.status_code == 422


# ── 9. Ownership ────────────────────────────────────────────────────────────


async def test_another_session_cannot_read_rename_or_delete(client, session):
    created = await start_conversation(client, session["headers"])
    conversation_id = created["conversation_id"]
    other = await new_session(client)

    assert (
        await client.get(f"/conversations/{conversation_id}", headers=other["headers"])
    ).status_code == 404
    assert (
        await client.patch(
            f"/conversations/{conversation_id}",
            json={"title": "Hijacked"},
            headers=other["headers"],
        )
    ).status_code == 404
    assert (
        await client.delete(f"/conversations/{conversation_id}", headers=other["headers"])
    ).status_code == 404

    detail = await client.get(
        f"/conversations/{conversation_id}", headers=session["headers"]
    )
    assert detail.status_code == 200
    assert detail.json()["title"] == TOMATO_QUESTION
    assert await count_rows("messages") == 2


# ── 10. Delete returns 204 ──────────────────────────────────────────────────


async def test_delete_conversation_returns_204(client, session):
    created = await start_conversation(client, session["headers"])

    response = await client.delete(
        f"/conversations/{created['conversation_id']}", headers=session["headers"]
    )
    assert response.status_code == 204
    assert response.content == b""

    assert await count_rows("conversations") == 0
    assert (await client.get("/conversations", headers=session["headers"])).json() == []


# ── 11. Delete removes records and stored files ─────────────────────────────


async def test_delete_removes_related_records_and_image_files(
    client, session, uploads_root
):
    created = await start_conversation_with_image(client, session["headers"])

    stored_row = (await fetch_all("select storage_key from images"))[0]
    stored_file = uploads_root / stored_row["storage_key"]
    assert stored_file.is_file()

    assert await count_rows("messages") == 2
    assert await count_rows("message_sources") == 1
    assert await count_rows("message_kg_edges") > 0
    assert await count_rows("images") == 1

    response = await client.delete(
        f"/conversations/{created['conversation_id']}", headers=session["headers"]
    )
    assert response.status_code == 204

    for table in (
        "conversations",
        "messages",
        "message_sources",
        "message_kg_edges",
        "images",
    ):
        assert await count_rows(table) == 0, table
    assert not stored_file.exists()


# ── 12. Second delete ───────────────────────────────────────────────────────


async def test_second_delete_returns_404(client, session):
    created = await start_conversation(client, session["headers"])
    url = f"/conversations/{created['conversation_id']}"

    assert (await client.delete(url, headers=session["headers"])).status_code == 204
    assert (await client.delete(url, headers=session["headers"])).status_code == 404


# ── 13. Image bytes for the owner ───────────────────────────────────────────


async def test_image_owner_receives_bytes_and_content_type(client, session):
    created = await start_conversation_with_image(client, session["headers"])
    detail = (
        await client.get(
            f"/conversations/{created['conversation_id']}", headers=session["headers"]
        )
    ).json()
    image_id = detail["messages"][0]["image"]["id"]

    response = await client.get(f"/images/{image_id}", headers=session["headers"])

    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"] == "image/png"
    assert "sessions/" not in str(response.headers)


# ── 14. Image ownership ─────────────────────────────────────────────────────


async def test_another_session_receives_404_for_the_image(client, session):
    created = await start_conversation_with_image(client, session["headers"])
    detail = (
        await client.get(
            f"/conversations/{created['conversation_id']}", headers=session["headers"]
        )
    ).json()
    image_id = detail["messages"][0]["image"]["id"]

    other = await new_session(client)
    response = await client.get(f"/images/{image_id}", headers=other["headers"])

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found."


# ── 15. Unknown and malformed ids ───────────────────────────────────────────


async def test_unknown_and_malformed_ids_are_reported_as_missing(client, session):
    unknown = str(uuid.uuid4())

    for path in (f"/conversations/{unknown}", "/conversations/not-a-uuid"):
        response = await client.get(path, headers=session["headers"])
        assert response.status_code == 404, path
        assert response.json()["detail"] == "Conversation not found."

        renamed = await client.patch(
            path, json={"title": "Anything"}, headers=session["headers"]
        )
        assert renamed.status_code == 404, path

        deleted = await client.delete(path, headers=session["headers"])
        assert deleted.status_code == 404, path

    for path in (f"/images/{unknown}", "/images/not-a-uuid", "/images/123"):
        response = await client.get(path, headers=session["headers"])
        assert response.status_code == 404, path
        assert response.json()["detail"] == "Image not found."
