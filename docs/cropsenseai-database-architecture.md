# CropSenseAI — Database Architecture

Persistence for conversations and plant-diagnosis history. This is a college project
optimised for getting a working application early: one developer, one backend server and
one PostgreSQL instance. Nothing in this document has been implemented yet.

## 1. Architecture decisions

| Decision | Choice |
|---|---|
| Primary database | **PostgreSQL 16** |
| Driver | `asyncpg` via **SQLAlchemy 2.0 async** |
| ORM | SQLAlchemy 2.0 `DeclarativeBase` + `AsyncSession` |
| Migrations | **Alembic**, one initial revision |
| Image bytes | **Filesystem** under `backend/data/uploads/`, metadata in Postgres |
| Identity | **Anonymous session token** in `localStorage`, sent as `X-Session-Token` |
| ID strategy | UUID v4 (`gen_random_uuid()`) |
| Timestamps | `TIMESTAMPTZ` in UTC |

### Why PostgreSQL

- One database process for the whole app; native `UUID` and `TIMESTAMPTZ` avoid custom
  type handling.
- Real `ON DELETE CASCADE` carries the integrity rules instead of application code.
- Alembic works with its normal path — no batch-ALTER workarounds.

A local Postgres is required for development and is started with the
`docker-compose.yml` that pins `postgres:16` and a named volume (see §5).

### Why filesystem images (not BLOB)

- Uploads are capped at 10 MB; keeping bytes out of the database keeps rows small and
  backups cheap.
- Serving a file is a simple stream; a BLOB row would hold a database connection for the
  whole transfer.
- Deletion is a two-part operation (row + file), handled explicitly when a conversation is
  permanently deleted (§3, §6).

### Why a header token and not a cookie

The browser talks to the API cross-origin (`http://localhost:3000` →
`http://127.0.0.1:8000`). A cookie would need `SameSite=None; Secure`, and `Secure` is
unusable over plain HTTP. `X-Session-Token` works with the existing CORS config and needs
no proxy. There is no cookie migration path in this project.

### Current state

- `/chat` (`main.py:389-542`) is `async def`, accepts `message: Form("")` +
  `image: File(None)`, returns `{user_message, reply, mode, sources, knowledge_graph}`.
- Images are read into memory (`await image.read()`, 10 MB cap) and passed straight to
  Gemini — no temp file and no persisted path.
- No database, ORM, migrations, or test runner installed.
- Frontend persists nothing: sidebar conversations are hardcoded `SAMPLE_CONVERSATIONS`.
- Backend Python 3.13.14; FastAPI 0.143, Pydantic 2.14, `python-multipart` present.

---

## 2. Database schema

Six tables. No `users` table, no `diagnoses` table.

```
sessions
  id             UUID PK default gen_random_uuid()
  token_hash     CHAR(64) NOT NULL UNIQUE        -- sha256 hex of the client token
  created_at     TIMESTAMPTZ NOT NULL default now()
  expires_at     TIMESTAMPTZ NOT NULL            -- created_at + 30 days
  INDEX (expires_at)

conversations
  id             UUID PK
  session_id     UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE
  title          TEXT NOT NULL default 'New conversation'
  crop           TEXT NULL                        -- 'tomato' | 'potato' from _resolve_crop_scope
  created_at     TIMESTAMPTZ NOT NULL default now()
  updated_at     TIMESTAMPTZ NOT NULL default now()
  CHECK (char_length(title) BETWEEN 1 AND 200)
  INDEX (session_id, updated_at DESC)

images
  id                UUID PK
  session_id        UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE
  storage_key       TEXT NOT NULL UNIQUE          -- 'sessions/<sid>/<uuid>.png', always relative
  original_filename TEXT NULL                     -- display only, never used as a path
  mime_type         TEXT NOT NULL
  byte_size         INTEGER NOT NULL CHECK (byte_size > 0 AND byte_size <= 10485760)
  sha256            CHAR(64) NOT NULL
  created_at        TIMESTAMPTZ NOT NULL default now()
  INDEX (session_id)

messages
  id                UUID PK
  conversation_id   UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE
  sequence          INTEGER NOT NULL              -- 1,2,3… strict per-conversation order
  role              TEXT NOT NULL CHECK (role IN ('user','assistant'))
  content           TEXT NOT NULL                 -- exactly what the user saw/typed (canonical)
  mode              TEXT NULL                     -- assistant outcome, e.g. 'vision_and_grounded_rag'
  vision_assessment TEXT NULL                     -- assistant: image-observation section only
  grounded_answer   TEXT NULL                     -- assistant: reference-grounded section only
  error_category    TEXT NULL                     -- sanitized reason, never raw SDK text
  image_id          UUID NULL REFERENCES images(id) ON DELETE SET NULL   -- user turns only
  created_at        TIMESTAMPTZ NOT NULL default now()
  UNIQUE (conversation_id, sequence)
  CHECK (role = 'user' OR mode IS NOT NULL)
  INDEX (conversation_id, sequence)

message_sources                                  -- RAG citations for one assistant message
  id               UUID PK
  message_id       UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE
  citation_number  INTEGER NOT NULL CHECK (citation_number >= 1)
  chunk_id         TEXT NOT NULL
  filename         TEXT NULL
  title            TEXT NULL
  url              TEXT NULL
  institution      TEXT NULL
  evidence_level   TEXT NULL
  disease_name     TEXT NULL                     -- denormalized from chunk metadata
  crop             TEXT NULL                     -- denormalized from chunk metadata
  UNIQUE (message_id, citation_number)
  INDEX (chunk_id)

message_kg_edges                                 -- knowledge-graph relationships for one message
  id          UUID PK
  message_id  UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE
  subject     TEXT NOT NULL
  relation    TEXT NOT NULL
  object      TEXT NOT NULL
  UNIQUE (message_id, subject, relation, object)
```

### Requirement → schema mapping

| # | Requirement | Where |
|---|---|---|
| 1 | Multiple conversations | `conversations` (many per `session_id`) |
| 2 | Persistent user + assistant messages | `messages.role`, `UNIQUE(conversation_id, sequence)` |
| 3 | Titles and timestamps | `conversations.title/created_at/updated_at`, `messages.created_at` |
| 4 | Uploaded image references | `images` + `messages.image_id` |
| 5 | Disease assessment results | `messages.vision_assessment` (+ `conversations.crop`) |
| 6 | RAG source citations | `message_sources` (+ denormalized `disease_name`/`crop`) |
| 7 | Knowledge-graph relationships | `message_kg_edges` |
| 8 | Retrieve previous conversations | `GET /conversations`, `GET /conversations/{id}` |
| 9 | Delete conversation + records | `DELETE` cascades to messages/sources/kg; image rows and files removed |

### Message ordering

Append messages in one transaction:

1. `SELECT ... FROM conversations WHERE id = :id FOR UPDATE` (locks this conversation's
   row).
2. `SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE conversation_id = :id`.
3. Insert the message with that `sequence`.
4. `UPDATE conversations SET updated_at = now() WHERE id = :id`.

The row lock plus `UNIQUE (conversation_id, sequence)` makes ordering correct on a single
backend server without any distributed locking. The first user message and the assistant
reply use separate transactions (see §3) so a generation failure still leaves the user
turn saved.

### Deliberate modelling notes

- **`content` is canonical.** `vision_assessment` and `grounded_answer` are derived splits
  of the same reply so history is queryable without re-parsing markdown. Both are written
  from one code path to avoid drift.
- **No separate `diagnoses` table.** The vision narrative plus the cited chunks'
  `disease_name`/`crop` already give a queryable diagnosis history.
- **No users table.** Anonymous sessions only; there is no login, registration or
  authentication system.
- **No soft delete.** Deleting a conversation is permanent, so there is no Trash, restore
  or purge scheduler.

---

## 3. Backend API design

Endpoints keep the existing root-level style (no `/api/v1` prefix; `/chat`, `/health`,
`/` stay at their current paths).

### Session

| Method | Path | Purpose |
|---|---|---|
| POST | `/sessions` | Create an anonymous session. Returns the raw token **once** + `expires_at` |
| GET | `/sessions/me` | Validate `X-Session-Token`; returns `{session_id, created_at, expires_at}`; 401 if bad or expired |

Token: `secrets.token_urlsafe(32)`; only `sha256(token)` is stored. Validation hashes the
presented token and looks up the row. Expiry is 30 days from creation (`expires_at`);
expired tokens return 401. There is no revocation or refresh flow.

### Conversations

| Method | Path | Purpose |
|---|---|---|
| GET | `/conversations` | Session's conversations, newest first. Returns `id, title, crop, created_at, updated_at, message_count, has_diagnosis` |
| GET | `/conversations/{id}` | Full detail: conversation + ordered messages + sources + kg |
| PATCH | `/conversations/{id}` | Rename (`{"title": "…"}`, 1–200 chars) |
| DELETE | `/conversations/{id}` | Permanently delete conversation, cascade children, delete image rows + files. `204` |

Ownership rule: every query is `WHERE id = :id AND session_id = :session_id`. A row that
exists but belongs to another session returns **404** (not 403) so IDs cannot be
enumerated.

### Chat (extended, backward compatible)

`POST /chat` keeps its current signature and gains two *optional* inputs:

- `conversation_id: Form(None)` — continue an existing conversation.
- `X-Session-Token` header — when present and valid, the exchange is persisted.

Behaviour:

1. **No `X-Session-Token` → stateless**, exactly today's path. The Gemini Vision, RAG,
   FAISS, BM25, knowledge-graph pipeline and response keys are untouched.
2. **Valid token** → session resolved, conversation created (if no `conversation_id`) or
   ownership-verified (if given); the exchange is persisted.
3. **Invalid or expired token → HTTP 401.** A supplied token that fails validation is an
   error, never a silent downgrade to stateless mode.
4. `conversations.crop` and `title` are set from `_resolve_crop_scope()` and a heuristic
   title (first ~60 chars of the first user message, trimmed at a word boundary). No extra
   Gemini call for titles.
5. Persist the **user message first and commit**, then run retrieval/generation unchanged,
   then persist the assistant message with its sources and KG edges. A crash
   mid-generation therefore loses only the assistant turn, never the user's input.
6. Failure outcomes persist the *user-visible fallback text* plus `mode` and
   `error_category` — the same sanitized categories the endpoint already produces, never
   raw SDK text.

Additive response fields (existing clients ignore them):

```json
{ "user_message": "...", "reply": "...", "mode": "…", "sources": [...],
  "knowledge_graph": [...],
  "conversation_id": "uuid", "user_message_id": "uuid", "assistant_message_id": "uuid" }
```

### Images

| Method | Path | Purpose |
|---|---|---|
| GET | `/images/{id}` | Stream the image to its owner. 404 for other sessions. `Content-Type` from the row |

Serving is ownership-checked and reads only the stored relative `storage_key`, resolved and
verified to stay inside `UPLOADS_DIR`. Image upload metadata is created during the chat
request that first references the image.

### Response shape for `GET /conversations/{id}`

```json
{
  "id": "…", "title": "…", "crop": "tomato",
  "created_at": "…", "updated_at": "…",
  "messages": [
    { "id": "…", "role": "user", "sequence": 1, "content": "…",
      "created_at": "…",
      "image": { "id": "…", "url": "/images/…", "mime_type": "image/png", "byte_size": 123 } },
    { "id": "…", "role": "assistant", "sequence": 2, "content": "…",
      "mode": "vision_and_grounded_rag",
      "vision_assessment": "…", "grounded_answer": "…",
      "sources": [ { "citation": 1, "filename": "…", "title": "…", "url": "…",
                     "institution": "…", "evidence_level": "full_page",
                     "chunk_id": "…" } ],
      "knowledge_graph": [ { "subject": "…", "relation": "…", "object": "…" } ],
      "created_at": "…" }
  ]
}
```

The `sources` entry matches the shape `_frontend_sources()` already emits, so the existing
`SourceItem` renderer works on history with no changes.

---

## 4. Frontend integration

### New: `src/lib/session.ts`

- `ensureSession()`: read `localStorage["cropsense.session"]`; if absent, `POST /sessions`
  and store the token. Memoised in a module-level promise so concurrent callers share one
  bootstrap.
- `getSessionToken()` / `clearSession()`.

### `src/lib/api.ts`

- All requests send `X-Session-Token` (when one exists) via a shared `authHeaders()`
  helper.
- Add: `listConversations()`, `getConversation(id)`, `renameConversation(id, title)`,
  `deleteConversation(id)`.
- `sendChatMessage(message, imageFile?, conversationId?)` adds `conversation_id` to the
  form and returns the extended `ChatApiResponse` (optional `conversation_id`,
  `user_message_id`, `assistant_message_id`).

### `page.tsx` (owner of conversation identity)

- State: `activeConversationId: string | null`, `conversations: Conversation[]`,
  `resetSignal: number`.
- On mount: `ensureSession()` → `listConversations()`.
- `handleSelect(id)`: fetch `getConversation(id)`, hold as `initialMessages`, set
  `activeConversationId`, bump `resetSignal`.
- `handleNewChat()`: `activeConversationId = null`, `initialMessages = []`, bump
  `resetSignal` (existing key-remount behaviour preserved).
- `handleConversationCreated(id)`: called by `ChatView` after the first persisted
  response; sets `activeConversationId` and refetches the sidebar list.

### `ChatView.tsx`

- Accept `initialMessages?: ChatMessage[]` and
  `onConversationCreated?: (id: string) => void`.
- Send `conversationId` with each message; after a response carrying `conversation_id`,
  notify the parent (once).
- Prevent concurrent submissions for the same active conversation: disable the send
  action (and ignore Enter) while a request is in flight, then re-enable it once the
  request completes. This keeps at most one in-flight message per `ChatView`; the backend
  row-lock sequence allocation (§2) remains the correctness guarantee.
- Everything else (loading state, error banner, source rendering) unchanged.

### `Sidebar.tsx`

- Replace `SAMPLE_CONVERSATIONS` with the fetched list; props become `conversations`,
  `onSelect`, `onDelete`, `onRename`, `isLoading`.
- States: loading skeleton, empty ("No saved conversations yet"), list.
- Per-item actions on hover/focus: rename (inline) and delete (confirm dialog).
- Remove the "Sample conversations / Demo entries" block once real data lands.
- Keep the existing visual language and the mobile drawer behaviour.

### Data flow

```
page.tsx  ──list──▶  Sidebar          (select / delete / rename)
   │                    │
   │  activeConversationId, initialMessages
   ▼                    ▼
ChatView ──sendChatMessage(conversationId)──▶ POST /chat ──▶ Postgres
   │                                             │
   └──onConversationCreated(id)──────────────────┘
```

---

## 5. Local development and migrations

### Docker Compose

`docker-compose.yml` at repo root: `postgres:16`, named volume, `.env`-driven password,
port `5432`. Compose reads `POSTGRES_PASSWORD` from `.env`; secrets are not committed.

Environment keys (documented in `.env.example`, actual values in `.env`):
`DATABASE_URL`, `UPLOADS_DIR`, `SESSION_TTL_DAYS` (default 30), `MAX_UPLOAD_BYTES`
(default 10485760).

### Alembic

- `alembic init -t async backend/migrations`; `env.py` reads `DATABASE_URL` and imports the
  metadata from `db.py`, so `--autogenerate` sees every model.
- One initial revision (`0001_initial`) creating all six tables + indexes + constraints.
- Migrations run explicitly with `alembic upgrade head`; the app does not auto-create
  tables at import or startup.

---

## 6. Security and validation

Keep only the checks needed for reliable operation:

- **No secrets in the database.** Gemini keys stay in `.env`.
- **Session tokens:** 256-bit random; only SHA-256 stored; expired tokens → 401.
- **Ownership:** every conversation and image query is session-scoped; foreign rows
  return 404.
- **Path traversal:** filenames are server-generated UUIDs; `storage_key` is always
  relative. Before read/delete, resolve the final path with `pathlib` and enforce
  containment explicitly:

  ```python
  target = (UPLOADS_DIR / storage_key).resolve()   # resolves symlinks
  if not target.is_relative_to(UPLOADS_DIR.resolve()):
      raise ValueError("storage_key escapes UPLOADS_DIR")
  ```

  `Path.relative_to()` (or `is_relative_to()`, Python 3.9+) rejects `../` traversal and
  symlinks that resolve outside the uploads directory. Never compare raw string prefixes.
- **Upload validation:** MIME allowlist (`image/png`, `image/jpeg`, `image/webp`) plus the
  existing 10 MB cap enforced in `_read_image`.
- **No raw SDK text** in `error_category` — reuse the existing sanitizers in
  `generator.py`/`vision.py`.
- **Error hygiene:** DB errors surface as generic 500s; never echo SQL or driver messages.
- **Deletion:** deleting a conversation hard-deletes messages, sources and KG edges via
  cascade, deletes `images` rows, then unlinks the related files (best effort). A crash
  between the two steps can leave an unused file under `backend/data/uploads/`; that is
  acceptable for this project and can be cleaned manually.
- **CORS:** keep the `localhost:3000`/`127.0.0.1:3000` allowlist; `allow_headers` already
  permits `X-Session-Token`.

---

## 7. Implementation task sequence

Small, practical tasks. Dependencies in brackets.

1. **[—]** Install database dependencies (`sqlalchemy[asyncio]`, `asyncpg`, `alembic`,
   `greenlet`; dev: `pytest`, `pytest-asyncio`) and add `docker-compose.yml` for Postgres
   16. Add the uploads directory to `.gitignore` and the env keys to `.env.example`.
2. **[1]** `backend/settings.py` and `backend/db.py` — typed env loading, async engine,
   `async_sessionmaker`, `Base` and a `get_session()` dependency.
3. **[2]** `backend/models.py` — the six models with constraints, indexes and
   relationships.
4. **[3]** Alembic init and `0001_initial`; verify `upgrade head` / `downgrade base`
   round-trips on a scratch database.
5. **[3,4]** `backend/sessions.py` — token issue/hash/validate with 30-day expiry,
   `POST /sessions`, `GET /sessions/me` and the `X-Session-Token` dependency.
6. **[2,5]** `backend/storage.py` — write (temp file + atomic rename), read, delete, sha256,
   MIME validation and path containment.
7. **[3,5,6]** `backend/repository.py` — create conversation (with title/crop), append user
   message, append assistant message + sources + kg, list/get/rename/delete (all
   session-scoped, permanent delete removes image files).
8. **[5,6,7]** `/chat` integration — optional token + `conversation_id`, persist before and
   after generation, additive response fields. **Existing RAG pipeline untouched.**
9. **[7,8]** Conversation and image endpoints (`GET /conversations`,
   `GET /conversations/{id}`, `PATCH`, `DELETE`, `GET /images/{id}`).
10. **[9]** Frontend `lib/session.ts` + `api.ts` additions.
11. **[10]** `page.tsx` conversation state, `ChatView` `initialMessages`/callback,
    `Sidebar` real data + rename/delete.
12. **[8-11]** Test the complete application and fix issues (§8).
13. **[12]** Update project setup documentation (Compose, migrations, env vars).

---

## 8. Testing strategy

Run against a dedicated Postgres test database (`cropsense_test`) created per session and
migrated with Alembic. The Gemini SDK is always mocked — **no live API calls in tests.**

**Database and migrations**
- `alembic upgrade head` creates all six tables; `downgrade base` removes them.

**Sessions**
- Token generation/hash-only storage; `GET /sessions/me` validates; expired token → 401.

**Repository**
- Create conversation with first message; `sequence` increments correctly.
- Message ordering under sequential appends; `updated_at` changes when a message is added.
- Assistant message with N sources + M KG edges round-trips in insertion order.
- Cascade: deleting a conversation removes messages, sources and KG edges.
- Session scoping: another session cannot read/rename/delete the conversation (404).

**API (httpx AsyncClient)**
- `POST /chat` **without** a token → same five response keys, no rows written.
- `POST /chat` **with** a token → rows created; response carries `conversation_id` and
  message ids; a second turn appends to the same conversation.
- Generation failure → assistant message persisted with `mode` + `error_category`, no raw
  SDK text.
- `GET /conversations` ordering; `GET /conversations/{id}` message ordering and source
  shape.
- `DELETE` → 204, rows gone, image file gone; a second DELETE → 404.
- `GET /images/{id}`: owner 200 with correct `Content-Type`; other session 404.

**Frontend**
- `npx tsc --noEmit`, `npm run lint`, `npm run build`.
- Manual checklist: reload keeps history; new chat starts empty; select loads messages +
  sources; rename/delete update the list; image renders from `/images/{id}`.

**Regression**
- The existing grounded-RAG behaviour must be unchanged without a session token — assert
  the same five response keys and the same citation/disclaimer rules.

---

## 9. Risks and trade-offs

| Risk | Impact | Mitigation |
|---|---|---|
| Postgres required locally | Dev friction | `docker-compose.yml` + documented one-liner |
| `localStorage` token readable by XSS | Hijack of anonymous, low-value history | Accepted for this project |
| Disk growth from 10 MB images | Storage exhaustion | Files are deleted with their conversation |
| Row + file not atomic | Orphaned file after a crash | Documented; clean `uploads/` manually if needed |
| Two-step write | Crash leaves an unanswered user turn | User message committed first; history still renders |
| `content` vs split fields drift | Inconsistent history | All three written from one function |
| Denormalized chunk metadata | Stale after a KB rebuild | Intentional — history shows what was cited |
| Adding `conversation_id` to `/chat` | Could disturb the working RAG path | Persistence is strictly additive; stateless path is the baseline |

## Out of scope

Authentication and user accounts, rate limiting, storage quotas, background cleanup
schedulers, distributed locking, multi-server deployment, cloud object storage,
httpOnly-cookie migration, monitoring, structured LLM-extracted diagnosis labels,
multilingual content, cross-device sync, and sharing/export.
