# CropSenseAI

CropSenseAI is an educational crop-disease assistant for **tomato** and **potato**.
You ask a question (optionally with a photo of a plant), and the application answers
from a small, cited agricultural reference library, plus a separate visual assessment
of the image produced by Google Gemini.

The project is a college MVP: a FastAPI backend, a Next.js frontend, and a PostgreSQL
database that stores anonymous conversations, messages, citations and uploaded images.

**Supported crops: tomato and potato only.** Questions about other crops are politely
declined rather than answered from unrelated material.

---

## 1. Project overview

### What it does

- Answers tomato/potato questions using hybrid retrieval over a local reference
  library (semantic FAISS search + BM25 keyword search, fused by reciprocal rank
  fusion) and generates a grounded answer with numbered citations.
- Analyses an uploaded plant photo with Gemini Vision and shows that assessment
  separately from the reference-grounded text.
- Shows knowledge-graph relationships (e.g. *Tomato — may_show_symptom — Yellow leaves*)
  for the crop and symptom in the question.
- Saves each exchange to PostgreSQL so the sidebar can list, reopen, rename and
  delete past conversations, including their uploaded images.

### Technology stack

| Layer | Technology |
| --- | --- |
| Backend | Python 3.13, FastAPI, Uvicorn, SQLAlchemy 2 (async) + asyncpg, Alembic |
| AI | Google Gemini (generation, vision, query embeddings) via `google-genai` |
| Retrieval | FAISS (`faiss-cpu`), BM25, reciprocal rank fusion |
| Database | PostgreSQL 16, run with Docker Compose |
| Frontend | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS |
| Images | Files on disk under `backend/data/uploads/`, metadata in PostgreSQL |

The approved design document is [`docs/cropsenseai-database-architecture.md`](docs/cropsenseai-database-architecture.md).

---

## 2. Prerequisites

Install these before starting:

| Software | Version used by this project | Notes |
| --- | --- | --- |
| Python | **3.13** (verified with 3.13.14) | `python --version` |
| Node.js + npm | **Node 22** (verified with v22.13.1), npm 10.9.2 | `node --version`, `npm --version` |
| Docker Desktop | Docker 29.9.2 with Compose v2 (`docker compose`) | Must be running before starting PostgreSQL |
| Google Gemini API key | — | Free tier available at <https://aistudio.google.com/app/apikey> |

The Gemini key is needed for **three** things: answer generation, image analysis and
**query embeddings for retrieval**. Without it the backend still starts, but chat
answers fall back to an "unavailable" message.

---

## 3. Project structure

```
CropSenseAI/
├── docker-compose.yml          # PostgreSQL 16 container (host port 5433)
├── docs/
│   └── cropsenseai-database-architecture.md
├── backend/
│   ├── main.py                 # FastAPI app: /chat, /sessions, /conversations, /images
│   ├── settings.py             # reads backend/.env into a typed settings object
│   ├── db.py                   # async SQLAlchemy engine, session factory, dependency
│   ├── models.py               # the six ORM models
│   ├── sessions.py             # anonymous session token issue/validate
│   ├── repository.py           # conversation/message/image database operations
│   ├── storage.py              # safe image file storage (path containment)
│   ├── vision.py               # Gemini Vision (image analysis)
│   ├── generator.py            # grounded answer generation with citations
│   ├── hybrid_retrieval.py     # FAISS + BM25 retrieval and rank fusion
│   ├── vector_store.py         # loads the FAISS index and chunk metadata
│   ├── embeddings.py           # Gemini query embeddings
│   ├── knowledge_graph.py      # small crop/symptom relationship table
│   ├── index_builder.py        # rebuilds the vector store from the guide
│   ├── alembic.ini             # Alembic configuration
│   ├── migrations/             # migrations (single revision: 0001_initial)
│   ├── requirements.txt        # pinned Python dependencies (incl. pytest)
│   ├── pytest.ini              # test configuration (testpaths = tests)
│   ├── tests/                  # backend integration tests
│   ├── .env.example            # environment template (copy to .env)
│   └── data/
│       ├── agricultural_guide.md        # knowledge base — not included here
│       ├── tomato_potato_guide.txt      # legacy sample text (not used for answers)
│       ├── vector_store/                # generated FAISS index — not included here
│       └── uploads/                     # uploaded images (created at runtime)
└── frontend/
    ├── package.json            # scripts: dev, build, start, lint
    ├── .env.local.example      # frontend environment template
    └── src/
        ├── app/                # page.tsx (state owner) and layout
        ├── components/         # ChatView, Sidebar, MessageList, MessageComposer, …
        ├── lib/                # api.ts (HTTP client), session.ts (token), mockData.ts
        └── types/chat.ts       # shared TypeScript types
```

---

## 4. Environment configuration

Both `.env.example` files are templates. **Copy them, then edit your copy.** Never
commit a real `.env` file; both are already git-ignored.

### Backend — `backend/.env`

```powershell
Copy-Item backend\.env.example backend\.env
```

Then open `backend\.env` and fill in your own values.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | — | Gemini key for generation, vision and retrieval embeddings |
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Model name used for generation and vision |
| `POSTGRES_HOST` | No | `127.0.0.1` | Database host |
| `POSTGRES_PORT` | No | `5433` | Host port published by Docker Compose |
| `POSTGRES_USER` | Yes | — | Database user (also used by Docker Compose) |
| `POSTGRES_PASSWORD` | Yes | — | Database password (also used by Docker Compose) |
| `POSTGRES_DB` | Yes | — | Database name (recommended: `cropsense`) |
| `DATABASE_URL` | No | built from the `POSTGRES_*` values | Full SQLAlchemy URL; if set it wins over the `POSTGRES_*` values. `postgresql://` is converted to `postgresql+asyncpg://` automatically |
| `UPLOADS_DIR` | No | `backend/data/uploads` | Where uploaded images are stored |
| `SESSION_TTL_DAYS` | No | `30` | Anonymous session lifetime |
| `MAX_UPLOAD_BYTES` | No | `10485760` | Maximum upload size (10 MB) |

`POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB` are read by Docker Compose
**and** by the backend, so keep them consistent.

### Frontend — `frontend/.env.local`

Optional. Only needed if your backend is not at `http://127.0.0.1:8000`.

```powershell
Copy-Item frontend\.env.local.example frontend\.env.local
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI backend, read at build/start time |

---

## 5. Installation and setup

Run these commands in **Windows PowerShell**, in order. Start at the project root —
the folder that contains this `README.md`, `backend\`, `frontend\` and
`docker-compose.yml`.

### 5.1 Go to the project root

```powershell
# Replace the placeholder with the folder you cloned/downloaded into.
cd C:\path\to\CropSenseAI
```

### 5.2 Backend dependencies

```powershell
cd backend
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If you prefer not to activate the environment, prefix every command with
> `.\.venv\Scripts\python.exe -m …` instead (for example
> `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`).

> `Set-ExecutionPolicy -Scope Process` only affects the current PowerShell window and
> is needed once per window because Windows blocks activation scripts by default.

### 5.3 Configure the environment

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in `GEMINI_API_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB`.
Leave `POSTGRES_PORT=5433` unless you changed the Compose port.

### 5.4 Start PostgreSQL

Docker Desktop must be running. From the **project root**:

```powershell
cd ..
docker compose --env-file backend/.env up -d
docker compose --env-file backend/.env ps
```

The container is named `cropsenseai-postgres` and should become `healthy`.
`--env-file backend/.env` is required: Compose reads `POSTGRES_USER` and
`POSTGRES_PASSWORD` from that file.

### 5.5 Run the database migrations

The application never creates tables by itself — migrations do.

```powershell
cd backend
alembic upgrade head
alembic current
```

`alembic current` should print `0001_initial (head)`.

### 5.6 Build the retrieval index

**This repository does not ship the retrieval artifacts.** Without them the backend
still runs, but chat answers fall back to "I couldn't find enough relevant
information…". Two inputs are absent on purpose:

- `backend/data/agricultural_guide.md` — the knowledge base. It is a compilation of
  extracted third-party source pages, so it is not redistributed here. Supply your own
  appropriately licensed tomato/potato reference material in the format
  `index_builder.py` expects: a `## Part 1` heading, one `###` section per disease, and
  a bold source block per section (`Source-Title`, `Source-URL`,
  `Source-Institution`, `Source-Access-Date`, `Source-Access-Status`).
- `backend/data/vector_store/` — the generated index (`index.faiss`, `chunks.json`,
  `manifest.json`). It is derived from that guide and is git-ignored.

The builder defaults to a **dry run** — no network calls, no files written — so
validate your guide before spending any embedding quota:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python index_builder.py
```

Read the report. It exits `0` only when every chunk has complete source metadata, a
unique ID, non-empty text and a known evidence level. Then build for real, which embeds
one text per chunk with Gemini and therefore needs a valid `GEMINI_API_KEY`:

```powershell
python index_builder.py --build
```

> `vector_store.py` will still report the store as unavailable after a rebuild: it
> compares three SHA-256 fingerprints against the `EXPECTED_*` constants pinned to the
> build the project was developed against. A new index fails closed by design. After an
> intentional rebuild, inspect the values and update those constants by hand:
>
> ```powershell
> python vector_store.py --print-actual-fingerprints
> ```
>
> That command only reports fingerprints; it writes nothing and does not approve a
> build. Change the constants only once you are sure the new index is the one you want
> to serve.

### 5.7 Frontend dependencies

Open a **second** PowerShell window:

```powershell
cd C:\path\to\CropSenseAI\frontend
npm install
```

Optionally configure the API URL:

```powershell
Copy-Item .env.local.example .env.local
```

---

## 6. Running the application

The backend and the frontend each need their **own terminal**, and both must stay
running while you use the app.

### Terminal 1 — backend (http://127.0.0.1:8000)

```powershell
cd C:\path\to\CropSenseAI\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

| URL | Purpose |
| --- | --- |
| <http://127.0.0.1:8000/health> | Health check — should return `{"status":"healthy",…}` |
| <http://127.0.0.1:8000/docs> | Interactive API documentation |
| <http://127.0.0.1:8000/> | Simple welcome message |

Uvicorn must be started **from the `backend` folder**, because `main.py` imports its
sibling modules by plain name (`import repository`, `from db import …`).

### Terminal 2 — frontend (http://localhost:3000)

```powershell
cd C:\path\to\CropSenseAI\frontend
npm run dev
```

Open <http://localhost:3000>. The chat header shows a backend status badge; if it says
"Backend offline", check Terminal 1 and the troubleshooting section below.

CORS is restricted to `http://localhost:3000` and `http://127.0.0.1:3000`, so open the
app on one of those two origins.

---

## 7. Database information

| Item | Value |
| --- | --- |
| Engine | PostgreSQL **16** (`postgres:16` image) |
| Container | `cropsenseai-postgres` (Docker Compose service `postgres`) |
| Volume | `cropsense_pgdata` (named volume — holds your data) |
| Host port | **5433** (`127.0.0.1:5433`) |
| Container port | **5432** |
| Database name | value of `POSTGRES_DB` in `backend/.env` (recommended: `cropsense`) |
| Image storage | files under `backend/data/uploads/`; rows in `images` |

### Tables

| Table | Purpose |
| --- | --- |
| `sessions` | Anonymous sessions; stores only the SHA-256 hash of the client token and its expiry |
| `conversations` | One row per conversation: title, resolved crop, created/updated timestamps |
| `messages` | Stored turns in order (`role`, `sequence`, `content`, mode, vision assessment, grounded answer, sanitized error category) |
| `images` | Metadata for uploaded images: relative storage key, MIME type, size, SHA-256 |
| `message_sources` | The numbered citations attached to one assistant message |
| `message_kg_edges` | The knowledge-graph relationships attached to one assistant message |

Rows are session-scoped: a conversation or image belonging to another session is
reported as "not found".

### Migrations

There is a single revision, `0001_initial`, which creates all six tables.

```powershell
cd backend
.\.venv\Scripts\Activate.ps1

alembic upgrade head    # apply migrations
alembic current         # show the revision in the database
alembic history         # list available revisions
```

> **Do not delete the Docker volume (`docker volume rm …`) or drop the database to
> "fix" a problem.** That permanently deletes every saved conversation and image.
> Use the commands above to inspect and repair the schema instead.

---

## 8. Application features

- **Anonymous sessions.** On first load the frontend calls `POST /sessions` and stores
  the returned token in `localStorage` under the key `cropsense.session`. Every
  history request sends it as the `X-Session-Token` header. There are no accounts or
  passwords; the token expires after `SESSION_TTL_DAYS` (30 by default) and the client
  creates a new session automatically once if the server rejects it.
- **Conversation history.** The sidebar lists this browser's saved conversations,
  newest first, with their title, relative update time, message count and an
  "Image analyzed" marker when a turn included a vision assessment.
- **Sending text and images.** Type a question, attach a JPEG/PNG/WebP up to 10 MB, or
  do both. The reply contains the cited, reference-grounded answer and, when an image
  was analysed, a separate "Image assessment" section.
- **Saved image retrieval.** Images uploaded in a saved conversation can be shown again
  later. They are fetched with the session token (`GET /images/{id}`) and displayed
  from a temporary in-memory URL, so the token never appears in an image URL.
- **Rename and permanent deletion.** Conversations can be renamed inline in the
  sidebar, and deleted after a confirmation dialog. Deletion is permanent: messages,
  citations, graph edges and any images that are no longer referenced are removed.
- **Source citations and knowledge graph.** Answers list the cited documents (title,
  institution, evidence level, link) and any knowledge-graph relationships that were
  recorded with the reply.
- **Graceful fallbacks.** Unsupported crops, ambiguous crop questions, an unavailable
  reference index or a failed Gemini call all produce a clear message instead of an
  invented answer.

---

## 9. Troubleshooting

### Docker Compose complains about missing environment variables

`docker-compose.yml` requires `POSTGRES_USER` and `POSTGRES_PASSWORD` and stops with
`set POSTGRES_USER in backend/.env` when they are missing. Point Compose at the
backend env file:

```powershell
docker compose --env-file backend/.env up -d
```

Check that both variables exist and are not empty in `backend\.env`.

### The PostgreSQL container is not running

```powershell
docker compose --env-file backend/.env ps
docker compose --env-file backend/.env up -d
docker compose --env-file backend/.env logs postgres
```

If Docker Desktop itself is not running, start it and wait until it reports that the
engine is ready.

### The database tables are missing

The app does not create tables at startup. Apply the migrations:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
alembic upgrade head
alembic current
```

### Alembic migration problems

- `alembic` must be run from the `backend` folder (that is where `alembic.ini` lives).
- It reads the database URL from `backend/.env`. Verify `POSTGRES_*` (or `DATABASE_URL`)
  are correct, then retry `alembic current`.
- `Can't locate revision` usually means the database was migrated by a different
  version of the project; run `alembic history` to compare.

### Port 3000, 8000 or 5433 is already in use

Find the owning process and stop it:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
Stop-Process -Id <PID>
```

Use the same pattern with `-LocalPort 3000` for the frontend. To run the backend on a
different port instead, start it with `uvicorn main:app --reload --port 8001` and set
`NEXT_PUBLIC_API_URL=http://127.0.0.1:8001` in `frontend\.env.local` (restart the
frontend afterwards).

### The frontend says "Backend offline"

1. Check <http://127.0.0.1:8000/health> in a browser.
2. Make sure the backend terminal is still running and shows no traceback.
3. Open the app at `http://localhost:3000` (CORS only allows that and
   `http://127.0.0.1:3000`).
4. If the backend runs on another port, update `NEXT_PUBLIC_API_URL` and restart the
   frontend — the value is read when the dev server starts.

### Gemini API configuration errors

- `GEMINI_API_KEY is not set` — add the key to `backend\.env`; the backend must be
  restarted after editing it.
- Invalid key, quota or model errors appear as a generic fallback message in the chat.
  Confirm the key in Google AI Studio and check `GEMINI_MODEL` in `backend\.env`.
- Without a valid key, retrieval embeddings also fail, so answers become
  "I couldn't find enough relevant information…" rather than real citations.

### Invalid or expired anonymous session

A rejected token normally repairs itself: the client discards it, creates one new
session and retries the request once. If the UI still fails, clear the stored token in
the browser (Developer Tools → Application → Local Storage → `cropsense.session`) and
reload the page; a fresh session is created automatically.

---

## 10. Limitations and safety

- This is an **educational college MVP**, intended to demonstrate the retrieval,
  vision and persistence pipeline — not production infrastructure.
- It is **not a certified agricultural diagnosis system**. Visual assessments are
  observations from a single photo, not confirmed diagnoses.
- Reference coverage is deliberately small: only **tomato** and **potato** are served,
  from a limited set of agricultural sources. Other crops are declined.
- Gemini API calls (generation, vision and embeddings) may incur **quota limits or
  costs** on your Google account.
- Always check recommendations against reliable local agricultural guidance or an
  extension officer before acting on them.

---

## 11. Testing (optional)

### Backend integration tests

The tests need a **separate** database and the Gemini pipeline is mocked — no live API
calls are made. Never point the tests at your application database: the test suite
refuses to run unless the target database is named `cropsense_test`.

Create and migrate that database once (using a PostgreSQL superuser, here via the
Compose container):

```powershell
docker exec -it cropsenseai-postgres psql -U cropsense -d postgres -c "CREATE DATABASE cropsense_test;"

Replace cropsense with the value of POSTGRES_USER configured in backend/.env, if different.
```

Then run the migrations for the test database and the suite:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1

$env:DATABASE_URL = "postgresql+asyncpg://<user>:<password>@127.0.0.1:5433/cropsense_test"
alembic upgrade head

$env:TEST_DATABASE_URL = $env:DATABASE_URL
Remove-Item Env:DATABASE_URL
python -m pytest
```

The test fixtures clean up their own rows and image files, so the test database is
empty again when the suite finishes.

### Frontend checks

```powershell
cd frontend
npx tsc --noEmit
npm run lint
npm run build
```
