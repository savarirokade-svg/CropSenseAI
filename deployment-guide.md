# CropSenseAI — Deployment Guide

> This guide walks you through a fresh production deployment:
> **Render** (FastAPI backend) + **Vercel** (Next.js frontend) + **Render PostgreSQL** (database).
> No credentials are stored here. All secrets are set through each platform's dashboard.

---

## Prerequisites

| Tool | Minimum version | Purpose |
|---|---|---|
| Python | 3.11+ | Local Alembic migration run |
| Node.js | 18+ | Frontend build (Vercel handles this automatically) |
| Git | any | Push to GitHub |
| Render account | — | Backend + PostgreSQL |
| Vercel account | — | Frontend |
| Google AI Studio key | — | Gemini API (`GEMINI_API_KEY`) |

---

## Step 1 — Push the repository to GitHub

```bash
# From the repository root
git add .
git commit -m "feat: add deployment configuration and full RAG pipeline"
git remote add origin https://github.com/<your-org>/cropsenseai.git
git push -u origin main
```

---

## Step 2 — Create a Render PostgreSQL database

1. In the Render dashboard click **New → PostgreSQL**.
2. Choose a name (e.g. `cropsenseai-db`), region **Oregon**, plan **Free**.
3. Click **Create Database** and wait for it to become `Available`.
4. Copy the **External Database URL** (starts with `postgres://…`) — you will need it in Step 3.

> [!IMPORTANT]
> Use the **External Connection URL**, not the Internal one. The Internal URL only works from services inside the same Render private network; migrating from your local machine requires the External URL.

---

## Step 3 — Deploy the FastAPI backend on Render

1. In Render click **New → Web Service**.
2. Connect your GitHub repository.
3. Render reads `render.yaml` automatically and pre-fills:
   - **Root directory:** `backend`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Click **Environment** and add the following variables:

| Key | Value | Notes |
|---|---|---|
| `GEMINI_API_KEY` | `AIza…` | From [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `DATABASE_URL` | `postgres://user:pass@host/db` | External Connection URL from Step 2 |
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` | Set after Vercel deploy (Step 5) |
| `SESSION_TTL_DAYS` | `30` | Pre-filled by render.yaml |
| `MAX_UPLOAD_BYTES` | `10485760` | Pre-filled by render.yaml |

> [!NOTE]
> `DATABASE_URL` may start with `postgres://` or `postgresql://` — the backend normalises both to `postgresql+asyncpg://` automatically before passing it to SQLAlchemy.

> [!NOTE]
> `GEMINI_MODEL` and `GEMINI_EMBEDDING_MODEL` default to `gemini-3.5-flash-lite` and `gemini-embedding-2` respectively. Set them explicitly only if you need a different model.

5. Click **Create Web Service**. The first build installs dependencies and starts Uvicorn.
6. Wait for the service status to show **Live** and confirm:

```bash
curl https://<your-service>.onrender.com/health
# Expected: {"status":"ok"}
```

> [!WARNING]
> On Render's free plan the service **spins down after 15 minutes of inactivity** and takes ~30 seconds to wake on the next request. Upgrade to Starter ($7/mo) to avoid cold starts.

---

## Step 4 — Run Alembic migrations against the production database

Run migrations from your local machine using the **External Database URL** from Step 2.

```bash
cd backend

# Activate your local virtual environment
source .venv/bin/activate        # macOS / Linux
.\.venv\Scripts\Activate.ps1    # Windows PowerShell

# Point the backend at the production database for this shell only
$env:DATABASE_URL = "postgres://user:pass@host/db"   # PowerShell
# export DATABASE_URL="postgres://user:pass@host/db" # bash

# Apply all migrations
alembic upgrade head
```

Expected output ends with:

```
INFO  [alembic.runtime.migration] Running upgrade  -> 0001_initial, 0001_initial
```

> [!IMPORTANT]
> The migrations directory is `backend/migrations/` and `alembic.ini` is in `backend/`. Run `alembic` from the `backend/` directory, or pass `-c backend/alembic.ini` from the repo root.

> [!CAUTION]
> Never run `alembic downgrade` against the production database without a verified backup.

---

## Step 5 — Deploy the Next.js frontend on Vercel

1. Go to [vercel.com/new](https://vercel.com/new) and import your GitHub repository.
2. Set the **Root Directory** to `frontend`.
3. Framework preset will be detected as **Next.js** automatically.
4. Under **Environment Variables** add:

| Key | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://<your-service>.onrender.com` |

5. Click **Deploy**. Vercel builds and publishes the frontend.
6. Copy the Vercel deployment URL (e.g. `https://cropsenseai.vercel.app`).

---

## Step 6 — Update ALLOWED_ORIGINS on Render

1. Return to the Render backend service → **Environment** tab.
2. Set `ALLOWED_ORIGINS` to your Vercel URL:

```
ALLOWED_ORIGINS=https://cropsenseai.vercel.app
```

3. Render automatically redeploys the backend with the new CORS configuration.

---

## Step 7 — Test the deployed application

### Health check
```bash
curl https://<your-service>.onrender.com/health
# {"status":"ok"}
```

### Chat endpoint (text query)
```bash
curl -s -X POST https://<your-service>.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of tomato early blight?"}' | python -m json.tool
```

### Image upload endpoint
```bash
curl -s -X POST https://<your-service>.onrender.com/upload-image \
  -F "file=@/path/to/leaf.jpg" | python -m json.tool
```

### Frontend smoke test
Open your Vercel URL in a browser and send a test question from the chat UI.

---

## Persistent image uploads (optional)

The free plan does not support Render persistent disks. Uploaded images are stored in `backend/data/uploads/` and **will be lost on each restart**.

To retain uploads:
- **Option A:** Upgrade to Render Starter ($7/mo), uncomment the `disk:` stanza in `render.yaml`, and set `UPLOADS_DIR` to the mount path.
- **Option B:** Integrate an external object store (e.g. AWS S3, Cloudflare R2) and update `storage.py` accordingly.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` returns 500 | `DATABASE_URL` not set or unreachable | Check Render env vars; confirm Postgres is `Available` |
| `CORS` errors in browser | `ALLOWED_ORIGINS` missing or wrong URL | Update env var and redeploy |
| `GEMINI_API_KEY is not set` in logs | Key not added to Render environment | Add via Render dashboard → Environment tab |
| `alembic upgrade head` fails with auth error | Wrong `DATABASE_URL` in local shell | Re-copy External Connection URL from Render |
| Cold start delays | Free plan spin-down | Upgrade to Starter or use a cron ping |

---

## Environment variable reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Google AI API key |
| `DATABASE_URL` | ✅ | — | PostgreSQL connection URL |
| `ALLOWED_ORIGINS` | ✅ (prod) | `http://localhost:3000` | Comma-separated CORS origins |
| `GEMINI_MODEL` | ❌ | `gemini-3.5-flash-lite` | Generation model override |
| `GEMINI_EMBEDDING_MODEL` | ❌ | `gemini-embedding-2` | Embedding model override |
| `SESSION_TTL_DAYS` | ❌ | `30` | Anonymous session lifetime |
| `MAX_UPLOAD_BYTES` | ❌ | `10485760` (10 MB) | Maximum image upload size |
| `UPLOADS_DIR` | ❌ | `backend/data/uploads` | Absolute path to uploads directory |
