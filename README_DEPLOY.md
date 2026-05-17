# BRIO Deploy — Railway Deployment Guide

## Quick Start

### 1. Required Environment Variables (set in Railway dashboard)
| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key |
| `API_KEY` | Secret key for protecting `/chat` endpoint |

### 2. Optional Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase anon/service key |
| `BUSINESS_SYSTEM_PROMPT` | (built-in) | Custom system prompt for BRIO |
| `BRIO_MEMORY_WINDOW` | `10` | Number of past turns to include in context |
| `BRIO_TOP_K_DOCS` | `3` | Number of RAG docs to retrieve |
| `BRIO_MIN_CONFIDENCE` | `0.70` | Confidence threshold below which to escalate |
| `CONVERSATION_HISTORY_PATH` | system temp | Writable path for conversation log |

### 3. Deploy on Railway
1. Push this project to a GitHub repo.
2. Create a new Railway project → link the repo.
3. In **Variables**, add `GROQ_API_KEY` and `API_KEY`.
4. Railway auto-detects `Procfile` and runs:
   ```
   web: uvicorn src.app:app --host 0.0.0.0 --port $PORT
   ```
5. Hit **Deploy**.

### 4. Test after deployment
```bash
# Health check (no auth needed)
curl https://<your-railway-url>/health

# Chat endpoint
curl -X POST https://<your-railway-url>/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your API_KEY>" \
  -d '{"customer_message": "Hello, what services do you offer?"}'
```

## Local Development
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY and API_KEY
python -m uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

## Notes
- `conversation_history.json` is written to the system temp directory on Railway (not the source folder, which is read-only).
- The `/health` endpoint requires no authentication and is suitable for Railway's health checks.
