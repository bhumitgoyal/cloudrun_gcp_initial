# GoHappy Club — WhatsApp AI Support Bot

An intelligent WhatsApp chatbot for **GoHappy Club**, India's senior community platform for people aged 50+. The bot answers member queries about sessions, memberships, Happy Coins, trips, and more — powered by Vertex AI RAG and Gemini 2.5 Flash, backed by Firestore for conversation memory, and deployed fully serverless on Google App Engine Standard.

---

## Architecture

```
┌──────────────────┐
│  WhatsApp User   │
│  (Senior Citizen)│
└────────┬─────────┘
         │ sends message
         ▼
┌──────────────────────────────────┐
│  Meta WhatsApp Business API      │
│  POST /webhook                   │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  App Engine  (FastAPI)                                       │
│                                                              │
│  1. Parse & deduplicate incoming message                     │
│  2. Filter junk (links, social media, emojis, greetings)     │
│  3. Load conversation state from Firestore                   │
│  4. Rewrite user query (Gemini — handles Hinglish/typos)     │
│  5. Semantic cache check (in-memory cosine similarity)       │
│     ├── HIT → skip steps 6–7, serve cached answer            │
│     └── MISS → continue                                      │
│  6. Retrieve relevant knowledge chunks (Vertex AI RAG)       │
│  7. Generate response (Gemini — structured JSON output)      │
│  8. Store response in cache (skip if escalation)             │
│  9. Send reply via WhatsApp Business API                     │
│ 10. Persist turn to Firestore                                │
│ 11. Escalate to human admin if needed                        │
│ 12. Compress rolling summary when threshold is hit           │
│                                                              │
│  Endpoints:                                                  │
│    GET  /webhook            — Meta webhook verification      │
│    POST /webhook            — Incoming messages              │
│    GET  /health             — Health check                   │
│    GET  /cache/stats        — Cache hit/miss counters        │
│    POST /cache/invalidate   — Flush all cached entries       │
└──────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐      ┌──────────────┐     ┌──────────────┐
   │ Firestore │      │ Vertex AI    │     │ Gemini 2.5   │
   │ (Memory)  │      │ RAG Engine   │     │ Flash        │
   └──────────┘      └──────────────┘     └──────────────┘
```

**No external cache servers** — the semantic cache runs entirely in-process on App Engine. No Redis, no VMs, no extra infrastructure.

---

## GCP Services Required

This project runs entirely on serverless/managed GCP services. **No VMs, no Redis, no additional servers.**

| Service | Purpose | Required? |
|---------|---------|-----------|
| **App Engine** | Hosts the FastAPI app | ✅ Yes |
| **Vertex AI (Gemini 2.5 Flash)** | Query rewrite, answer generation, summary compression | ✅ Yes |
| **Vertex AI RAG Engine** | Knowledge retrieval from your document corpus | ✅ Yes |
| **Cloud Firestore** | Conversation memory (per-user state) | ✅ Yes |
| **Secret Manager** | Stores WhatsApp API tokens securely | ✅ Yes |
| **Cloud Build** | Builds Docker container on deploy | ✅ Yes |
| **Artifact Registry** | Stores the built Docker image | ✅ Yes |
| ~~Redis / Memorystore~~ | ~~Cache backend~~ | ❌ Not needed |
| ~~Compute Engine~~ | ~~Any VM~~ | ❌ Not needed |
| ~~VPC Connector~~ | ~~Network bridge~~ | ❌ Not needed |

---

## Project Structure

```
cloudrun_gcp_initial/
├── main.py                  # FastAPI app — webhook + cache + health endpoints
├── bot/
│   ├── __init__.py
│   ├── whatsapp.py          # WhatsApp Cloud API client (send/receive)
│   ├── rag.py               # Vertex AI RAG Engine retrieval wrapper
│   ├── memory.py            # Firestore-backed conversation state manager
│   ├── llm.py               # Gemini prompt engineering + JSON output parser
│   ├── pipeline.py          # Orchestrates the full message handling flow
│   ├── rag_cache.py         # Serverless in-memory semantic cache
│   └── message_filter.py    # Filters links, social media, emojis, greetings
├── .env                     # Environment variables (local dev only)
├── requirements.txt         # Python dependencies
├── app.yaml                 # App Engine configuration
├── DEPLOY.md                # Step-by-step GCP deployment guide
├── run_dev.sh               # Local dev launcher (cloudflared tunnel)
├── run.sh                   # Simple launcher with ngrok
├── Test/
│   ├── test_rag_cache.py        # Cache + filter tests
│   ├── test_cache_pipeline.py   # End-to-end pipeline integration tests
│   ├── test_pipeline.py         # Pipeline unit tests
│   ├── test_rag.py              # RAG retrieval test
│   ├── test_bad_queries.py      # Query rewriter test (Hinglish, typos, shortforms)
│   ├── test_send_receive.py     # End-to-end WhatsApp API test
│   ├── test_kb_insights.py      # KB Insights mock test
│   └── test_full_simulation.py  # Full conversation simulation test
```

---

## Modules

### `main.py` — Application Entry Point

FastAPI application with five endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/webhook` | `GET` | One-time Meta webhook verification (responds with `hub.challenge`) |
| `/webhook` | `POST` | Receives all incoming WhatsApp messages; responds `200` immediately and processes in a background task |
| `/health` | `GET` | Health check — returns `{"status": "ok"}` |
| `/cache/stats` | `GET` | Cache hit/miss counters, hit rate, and total cached entries |
| `/cache/invalidate` | `POST` | Flush all cached entries (admin use) |

On startup, initialises the `MessagePipeline` with all dependencies (WhatsApp client, RAG engine, Firestore memory, Gemini LLM, semantic cache).

---

### `bot/message_filter.py` — Message Filter

Filters out non-actionable messages **before** they enter the pipeline, saving Gemini and RAG tokens:

| Blocked | Examples |
|---------|----------|
| **Facebook links** | `https://www.facebook.com/share/...` |
| **Instagram links** | `https://www.instagram.com/reel/...` |
| **YouTube links** | `https://youtu.be/...` |
| **Random URLs** | Any `http://` or `https://` link with no question text |
| **Short links** | `bit.ly`, `goo.gl`, `t.co`, `tinyurl.com` |
| **Emoji-only** | `🙏🙏🙏`, `👍`, `😀😀` |
| **Greeting-only** | `Good morning 🌸`, `Namaste 🙏`, `Ram Ram`, `Jai Shri Krishna` |
| **Too short** | Single characters, empty messages |

**Passes through:** Real questions ("How do I join GoHappy Club?"), Hinglish queries ("membership lena hai"), and questions that include a link alongside real text.

---

### `bot/rag_cache.py` — Semantic Cache (In-Memory, Serverless)

Intercepts after query rewriting, before Vertex AI RAG retrieval. Caches responses for semantically identical queries.

**How it works:**
1. Embeds the strictly normalized query into a 384-dimensional vector using `all-MiniLM-L6-v2`
2. Cosine similarity search against all cached entries
3. If similarity ≥ 0.75 → **HIT** — return cached response, skip RAG + Gemini
4. If below threshold → **MISS** — run full pipeline, cache the result

**Key properties:**
- **No external dependencies** — runs entirely in Cloud Run process memory
- **Max 1,000 entries** (~2.5 MB total — negligible for 2 GiB Cloud Run)
- **TTL: 24 hours** — entries auto-expire so knowledge updates propagate
- **Escalation-safe** — responses with `escalation: true` are never cached
- **Lazy model loading** — embedding model loads on first cache use, not at startup

**Configuration** (env vars):

| Variable | Description | Default |
|----------|-------------|---------|
| `CACHE_SIMILARITY_THRESHOLD` | Cosine similarity for a cache hit | `0.75` |
| `CACHE_TTL_SECONDS` | Entry expiry in seconds | `86400` (24h) |
| `CACHE_MAX_ENTRIES` | Max cached entries | `1000` |

---

### `bot/whatsapp.py` — WhatsApp Cloud API Client

- **`IncomingMessage`** — dataclass representing a parsed incoming text message (message ID, sender phone, display name, phone number ID, text body).
- **`WhatsAppClient`** — handles:
  - `parse_message(payload)` — extracts text messages from Meta's nested webhook JSON. Returns `None` for status updates, reactions, images, videos, and non-text messages.
  - `send_text(to, body)` — sends a plain-text reply via the Graph API (`v19.0`).
- Uses `httpx.AsyncClient` for non-blocking HTTP.

---

### `bot/rag.py` — Vertex AI RAG Engine

- **`RAGEngine`** — retrieves the most relevant knowledge chunks from a pre-built Vertex AI RAG Corpus.
  - `query(query_text)` — calls `rag.retrieval_query()` with configurable `top_k` and `distance_threshold`. Returns a list of `RetrievedChunk` objects.
  - `format_for_prompt(chunks)` — serialises chunks into a labelled `[DOC_N]` block for LLM prompt injection.
- Degrades gracefully — if RAG fails, returns an empty list and the LLM answers from its system prompt alone.

**Configuration** (env vars):

| Variable | Description | Default |
|----------|-------------|---------|
| `VERTEX_RAG_CORPUS` | Full corpus resource name | Required |
| `RAG_TOP_K` | Number of chunks to retrieve | `8` |
| `RAG_DISTANCE_THRESHOLD` | Max vector distance | `0.5` |

---

### `bot/memory.py` — Firestore Conversation Memory

- **`ConversationMemory`** — per-user conversation state stored in a `conversations` collection.
  - `get_state(phone)` — returns the user's full state, or a clean default for new users.
  - `append_turn(phone, display_name, user_text, bot_text)` — atomically appends a user+bot turn pair using `ArrayUnion` and `Increment`. Creates the document if it doesn't exist.
  - `update_summary(phone, new_summary)` — replaces the rolling summary and trims the turn buffer.
  - `set_escalation_status(phone, status)` — toggles the human handoff pause flag.
  - `should_summarise(state)` — returns `True` when a compression cycle is due (every `SUMMARISE_EVERY` turns).
  - `build_customer_summary(state)` / `format_history_for_prompt(state)` — format state into prompt-ready strings.

**Configuration** (env vars):

| Variable | Description | Default |
|----------|-------------|---------|
| `FIRESTORE_DB` | Firestore database ID | `(default)` |
| `MAX_RECENT_TURNS` | Max raw turns kept in ring buffer | `10` |
| `SUMMARISE_EVERY` | Compress summary after N turns | `6` |

---

### `bot/llm.py` — Gemini LLM Wrapper

- **`GeminiChat`** — manages three Gemini model instances:

  | Instance | Purpose |
  |----------|---------|
  | `self.model` | Main chat — generates customer support replies with structured JSON output |
  | `self.rewrite_model` | Canonical Normalizer — rewrites broken English and Hinglish into strict, identical standard English queries (e.g., 'membership kaise lu' → 'How do I join GoHappy Club?'). This fuels the 70%+ hit rate of the semantic cache. |
  | `self.summary_model` | Conversation compressor — generates rolling summaries |

- **System Prompt** — a detailed, production-grade prompt baked into the module as `SYSTEM_PROMPT`. Includes:
  - Role & identity (GoHappy Club support agent)
  - Full company context (plans, pricing, policies, contact info)
  - Structured response instructions (understand → evaluate → answer / reject / escalate)
  - Tone rules (warm, concise, English-only, no filler, max 1 emoji)
  - Strict Guardrails & Anti-Hallucination (SILENT REJECT on trivia, ESCALATE gracefully if answers are missing without making things up)
  - Strict JSON output schema: `{"answer": "...", "escalation": true/false}`

- **Output Parsing** — enforces JSON output via Gemini's `response_mime_type="application/json"` and a response schema. Falls back to best-effort text extraction if parsing fails.

- **Language Policy** — the bot always replies in **English only**, regardless of the user's input language.

---

### `bot/pipeline.py` — Message Pipeline Orchestrator

The core orchestration layer. Handles the full lifecycle of an incoming message:

```
Incoming Webhook
      │
      ▼
Parse message (skip non-text, images, status updates)
      │
      ▼
Admin override check (/resolve <PHONE>)
      │
      ▼
Deduplication (in-memory set, last 500 IDs)
      │
      ▼
Message filter (block links, social media, emojis, greetings)
      │
      ▼
Load conversation state from Firestore
      │
      ▼
Check escalation pause flag → skip if escalated
      │
      ▼
Rewrite/polish user query (Gemini)
      │
      ▼
Semantic cache check → serve cached response on HIT
      │
      ▼ (MISS only)
RAG retrieval (Vertex AI)
      │
      ▼
Build prompt → call Gemini → get JSON response
      │
      ▼
Cache the response (skip if escalation)
      │
      ▼
Send reply to user (WhatsApp API)
      │
      ▼
Persist turn to Firestore
      │
      ▼
Handle escalation if flagged
      │
      ▼
Trigger rolling summary compression if due (async)
```

**Key features:**

- **Message Filtering** — blocks links, social media URLs, emoji-only, and greeting-only messages before they consume any Gemini or RAG tokens.
- **Semantic Caching** — identical and semantically similar queries return cached responses in <50ms, skipping RAG and Gemini entirely. Saves ~60-70% of per-message cost on hits.
- **Deduplication** — Meta may deliver the same webhook multiple times. An in-memory set of the last 500 message IDs prevents duplicate processing.
- **Admin Override** — the `ADMIN_PHONE_NUMBER` can send `/resolve <PHONE>` to unpause a user after a human support interaction.
- **Escalation** — when Gemini flags `escalation: true`, the bot pauses itself for that user, sends the user an escalation message, and alerts the admin via WhatsApp with full context.
- **Background Summary Compression** — when the turn count hits the threshold, kicks off a Gemini call to compress recent turns into a rolling summary (runs as an `asyncio` task, doesn't block the reply).

---

## Firestore Data Model

```
conversations/                         # Collection
  └── <phone_number>/                  # Document (e.g. "919876543210")
        display_name:       str        # "Ramesh Kumar"
        summary:            str        # AI-generated rolling summary
        turn_count:         int        # Total turns in this session
        last_seen:          timestamp  # Last interaction time (UTC)
        escalated_to_human: bool       # Human handoff pause flag
        recent_turns:       list       # Ring buffer of recent turns
          [
            { role: "user",      content: "...", ts: <timestamp> },
            { role: "assistant", content: "...", ts: <timestamp> },
            ...
          ]
```

---

## Environment Variables

Create a `.env` file for local development:

```env
# GCP
GCP_PROJECT_ID=ghc-chatbot
GCP_LOCATION=asia-south1
VERTEX_RAG_CORPUS=projects/ghc-chatbot/locations/asia-south1/ragCorpora/<CORPUS_ID>
GEMINI_MODEL=gemini-2.5-flash

# RAG Tuning
RAG_TOP_K=4
RAG_DISTANCE_THRESHOLD=0.5

# WhatsApp Cloud API (from Meta Business Suite)
WHATSAPP_ACCESS_TOKEN=<your-access-token>
WHATSAPP_PHONE_NUMBER_ID=<your-phone-number-id>
WHATSAPP_BUSINESS_ACCOUNT_ID=<your-business-account-id>
WHATSAPP_VERIFY_TOKEN=<your-webhook-verify-token>

# Admin Routing
ADMIN_PHONE_NUMBER=<admin-phone-in-E164-format>

# Semantic Cache (in-memory, serverless)
CACHE_SIMILARITY_THRESHOLD=0.92
CACHE_TTL_SECONDS=86400
CACHE_MAX_ENTRIES=1000

# Optional
PORT=8080
FIRESTORE_DB=(default)
MAX_RECENT_TURNS=10
SUMMARISE_EVERY=6
```

> **⚠️ Never commit `.env` to version control.** For production, use GCP Secret Manager (see [DEPLOY.md](DEPLOY.md)).

---

## Getting Started

### Prerequisites

- Python 3.11+
- GCP project with billing enabled
- APIs enabled: **Vertex AI**, **Firestore**, **Cloud Run**, **Cloud Build**, **Secret Manager**
- Meta Developer account with a **WhatsApp Business App**
- A **Vertex AI RAG Corpus** created and populated with your knowledge base documents
- A **Firestore (Native mode)** database created in your GCP project
- `gcloud` CLI authenticated (`gcloud auth login`)
- `cloudflared` installed for local tunnelling (`brew install cloudflared`)

### Local Development

```bash
# 1. Clone the repo
git clone <repo-url> && cd cloudrun_gcp_initial

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up .env
cp .env.example .env   # or create from the template above
# Fill in all values

# 4. Run with the dev launcher (starts server + Cloudflare tunnel)
chmod +x run_dev.sh && ./run_dev.sh
```

The dev script will:
1. Load `.env` variables
2. Run pre-flight checks (Python, cloudflared, required env vars)
3. Install pip dependencies
4. Start the FastAPI server on port 8080
5. Start a Cloudflare Quick Tunnel (no account needed)
6. Print the public webhook URL to paste into Meta Developer Console

### Register the Webhook

1. Go to [developers.facebook.com](https://developers.facebook.com) → your App → WhatsApp → Configuration
2. Set **Callback URL** to the tunnel URL printed by `run_dev.sh` (e.g. `https://xxx.trycloudflare.com/webhook`)
3. Set **Verify Token** to the value of `WHATSAPP_VERIFY_TOKEN` in your `.env`
4. Subscribe to the **messages** webhook field

---

## Production Deployment (App Engine)

See [DEPLOY.md](DEPLOY.md) for the full step-by-step guide covering:

1. Creating the Vertex AI RAG Corpus
2. Setting up Firestore
3. Setting local config values in env_variables.yaml
4. Creating a service account with IAM roles
5. Building and deploying the app configuration to App Engine
6. Registering the WhatsApp webhook

### Quick Deploy

```bash
gcloud app deploy app.yaml --project=ghc-chatbot --quiet
```

> **Memory**: 2 GiB is required because the `sentence-transformers` embedding model (`all-MiniLM-L6-v2`) uses ~500 MB at runtime. The model is pre-downloaded during Docker build — no internet download on first request.

---

## Escalation & Rejection Flow

When the bot processes a message, it enforces strict guardrails to prevent hallucinations and spam:

### 1. Out-of-Context Responses (Rejections)
If the user asks random trivia (e.g. "what are the top 10 schools in India?") or for medical/financial advice, the bot **rejects** the message. It politely states it can only help with GoHappy Club and explicitly **avoids** alerting humans (escalation: false).

### 2. Standard Escalations (Missing context or account issues)
When the query *is* related to GoHappy Club, but the bot cannot answer accurately (e.g. missing policies), or the user asks for a human:

```
1. Gemini returns { "escalation": true }
             │
             ▼
2. Bot sends an escalation message to the user
   ("Let me connect you with our team...")
             │
             ▼
3. Bot sets escalated_to_human = true in Firestore
   → All further messages from this user are silently ignored by the bot
             │
             ▼
4. Admin receives a WhatsApp alert:
   🚨 ESCALATION REQUIRED
   User: <name> (<phone>)
   Issue: <their message>
   Bot replied: <what the bot said>
             │
             ▼
5. Admin replies to the user directly from their WhatsApp app
             │
             ▼
6. When resolved, admin sends:
   /resolve <PHONE_NUMBER>
   → Bot resumes handling messages for that user
```

---

## Automated KB Insights

The bot includes an automated insight generator to help you continuously improve the knowledge base. It reads recent conversations logged in the Audit Spreadsheet to find missing information that caused failures or escalations.

### How to Trigger:
1. From the designated admin WhatsApp number (`+919818646823`), send the message: **`insights`**
2. The bot will automatically:
   - Identify all unprocessed queries in your Audit Google Sheet.
   - Send the recent difficult queries along with your existing system prompt and context to Gemini.
   - Generate actionable recommendations of exactly what paragraphs or facts to add to your knowledge base to prevent those hallucinations or escalations.
   - Save these insights into a new **"KB Insights"** tab in your Google Spreadsheet.
   - Mark the processed rows as "viewed" so they aren't analyzed twice.
3. The bot will reply to you on WhatsApp with a success message once the analysis is complete.

---

## Test Suite

| Script | Tests | Purpose |
|--------|-------|---------|
| `test_rag_cache.py` | 17 | In-memory cache (hit/miss/semantic/TTL/escalation) + message filter (links/emoji/greetings) |
| `test_cache_pipeline.py` | 18 | End-to-end pipeline: proves RAG + Gemini are SKIPPED on cache HIT |
| `test_pipeline.py` | — | Pipeline unit tests — simulates webhook payloads locally |
| `test_rag.py` | — | Tests RAG retrieval quality against the knowledge corpus |
| `test_bad_queries.py` | — | Tests the query rewriter with broken English, Hinglish, and shortforms |
| `test_full_simulation.py` | — | Full conversation simulation — multi-turn flows, escalation, dedup, summary compression |
| `test_send_receive.py` | — | End-to-end test — sends real messages via WhatsApp API |

```bash
# Run cache + filter tests
python Test/test_rag_cache.py

# Run pipeline integration tests (no GCP creds needed)
python Test/test_cache_pipeline.py

# Run the full simulation
python Test/test_full_simulation.py
```

---

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Runtime** | Python | 3.11 |
| **Web Framework** | FastAPI + Uvicorn | 0.111.0 / 0.30.1 |
| **LLM** | Google Gemini 2.5 Flash (via Vertex AI) | SDK 1.71.1 |
| **Knowledge Retrieval** | Vertex AI RAG Engine | SDK 1.71.1 |
| **Conversation Memory** | Google Cloud Firestore (Native mode) | SDK 2.16.0 |
| **Messaging** | WhatsApp Business Cloud API (Meta) | Graph API v19.0 |
| **Semantic Cache** | sentence-transformers + numpy (in-memory) | ≥3.0.0 / ≥1.26.0 |
| **Embedding Model** | all-MiniLM-L6-v2 (384-dim, Apache 2.0) | — |
| **HTTP Client** | httpx (async) | 0.27.0 |
| **Deployment** | GCP App Engine Standard (F2 Instance) | — |
| **Secrets** | GCP Secret Manager | — |
| **Local Tunnelling** | Cloudflare Quick Tunnel (`cloudflared`) | — |

---

## Key Design Decisions

1. **Background processing** — webhook POSTs return `200` immediately. The actual pipeline runs as a FastAPI `BackgroundTask` to avoid Meta timeout retries.

2. **Message filtering** — seniors frequently forward Facebook posts, YouTube videos, "Good morning" images, and random links. The filter blocks these at the gate, saving Gemini and RAG tokens.

3. **In-memory semantic cache** — no external Redis or Memorystore needed. The cache runs inside Cloud Run process memory (~2.5 MB for 1,000 entries). On cache HIT, response time drops from 2-4s to <50ms and RAG+Gemini calls are skipped entirely. With `--min-instances 1`, the cache stays warm across requests.

4. **Query rewriting** — senior users often type in Hinglish, broken English, or shortforms. A lightweight Gemini call polishes the query before RAG retrieval for much better chunk matching.

5. **Strict JSON output** — the main Gemini call uses `response_mime_type="application/json"` with a schema to guarantee parseable `{answer, escalation}` output. A regex fallback handles rare edge cases.

6. **Rolling summary compression** — instead of passing the full conversation history to every LLM call (which would hit context limits and increase cost), the bot compresses older turns into a summary every N turns. Only the summary + last 10 raw turns are sent.

7. **In-process deduplication** — Meta's webhook delivery can fire multiple times for the same message. A simple in-memory set (capped at 500 entries) prevents duplicate processing.

8. **Graceful degradation** — if RAG fails, the bot still answers from its system prompt. If Gemini fails, it returns a polite error with a phone number. If Firestore fails, the error is logged but the webhook still returns 200. If the cache fails, the pipeline runs normally without it.

9. **English-only replies** — the bot understands messages in any language (Hindi, Hinglish, Tamil, etc.) but always replies in simple, clear English to maintain consistency.

10. **Pre-downloaded model** — the `all-MiniLM-L6-v2` embedding model (80 MB) is downloaded during Docker build, not at runtime. This eliminates cold-start network downloads.

11. **Anti-Hallucination Guardrails** — The prompt explicitly forces the bot to reject unrelated trivia and completely forbids answering medical or financial inquiries.

---

## License

Private — GoHappy Club internal project.
