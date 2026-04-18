# GoHappy Club — WhatsApp Chatbot on GCP Cloud Run

## Architecture

```
WhatsApp User
     │
     ▼
Meta WhatsApp Business API
     │  POST /webhook
     ▼
Cloud Run (FastAPI)
     │
     ├─► Message Filter  ──► Blocks links, spam, greetings
     │
     ├─► Semantic Cache (in-memory)  ──► HIT? Skip RAG + Gemini
     │
     ├─► Vertex AI RAG Engine (.query)  ──► Ranked Knowledge Chunks
     │
     ├─► Firestore  ──► Conversation History + Rolling Summary
     │
     ├─► Gemini 2.5 Flash  ──► Structured JSON { answer, escalation }
     │
     └─► WhatsApp Business API  ──► Reply to User
```

---

## Prerequisites

- GCP project with billing enabled
- APIs enabled: Vertex AI, Firestore, Cloud Run, Cloud Build, Secret Manager
- Meta Developer account with WhatsApp Business App
- A Vertex AI RAG Corpus already created and populated

---

## Step 1 — Create Vertex AI RAG Corpus

```bash
# Create corpus
gcloud ai rag-corpora create \
  --display-name="gohappy-knowledge-base" \
  --location=asia-south1 \
  --project=YOUR_PROJECT_ID

# Note the corpus name from output:
# projects/YOUR_PROJECT_ID/locations/us-central1/ragCorpora/CORPUS_ID

# Import your documents (PDF, GCS, Drive, etc.)
gcloud ai rag-corpora import-rag-files \
  --rag-corpus=CORPUS_ID \
  --location=asia-south1 \
  --gcs-uris=gs://your-bucket/gohappy-docs/
```

---

## Step 2 — Set up Firestore

```bash
# Create Firestore database (Native mode)
gcloud firestore databases create \
  --location=asia-south1 \
  --project=YOUR_PROJECT_ID
```

The `conversations` collection is created automatically on first message.

---

## Step 3 — Store secrets in Secret Manager

```bash
PROJECT=your-gcp-project-id

# Store each secret
echo -n "EAAxxxxxxx" | gcloud secrets create WHATSAPP_ACCESS_TOKEN \
  --data-file=- --project=$PROJECT

echo -n "123456789" | gcloud secrets create WHATSAPP_PHONE_NUMBER_ID \
  --data-file=- --project=$PROJECT

echo -n "my-verify-token" | gcloud secrets create WHATSAPP_VERIFY_TOKEN \
  --data-file=- --project=$PROJECT
```

---

## Step 4 — Create Service Account

```bash
gcloud iam service-accounts create gohappy-bot \
  --display-name="GoHappy Bot SA" \
  --project=$PROJECT

SA=gohappy-bot@${PROJECT}.iam.gserviceaccount.com

# Grant required roles
for ROLE in \
  roles/aiplatform.user \
  roles/datastore.user \
  roles/secretmanager.secretAccessor \
  roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:${SA}" \
    --role="$ROLE"
done
```

---

## Step 5 — Deploy to Cloud Run

```bash
# Build and push container
gcloud builds submit \
  --tag gcr.io/$PROJECT/gohappy-bot \
  --project=$PROJECT

# Deploy
gcloud run deploy gohappy-bot \
  --image gcr.io/$PROJECT/gohappy-bot \
  --platform managed \
  --region asia-south1 \
  --service-account gohappy-bot@${PROJECT}.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "GCP_PROJECT_ID=$PROJECT,GCP_LOCATION=asia-south1,GEMINI_MODEL=gemini-2.5-flash,VERTEX_RAG_CORPUS=projects/$PROJECT/locations/asia-south1/ragCorpora/CORPUS_ID,CACHE_SIMILARITY_THRESHOLD=0.75,CACHE_TTL_SECONDS=86400,CACHE_MAX_ENTRIES=1000" \
  --set-secrets "WHATSAPP_ACCESS_TOKEN=WHATSAPP_ACCESS_TOKEN:latest,WHATSAPP_PHONE_NUMBER_ID=WHATSAPP_PHONE_NUMBER_ID:latest,WHATSAPP_VERIFY_TOKEN=WHATSAPP_VERIFY_TOKEN:latest" \
  --project=$PROJECT

# Allow public access to webhook endpoint ONLY
gcloud run services add-iam-policy-binding gohappy-bot \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --region=asia-south1 \
  --project=$PROJECT
```

> **Note**: For production, use Cloud Armor or validate the `X-Hub-Signature-256` header instead of making the whole service public.

---

## Step 6 — Register WhatsApp Webhook

1. Go to [developers.facebook.com](https://developers.facebook.com) → your App → WhatsApp → Configuration
2. Set **Webhook URL**: `https://YOUR_CLOUD_RUN_URL/webhook`
3. Set **Verify Token**: same value as your `WHATSAPP_VERIFY_TOKEN` secret
4. Subscribe to the **messages** webhook field

---

## Step 7 — Verify it works

```bash
# Check health
curl https://YOUR_CLOUD_RUN_URL/health

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=gohappy-bot" \
  --limit=50 --format="table(timestamp,textPayload)" \
  --project=$PROJECT
```

---

## Firestore Data Model

```
conversations/
  └── <phone_number>/          # e.g. "919876543210"
        display_name:  "Ramesh Kumar"
        summary:       "Member on Silver plan. Asked about Happy Coins..."
        turn_count:    14
        last_seen:     2024-01-15T10:30:00Z
        recent_turns:  [
          { role: "user",      content: "...", ts: ... },
          { role: "assistant", content: "...", ts: ... },
          ...  (last 10 turns kept verbatim)
        ]
```

---

## Escalation Handling

When `escalation: true` is returned by Gemini, the pipeline currently:
1. Logs a `WARNING` level entry to Cloud Logging (searchable in GCP Log Explorer)
2. Sends the escalation reply to the user

**To route escalations to a human agent**, extend `pipeline._handle_escalation()`:

```python
# Option A: Pub/Sub → any CRM
await pubsub_client.publish("gohappy-escalations", json.dumps({
    "phone": msg.from_number,
    "name":  msg.display_name,
    "query": msg.text,
}))

# Option B: POST to your internal ticketing system
await httpx_client.post("https://your-crm.com/tickets", json={...})

# Option C: Send WhatsApp message to support staff number
await self.wa.send_text(to="91XXXXXXXXXX", body=f"Escalation from {msg.from_number}: {msg.text}")
```

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill env vars
cp .env.example .env

# Load env and run
set -a && source .env && set +a
python main.py

# Expose locally with ngrok for webhook testing
ngrok http 8080
# Use the ngrok URL as your webhook in Meta Developer Console
```
