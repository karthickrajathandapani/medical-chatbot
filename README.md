# 🩺 MedInquire — Grounded Medical RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot that answers
medical questions using **only** the content of a medical reference PDF
(tested with the *Gale Encyclopedia of Medicine*, 636 pages), with a
clinical, lab-report-themed chat UI.

```
Question -> Hybrid Search (dense + BM25) -> Cross-Encoder Rerank
         -> Memory-aware Prompt -> OpenAI (streamed) -> Cited Answer
```

---

## ✨ What's inside

| Requirement                     | Implementation |
|----------------------------------|----------------|
| Best PDF loader                  | **PyMuPDF (fitz)** — fast, layout-accurate, page-cited |
| Best chunking strategy           | **Recursive character splitting**, medical-aware separators |
| Best embedding model              | **`pritamdeka/S-PubMedBert-MS-MARCO`** (Sentence-Transformers, biomedical) |
| Vector database                  | **Pinecone** (serverless, `dotproduct` metric) |
| LLM                               | **OpenAI `gpt-4o-mini`** (streamed) |
| Best retrieval method             | **Hybrid search** — dense + BM25 sparse, alpha-blended |
| Reranker                          | **Cross-encoder** (`ms-marco-MiniLM-L-6-v2`) |
| Conversational memory             | **Buffer memory** — last N turns, per session |
| One file, one purpose             | See [Architecture](#-architecture) below |
| Separate API keys                 | All secrets live only in `.env` (see `config/config.py`) |
| `#` explanations                  | Every module is heavily commented with **why**, not just what |
| Full end-to-end README            | You're reading it |
| World-class UI                    | Clinical "lab-report" themed chat, live retrieval telemetry |
| Response speed                    | Streaming tokens, cached query embeddings, fast reranker/model |

---

## 🗂 Architecture

```
medical-chatbot/
├── .env.example          # template for all API keys/settings (copy -> .env)
├── requirements.txt
├── ingest.py             # ONE-TIME script: PDF -> chunks -> vectors -> Pinecone
├── app.py                # Flask server: serves UI + /api/chat streaming endpoint
├── config/
│   └── config.py         # single source of truth for all settings & keys
├── src/
│   ├── pdf_loader.py      # PDF -> clean, page-tagged text        (PyMuPDF)
│   ├── chunking.py        # text -> overlapping semantic chunks   (recursive splitter)
│   ├── embeddings.py      # text -> dense vectors                 (Sentence-Transformers)
│   ├── sparse_encoder.py  # text -> sparse vectors                (BM25)
│   ├── vector_store.py    # ALL Pinecone I/O                      (hybrid upsert + query)
│   ├── reranker.py        # candidates -> re-scored, re-ordered   (cross-encoder)
│   ├── memory.py          # per-session conversation buffer
│   ├── llm.py             # ALL OpenAI calls                      (grounded prompt + streaming)
│   └── rag_pipeline.py    # orchestrates all of the above, end to end
├── templates/
│   └── index.html         # chat UI markup
├── static/
│   ├── css/style.css      # design system (see "UI design" below)
│   └── js/chat.js         # SSE streaming client, vitals-strip rendering
└── data/
    └── Medical_book.pdf   # your source PDF(s) go here
```

**Rule of thumb followed throughout:** `app.py` never imports from `src/*`
except `rag_pipeline.py`. Every other file does exactly one job and is
imported only by the pipeline that needs it.

---

## 🔀 How hybrid retrieval + reranking works here

1. **Dense embedding** (`embeddings.py`): the question is embedded with a
   PubMedBERT-based Sentence-Transformer — captures *meaning* ("high blood
   pressure" ≈ "hypertension").
2. **Sparse embedding** (`sparse_encoder.py`): the same question is BM25-encoded
   — captures *exact terms* (drug names, dosages, "Type 2" vs "Type 1").
3. **Hybrid query** (`vector_store.py`): both vectors are alpha-weighted and
   sent to Pinecone in a single `dotproduct` query, returning the top ~20
   candidates that score well on *either* signal.
4. **Rerank** (`reranker.py`): a cross-encoder reads `(question, chunk)`
   pairs together (not separately, like the bi-encoder above) and keeps only
   the best 4 — trading a little latency for much higher precision.
5. **Grounded generation** (`llm.py`): the top 4 chunks + last 6 conversation
   turns are placed into a strict system prompt that forbids answering
   outside the provided context, and the answer streams back token-by-token.

Tune the balance in `.env`:
```
HYBRID_ALPHA=0.6      # 1.0 = pure semantic, 0.0 = pure keyword
TOP_K_RETRIEVE=20     # candidates pulled from Pinecone
TOP_K_RERANK=4         # candidates kept after cross-encoder rerank
```

---

## 🎨 UI design

The interface is deliberately **not** a generic chat-bubble template. It's
styled like a clinical lab report / apothecary ledger:

- **Sidebar** (deep pine ink `#16302B`) shows the live pipeline stages and a
  standing safety disclaimer.
- **Chat panel** (sage-paper `#EDEFE7`) renders assistant answers as
  "report cards" — a thin teal rule on the left, like an excerpted clinical
  note — with amber, monospace **source tags** (`Medical_book.pdf · p.42`)
  underneath every answer.
- **Vitals strip** — the signature element. Under every answer, a
  monospace telemetry readout (`retrieve 42ms · rerank 18ms · top score
  0.91 · generate 640ms`) shows the *real* pipeline metrics for that
  message, like a lab print-out. This makes the RAG pipeline transparent
  instead of a black box.
- Type system: **Source Serif 4** for headings (encyclopedia voice),
  **Inter** for body/UI, **JetBrains Mono** for all data/telemetry/citations.

---

## ⚡ Speed optimizations (requirement #14)

- **Streaming** OpenAI responses (`llm.py`) — first tokens appear almost
  instantly instead of waiting for the full answer.
- **`gpt-4o-mini`** — fast, low-latency model tuned for RAG synthesis.
- **Small cross-encoder** (`ms-marco-MiniLM-L-6-v2`) — reranking 20
  candidates costs well under 100ms on CPU.
- **LRU-cached query embeddings** (`embeddings.py`) — repeated/similar
  questions in a session skip re-embedding.
- **Normalized embeddings at ingest time** — avoids extra normalization
  math on every query.
- **Batch upserts** at ingestion (100 vectors/request) to avoid per-item
  network overhead.

---

## 🚀 Setup guide (end-to-end)

### 1. Prerequisites
- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Pinecone API key](https://app.pinecone.io) (free tier is enough to start)

### 2. Clone / unzip the project, then install dependencies
```bash
cd medical-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure your API keys
```bash
cp .env.example .env
```
Open `.env` and fill in:
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...
```
All other settings in `.env` have sensible defaults — change them later if
you want to experiment (chunk size, hybrid alpha, model names, etc).

### 4. Add your medical PDF(s)
Put one or more PDFs into the `data/` folder. This project ships with
`data/Medical_book.pdf` (Gale Encyclopedia of Medicine) already in place —
swap in your own PDF(s) if you'd like.

### 5. Run ingestion (one time, or whenever your PDFs change)
```bash
python ingest.py
```
This will:
- Load and clean every PDF in `data/`
- Chunk the text
- Fit the BM25 sparse encoder over your corpus
- Embed every chunk with the biomedical Sentence-Transformer
- Create the Pinecone index (if it doesn't exist) and upsert everything

You'll see progress + timing for each of the 5 stages. For the included
636-page book, expect ingestion to take a few minutes (mostly spent on
embedding — CPU-only machines will be slower than GPU).

### 6. Launch the chatbot
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

### 7. Ask away
Try the suggested chips, or ask your own question. Every answer shows:
- The streamed, cited answer
- A "vitals strip" of real retrieval/generation timings
- Source page tags for every chunk actually used

---

## ☁️ Deploying for free

Want a public URL instead of just `localhost`? See **[DEPLOYMENT.md](DEPLOYMENT.md)**
for a complete, beginner-friendly, step-by-step guide to deploying this
project for free on Hugging Face Spaces (Docker) — no prior deployment
experience needed.

---

## 🔧 Troubleshooting

| Symptom | Fix |
|---|---|
| `EnvironmentError: Missing required environment variable(s)` | You haven't filled in `.env` — see step 3. |
| `BM25 params not found. Run ingest.py first.` | Run `python ingest.py` before `python app.py`. |
| Ingestion is slow | Normal on CPU for large PDFs — the embedding step dominates. Consider a smaller PDF for testing, or run on a machine with a GPU (sentence-transformers will use it automatically). |
| `No extractable text found` | Your PDF is likely scanned/image-only — you'd need OCR (e.g. `pytesseract`) first; this pipeline expects text-based PDFs. |
| Pinecone dimension mismatch error | If you change `EMBEDDING_MODEL` to one with a different output size, also update `EMBEDDING_DIM` in `.env` **and** delete/recreate the Pinecone index. |
| `ValueError: ...we now require users to upgrade torch to at least v2.6...` | Your installed `torch` is older than what a recent `transformers` version requires (security fix for CVE-2025-32434). Run `pip install --upgrade -r requirements.txt` — this project's `requirements.txt` already pins `torch==2.6.0`, which fixes it. |
| `torch` install is huge / very slow | You likely don't need GPU support for this project (CPU is fine for these model sizes). Install the smaller CPU-only build instead: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt` again (pip will skip torch since it's already installed). |

---

## 🧪 Extending this project

- **Swap the reranker for higher quality:** set `RERANKER_MODEL=BAAI/bge-reranker-base` in `.env` (slower, more accurate).
- **Persist memory across restarts:** replace the in-memory dict in `src/memory.py` with Redis.
- **Add authentication:** wrap `app.py` routes with your preferred auth (Flask-Login, etc).
- **Multi-PDF corpora:** just drop more PDFs into `data/` and re-run `ingest.py` — `source_file` metadata keeps citations distinct per book.

---

## ⚕️ Disclaimer

This chatbot provides general medical information sourced from a reference
text. It is **not** a substitute for professional medical advice, diagnosis,
or treatment. Always consult a qualified healthcare provider with questions
about a medical condition.
