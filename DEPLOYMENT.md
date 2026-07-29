# 🚀 Deploying MedInquire for Free (Beginner's Guide)

> **Update:** This guide originally recommended Hugging Face Spaces. In
> late July 2026, Hugging Face changed their policy so that Docker and
> Gradio Spaces now require a paid PRO plan — only static (no backend)
> Spaces stay free. Since this project needs a real Python backend, we've
> switched this guide to **Render**, which still has a genuine free tier
> with no credit card required.

---

## Honest expectations before you start

Render's free web service tier gives you **512MB of RAM**. This project's
dependencies (`torch` + `sentence-transformers`) are on the heavier side
for that limit — it may run absolutely fine, or it may crash with an
out-of-memory error under load. This guide's Step 6 tells you exactly what
that failure looks like and what to do about it (a small code tweak, or a
$7/month upgrade if you want guaranteed headroom). We'd rather tell you
this upfront than have you discover it after 20 minutes of setup.

Also: like every free tier, Render's free web services **sleep after 15
minutes of no traffic** and take 30-60 seconds to wake back up on the next
visit. Normal, not a bug.

---

## Overview of what you'll do

1. Run ingestion **on your own computer** (one time) so your PDF's data is
   in Pinecone and a small results file is ready to ship.
2. Push your project to a **GitHub repository** (Render deploys from Git).
3. Create a free Render account and connect that repo.
4. Add your API keys as **Environment Variables** (never in code).
5. Deploy and get a public URL.
6. If it crashes on memory — here's the fix.

---

## Step 1 — Run ingestion locally first

(Skip this if you've already done it and `data/bm25_params.json` exists.)

```bash
cd medical-chatbot
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`, add your real keys, then:
```bash
python ingest.py
```

This uploads your data to Pinecone (cloud — stays there) and creates
`data/bm25_params.json` on your computer, which you'll need to ship to
the server.

---

## Step 2 — Push your project to GitHub

If you don't already have a GitHub account, make one free at
**https://github.com/join**.

1. Go to **https://github.com/new**, create a new repository (e.g.
   `medical-chatbot`), keep it **Public** or **Private** — either works
   with Render.
2. **Do not upload `.env`** — it's already excluded by `.gitignore`.
3. **Do upload `data/bm25_params.json`** — you need this one.
4. From your project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/medical-chatbot.git
   git push -u origin main
   ```
   (If you've never used git before, GitHub's "quick setup" page shown
   right after creating the repo gives you these exact commands with your
   real URL already filled in — just copy them from there.)

---

## Step 3 — Create your Render account + web service

1. Go to **https://render.com** → **Get Started** → sign up free (you can
   use your GitHub account to sign in, which also makes Step 4 easier).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account if prompted, then select your
   `medical-chatbot` repository.
4. Fill in the settings:
   - **Name**: anything, e.g. `medinquire-chatbot`
   - **Region**: closest to you
   - **Branch**: `main`
   - **Runtime**: **Docker** (Render will detect your `Dockerfile` automatically)
   - **Instance Type**: **Free**
5. Don't click "Create Web Service" yet — first add your secrets (next step).

---

## Step 4 — Add your API keys as Environment Variables

Still on the same setup page, scroll to **Environment Variables** and add:

| Key | Value |
|---|---|
| `OPENAI_API_KEY` | your real OpenAI key |
| `PINECONE_API_KEY` | your real Pinecone key |

These become available to your app exactly like they were in your local
`.env` file — `config/config.py` picks them up automatically.

---

## Step 5 — Deploy

Click **Create Web Service**. Render will:
- Clone your repo
- Build your `Dockerfile` (installs torch, sentence-transformers, etc — takes 5-10 minutes the first time)
- Start your app and give you a URL like:
  ```
  https://medinquire-chatbot.onrender.com
  ```

Watch the **Logs** tab while it builds — that's where you'll see if
anything goes wrong.

---

## Step 6 — If it crashes with an out-of-memory error

You'll know this is what happened if the Logs show something like `Killed`
or the service repeatedly restarts right after loading the embedding
model. If that happens, try these in order:

1. **Cheapest fix — reduce memory at startup** by loading the model in a
   lower-memory mode. Open `src/embeddings.py` and change:
   ```python
   self._model = SentenceTransformer(model_name)
   ```
   to:
   ```python
   self._model = SentenceTransformer(model_name, device="cpu")
   self._model.max_seq_length = 256  # smaller than the default 512 — cuts memory use
   ```
   Commit and push — Render redeploys automatically.

2. **If it still crashes**, Render's cheapest paid tier (**Starter, $7/month**)
   gives you 512MB → 2GB RAM, which comfortably fits this project. Not
   free, but worth knowing it's a small, one-line upgrade rather than a
   redesign.

3. **Alternative that's still no-cost to try**: Google Cloud Run has a
   genuinely generous free quota (more RAM headroom than Render's free
   tier) and doesn't charge unless you exceed it — but it requires adding
   a credit card to your Google Cloud account during signup (won't be
   charged if you stay in the free tier). Ask if you'd like the full
   walkthrough for this option.

---

## Updating your chatbot later

- **Changed the PDF or want to re-ingest?** Run `python ingest.py`
  locally again, commit + push the new `data/bm25_params.json`.
- **Changed code?** Just `git push` — Render redeploys automatically on
  every push to `main`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails immediately | Check the Logs tab for the actual pip/docker error — usually a typo in `requirements.txt` or a missing file. |
| "BM25 params not found" at runtime | You forgot to commit/push `data/bm25_params.json`. Run `git add data/bm25_params.json`, commit, push. |
| App works locally but not on Render | Double check both environment variables are set exactly as `OPENAI_API_KEY` and `PINECONE_API_KEY` (case-sensitive) in Render's dashboard, not just your local `.env`. |
| Page takes 30-60s to load the first time | Normal — free tier sleeps after 15 min idle and wakes on the next visit. |
| Out of memory / app keeps restarting | See Step 6 above. |
