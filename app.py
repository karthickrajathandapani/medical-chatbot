"""
app.py
======
# PURPOSE (one file, one purpose):
#   The web server. Serves the chat UI (templates/index.html) and exposes
#   a single streaming endpoint (/api/chat) that the frontend calls.
#   Contains NO retrieval/LLM logic itself — everything is delegated to
#   src/rag_pipeline.py, keeping this file thin (one file, one purpose).
"""

import json
import uuid

from flask import Flask, Response, render_template, request, session

from config.config import settings, validate_keys
from src.memory import conversation_memory
from src.rag_pipeline import answer_stream
from src.sparse_encoder import sparse_encoder
from src.vector_store import vector_store

app = Flask(__name__)
app.secret_key = "medical-chatbot-session-key"  # only used for the anonymous session id cookie


def _bootstrap():
    """Validate keys and connect to already-ingested resources at startup."""
    validate_keys()
    vector_store.connect()
    if not sparse_encoder.load():
        raise RuntimeError(
            "BM25 params not found. Run `python ingest.py` first to ingest your PDF(s)."
        )


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Streams Server-Sent Events (SSE) back to the browser.
    Each event is one JSON object: {"type": ..., "data": ...}
    See src/rag_pipeline.answer_stream() for the exact event shapes.
    """
    payload = request.get_json(force=True)
    question = (payload.get("message") or "").strip()
    session_id = session.get("session_id", "anonymous")

    if not question:
        return {"error": "Empty message"}, 400

    def event_stream():
        for event in answer_stream(session_id, question):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clears this browser session's conversational memory."""
    session_id = session.get("session_id", "anonymous")
    conversation_memory.clear(session_id)
    return {"status": "cleared"}


if __name__ == "__main__":
    _bootstrap()
    app.run(host=settings.FLASK_HOST, port=settings.FLASK_PORT, debug=settings.FLASK_DEBUG, threaded=True)
