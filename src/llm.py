"""
llm.py
======
# PURPOSE (one file, one purpose):
#   Own ALL calls to OpenAI (requirement #5). Builds the grounded RAG
#   prompt (retrieved context + chat history + question) and streams
#   the answer back token-by-token.

# WHY STREAMING (requirement #14 — improve response speed):
#   Streaming doesn't reduce total generation time, but it slashes
#   PERCEIVED latency to near-zero — the user sees the first words
#   appear almost immediately instead of waiting for the full answer.
#   Combined with gpt-4o-mini (fast + cheap) this keeps the chat feeling
#   instant even though a full RAG pipeline runs underneath.
"""

from typing import Dict, Generator, List

from openai import OpenAI

from config.config import settings

_client = OpenAI(api_key=settings.OPENAI_API_KEY)


SYSTEM_PROMPT = """You are MedInquire, a careful medical-information assistant.

Rules you MUST follow:
1. Answer ONLY using the CONTEXT provided below. If the context doesn't
   contain the answer, say so plainly — do not guess or use outside knowledge.
2. Always cite the source page(s) you used, like: (Source: p. 42).
3. Keep language clear and plain-English; briefly define technical terms.
4. End every answer with this exact disclaimer on its own line:
   "⚕️ This is general medical information, not a diagnosis. Please consult a licensed healthcare professional for personal medical advice."
5. If the question describes a medical emergency (e.g. chest pain, severe
   bleeding, difficulty breathing, suicidal ideation), tell the user to
   contact local emergency services immediately, before anything else.
"""


def _build_context_block(chunks: List[Dict]) -> str:
    """Turn reranked chunks into a numbered context block for the prompt."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (Source: {c['source_file']}, p. {c['page_number']})\n{c['text']}")
    return "\n\n".join(lines)


def stream_answer(
    question: str,
    context_chunks: List[Dict],
    history_messages: List[Dict[str, str]],
) -> Generator[str, None, None]:
    """
    Stream the LLM's answer as a sequence of text deltas.

    Args:
        question: the user's latest message.
        context_chunks: reranked chunks from src/reranker.py.
        history_messages: prior turns from src/memory.py (OpenAI format).

    Yields:
        str chunks of the answer, in order, as they're generated.
    """
    context_block = _build_context_block(context_chunks)

    user_content = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{question}"
    )

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history_messages
        + [{"role": "user", "content": user_content}]
    )

    stream = _client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        messages=messages,
        stream=True,  # <-- key speed/UX lever, see module docstring
    )

    for event in stream:
        delta = event.choices[0].delta
        if delta and delta.content:
            yield delta.content
