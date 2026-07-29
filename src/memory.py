"""
memory.py
=========
# PURPOSE (one file, one purpose):
#   Give each chat session a "Conversational Buffer Memory" (requirement
#   #8) — a rolling window of the last N (question, answer) turns — so
#   the LLM can resolve follow-ups like "what about in children?" that
#   only make sense with prior context.

# DESIGN NOTE:
#   This is a lightweight, dependency-free re-implementation of
#   LangChain's ConversationBufferMemory concept: a simple FIFO buffer
#   per session_id, capped at MEMORY_MAX_TURNS. In-memory dict is fine
#   for a single-process demo; swap `_store` for Redis in production
#   so memory survives restarts / works across multiple server workers.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from config.config import settings


@dataclass
class Turn:
    question: str
    answer: str


class ConversationBufferMemory:
    def __init__(self, max_turns: int = settings.MEMORY_MAX_TURNS):
        self.max_turns = max_turns
        # session_id -> deque of Turn, oldest evicted first once full
        self._store: Dict[str, Deque[Turn]] = defaultdict(lambda: deque(maxlen=self.max_turns))

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        self._store[session_id].append(Turn(question=question, answer=answer))

    def get_history(self, session_id: str) -> List[Turn]:
        return list(self._store[session_id])

    def as_openai_messages(self, session_id: str) -> List[Dict[str, str]]:
        """
        Format buffered turns as OpenAI chat "messages" so llm.py can
        splice them straight into the API call.
        """
        messages = []
        for turn in self._store[session_id]:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


# Singleton — shared across all requests in this process.
conversation_memory = ConversationBufferMemory()
