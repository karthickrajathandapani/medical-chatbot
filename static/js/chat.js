/**
 * chat.js
 * =======
 * PURPOSE (one file, one purpose):
 *   All browser-side behavior for the chat UI — sending messages,
 *   reading the streamed SSE response from /api/chat, and rendering
 *   assistant text + the "vitals strip" telemetry + source citations.
 */

const chatScroll = document.getElementById("chatScroll");
const messagesEl = document.getElementById("messages");
const form = document.getElementById("composerForm");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const userTemplate = document.getElementById("userTemplate");
const assistantTemplate = document.getElementById("assistantTemplate");

// Auto-grow the textarea as the user types (up to CSS max-height)
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
});

// Suggestion chips fill the input and submit
document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.q;
    form.requestSubmit();
  });
});

newChatBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  messagesEl.innerHTML = "";
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  appendUserMessage(question);
  input.value = "";
  input.style.height = "auto";

  const assistantEls = appendAssistantSkeleton();
  setLoading(true);

  try {
    await streamAnswer(question, assistantEls);
  } catch (err) {
    assistantEls.reportText.textContent =
      "Something went wrong reaching the server. Please try again.";
    console.error(err);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  sendBtn.classList.toggle("loading", isLoading);
  sendBtn.disabled = isLoading;
}

function appendUserMessage(text) {
  const node = userTemplate.content.cloneNode(true);
  node.querySelector(".user-bubble").textContent = text;
  messagesEl.appendChild(node);
  scrollToBottom();
}

function appendAssistantSkeleton() {
  const node = assistantTemplate.content.cloneNode(true);
  const wrapper = node.querySelector(".message");
  const reportText = node.querySelector(".report-text");
  const vitalsStrip = node.querySelector(".vitals-strip");
  const sourcesRow = node.querySelector(".sources-row");

  reportText.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';

  messagesEl.appendChild(node);
  scrollToBottom();

  return { wrapper, reportText, vitalsStrip, sourcesRow };
}

async function streamAnswer(question, els) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: question }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answerText = "";
  let firstToken = true;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop(); // keep any incomplete trailing event in buffer

    for (const rawEvent of events) {
      if (!rawEvent.startsWith("data: ")) continue;
      const payload = JSON.parse(rawEvent.slice(6));
      handleEvent(payload, els, () => {
        if (firstToken) {
          els.reportText.textContent = ""; // clear typing dots on first token
          firstToken = false;
        }
      });
      if (payload.type === "token") {
        answerText += payload.data;
        els.reportText.textContent = answerText;
        scrollToBottom();
      }
    }
  }
}

function handleEvent(payload, els, onFirstToken) {
  switch (payload.type) {
    case "token":
      onFirstToken();
      break;
    case "metrics":
      renderVitals(els.vitalsStrip, payload.data);
      break;
    case "sources":
      renderSources(els.sourcesRow, payload.data);
      break;
    case "done":
      scrollToBottom();
      break;
  }
}

function renderVitals(stripEl, metrics) {
  stripEl.hidden = false;
  const map = {
    retrieve: metrics.retrieve_ms,
    rerank: metrics.rerank_ms,
    score: metrics.top_score,
    generation: metrics.generation_ms,
  };
  for (const [key, value] of Object.entries(map)) {
    if (value === undefined) continue;
    const item = stripEl.querySelector(`[data-key="${key}"] b`);
    if (item) item.textContent = value;
  }
}

function renderSources(rowEl, sources) {
  if (!sources || sources.length === 0) return;
  rowEl.hidden = false;
  rowEl.innerHTML = "";
  sources.forEach((s) => {
    const tag = document.createElement("span");
    tag.className = "source-tag";
    tag.textContent = `${s.source_file} · p.${s.page_number}`;
    rowEl.appendChild(tag);
  });
}

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

// Enter to send, Shift+Enter for newline
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
