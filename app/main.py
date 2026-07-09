from dotenv import load_dotenv
load_dotenv()  # must run before any module that constructs an OpenAI() client at import time

import asyncio
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.retriever import retrieve
from app.llm import generate_answer
from app.memory import init_db, add_message, get_history, save_lead_summary
from app.scoring import score_lead
app = FastAPI()
# actual web application object — this is what will run as your server

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# Allows requests from any website domain to reach this API 

init_db()
# Runs the database setup function once when the server starts, making sure the tables exist

STATIC_DIR = Path(__file__).parent / "static"

@app.get("/")
def index():
    # Serves the chat web page at the root URL
    return FileResponse(STATIC_DIR / "index.html")

class ChatRequest(BaseModel):
    session_id: str
    message: str
# FastAPI uses this to automatically validate incoming requests

# Tracks the last message timestamp per session, so we know if a newer message has arrived
last_message_time = {}
# Tracks the last time the user did ANYTHING (typed a key or sent a message) per session.
# Both /chat and /typing bump this, so an in-flight reply keeps waiting while the user types.
last_activity_time = {}

QUIET_SECONDS = 3        # reply only after this many seconds of no typing / no new messages
MAX_WAIT_SECONDS = 30    # safety cap so a reply can never wait forever


class TypingRequest(BaseModel):
    session_id: str


@app.post("/typing")
def typing(req: TypingRequest):
    # Frontend pings this on every keystroke. It just tells any waiting reply
    # "the user is still typing — don't answer yet."
    last_activity_time[req.session_id] = time.time()
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest):
    add_message(req.session_id, "user", req.message)

    # Record this message's arrival time and mark it as fresh activity
    this_message_time = time.time()
    last_message_time[req.session_id] = this_message_time
    last_activity_time[req.session_id] = this_message_time

    # Wait until the user has been quiet (not typing, no new message) for QUIET_SECONDS.
    # Every keystroke bumps last_activity_time, which keeps extending this wait.
    started = time.time()
    while True:
        await asyncio.sleep(0.4)

        # A newer message arrived — let that one respond to the whole burst instead.
        if last_message_time[req.session_id] != this_message_time:
            return {"answer": None, "skipped": True}

        quiet_for = time.time() - last_activity_time[req.session_id]
        if quiet_for >= QUIET_SECONDS:
            break  # user has stopped — safe to reply
        if time.time() - started >= MAX_WAIT_SECONDS:
            break  # safety cap reached — reply anyway

    # No newer message came in — safe to respond now.
    history = get_history(req.session_id, limit=10)

    # Gather the burst: every user message since the last assistant reply, so
    # rapid-fire messages like "hi" + "can I know more?" are answered as ONE question.
    burst = []
    prior_history = history
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] != "user":
            break
        burst.insert(0, history[i]["content"])
        prior_history = history[:i]
    combined_message = "\n".join(burst) if burst else req.message

    context_chunks = retrieve(combined_message, top_k=5)
    answer = generate_answer(combined_message, context_chunks, prior_history)

    add_message(req.session_id, "assistant", answer)

    summary = score_lead(get_history(req.session_id, limit=20))
    save_lead_summary(req.session_id, summary)

    return {"answer": answer, "lead_summary": summary}

#1. Save the user's message
#2. Pull up the recent conversation history
#3. Search the knowledge base for relevant facts
#4. Ask the AI to write an answer, using the facts + history
#5. Save the AI's reply
#6. Re-check the whole conversation to score how promising this lead is
#7. Save that score
#8. Send back the answer (and the score) to whoever asked