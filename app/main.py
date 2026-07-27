from dotenv import load_dotenv
load_dotenv()  # must run before any module that constructs an OpenAI() client at import time

import asyncio
import os
import random
import secrets
import time
from pathlib import Path
from fastapi import BackgroundTasks, Cookie, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.retriever import retrieve
# LLM_IMPL=min swaps in app/llm_min.py — a much shorter, bare-minimum rewrite kept alongside
# app/llm.py for A/B testing whether a plain system prompt + one model call holds up without
# the directive/regex scaffolding. Same function names/signatures on both sides.
if os.environ.get("LLM_IMPL") == "min":
    from app.llm_min import (
        generate_answer,
        decisive_language,
        last_assistant_offered_call,
        onsite_meeting_confirmed,
        bot_admits_unknown,
        is_pure_greeting_only,
        is_small_talk,
    )
else:
    from app.llm import (
        generate_answer,
        decisive_language,
        last_assistant_offered_call,
        onsite_meeting_confirmed,
        bot_admits_unknown,
        is_pure_greeting_only,
        is_small_talk,
    )
from app.memory import (
    init_db, add_message, get_history, save_lead_summary,
    get_profile, save_profile, log_analytics, get_analytics_summary,
    log_knowledge_gap, get_knowledge_gaps,
    ensure_conversation, touch_conversation, set_bot_active, get_bot_active,
    list_conversations, get_conversation_messages,
)
from app.profile import merge_profile
from app.understand import understand_turn
from app.notify import send_booking_email, send_knowledge_gap_email
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
    # Optional pre-chat capture (2026-07-21 user request): the frontend can offer two fields —
    # company name and field/industry — before the conversation starts. Both are optional (the
    # frontend's "skip" case sends neither), and safe to resend on every message: merge_profile's
    # sticky semantics mean a value only ever gets set once and is never overwritten by a later
    # None. This is purely an upfront shortcut for the same profile.company_name/profile.industry
    # the backend already fills in from the conversation itself when these aren't provided.
    company_name: str | None = None
    industry: str | None = None
# FastAPI uses this to automatically validate incoming requests


@app.get("/analytics")
def analytics():
    # Read-only rollup of the per-turn telemetry: average latency per stage, average lead
    # quality, and how turns spread across conversation stages and languages. Open
    # http://localhost:8000/analytics in a browser to eyeball how the bot is performing.
    return get_analytics_summary()


@app.get("/knowledge-gaps")
def knowledge_gaps():
    # Read-only internal list of questions the knowledge base couldn't answer well (weakest match
    # first). This is the "what to improve" queue — review these and add the missing answers to
    # data/blnk_chunks.jsonl. Open http://localhost:8000/knowledge-gaps to see them.
    return get_knowledge_gaps()


# A turn counts as a knowledge gap when the BEST retrieved chunk's similarity is below this.
# bge-m3 cosine similarity runs ~0..1; relevant matches usually score well above this. Tune the
# threshold after watching real data (env override: KNOWLEDGE_GAP_SCORE).
KNOWLEDGE_GAP_SCORE = float(os.environ.get("KNOWLEDGE_GAP_SCORE", "0.45"))


# ---------------------------------------------------------------------------
# Live takeover: lets the owner watch chats live, get notified the instant a
# new one starts, and take over from the bot to reply manually.
# ---------------------------------------------------------------------------

# In-memory only (single-process, like last_message_time/last_activity_time above) — fine for
# this app's current deployment shape; a multi-worker setup would need a shared store instead.
admin_sockets: set[WebSocket] = set()          # every open /admin dashboard tab
customer_sockets: dict[str, WebSocket] = {}    # session_id -> its one open widget tab
admin_tokens: set[str] = set()                 # valid logged-in dashboard session tokens

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
# No default on purpose: if it's unset, login must fail closed (never silently accept anything).


def _admin_authed(token: str | None) -> bool:
    return bool(token) and token in admin_tokens


async def _broadcast_admin(event: dict):
    # Push one event to every open dashboard tab; drop any socket that errors on send (closed
    # tab, dropped connection) rather than letting one dead socket break the others.
    dead = set()
    for ws in admin_sockets:
        try:
            await ws.send_json(event)
        except Exception:
            dead.add(ws)
    admin_sockets.difference_update(dead)


async def _send_customer(session_id: str, event: dict):
    # No-op if that customer's widget isn't currently connected (e.g. they closed the tab) —
    # the owner's reply is still saved to the DB either way, just not delivered live.
    ws = customer_sockets.get(session_id)
    if not ws:
        return
    try:
        await ws.send_json(event)
    except Exception:
        customer_sockets.pop(session_id, None)


class AdminLoginRequest(BaseModel):
    password: str


@app.post("/admin/login")
def admin_login(req: AdminLoginRequest):
    if not ADMIN_PASSWORD or not secrets.compare_digest(req.password, ADMIN_PASSWORD):
        return JSONResponse({"ok": False}, status_code=401)
    token = secrets.token_urlsafe(32)
    admin_tokens.add(token)
    resp = JSONResponse({"ok": True})
    resp.set_cookie("admin_token", token, httponly=True, samesite="lax")
    return resp


@app.get("/admin")
def admin_page():
    # Always serves the page — it shows a login form client-side until the WS/API calls it
    # makes succeed. The actual data (conversations, replies, takeover) is gated below.
    return FileResponse(STATIC_DIR / "admin.html")


class TakeoverRequest(BaseModel):
    session_id: str


@app.post("/admin/takeover")
async def admin_takeover(req: TakeoverRequest, admin_token: str = Cookie(default=None)):
    if not _admin_authed(admin_token):
        return JSONResponse({"ok": False}, status_code=401)
    set_bot_active(req.session_id, False)
    await _broadcast_admin({"type": "takeover_changed", "session_id": req.session_id, "bot_active": False})
    return {"ok": True}


@app.post("/admin/handback")
async def admin_handback(req: TakeoverRequest, admin_token: str = Cookie(default=None)):
    if not _admin_authed(admin_token):
        return JSONResponse({"ok": False}, status_code=401)
    set_bot_active(req.session_id, True)
    await _broadcast_admin({"type": "takeover_changed", "session_id": req.session_id, "bot_active": True})
    return {"ok": True}


class AdminTypingRequest(BaseModel):
    session_id: str


@app.post("/admin/typing")
async def admin_typing(req: AdminTypingRequest, admin_token: str = Cookie(default=None)):
    # Owner is actively typing a reply — let the customer's widget show a typing indicator,
    # mirroring the existing customer -> bot /typing ping but in the other direction.
    if not _admin_authed(admin_token):
        return JSONResponse({"ok": False}, status_code=401)
    await _send_customer(req.session_id, {"type": "owner_typing"})
    return {"ok": True}


class AdminReplyRequest(BaseModel):
    session_id: str
    message: str


@app.post("/admin/reply")
async def admin_reply(req: AdminReplyRequest, admin_token: str = Cookie(default=None)):
    if not _admin_authed(admin_token):
        return JSONResponse({"ok": False}, status_code=401)
    # Stored with role="assistant" (so chat_history stays valid as OpenAI messages if the bot
    # is later handed the conversation back) and sender="owner" (so the dashboard/DB can still
    # tell this apart from a bot-generated reply).
    add_message(req.session_id, "assistant", req.message, sender="owner")
    touch_conversation(req.session_id)
    event = {"type": "new_message", "session_id": req.session_id, "sender": "owner", "content": req.message}
    await _send_customer(req.session_id, event)
    await _broadcast_admin(event)
    return {"ok": True}


@app.get("/admin/conversation/{session_id}")
def admin_conversation(session_id: str, admin_token: str = Cookie(default=None)):
    if not _admin_authed(admin_token):
        return JSONResponse({"ok": False}, status_code=401)
    return {
        "session_id": session_id,
        "bot_active": get_bot_active(session_id),
        "messages": get_conversation_messages(session_id),
    }


@app.websocket("/ws/admin")
async def ws_admin(websocket: WebSocket):
    token = websocket.cookies.get("admin_token")
    if not _admin_authed(token):
        await websocket.close(code=1008)  # policy violation
        return
    await websocket.accept()
    admin_sockets.add(websocket)
    try:
        await websocket.send_json({"type": "init", "conversations": list_conversations()})
        while True:
            await websocket.receive_text()  # dashboard doesn't send anything; just detect disconnect
    except WebSocketDisconnect:
        pass
    finally:
        admin_sockets.discard(websocket)


@app.websocket("/ws/customer/{session_id}")
async def ws_customer(websocket: WebSocket, session_id: str):
    # Same trust level as the public /chat endpoint — no auth, anyone with a session_id (which
    # is only ever known to that browser tab) can open this.
    await websocket.accept()
    customer_sockets[session_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # widget doesn't send anything; just detect disconnect
    except WebSocketDisconnect:
        pass
    finally:
        if customer_sockets.get(session_id) is websocket:
            customer_sockets.pop(session_id, None)

# Tracks the last message timestamp per session, so we know if a newer message has arrived
last_message_time = {}
# Tracks the last time the user did ANYTHING (typed a key or sent a message) per session.
# Both /chat and /typing bump this, so an in-flight reply keeps waiting while the user types.
last_activity_time = {}

QUIET_SECONDS = 0      # reply only after this many seconds of no typing / no new messages
# Removed entirely (2026-07-24, user's explicit latency request — was 1.5s, lowered from 3s on
# 2026-07-23): this is a fixed tax on EVERY reply regardless of content, and with gpt-5-mini's own
# per-call latency already much higher than gpt-4o-mini's, every added second here now stacks on top
# of a slower base instead of a faster one. Tradeoff worth knowing: this debounce is also what lets
# rapid-fire messages ("hi" then, half a second later, "can I know more?") get gathered into ONE
# combined reply (main.py's burst-gathering loop below) instead of two separate ones — at 0, that
# window is gone in practice, so two messages sent close together will now usually get answered
# individually. The /typing endpoint already bumps last_activity_time on every keystroke, so this
# only measures time since the customer went fully idle (stopped typing AND sent nothing new) — a
# genuine follow-up sent while still actively typing is unaffected; this just shortens how long the
# bot waits once they've actually stopped typing before replying.
MAX_WAIT_SECONDS = 30    # safety cap so a reply can never wait forever


class TypingRequest(BaseModel):
    session_id: str


@app.post("/typing")
def typing(req: TypingRequest):
    # Frontend pings this on every keystroke. It just tells any waiting reply
    # "the user is still typing — don't answer yet."
    last_activity_time[req.session_id] = time.time()
    return {"ok": True}


class MeetingBookedRequest(BaseModel):
    session_id: str
    meeting_type: str = "online"


@app.post("/meeting-booked")
def meeting_booked(req: MeetingBookedRequest, background_tasks: BackgroundTasks):
    # Fired by the frontend when Calendly confirms an ONLINE booking actually completed (see the
    # calendly.event_scheduled listener in index.html). The onsite path is detected server-side
    # instead, in /chat below, since there's no self-booking calendar for it.
    profile = get_profile(req.session_id)
    if profile.get("meeting_booked_sent"):
        return {"ok": True, "already_sent": True}
    profile["meeting_booked_sent"] = True
    save_profile(req.session_id, profile)
    background_tasks.add_task(send_booking_email, req.session_id, req.meeting_type)
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    add_message(req.session_id, "user", req.message)

    # Register the conversation on its very first message (fires the dashboard's "new
    # conversation" notification exactly once), then always bump recency + broadcast this
    # message live so an open dashboard shows activity as it happens, not just once the bot
    # has finished its debounce wait below.
    is_new_conversation = ensure_conversation(req.session_id)
    touch_conversation(req.session_id)
    if is_new_conversation:
        await _broadcast_admin({"type": "new_conversation", "session_id": req.session_id})
    await _broadcast_admin({
        "type": "new_message", "session_id": req.session_id,
        "sender": "user", "content": req.message,
    })

    # The owner has taken over this conversation — the bot stays silent entirely. Check this
    # BEFORE the debounce wait below so a customer waiting on a human isn't stuck in the bot's
    # own (up to 30s) quiet-wait window for no reason.
    if not get_bot_active(req.session_id):
        return {"answer": None, "awaiting_owner": True}

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

    # No newer message came in — safe to respond now. Pull a wider window so the model
    # can still reference things the user said earlier in a longer chat (their field,
    # which service they picked, etc.) rather than forgetting past ~5 turns.
    history = get_history(req.session_id, limit=16)

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

    # True when every message in this burst is either a bare greeting or a pure pleasantry ("hii" +
    # "how are you" sent together, etc.) — nothing with any real informational content. Broader than
    # is_fast_greeting below (which only covers a SINGLE bare "hii" and gates the retrieval-skip fast
    # path specifically) — this only feeds the knowledge-gap check further down, so it's fine for it to
    # also cover a multi-line all-smalltalk burst. Found live: "hii" + "how are you" got emailed to the
    # team as a knowledge gap — retrieval naturally has nothing relevant for small talk, and the bot's
    # conversational non-answer isn't a real content gap, just like a bare "hii" already wasn't.
    is_smalltalk_only = bool(burst) and all(
        is_pure_greeting_only(m) or is_small_talk(m) for m in burst
    )

    # Find the most recent assistant message. Short follow-ups like "yes please"
    # carry no topic on their own, so we fold the last assistant turn into the
    # retrieval query — that way a "yes" to "want me to explain brand identity?"
    # still pulls the brand-identity chunks.
    last_assistant = next(
        (m["content"] for m in reversed(prior_history) if m["role"] == "assistant"),
        "",
    )
    retrieval_query = f"{last_assistant}\n{combined_message}".strip() if last_assistant else combined_message

    # What we already know about this customer, carried forward from earlier turns.
    profile = get_profile(req.session_id)

    # One rep handles each conversation (user's explicit request, 2026-07-21) — assigned ONCE per
    # session and stuck to for its whole lifetime, not re-rolled every turn. Saved immediately
    # (rather than waiting for the batch save_profile below) so it's locked in even if this turn
    # hits one of the early returns (skipped/awaiting_owner) further down before reaching that save.
    if not profile.get("assigned_rep"):
        profile["assigned_rep"] = random.choice(["Sami", "Lana"])
        save_profile(req.session_id, profile)

    # Pre-chat capture, if the frontend collected it (see ChatRequest — both optional). Sticky-
    # merged the same way every other profile field is, so this is a no-op once already set and
    # safe to resend every message. Doing this BEFORE understand_turn/retrieval below means even
    # the very first reply is already tailored — no field_pitch_unclear detour, no mid-conversation
    # "what's your field?" ask, for anyone who filled this in upfront.
    if req.company_name or req.industry:
        profile = merge_profile(profile, {
            "company_name": req.company_name,
            "industry": req.industry,
        })
        save_profile(req.session_id, profile)

    # Latency fast-path (user's explicit request, 2026-07-23): a bare "hii" with nothing else
    # attached has no pricing/field/contact signal to extract and nothing to retrieve — the full
    # understand_turn() + retrieve() pair was costing ~5s of pure overhead for a message that needs
    # neither (measured live). Skip both entirely for this narrow case; "hii what are your services"
    # still goes through the full pipeline below, is_pure_greeting_only only matches when the ENTIRE
    # message is just the greeting. preferred_language deliberately stays UNSET here rather than
    # guessed from "hii" alone (a wrong first-turn guess would otherwise lock in via the "once, never
    # after" logic below) — the next substantive message establishes it properly instead.
    is_fast_greeting = is_pure_greeting_only(combined_message)
    if is_fast_greeting:
        signals = {}
        understand_ms = 0.0
        context_chunks = []
        retrieval_ms = 0.0
    else:
        # ONE model pass understands the whole turn — reply language, pricing intent, and the
        # profile/lead fields — instead of brittle regex/keyword matching. It handles any language
        # or arabizi spelling, and replaces the old separate scoring call (so no extra latency).
        # We time each expensive step with perf_counter (a monotonic clock meant for measuring
        # durations — unlike time.time() it can't jump backward if the system clock is adjusted).
        t_understand_start = time.perf_counter()
        signals = understand_turn(combined_message, prior_history)
        understand_ms = (time.perf_counter() - t_understand_start) * 1000

        # Anchor the language to what the customer STARTED with: set preferred_language once, and
        # never drift it after. decisive_language checked FIRST, not the model's own reply_language
        # guess — found live: an all-arabizi first message ("marhaba kifkon aam bhkekon la estafser
        # aktar...") got signals["reply_language"]="ar" from the LLM (a reasonable-sounding but wrong
        # read — it saw Arabic dialect content and guessed Arabic script, missing the 2026-07-17
        # product decision that arabizi always replies in English). That "ar" got locked into the
        # profile permanently, so every later short/ambiguous message in the SAME conversation (e.g.
        # a bare "print", "helo habet") fell back to the wrongly-stored "ar" and got Arabic-script
        # replies mid-conversation, even though the customer never once typed Arabic script.
        # decisive_language(combined_message) is the deterministic, script-based rule that already
        # correctly encodes "Latin script (English or arabizi) = en, Arabic script = ar" — the same
        # "reliable local read wins over the LLM guess" principle used everywhere else in llm.py. Only
        # fall back to the model's own guess when the message itself genuinely can't decide (no
        # letters at all, e.g. just an emoji or a number).
        if not profile.get("preferred_language"):
            lang = decisive_language(combined_message) or signals.get("reply_language")
            if lang in ("en", "ar"):
                profile["preferred_language"] = lang
        # An explicit "switch to Arabic"/"reply in English" request is a real, lasting change, not
        # just a one-off override for this reply — so it overwrites the sticky preferred_language
        # outright (unlike the block above, which only ever sets it once).
        requested_lang = signals.get("requested_reply_language")
        if requested_lang in ("en", "ar"):
            profile["preferred_language"] = requested_lang
        if signals.get("is_pricing"):
            profile["pricing_requested"] = True

        # When the customer names their field and wants to know how we can help, this is the moment
        # to convince them — so search HARDER: bias retrieval toward proof in their industry (client
        # lists and case studies) and pull more chunks so the model has material to make a strong,
        # tailored case. The fuller-answer behavior itself is driven by FIELD_PITCH_DIRECTIVE in
        # generate_answer.
        industry = signals.get("industry") or profile.get("industry")
        if industry:
            # We know the customer's field — always search for BLNK's own past work in it, so the
            # generation step actually has a real client it can name as proof. The full convince-them
            # pitch turn pulls a wider net (top_k=8); an ordinary turn where we merely know the field
            # pulls a little extra (top_k=6) to surface a client without crowding out the direct
            # answer.
            retrieval_query = f"{retrieval_query}\nBLNK clients, case studies and past work in the {industry} industry"
            top_k = 8 if signals.get("wants_field_pitch") else 6
        else:
            top_k = 5

        # On a "tell me more" turn, anchor retrieval to the service already under discussion so we
        # detail THAT (deeper) instead of surfacing brand-new offerings the customer didn't ask about.
        if signals.get("wants_more_detail") and profile.get("interested_service"):
            retrieval_query = f"{retrieval_query}\nwhat BLNK's {profile['interested_service']} service includes, in detail"

        # Skip retrieval for a pre-classified small_talk turn — mirrors is_fast_greeting above, just
        # for the broader small-talk set that regex alone doesn't catch (e.g. "kifak"). Scoped to
        # LLM_IMPL=min only: this is a latency optimization for the experimental path, not a change
        # to app/llm.py's tested behavior. Deliberately NOT applied to off_topic — a wrong
        # off_topic call with no context at all leaves the generation step no way to self-correct
        # (it can still override the pre-classification per its own instructions, but only if it
        # actually has context to look at), whereas a wrong small_talk call is low-stakes either way.
        if os.environ.get("LLM_IMPL") == "min" and signals.get("message_category") == "small_talk":
            context_chunks = []
            retrieval_ms = 0.0
        else:
            t_retrieval_start = time.perf_counter()
            context_chunks = retrieve(retrieval_query, top_k=top_k)
            retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

    # Best match similarity (cosine, higher = closer). A low top score means the knowledge base
    # doesn't really cover this question — we log it as a gap once we have the reply.
    top_score = max((c.get("score", 0.0) for c in context_chunks), default=0.0)

    t_generation_start = time.perf_counter()
    answer, primary_directive = generate_answer(combined_message, context_chunks, prior_history,
                                                profile=profile, signals=signals)
    generation_ms = (time.perf_counter() - t_generation_start) * 1000

    # A newer message can arrive WHILE we were generating (after our own quiet-wait already
    # passed, since that loop above only guards the pre-generation wait, not the several seconds
    # of OpenAI calls that follow it). Found live: two rapid-fire messages ("Monday at 10am?"
    # then, moments later, "and we're a boutique now") got answered as two SEPARATE turns instead
    # of one combined burst — because by the time the second message's own burst-gathering ran,
    # this reply had already been saved in between, and that gathering loop stops the instant it
    # sees a non-user message scanning backward. Discard this reply entirely instead of saving it:
    # the newer request will then find nothing saved between the two user messages and correctly
    # combine them into one burst when it runs.
    if last_message_time[req.session_id] != this_message_time:
        return {"answer": None, "skipped": True}

    # The owner can click "Take over" at any point during the several seconds this turn spent on
    # understand/retrieval/generation above — bot_active was only checked once, at the very top,
    # before any of that ran. Re-check now: if the owner has since taken over, discard this
    # reply entirely instead of saving/sending it, so the bot never answers a message the owner
    # already started handling.
    if not get_bot_active(req.session_id):
        return {"answer": None, "awaiting_owner": True}

    add_message(req.session_id, "assistant", answer, sender="bot")
    await _broadcast_admin({
        "type": "new_message", "session_id": req.session_id,
        "sender": "assistant", "content": answer,
    })

    # Did the reply we just gave offer the call? Record it so we don't re-offer later.
    if last_assistant_offered_call([{"role": "assistant", "content": answer}]):
        profile["call_offered"] = True

    # Onsite meetings have no self-booking calendar (unlike online, which is confirmed by Calendly
    # itself via the /meeting-booked endpoint) — this chat turn IS the booking confirmation: we
    # already gave the office address, and the customer just replied with a day/time for it. Guard
    # on the flag so a session only ever triggers this once. Pass understand_turn()'s own
    # confirms_meeting_daytime read alongside the regex check — either one firing is enough, so a
    # day/time phrasing the regex hasn't been taught yet still gets caught instead of silently
    # never sending the booking email.
    onsite_booked_now = (
        not profile.get("meeting_booked_sent")
        and onsite_meeting_confirmed(prior_history, combined_message,
                                      signals.get("confirms_meeting_daytime"))
    )
    if onsite_booked_now:
        profile["meeting_booked_sent"] = True

    # Fold the fields the understanding pass extracted into the profile, stickily (a null
    # extraction never erases a fact we already learned). Same signals feed the lead summary.
    profile = merge_profile(profile, {
        "industry": signals.get("industry"),
        "interested_service": signals.get("interested_service"),
        "customer_name": signals.get("customer_name"),
        "company_name": signals.get("company_name"),
        "budget_signal": signals.get("budget_signal"),
        "timeline": signals.get("timeline"),
    })
    # Conversation stage is the one field that is deliberately NOT sticky: it's meant to
    # move forward (greeting -> discovery -> ... -> closing) as the chat progresses, so the
    # latest read overwrites. We only keep the prior stage when this turn couldn't classify.
    stage = signals.get("conversation_stage")
    if stage:
        profile["conversation_stage"] = stage
    save_profile(req.session_id, profile)

    summary = {
        "industry": profile.get("industry"),
        "interested_service": profile.get("interested_service"),
        "budget_signal": signals.get("budget_signal"),
        "timeline": signals.get("timeline"),
        "score": signals.get("lead_score"),
        "conversation_stage": profile.get("conversation_stage"),
        "notes": signals.get("notes"),
    }
    save_lead_summary(req.session_id, summary)

    # Notify BLNK now that the profile/summary this email reports on are actually saved.
    if onsite_booked_now:
        background_tasks.add_task(send_booking_email, req.session_id, "onsite")

    # Record one telemetry row for this turn. total_ms is the sum of the three compute steps
    # (understanding + retrieval + generation) — it deliberately EXCLUDES the debounce wait at
    # the top of this handler, which is idle time spent waiting for the user to stop typing and
    # would otherwise swamp the numbers we actually want to watch. Wrapped so analytics can
    # never break a reply.
    try:
        log_analytics(req.session_id, {
            "understand_ms": understand_ms,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": understand_ms + retrieval_ms + generation_ms,
            "reply_language": profile.get("preferred_language"),
            "conversation_stage": profile.get("conversation_stage"),
            "lead_score": signals.get("lead_score"),
            "chunks_used": len(context_chunks),
            "is_pricing": bool(signals.get("is_pricing")),
            "wants_field_pitch": bool(signals.get("wants_field_pitch")),
        })
    except Exception:
        pass

    # If the bot's own reply admits it doesn't know (the standing "I'll confirm that for you shortly"
    # escalation hedge from SYSTEM_PROMPT, or a flat "I don't have that information" — see bot_admits_
    # unknown's docstring) — log it AND alert the team (2026-07-21 user request): an email with the
    # admin dashboard link, plus a live notification to any open dashboard, so a human can step in and
    # get the customer a real answer instead of it silently going unanswered.
    #
    # User's explicit correction (2026-07-23), then simplified further (2026-07-24): this used to ALSO
    # fire whenever retrieval score was low, which meant firing on turns the bot answered CORRECTLY —
    # a bare "hii" got emailed as a gap, and so would every pricing/timing question (there's no pricing
    # chunk in the data at all, so retrieval score is naturally low even though PRICING_DIRECTIVE
    # answers it perfectly every time) — which needed an ever-growing list of exemptions (is_fast_
    # greeting, is_smalltalk_only, a directive allowlist) to claw back. The user's explicit, simpler
    # product decision: an email is warranted ONLY when the bot itself told the customer it'll check
    # and follow up — that's the one unambiguous, self-contained signal that a human actually needs to
    # step in, and it needs no exemption list at all (a bare "hii" or a pricing answer never contains
    # that hedge in the first place).
    # primary_directive is llm_min's own structured category self-report when that path is active
    # (see its generate_answer) — reliable by construction, unlike regex-guessing from the reply
    # text. Found live: bot_admits_unknown's regex was being OR'd in regardless of what a VALID
    # category already said, so a turn correctly self-reported as "answerable" or "meeting_relevant"
    # could still trip a false knowledge-gap email purely because the reply happened to contain
    # "check...get back" wording — which, now that the onsite flow's own acknowledgment ("I will
    # check the schedule and get back to you shortly to confirm") reuses that exact shape on every
    # settled-appointment follow-up, not just the one turn a time was actually given, fired on
    # nearly every ordinary onsite conversation turn, not just genuine gaps. Only fall back to the
    # regex when there's no reliable category at all (app/llm.py, which never returns one of these
    # values, or a malformed/missing structured response) — never to second-guess a valid one.
    _MIN_CATEGORIES = {"small_talk", "off_topic", "meeting_relevant", "answerable", "unknown_info"}
    if primary_directive in _MIN_CATEGORIES:
        is_knowledge_gap = primary_directive == "unknown_info" and not onsite_booked_now
    else:
        is_knowledge_gap = bot_admits_unknown(answer) and not onsite_booked_now
    if is_knowledge_gap:
        try:
            log_knowledge_gap(req.session_id, combined_message, top_score, answer)
        except Exception:
            pass
        background_tasks.add_task(send_knowledge_gap_email, req.session_id, combined_message, answer)
        await _broadcast_admin({
            "type": "knowledge_gap", "session_id": req.session_id,
            "question": combined_message,
        })

    return {"answer": answer, "lead_summary": summary, "profile": profile}

#1. Save the user's message
#2. Pull up the recent conversation history
#3. Search the knowledge base for relevant facts
#4. Ask the AI to write an answer, using the facts + history
#5. Save the AI's reply
#6. Re-check the whole conversation to score how promising this lead is
#7. Save that score
#8. Send back the answer (and the score) to whoever asked