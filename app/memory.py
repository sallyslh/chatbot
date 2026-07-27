#this is the memory module that handles storing and retrieving conversation history and lead summaries in a SQLite database
import sqlite3, json, os
from datetime import datetime, timezone

DB_PATH = "data/conversations.db"

def init_db():
  
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    # Opens a connection to the database file (creates it if it doesn't exist yet)

    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT
    )""")

    conn.execute("""CREATE TABLE IF NOT EXISTS lead_summary (
        session_id TEXT PRIMARY KEY,
        summary_json TEXT
    )""")
    # Creates a second table to store the latest lead-scoring summary per conversation

    conn.execute("""CREATE TABLE IF NOT EXISTS conversation_profile (
        session_id TEXT PRIMARY KEY,
        profile_json TEXT
    )""")
    # Persistent, accumulating profile per conversation (industry, interested service,
    # preferred language, flags...). Unlike the messages, this is UPDATED not appended,
    # so the backend carries forward known facts instead of the model re-inferring them
    # every turn.

    conn.execute("""CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        created_at TEXT,
        understand_ms REAL,
        retrieval_ms REAL,
        generation_ms REAL,
        total_ms REAL,
        reply_language TEXT,
        conversation_stage TEXT,
        lead_score INTEGER,
        chunks_used INTEGER,
        is_pricing INTEGER,
        wants_field_pitch INTEGER
    )""")
    # One row per answered turn. Unlike the other tables this is APPEND-only (a time series):
    # every reply leaves a durable record of how long each stage took and what the turn looked
    # like, so we can measure latency and conversation health instead of guessing.

    conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_gaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        created_at TEXT,
        question TEXT,
        retrieval_score REAL,
        answer_given TEXT
    )""")
    # Append-only list of questions the knowledge base couldn't answer well (the top retrieval
    # similarity came back low). This is the invisible "what to improve" queue: review these
    # internally and add the missing answers to data/blnk_chunks.jsonl.

    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        session_id TEXT PRIMARY KEY,
        bot_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT,
        last_activity_at TEXT
    )""")
    # Live-takeover control state per conversation: whether the bot is currently allowed to
    # reply, so the owner can pause it and answer manually from the /admin dashboard.

    try:
        conn.execute("ALTER TABLE messages ADD COLUMN sender TEXT")
        # Distinguishes an assistant-role row written by the bot ("bot") from one written by
        # the human owner during a takeover ("owner"). Deliberately NOT a new role value —
        # role stays user/assistant so chat_history keeps working unmodified as the OpenAI
        # messages array (llm.py requires role in system/user/assistant/tool). NULL for
        # ordinary customer ("user") rows.
    except sqlite3.OperationalError:
        pass  # column already exists — safe to ignore, this migration is idempotent

    conn.commit()
    conn.close()

def add_message(session_id, role, content, sender=None):
    # Function to save one new message into the messages table.
    # sender: None for customer ("user") rows; "bot" or "owner" for assistant-role rows,
    # so the admin dashboard can tell a bot reply from a manual takeover reply.
    conn = sqlite3.connect(DB_PATH)
    # Opens a connection to the database

    conn.execute("INSERT INTO messages (session_id, role, content, sender) VALUES (?,?,?,?)",
                 (session_id, role, content, sender))
    # Inserts a new row with this conversation's ID, who sent it, and what was said
    # (the ?,?,? placeholders protect against SQL injection issues)

    conn.commit()
    conn.close()

def get_history(session_id, limit=10):
    # Function to retrieve recent messages for a specific conversation

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    # Fetches the most recent messages for this specific session_id,

    conn.close()

    return [{"role": r, "content": c} for r, c in reversed(rows)]
    # Converts the raw database rows into a clean list of dictionaries,
    # and reverses the order back to oldest-to-newest (since the query fetched newest-first)

def save_lead_summary(session_id, summary: dict):
    # Function to save (or update) the lead-scoring summary for a conversation

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""INSERT INTO lead_summary (session_id, summary_json) VALUES (?,?)
                    ON CONFLICT(session_id) DO UPDATE SET summary_json=excluded.summary_json""",
                 (session_id, json.dumps(summary)))
    # Tries to insert a new summary row; if one already exists for this session_id,
    # it updates it instead (so each conversation only ever has one, latest summary)
    # json.dumps() converts the Python dictionary into a text string for storage

    conn.commit()
    conn.close()

def get_lead_summary(session_id) -> dict:
    # Returns the latest saved lead-scoring summary for this session, or {} if none yet.
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT summary_json FROM lead_summary WHERE session_id=?",
        (session_id,)
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}

def get_profile(session_id) -> dict:
    # Returns the stored conversation profile for this session, or {} if none yet.
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT profile_json FROM conversation_profile WHERE session_id=?",
        (session_id,)
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}

def save_profile(session_id, profile: dict):
    # Saves (or updates) the single latest profile for a conversation.
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO conversation_profile (session_id, profile_json) VALUES (?,?)
                    ON CONFLICT(session_id) DO UPDATE SET profile_json=excluded.profile_json""",
                 (session_id, json.dumps(profile, ensure_ascii=False)))
    conn.commit()
    conn.close()


def log_analytics(session_id, metrics: dict):
    # Append one telemetry row for a turn. Kept intentionally forgiving: analytics must never
    # be able to break a reply, so missing keys just become NULL and the caller wraps this in
    # its own try/except.
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO analytics
           (session_id, created_at, understand_ms, retrieval_ms, generation_ms, total_ms,
            reply_language, conversation_stage, lead_score, chunks_used, is_pricing, wants_field_pitch)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            session_id,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            metrics.get("understand_ms"),
            metrics.get("retrieval_ms"),
            metrics.get("generation_ms"),
            metrics.get("total_ms"),
            metrics.get("reply_language"),
            metrics.get("conversation_stage"),
            metrics.get("lead_score"),
            metrics.get("chunks_used"),
            1 if metrics.get("is_pricing") else 0,
            1 if metrics.get("wants_field_pitch") else 0,
        ),
    )
    conn.commit()
    conn.close()


def get_analytics_summary(limit_recent: int = 20) -> dict:
    # Roll the raw per-turn rows up into the numbers you'd actually look at: average latency
    # for each stage, average lead quality, and how turns are distributed across conversation
    # stages and languages — plus the most recent turns for a quick eyeball.
    conn = sqlite3.connect(DB_PATH)
    agg = conn.execute(
        """SELECT COUNT(*), AVG(understand_ms), AVG(retrieval_ms),
                  AVG(generation_ms), AVG(total_ms), AVG(lead_score)
           FROM analytics"""
    ).fetchone()
    stages = conn.execute(
        "SELECT conversation_stage, COUNT(*) FROM analytics GROUP BY conversation_stage"
    ).fetchall()
    langs = conn.execute(
        "SELECT reply_language, COUNT(*) FROM analytics GROUP BY reply_language"
    ).fetchall()
    recent = conn.execute(
        """SELECT created_at, total_ms, reply_language, conversation_stage, lead_score, chunks_used
           FROM analytics ORDER BY id DESC LIMIT ?""",
        (limit_recent,),
    ).fetchall()
    conn.close()

    def _round(v):
        return round(v, 1) if v is not None else None

    return {
        "turns": agg[0] or 0,
        "avg_ms": {
            "understand": _round(agg[1]),
            "retrieval": _round(agg[2]),
            "generation": _round(agg[3]),
            "total": _round(agg[4]),
        },
        "avg_lead_score": _round(agg[5]),
        "by_stage": {(s or "unknown"): n for s, n in stages},
        "by_language": {(l or "unknown"): n for l, n in langs},
        "recent": [
            {"at": r[0], "total_ms": _round(r[1]), "language": r[2],
             "stage": r[3], "lead_score": r[4], "chunks_used": r[5]}
            for r in recent
        ],
    }


def log_knowledge_gap(session_id, question: str, retrieval_score: float, answer_given: str):
    # Record a question the knowledge base couldn't answer well. Forgiving on purpose (the caller
    # wraps it in try/except) — capturing a gap must never be able to break a reply.
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO knowledge_gaps (session_id, created_at, question, retrieval_score, answer_given)
           VALUES (?,?,?,?,?)""",
        (session_id, datetime.now(timezone.utc).isoformat(timespec="seconds"),
         question, retrieval_score, answer_given),
    )
    conn.commit()
    conn.close()


def get_knowledge_gaps(limit: int = 100) -> dict:
    # The internal "what to improve" list: questions the KB struggled with, weakest match first so
    # the biggest gaps surface at the top, plus a count.
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM knowledge_gaps").fetchone()[0]
    rows = conn.execute(
        """SELECT created_at, question, retrieval_score, answer_given
           FROM knowledge_gaps ORDER BY retrieval_score ASC, id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return {
        "total": total,
        "gaps": [
            {"at": r[0], "question": r[1],
             "retrieval_score": round(r[2], 3) if r[2] is not None else None,
             "answer_given": r[3]}
            for r in rows
        ],
    }


def get_conversation_messages(session_id) -> list:
    # Full message history for the admin dashboard's thread view (unlike get_history, this isn't
    # limited to the last N turns and includes `sender` so the UI can tell bot vs. owner apart).
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, sender FROM messages WHERE session_id=? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    return [{"role": r, "content": c, "sender": s} for r, c, s in rows]


def ensure_conversation(session_id) -> bool:
    # Registers a conversation the first time it's ever seen (bot_active defaults to True).
    # Returns True only if this call actually created the row, so the caller can fire a
    # "new conversation" event exactly once instead of on every turn.
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT OR IGNORE INTO conversations (session_id, bot_active, created_at, last_activity_at) "
        "VALUES (?, 1, ?, ?)",
        (session_id, now, now),
    )
    conn.commit()
    is_new = cur.rowcount > 0
    conn.close()
    return is_new


def touch_conversation(session_id):
    # Bumps last_activity_at so the admin dashboard's conversation list can sort by recency.
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE conversations SET last_activity_at=? WHERE session_id=?",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), session_id),
    )
    conn.commit()
    conn.close()


def set_bot_active(session_id, active: bool):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE conversations SET bot_active=? WHERE session_id=?",
        (1 if active else 0, session_id),
    )
    conn.commit()
    conn.close()


def get_bot_active(session_id) -> bool:
    # Defaults to True (bot replies normally) if the conversation was never registered —
    # ensure_conversation always runs first in /chat, so this is just a safe fallback.
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT bot_active FROM conversations WHERE session_id=?", (session_id,)
    ).fetchone()
    conn.close()
    return True if not row else bool(row[0])


def list_conversations() -> list:
    # Snapshot for the admin dashboard: every known conversation, newest activity first, with
    # its bot_active state, the customer's name if the bot has learned it, and a preview of the
    # latest message so the list is useful without opening every thread.
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT c.session_id, c.bot_active, c.created_at, c.last_activity_at,
                  (SELECT content FROM messages m WHERE m.session_id = c.session_id
                    ORDER BY m.id DESC LIMIT 1) AS last_message,
                  (SELECT role FROM messages m WHERE m.session_id = c.session_id
                    ORDER BY m.id DESC LIMIT 1) AS last_role,
                  p.profile_json
           FROM conversations c
           LEFT JOIN conversation_profile p ON p.session_id = c.session_id
           ORDER BY c.last_activity_at DESC"""
    ).fetchall()
    conn.close()

    result = []
    for session_id, bot_active, created_at, last_activity_at, last_message, last_role, profile_json in rows:
        customer_name = None
        if profile_json:
            try:
                customer_name = json.loads(profile_json).get("customer_name")
            except json.JSONDecodeError:
                pass
        result.append({
            "session_id": session_id,
            "bot_active": bool(bot_active),
            "created_at": created_at,
            "last_activity_at": last_activity_at,
            "last_message": last_message,
            "last_role": last_role,
            "customer_name": customer_name,
        })
    return result