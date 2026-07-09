#this is the memory module that handles storing and retrieving conversation history and lead summaries in a SQLite database
import sqlite3, json, os

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

    conn.commit()
    conn.close()

def add_message(session_id, role, content):
    # Function to save one new message into the messages table
    conn = sqlite3.connect(DB_PATH)
    # Opens a connection to the database

    conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
                 (session_id, role, content))
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