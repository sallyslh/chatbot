import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.memory import init_db, add_message, get_history, save_lead_summary

init_db()

add_message("test-session-1", "user", "Hello, do you work with real estate clients?")
add_message("test-session-1", "assistant", "Yes! We've worked with several real estate clients.")

history = get_history("test-session-1")
print("Conversation history:")
for msg in history:
    print(f"  {msg['role']}: {msg['content']}")

save_lead_summary("test-session-1", {"industry": "real estate", "score": 70})
print("\nSaved a fake lead summary successfully.")
