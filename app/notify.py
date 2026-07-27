# Sends BLNK an email with a PDF transcript whenever a customer books a meeting (onsite or
# online). Called as a FastAPI background task from main.py so it never adds latency to a chat
# reply; failures here must never surface to the customer.
import os
import re
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from fpdf import FPDF

from app.memory import get_history, get_profile, get_lead_summary

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _CAN_SHAPE_ARABIC = True
except ImportError:
    # Optional: without these, Arabic still renders (via the Unicode font below) but with
    # disconnected letterforms instead of properly joined/reordered script.
    _CAN_SHAPE_ARABIC = False

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
BOOKING_NOTIFY_EMAIL = os.environ.get("BOOKING_NOTIFY_EMAIL", "hello@blnkcs.com")
# Same inbox as the booking alert by default (one place the team already watches) — override via
# env if knowledge-gap alerts should route somewhere else.
KNOWLEDGE_GAP_NOTIFY_EMAIL = os.environ.get("KNOWLEDGE_GAP_NOTIFY_EMAIL", BOOKING_NOTIFY_EMAIL)
# Base URL of the deployed app, used to build the admin dashboard link in the alert email — no
# request object is available from a background task, so this can't be derived from the incoming
# request and needs its own env var. Defaults to the common local-dev address.
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8000").rstrip("/")

# fpdf2's built-in core fonts are Latin-1 only, which would mangle Arabic/arabizi transcripts.
# Tahoma ships with Windows and has solid Arabic glyph coverage; override via env for other
# deployment targets.
PDF_FONT_PATH = os.environ.get("PDF_FONT_PATH", r"C:\Windows\Fonts\tahoma.ttf")
PDF_FONT_BOLD_PATH = os.environ.get("PDF_FONT_BOLD_PATH", r"C:\Windows\Fonts\tahomabd.ttf")

_HAS_ARABIC_RE = re.compile(r"[؀-ۿ]")


def _shape(text: str) -> str:
    if not text or not _CAN_SHAPE_ARABIC or not _HAS_ARABIC_RE.search(text):
        return text
    return get_display(arabic_reshaper.reshape(text))


def _build_pdf(session_id: str, meeting_type: str, history: list, profile: dict, summary: dict) -> bytes:
    pdf = FPDF()
    use_unicode_font = os.path.exists(PDF_FONT_PATH)
    if use_unicode_font:
        pdf.add_font("Body", "", PDF_FONT_PATH)
        pdf.add_font("Body", "B", PDF_FONT_BOLD_PATH if os.path.exists(PDF_FONT_BOLD_PATH) else PDF_FONT_PATH)
        font = "Body"
    else:
        font = "Helvetica"  # Latin-1 fallback — Arabic text will not render correctly

    pdf.add_page()
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "BLNK Chatbot — New Meeting Booked", ln=True)

    pdf.set_font(font, "", 10)
    pdf.cell(0, 6, f"Session ID: {session_id}", ln=True)
    pdf.cell(0, 6, f"Meeting type: {meeting_type}", ln=True)
    pdf.cell(0, 6, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", ln=True)
    pdf.ln(4)

    pdf.set_font(font, "B", 12)
    pdf.cell(0, 8, "Lead summary", ln=True)
    pdf.set_font(font, "", 10)
    labels = {
        "customer_name": "Customer name",
        "industry": "Industry",
        "interested_service": "Interested service",
        "budget_signal": "Budget signal",
        "timeline": "Timeline",
        "score": "Lead score",
        "conversation_stage": "Stage",
        "notes": "Notes",
    }
    merged = {**summary, "customer_name": profile.get("customer_name")}
    for key, label in labels.items():
        value = merged.get(key)
        if value:
            pdf.multi_cell(0, 6, _shape(f"{label}: {value}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font(font, "B", 12)
    pdf.cell(0, 8, "Full conversation", ln=True)
    for msg in history:
        speaker = "Customer" if msg.get("role") == "user" else "BLNK Bot"
        pdf.set_font(font, "B", 10)
        pdf.multi_cell(0, 6, speaker + ":", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font, "", 10)
        pdf.multi_cell(0, 6, _shape(msg.get("content") or ""), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    return bytes(pdf.output())


def send_booking_email(session_id: str, meeting_type: str):
    # Best-effort: this runs in a background task after the chat response has already been sent,
    # so an SMTP/PDF failure here can only cost BLNK one notification — never a customer-visible
    # error. Logged to stderr so it's visible in the server logs for follow-up.
    if not SMTP_USER or not SMTP_PASS:
        print(f"[notify] SMTP_USER/SMTP_PASS not configured — skipping booking email for {session_id}")
        return
    try:
        history = get_history(session_id, limit=1000)
        profile = get_profile(session_id)
        summary = get_lead_summary(session_id)
        pdf_bytes = _build_pdf(session_id, meeting_type, history, profile, summary)

        msg = EmailMessage()
        customer = profile.get("customer_name") or "A customer"
        msg["Subject"] = f"New {meeting_type} meeting booked — session {session_id}"
        msg["From"] = SMTP_USER
        msg["To"] = BOOKING_NOTIFY_EMAIL
        msg.set_content(
            f"{customer} just booked a {meeting_type} meeting via the BLNK chatbot.\n"
            f"Session: {session_id}\n\n"
            "Full conversation transcript attached as PDF."
        )
        msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf",
                            filename=f"blnk-chat-{session_id}.pdf")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception as e:
        print(f"[notify] failed to send booking email for {session_id}: {e}")


def send_knowledge_gap_email(session_id: str, question: str, answer_given: str):
    # User's explicit request (2026-07-21): when the bot can't actually answer something, alert the
    # team by email — including the admin dashboard link so they can open the conversation directly
    # — rather than just letting the customer hear a plain "I don't know". Same best-effort contract
    # as send_booking_email: runs as a background task after the reply is already sent, so a failure
    # here can only cost BLNK one notification, never a customer-visible error.
    if not SMTP_USER or not SMTP_PASS:
        print(f"[notify] SMTP_USER/SMTP_PASS not configured — skipping knowledge-gap email for {session_id}")
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = f"BLNK chatbot couldn't answer a question — session {session_id}"
        msg["From"] = SMTP_USER
        msg["To"] = KNOWLEDGE_GAP_NOTIFY_EMAIL
        msg.set_content(
            "The BLNK chatbot wasn't able to properly answer a customer's question and could use a "
            "human follow-up.\n\n"
            f"Question: {question}\n\n"
            f"What the bot told them: {answer_given}\n\n"
            f"Session: {session_id}\n"
            f"Open this conversation in the admin dashboard: {APP_BASE_URL}/admin"
        )
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception as e:
        print(f"[notify] failed to send knowledge-gap email for {session_id}: {e}")
