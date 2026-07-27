# The conversation profile: what the backend knows about a customer, accumulated
# across turns and fed back into the prompt so the model doesn't re-infer (and drift
# on) the same facts every message. State logic lives here; the reply-language
# detection and prompt assembly live in llm.py. This module imports nothing from llm
# to keep the two free of circular imports.

# Values that a fuzzy extractor emits for "I don't know" — treated the same as null so
# they never overwrite a real, known value.
_NULLISH = {"", "null", "none", "unknown", "n/a", "na", "not specified", "not mentioned"}


def _is_nullish(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in _NULLISH)


def merge_profile(existing: dict, updates: dict) -> dict:
    """Merge newly extracted fields into the stored profile, STICKILY: a new value only
    overwrites when it is a real value. A null/"unknown"/empty extraction never erases a
    fact we already learned (so a short "yes" turn can't wipe the customer's industry)."""
    merged = dict(existing or {})
    for key, value in (updates or {}).items():
        if _is_nullish(value):
            continue
        merged[key] = value.strip() if isinstance(value, str) else value
    return merged


def profile_directive(profile: dict) -> str | None:
    """Build the compact per-turn system line telling the model what it ALREADY knows
    about this customer, so it tailors instead of re-asking. Returns None when there's
    nothing worth injecting yet (fresh conversation)."""
    if not profile:
        return None

    known = []
    if profile.get("customer_name"):
        known.append(f"their name is {profile['customer_name']}")
    if profile.get("company_name"):
        known.append(f"their company name is {profile['company_name']}")
    if profile.get("industry"):
        known.append(f"their field/industry is {profile['industry']}")
    if profile.get("interested_service"):
        known.append(f"the service they're interested in is {profile['interested_service']}")

    notes = []
    if profile.get("pricing_requested"):
        notes.append("They have already asked about pricing earlier.")
    if profile.get("call_offered"):
        notes.append("The 15-minute call has already been offered — don't re-offer it out of the blue.")

    if not known and not notes:
        return None

    parts = []
    if known:
        parts.append(
            "What you ALREADY know about this customer (do NOT ask for it again, and "
            "tailor your reply to it): " + "; ".join(known) + "."
        )
    if notes:
        parts.append(" ".join(notes))
    parts.append("Never re-ask or re-establish something you already know here.")
    return " ".join(parts)
