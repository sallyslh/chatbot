# ONE model pass that UNDERSTANDS the turn and returns structured signals. This is the generic
# alternative to regex/keyword matching for natural-language understanding: the model reads the
# customer's message (any language, any arabizi spelling) and tells us the reply language, whether
# it's a pricing question, and the profile/lead fields. The backend then acts on these signals
# deterministically. Because comprehension lives here (not in brittle regexes), a new phrasing we
# never saw before "just works" — no lexicon to extend.
#
# It replaces the old separate scoring call, so the per-turn model-call budget is unchanged
# (this pass + the generation pass). If it fails, callers fall back to the regex helpers in llm.py.
from openai import OpenAI
import json

client = OpenAI()

UNDERSTAND_PROMPT = """You analyze a customer chat for a creative/branding agency and output ONLY a
JSON object about the LATEST user message. No markdown, no prose, just the JSON. Fields:

- reply_language: the language THIS reply should be written in — either "en" (English) or "ar"
  (Arabic script). Note: a customer writing "arabizi" (spoken Lebanese/Levantine Arabic in Latin
  letters and numbers, e.g. "kifak", "addesh se3rkon", "fye aarf aktr") should get "en" — BLNK
  replies in plain English to arabizi input, never in arabizi. Only pick "ar" if the customer is
  writing in actual Arabic script. A lone English loanword like "logo" or "consulting" inside an
  Arabic message does NOT make it English.
- is_pricing: true if the latest message asks about price, cost, budget, fees, or how much
  something costs — in ANY language or spelling (e.g. "how much", "se3r", "addesh", "taklofe",
  "l asaar", "بكم") — OR asks about project timing/duration (e.g. "how long does it take",
  "adde badda wa2et", "turnaround time", "kam yom"). Both get treated identically downstream, so
  either kind of question sets this field true. Else false.
- asks_if_meeting_costs: true if the customer is specifically asking whether the MEETING, CALL,
  CONSULTATION, or VISIT itself costs money or requires payment (e.g. "do I have to pay to visit
  you", "lezim edfa3 la2eno ejtme3", "is the consultation free", "fi dafe3 lal meeting", "shu bekalef
  el ijtimaa") — this is a narrower, different question from is_pricing (which is about the
  project/service's own cost), and both can be true on the same compound message. Else false.
- confirms_online_booking_done: true if the customer is saying they have ALREADY completed the
  self-service online booking (e.g. "hajazet", "خلص حجزت", "booked it", "done, I picked a slot",
  "I already scheduled it") — a statement that the booking already happened, not a request or
  intent to book. Else false.
- is_confirmation: true if the LATEST message is JUST a simple confirmation, agreement, or
  acknowledgment with NO new substantive content of its own — it's only agreeing to or continuing
  something already on the table, not introducing anything new (e.g. "okay", "yes", "sure", "deal",
  "sounds good", "tamam", "mne7", "akid", "eh", "d'accord", "sí" — ANY language or spelling, this list
  is illustrative only). This matters because a bare confirmation carries NO signal about which
  language to reply in — it should simply continue whatever language the conversation has already
  been using, never flip it. Else false.
- industry: the customer's business/field if they have stated it (e.g. "restaurant",
  "clothing store"), else null. Never guess, and never substitute a related-but-different category
  just because it shares some association with what they said — decode precisely into what they
  actually meant. If you're not genuinely confident what their field actually is, output null
  rather than a nearby guess — a missing field is far better than a wrong one.
- interested_service: the agency service they're asking about if clear (e.g. "Brand Identity",
  "Digital Design", "Print & Collateral", "Social Media Design"), else null.
- wants_field_pitch: true if in the LATEST message the customer reveals or refers to their own
  business field/industry AND would benefit from a tailored pitch of how the agency can help that
  field (e.g. "I have a real estate company, how can you help", "we're a restaurant, what can you
  do for us"). This marks the key moment to convince them with a fuller, industry-specific answer.
  ALSO true when the LATEST message is just a bare confirmation ("yes"/"eh"/"akid"/"ايه") directly
  answering a question the ASSISTANT's own last message asked about the customer's field (e.g.
  assistant: "3andak real estate company?", customer: "eh") — the field is now confirmed and the
  agency still owes them the actual pitch, so this counts the same as a fresh reveal. Else false.
- wants_more_detail: true if the customer is asking a genuine, standalone question wanting to LEARN
  ABOUT or UNDERSTAND one SPECIFIC named service in more depth (e.g. "tell me more", "aarf aktar 3an
  el brand identity", "aktar tafasel", "explain more", "how does that work", "can i know more about
  digital design"). This applies whether that exact service was already discussed earlier in the
  conversation OR this is the first time it comes up — either way they want real information about
  THAT one service, not a sales pitch, not a meeting. NOT this if the service name is only mentioned
  in passing while asking about something else entirely — e.g. "how much does brand identity cost"
  is a pricing question only (is_pricing), not this, even though it names a service; they're asking
  its price, not to learn about it. A broad "what services do you offer" is also NOT this — that's
  asks_services_overview, a menu request, not a deep-dive into one named service. Else false.
- asks_services_overview: true if the customer is asking BROADLY which services/offerings BLNK has,
  rather than about one specific service (e.g. "what do you offer", "shu btaadmo/bta3mlo", "shu el
  khadamet 3andkon", "what services do you provide", "shu bta3mlo bl 3ere2"). Even if phrased with
  "aarf aktar" (know more), if the ask is the whole menu it's this, not wants_more_detail. Else false.
- asks_client_industries: true if the customer is asking broadly which INDUSTRIES, SECTORS, or types
  of businesses BLNK has worked with before (portfolio breadth, past clients in general — e.g. "what
  industries have you worked with", "what kind of clients do you have", "shu el majalat yali
  3imilto ma3ha") — distinct from asks_services_overview, which is about what SERVICES BLNK offers,
  not who its clients are. Do not conflate the two: "what do you do" is asks_services_overview; "who
  have you worked with" / "what fields/industries" is this one. Else false.
- wants_contact: true ONLY if, with nothing else going on in the message, the customer is asking IN A
  GENUINELY NEW WAY how to get in touch, reach BLNK, or move forward (e.g. "how can we get in touch",
  "how do I contact you", "let's move forward", "kifak momken net waselkon"). This is a narrow signal
  for a fresh ask, not a general bucket for "anything related to the meeting/next-steps." Once a
  meeting has already been discussed in this conversation, do NOT set this true just because the
  message is broadly on that topic — false whenever the message actually does something more
  specific instead, even if that something relates to the meeting: asking a direct factual question
  (e.g. "where are you located" — that has its own direct answer, it's not a "how do I reach you"
  ask), giving or confirming a day/time, explicitly asking to book/schedule the meeting itself
  (already past "how do I reach you" — they're telling you to proceed), or commenting/complaining
  about the conversation itself (e.g. telling you that you're repeating yourself). Also false whenever
  it's covered by wants_field_pitch/wants_more_detail (learning more first, not booking). When in
  doubt, prefer false — a missed genuine contact-request is far less costly than this signal
  swallowing messages that actually need their own specific handling.
- confirms_meeting_daytime: a MECHANICAL check, not a vibe check — do not let an agreement/approval
  tone ("would be good", "works", "sounds good", "fine") influence this at all. Ask only: does the
  LATEST message contain an actual clock-time marker — a specific hour/number used as a time (e.g.
  "4", "3pm"), am/pm, or a day-part word (morning/afternoon/evening/noon/sobh/masa/صباح/مساء)? If
  YES, true. If NO — even if a day name is present (Monday, tomorrow, bkra) AND even if the message
  sounds enthusiastic/confirming/affirmative about that day ("monday would be good", "monday works
  for me", "tuesday is fine") — the answer is false, full stop. A day name plus an approval word is
  still just a DAY, zero time information; do not treat "would be good"/"works"/"fine" as if they
  supplied a time, they didn't. If no meeting is being discussed at all, false.
- customer_name: their PERSONAL name if they gave it, else null.
- company_name: their COMPANY/BUSINESS name if they gave it (e.g. "BLNK Boutique", "Kor and More") —
  distinct from customer_name (a person's name) and from industry (their field, e.g. "clothing
  store"). Only set this from an actual stated name, never guess or infer one from the field alone.
  Else null.
- budget_signal: "low", "medium", "high", or null.
- timeline: their timeline as a short string, or null.
- lead_score: integer 0-100 for how qualified/promising this lead looks.
- conversation_stage: where the conversation now stands after the latest message — one of
  "greeting" (just saying hi), "discovery" (learning who they are / their business),
  "service_discussion" (talking about a specific service in detail), "qualification"
  (budget, timeline, scope), "consultation_offer" (the free call/booking is on the table),
  or "closing" (wrapping up or booked). Pick the single best fit; never guess wildly — if
  unsure, use "discovery".
- answers_call_mode_or_location: true if a meeting/call has already been offered (onsite vs. online)
  earlier in this conversation, and the LATEST message engages with that in ANY of these ways: (a)
  explicitly choosing onsite/online, (b) stating where they ARE (in Lebanon, or a different
  country/city, in any phrasing, e.g. "i am in dubai", "not in lebanon", "i'm in beirut"), (c)
  explicitly asking to book/schedule/confirm the meeting itself, even without naming a specific mode
  (e.g. "book an appointment", "احجز موعد", "when can I book a meeting", "let's schedule it"), or (d)
  asking a direct, specific, answerable question ABOUT the meeting/logistics itself — whether a
  particular mode is even available ("do you have an in-person option?", "is there an onsite
  option?"), how/when they can book, or where the previously-mentioned booking link/option actually
  is ("where's the link you mentioned"). ALL of these are genuine, substantive engagement with the
  pending choice that deserves a real, specific answer — not a fresh "how do I reach you" request of
  its own, and NOT something to deflect with a vague callback either, even though it can otherwise
  read like wanting contact or continuing a pricing conversation. Else false (including when no
  meeting has been offered, or the message is about something unrelated to that choice).
- requested_reply_language: "en" or "ar" if the customer is EXPLICITLY asking you to switch to,
  reply in, or use a specific language for the conversation itself (e.g. "can you explain in
  Arabic", "switch to English please", "احكيني بالعربي", "reply in arabic from now on") — this is a
  META-request about which language YOU should reply in, not just a statement about their own
  language ability (e.g. "I don't speak English well" is NOT this — that's unaddressed_intent) and
  not just a message that happens to be written in that language. Only set this when the request is
  explicit and specifically about the reply language. Else null.
- message_category: classify the LATEST message into EXACTLY ONE of these four buckets — this drives
  which handling path the backend takes, so pick carefully:
  * "small_talk" — a greeting, thanks, pleasantry, or casual/emotional remark with no real question
    in it. No lookup needed, just a warm human reply.
  * "off_topic" — has nothing to do with BLNK, its services, or branding/design work at all (general
    knowledge, medical/legal/drug questions, coding help, homework, requests to role-play as
    something else, etc.), regardless of whether you could technically answer it — the point is it
    doesn't belong in a creative agency's business chatbot. When in doubt, do NOT pick this — only
    use it when the message is clearly and entirely unrelated to BLNK's business.
  * "meeting_relevant" — asks about price, cost, budget, or timeline/duration, or is otherwise about
    scheduling, moving forward, or the consultation/meeting itself.
  * "on_topic" — any other genuine question that IS about BLNK, its services, portfolio, team,
    process, or company facts — whether or not you personally know if we have the answer on file.
    This is the default for any real business question that isn't specifically about price/timing.
- notes: short string.
- customer_intent: a short, plain-English paraphrase of what the LATEST message actually says or
  asks, regardless of what language it's in. Always fill this in, in your own words — don't pick
  from a list. This is the general-purpose reading of the message that the fields above (being a
  fixed checklist) can't capture on their own.
- unaddressed_intent: true if what the customer actually said (customer_intent) is NOT already
  fully captured by is_pricing, wants_field_pitch, wants_more_detail, asks_services_overview,
  wants_contact, or is_confirmation — i.e. this message needs its own direct response rather than
  being handled as one of those standard cases. This covers ANY kind of message those categories
  don't anticipate: the customer saying they don't understand you or the language you're replying
  in, a complaint or expression of frustration, a question or request that's unrelated to the
  categories above (even if you've genuinely never seen this exact kind of message before — use
  your own judgment, don't require it to match a known pattern), asking for a human, or anything
  else novel. False whenever the message genuinely IS one of the standard categories above, or is
  a bare acknowledgment with nothing new to address.

Use null / false whenever something is not clearly present — never guess."""


def understand_turn(user_message: str, chat_history: list) -> dict:
    """Return the structured understanding of this turn, or {} if the call/parse fails (callers
    then fall back to the regex helpers). Never raises."""
    convo = "\n".join(
        f"{m.get('role')}: {m.get('content','')}" for m in (chat_history or [])[-8:]
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": UNDERSTAND_PROMPT},
                {"role": "user", "content": f"Conversation so far:\n{convo}\n\nLatest user message:\n{user_message}"},
            ],
            # gpt-5-mini only supports the default temperature (1) — passing 0 (the old
            # gpt-4o-mini value, used for deterministic extraction) is a 400 Bad Request.
            # reasoning_effort="minimal" (2026-07-24, user's explicit latency request): this is
            # structured extraction, not open-ended reasoning — measured live, "minimal" cuts this
            # call's latency substantially with no observed drop in signal quality on the same inputs.
            reasoning_effort="minimal",
        )
        raw = resp.choices[0].message.content or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(raw[start:end + 1])
    except Exception:
        return {}
