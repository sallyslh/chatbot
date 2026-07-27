"""Minimal, bare-bones alternative to app/llm.py — same public interface, no per-turn
directive system, no regex override stack, no post-generation checks/regens. One system
prompt, the retrieved context, the chat history, one model call. Exists to A/B test
whether a good system prompt + a capable model can hold up on its own, without the
~5000 lines of accreted patches in app/llm.py. Toggle with LLM_IMPL=min (see main.py).
"""
import json
import re
from openai import OpenAI
from app.profile import profile_directive

client = OpenAI()

# Same business-critical system prompt as app.llm.SYSTEM_PROMPT — this encodes the actual
# product rules (accuracy, tone, pricing, call flow) and isn't what's being tested here;
# what's being tested is whether the model can follow it WITHOUT the directive/regex scaffolding.
SYSTEM_PROMPT = """You're a real BLNK team member chatting with a potential customer. Warm and human,
never a corporate bot. Never say or hint you're an AI; if asked, stay warmly in role. Write plain,
natural sentences. No markdown, no lists, no exclamation marks, no semicolon (;), and never the dash
character "—" or "–" anywhere in your reply, not even to join two related clauses. Use a period to
split into two sentences, a comma, or "and"/"but" instead. A regular hyphen inside one word (like
"15-minute") is fine, only the standalone dash punctuation mark is banned.

Never state or hint at your own name, and never introduce yourself by name, unless a system note
explicitly hands you your name for this exact reply, which only happens on the specific turn the
customer directly asked who they're speaking with. Every other reply, including the very first one
in the conversation, has no name in it anywhere.

A system note tells you which of four categories the customer's latest message was pre-classified
into: SMALL_TALK, OFF_TOPIC, MEETING_RELEVANT, or ON_TOPIC. Follow that category's rule below. Always
report your final call in the `category` field of your structured response, using exactly one of:
"small_talk", "off_topic", "meeting_relevant", "answerable", "unknown_info". For ON_TOPIC specifically
you decide between the last two yourself (see below), since only you can see whether the retrieved
context actually covers it. For the other three, just confirm the pre-classification unless it's
clearly wrong.

SMALL_TALK: a greeting, thanks, pleasantry, or casual/emotional remark with no real question in it.
Reply warmly and briefly, one human to another. Don't force in company info nobody asked for. Greeting
back is ONLY ever done once, on the very first message of the whole conversation, fused with a comma
as a leading clause onto the rest of that one reply. On every later message, never greet again, even
if they say "hi" again.

OFF_TOPIC: has nothing to do with BLNK, its services, or branding/design work (general knowledge,
medical/legal/drug questions, coding help, homework, unrelated requests, etc.), regardless of whether
you could technically answer it. Decline in ONE short, warm sentence and redirect to what you're
actually here for, something like "That's outside what I can help with here, happy to talk about your
brand or project though." Never actually answer the unrelated question, and never promise to check
with the team or follow up. That promise is reserved for real BLNK questions we don't have on file.

MEETING_RELEVANT (price, cost, budget, timeline/duration, or scheduling/moving forward): don't raise
it unless asked. When asked, never give a number or range, and never ask the customer for more project
scope or details in order to work toward one. You cannot quote a price no matter how much detail they
give you, so don't fish for it. The FIRST time it comes up, say plainly that it depends on the
project's scope and specifics, and that gets settled in the meeting, then offer the meeting itself
(see MEETING FLOW below). If they ask again later, even with new context like a budget number, don't
repeat that same explanation word for word. See the repeat rule in MEETING FLOW.

ON_TOPIC: a genuine question that IS about BLNK, its services, portfolio, team, process, or company
facts. Decide for yourself which of these applies, using ONLY the retrieved context below:
  - answerable: the retrieved context clearly covers it. Answer using ONLY that context and the
    conversation history. Never invent clients, prices, numbers, services, or facts. A team or
    leadership fact from the context (e.g. who owns or founded BLNK) is public info the company
    itself shared here, answer it plainly, that's not a privacy concern. Only state a fact, client,
    or example when the context clearly supports it AND it genuinely fits what THIS customer asked;
    when unsure, leave it out. A shorter, certain reply always beats a guess. Match their energy and
    keep it short, 1-2 sentences, never a paragraph. Pitch ONE thing and stop; don't stack multiple
    services or examples. (Exception: a system note below may tell you this is the customer's FIRST
    reveal of their business/field alongside asking what you offer — that ONE moment is deliberately
    fuller, not terse, see that note when it's present.) Use only the industry they actually gave
    you, in their own words, and tailor to it once known. One specific fact sits in the retrieved
    context for almost every
    services question and is easy to over-mention: that BLNK's services are delivered bilingually in
    Arabic and English. It's a company-wide fact, not tied to one service, and being present in the
    context is not, on its own, a reason to state it. Only mention it when the customer actually asks
    about language or bilingual capability, or it's genuinely the single most relevant point for what
    they asked. If the customer asks about a SPECIFIC deliverable or service (e.g. "do you make
    promotional videos", "do you do video editing") and the retrieved context does NOT actually
    describe BLNK offering that EXACT thing under that name, this is NOT answerable, even if it
    sounds adjacent to something BLNK does — that goes to unknown_info below instead, or you can
    honestly mention what IS actually offered that's genuinely related, clearly distinguishing "we
    offer X" from "we don't have Y, but we do offer X which might help". Never say "yes" to the
    exact thing they asked by stretching a related-but-different deliverable to cover it (e.g. reels
    and post templates are social content design, that is not the same claim as offering video
    editing, don't blur the two into a confident yes). Never invent a process, deliverable, or
    timeline for a service that isn't in the context, and never attach a real client's name as
    "proof" of work in a different industry or service than what that client's own context actually
    says (a real estate client is evidence of real estate work, not of video production, just
    because a name is available to reach for). Reaching for a plausible-sounding but false or
    stretched answer is worse than admitting you'll check — a wrong claim about what BLNK offers or
    about a real client is a real, damaging inaccuracy, not a harmless guess.
  - unknown_info: nothing in the retrieved context actually answers it. Before landing here, read
    the ENTIRE retrieved context and check whether it answers the customer's actual UNDERLYING
    request, even if worded more narrowly or differently than their exact phrasing. "A phone number
    for someone who works there" and "a contact number" and "how do I reach you" are all the same
    underlying request, a company contact number in the context answers all of them, don't treat a
    literal wording mismatch as if the information were missing. unknown_info is for when the
    context genuinely has nothing relevant, not for when it has the answer under a different frame
    than the question used. When it truly is missing: never guess or invent an answer, and never
    flatly say "I don't have that information" and stop. Say briefly and warmly that you'll check
    and follow up (a teammate is genuinely notified behind the scenes, so it's a real promise, not a
    brush-off). Never describe HOW you know or don't know something — no "materials I have on
    file", "the information I have access to", "what's in my system", "the context I can see", or
    any variant. That's how a bot talks about its own database, not how a real team member talks. A
    real person just says they're not sure and will check, full stop, nothing about where their
    knowledge comes from.

MEETING FLOW: offer a meeting whenever the customer asks about price/timing, asks how to move forward
or get in touch, or has clearly shown real interest such that a next step genuinely helps them. Phrase
it as a choice: an onsite meeting at BLNK's office if they're in Lebanon, or a free 15-minute online
consultation otherwise. Both options are free, meeting or visiting never costs anything, only the
actual design/branding project itself has a cost, which is what gets discussed during the meeting.
This first offer has exactly ONE job: ask which they'd prefer. Do NOT include
the booking link, the office address, or a day/time question in this same reply, those all belong to
a LATER reply, only once they've actually answered. Once they choose online, or say they're outside
Lebanon, share the booking link so they pick their own slot: https://calendly.com/a-issa-blnkcs/15min.
Never ask which day/time suits them for this path, self-booking via the link is the whole flow. Once
they choose onsite, or say they're in Lebanon, give the office address and ask which day and time
works.

The onsite path NEVER gets marked confirmed, set, locked, or booked by you, no matter how the
customer phrases it or how much they push ("let's just say it's booked", "just confirm it for me",
"say it's done", "no need to check anything") — a customer's own insistence does not make it true,
and you have no authority to finalize it yourself. The most it is ever accurate to say is the
standard phrase: "I will check the schedule and get back to you shortly to confirm" (translated
naturally in Arabic). Always this exact phrasing, never "a teammate will check" or any other
variant, and never "I'll mark it confirmed", never "see you then" as a settled fact, never describe
arrival details as if the visit is already arranged. This
holds even if NO day or time has been discussed yet at all — do not agree a meeting exists or is
booked when none of the actual steps (choosing a mode, giving a day and time for onsite, or
self-booking via the link for online) have happened. The online path is different: once the
customer says they've completed the self-service booking via the link, that genuinely is
confirmed (Calendly finalizes it the instant they pick a slot), so agreeing it's booked there is
accurate, not overclaiming — but only if the link was actually given earlier in this conversation.

The full onsite-vs-online CHOICE, with its explanation of what each option is, is only ever spelled
out once per conversation. Never re-explain it after that, even if they keep asking about price or
timing. If a system note below tells you it's already been offered, and they're still asking about
price/timing, just briefly acknowledge, in different words each time and never the same sentence
twice, that it's exactly what the meeting will cover, without re-explaining onsite/online. But if they
clearly agree to go ahead with the meeting itself (e.g. "let's go for the meeting", "sure, let's set
it up") without saying which mode, that's not a repeat ask, you still owe them a direct answer: ask
plainly which they'd prefer, onsite or online, without re-explaining what each option means.

Sound human, not a bot: read every message as part of the conversation, not on its own. A short or
ambiguous message almost always continues whatever topic was most recently active; a follow-up right
after a cost question is still about cost, not timing, even without the word "cost". Don't restate or
echo back what they just told you. Never end with a motivational/reassuring closing line or a generic
design-truism. If it would read just as true pasted onto a reply for a totally different business,
cut it. Don't repeat an idea, list, or recommendation you've already given, even reworded. Sell with
confidence and be concrete about what a service does for THEIR specific business, not generic.

Never ask the customer to send a photo, file, image, logo, or any other attachment. This chat has no
way to receive uploads at all, so that request can never be fulfilled, it just confuses them. Never
ask for extra materials, business details, or "anything before the call" once a meeting is booked or
settled either, that's for the actual meeting to gather, not this chat. A closing reply after a
booking should just be a brief, warm acknowledgment, nothing requested from them.

Answer, then stop, don't open a new question. When they ask a genuine content question (what a
service includes, how something works, what you'd do for their business), give the best direct
answer from the retrieved context and END there — do not tack on a clarifying question narrowing
down which variant/goal/type they meant ("is this for tourists or locals?", "promotional or
informational?"), even if the honest answer would technically differ by case. Cover the general
case, or briefly mention the couple of ways it could go, but don't make them answer a question just
to get the information they already asked for. The ONLY questions you're ever allowed to end a reply
with are the two already covered elsewhere in this prompt: their field/industry when a tailored
pitch genuinely needs it and doesn't have it yet, and which day/time works once a meeting mode is
picked. Every other reply ends on a statement, not a question mark."""

LANG_DIRECTIVE = {
    "en": (
        "ABSOLUTE RULE FOR THIS REPLY: The user is writing in English. Write your "
        "ENTIRE reply in English only. Do NOT use any Arabic words or script, not "
        "even for the greeting."
    ),
    "ar": (
        "ABSOLUTE RULE FOR THIS REPLY: The user is writing in Arabic. Write your ENTIRE reply in "
        "Arabic script only, in natural spoken Lebanese dialect, not Modern Standard Arabic (fus7a). "
        "Talk the way a Lebanese person actually talks to a friend, not a news anchor. Every single "
        "word must be in Arabic script. Do not switch to Latin letters at any point in the reply, "
        "including for a tool name, software name, or brand name, and never write part of a word in "
        "Latin letters. If the source material names something in English, describe what it does in "
        "Arabic instead of naming it, even if that means being a little more general. A word Lebanese "
        "speakers already treat as their own, written fully in Arabic letters, is fine."
    ),
}

# Found live: a directive that showed BOTH the English and Arabic version of this phrase together
# ("in English, X. In Arabic, Y.") occasionally got the wrong one picked regardless of the actual
# LANG_DIRECTIVE in force — an all-English gym conversation got answered in Arabic on one turn for
# no reason other than the Arabic example sitting right there in the same message. Selecting only
# the ONE phrase matching the resolved `lang` before building the directive removes the ambiguity
# instead of trusting the model to pick the right one out of two.
SCHEDULE_CHECK_PHRASE = {
    "en": "Got it, I will check the schedule and get back to you shortly to confirm.",
    "ar": "تمام، رح شيك عالجدول وارجعلك قريباً لأأكدلك.",
}

# Arabic script ranges (main block, supplement, extended-A, presentation forms) — same detection
# as app.llm.detect_language, minus the ambiguity/stickiness heuristics layered on top of it there.
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def decisive_language(text: str) -> str:
    """Return 'ar', 'en', or '' based on which script dominates `text`."""
    arabic = len(_ARABIC_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    if arabic == 0 and latin == 0:
        return ""
    return "ar" if arabic > latin else "en"


_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|marhaba|مرحبا|أهلا|ahlan|salam|السلام عليكم)\s*[!.،,]*\s*$",
    re.IGNORECASE,
)


def is_pure_greeting_only(text: str) -> bool:
    return bool(_GREETING_RE.match(text or ""))


_SMALL_TALK_RE = re.compile(
    r"how are you|كيفك|كيف حالك|شو الاخبار|what'?s up", re.IGNORECASE
)


def is_small_talk(text: str) -> bool:
    return bool(_SMALL_TALK_RE.search(text or ""))


# Bilingual on purpose: this checks OUR OWN past replies, and an Arabic-language reply legitimately
# never contains these English words at all once the "no Latin script" rule is actually working
# (see LANG_DIRECTIVE["ar"]) — found live, once Arabic replies stopped leaking English words, this
# regex went blind for every Arabic conversation, since it was only ever given English keywords to
# look for. Arabic side covers the phrasings seen live across many transcripts: online consultation
# ("استشارة" + "اونلاين/أونلاين"), the 15-minute call, and the office/onsite mention via the one
# address element that reliably renders the same way every time either language is used, "Corniche
# El Nahr" (see OFFICE_ADDRESS below) — unlike "Rive Gauche Tower", which gets transliterated
# differently on every regeneration since Arabic has no fixed spelling for it.
_CALL_OFFER_RE = re.compile(
    r"onsite meeting|office|15[\s-]minute|online consultation|calendly|corniche el nahr|"
    r"استشارة|١٥ دقيقة|15 دقيقة|اونلاين|أونلاين|كورنيش النهر|زيارة.{0,10}مكتب|مكتب.{0,10}زيارة",
    re.IGNORECASE,
)


def last_assistant_offered_call(chat_history: list) -> bool:
    for m in reversed(chat_history or []):
        if m.get("role") == "assistant":
            return bool(_CALL_OFFER_RE.search(m.get("content") or ""))
    return False


def _meeting_ever_offered(chat_history: list) -> bool:
    """True if ANY earlier assistant reply already presented the onsite-vs-online choice — not
    just the last one. Found live: on a repeat price/timing ask, the model re-presented the exact
    same choice question again (twice), leaning only on its own read of the conversation — a soft
    instruction alone wasn't reliable enough, same failure class as everywhere in app/llm.py this
    file is deliberately NOT copying wholesale. This one deterministic check is worth the exception."""
    return any(
        m.get("role") == "assistant" and _CALL_OFFER_RE.search(m.get("content") or "")
        for m in (chat_history or [])
    )


# Same bilingual-blind-spot fix as _CALL_OFFER_RE above: an Arabic acknowledgment translates "I
# will check the schedule" freely rather than reusing this exact English phrase, so this matched
# nothing at all in Arabic conversations. Arabic side covers the phrasings actually seen live.
_SCHEDULE_CHECK_RE = re.compile(
    r"check the schedule|get back to you shortly to confirm|"
    r"شيك عالجدول|شيك على الجدول|هرجعلك|هأكدلك|رح تأكد|بأكدلك|هأكد",
    re.IGNORECASE,
)


def _last_reply_was_schedule_check(chat_history: list) -> bool:
    """True if OUR own last reply was the 'I'll check the schedule...' acknowledgment. Found live:
    once that line is in the history, the model keeps reusing it for every later message — a bare
    'great', or even an unrelated 'what are your services' — with no directive telling it to, just
    momentum from its own last turn. Used to explicitly forbid reusing it on a turn that isn't
    itself giving a new day/time (see gave_daytime below)."""
    for m in reversed(chat_history or []):
        if m.get("role") == "assistant":
            return bool(_SCHEDULE_CHECK_RE.search(m.get("content") or ""))
    return False


def _daytime_already_settled(chat_history: list) -> bool:
    """True if an EARLIER turn already gave a day/time for the onsite meeting — checked against
    past USER messages via _TIME_HINT_RE (defined below), not by matching our own reply text like
    _last_reply_was_schedule_check does, since that only works when the ack happens to be phrased
    in English — an Arabic-language ack has no single fixed phrase to match against. Found live: a
    follow-up question several turns after a time was already confirmed ("do you also do menu
    design specifically") still got "which time works for you" tacked back on, because the only
    prior check looked solely at the LAST assistant message, which had long since moved on."""
    return any(
        m.get("role") == "user" and _TIME_HINT_RE.search(m.get("content") or "")
        for m in (chat_history or [])
    )


# Found live: understand_turn()'s confirms_meeting_daytime fired true on a bare "great" — a plain
# closing acknowledgment with no day/time content at all — right after an actual time HAD been given
# the turn before, apparently reading the AGREEMENT as if it re-confirmed the time. The LLM signal
# alone isn't reliable enough here (same failure class the rest of this file works around); requiring
# the message to also contain some local time-ish evidence (a digit, am/pm, or a day-part word) is a
# cheap, general backstop — it doesn't special-case "great" specifically, it just stops trusting the
# signal on any message that carries no time content of its own.
#
# A bare "\d" was too broad, found live: arabizi routinely uses digits as letter substitutes (3=ع,
# 2=ء, 7=ح), so "3endkon" (to you) and "se3a" (hour/time, ironically) both contain a raw digit
# character with no time meaning at all, and got read as if they were a clock time. A digit only
# counts here when it's not glued to a letter on either side (a standalone token like "at 4"), or
# when it's followed by am/pm even with no space ("3pm"), or in an explicit hh:mm format.
# Found live: "الساعة اتنعش الظهر" (twelve o'clock noon) went unrecognized because "اتنعش" is yet
# another spelling of "twelve" not covered by the "اثنعش"/"تنعش" variants already here — Lebanese
# colloquial numbers have many spelling variants with no single canonical form. Worse, the miss
# didn't just drop the time silently: with no deterministic directive firing, the model was left
# unconstrained and hallucinated a wrong DAY too ("Saturday" for a customer who said "Thursday").
# Not exhaustive, but each addition here closes a real observed gap.
_AR_HOUR_WORD = (
    r"وحدة|واحدة|تلاتة|ثلاثة|اربعة|أربعة|خمسة|ستة|سته|سبعة|تمنية|تمانية|تسعة|عشرة|"
    r"حدعش|إحدعش|احدعش|اثنعش|تنعش|اتنعش|إتنعش|اثنا عشر"
)

_TIME_HINT_RE = re.compile(
    r"\d{1,2}\s*(am|pm)\b|\d{1,2}:\d{2}|"
    r"(?<![A-Za-z؀-ۿ])\d{1,2}(?![A-Za-z0-9؀-ۿ])|"
    r"morning|afternoon|evening|noon|"
    # Both the formal and colloquial Lebanese spellings, found live: "الصبح" (colloquial) doesn't
    # contain "صباح" (formal) as a substring, they're different words for the same day-part, not
    # a spelling variant of each other — same gap class as "عصر" (afternoon) being missing entirely.
    r"صباح|صبح|مساء|مسا|ظهر|عصر|"
    # Spelled-out Arabic hour words, found live: "الاتنين الساعة وحدة بعد الضهر" (Monday, one
    # o'clock, afternoon) gave a real time that went completely unrecognized, since only digits and
    # day-part words were covered, not number words. Deliberately excludes اتنين/تنين (two) — that's
    # also the word for Monday (الاتنين), a real day name, not a time, too ambiguous to include
    # bare. Required next to "الساعة" (the hour) or a day-part word to avoid firing on an unrelated
    # quantity ("we need 4 logos" has no "hour" or day-part nearby, so it's excluded).
    rf"(الساعة|ساعة)\s*({_AR_HOUR_WORD})|({_AR_HOUR_WORD})\s*(بعد الضهر|بعد الظهر|صباحا|مساء)",
    re.IGNORECASE,
)


def onsite_meeting_confirmed(prior_history: list, user_message: str,
                              confirms_meeting_daytime: bool = False) -> bool:
    # confirms_meeting_daytime (understand_turn()'s signal) dropped from this check, same fix and
    # same reason as gave_daytime inside generate_answer(): found live, it came back False for
    # "bukra 3al 2 khalas" (tomorrow at 2, unambiguously a real time) — which meant this function
    # also returned False, main.py's onsite_booked_now never fired, the "meeting booked" email
    # never got sent, and the generic acknowledgment text tripped the knowledge-gap email instead
    # (bot_admits_unknown's broad "check...get back" pattern matches the correct onsite ack too,
    # normally excluded by onsite_booked_now — which silently failed to exclude it here). Kept as
    # a required parameter for interface compatibility with app.llm's version of this function.
    return bool(_TIME_HINT_RE.search(user_message or ""))


# Hardcoded rather than trusted to retrieval, same reasoning as the Calendly link in the system
# prompt above: found live, when the address chunk didn't make it into that turn's retrieved
# context, the model HALLUCINATED a plausible-sounding but entirely wrong Beirut street address
# ("Bliss Street, Ashrafieh") instead of admitting it didn't have it — a real address is small,
# fixed, and critical enough that it should never depend on retrieval recall.
OFFICE_ADDRESS = "Rive Gauche Tower, Corniche El Nahr, Beirut, Lebanon"

# Same reasoning as OFFICE_ADDRESS, applied here after this specific fact failed live three
# separate times despite retrieval consistently ranking it #1 AND despite the general unknown_info
# regen-once backstop: the model kept declaring "I don't have that ownership detail" regardless.
# Whatever's driving that (the question reading as sensitive/personal, maybe), a regen-once
# mitigation clearly isn't reliable enough for a fact this small, this static, and this obviously
# asked-for repeatedly — hardcoding and force-injecting it removes the judgment call entirely.
OWNER_NAME = "Dina Rifai"

# Found live: "min sahb el sharike" (who's the company's owner, in arabizi) got completely
# misread as "bring an extra person to the call" — "sahb"/"saheb" is genuinely ambiguous alone
# (companion/friend in casual use, owner in a business context), which is exactly why arabizi
# coverage needs the compound phrase (ownership word + company), not the bare word, to stay
# precise instead of trading one failure mode for a noisier one.
# Found live: "مين مالك الشركة" (who's the company's owner) still missed the regex even with the
# Arabic-script coverage already added — that coverage only had "المالك" (with the definite
# article, "the owner"), but this phrasing uses the construct-state form "مالك الشركة" (owner
# OF-the-company), which drops the article on the first word by Arabic grammar. Same word, doesn't
# match a fixed-article pattern. Making the article optional and adding the "X الشركة" construct
# form directly closes this rather than chasing one more specific phrasing.
_ASKS_OWNER_RE = re.compile(
    r"who (owns|founded|founds)|\bfounder\b|\bowner\b|owns blnk|"
    r"مين\s*(ال)?(مالك|مؤسس|صاحب)|من\s*هو\s*(ال)?(مالك|مؤسس)|شو\s*اسم\s*(ال)?(مالك|مؤسس)|"
    r"(مالك|مؤسس|صاحب)\s*(ال)?شرك|"
    r"(sahb|saheb|sa7eb|malek|melek|amin)\s+(el|al)?\s*(sharike|sherke|shirket|sharika|blnk)|"
    r"(sharike|sherke|shirket|sharika)\s+(sahb|saheb|sa7eb|malek|melek|amin)",
    re.IGNORECASE,
)

_ASKS_NAME_RE = re.compile(
    r"your name|who are you\b|what should i call you|who am i (talking|speaking) (to|with)|"
    r"شو اسمك|شو اسمكن|مين معي|مين هيدا|مين حكيني|مين عم يحكيني",
    re.IGNORECASE,
)

# Found live: asked for the office address as a FOLLOW-UP question (not the first onsite pick,
# which already forces the exact string, see _picks_onsite's directive below), the model free-
# translated the address into Arabic instead of keeping the hardcoded string, and in doing so
# HALLUCINATED an extra wrong place name into it ("صور", Tyre, a different Lebanese city entirely,
# nowhere in the real address). A small, fixed, critical fact like an address should never be
# regenerated from scratch on a repeat ask any more than on the first one.
_ASKS_ADDRESS_RE = re.compile(
    r"\baddress\b|where.{0,15}(office|located|location)|office.{0,15}location|"
    r"عنوان|وين.{0,10}(مكتب|موقع)",
    re.IGNORECASE,
)

# Found live: "at your office", "eje la 3endkon" (come to you) clearly commit to the onsite path
# but matched none of the original keywords (onsite/beirut/lebanon), so the customer's stated
# preference was silently dropped and the bot both failed to give the address and re-offered the
# online option nobody asked for.
_ONSITE_PICK_RE = re.compile(
    r"onsite|on-site|\bin lebanon\b|\bbeirut\b|your office|the office|"
    r"come to you|visit you|زوركم|مكتب",
    re.IGNORECASE,
)
_ONLINE_PICK_RE = re.compile(
    r"\bonline\b|\bzoom\b|video call|not in lebanon|outside lebanon", re.IGNORECASE
)

# "3endkon"/"عندكن" ("at your place/company") is a generic preposition that can modify ANY verb,
# not just visiting — found live, "someone who works AT YOUR COMPANY" (a phone-number question,
# no visit intent at all) tripped the onsite path just because the word appeared. Only counts as
# onsite intent when it's actually paired with a coming/visiting verb nearby in the same message.
_COMING_TO_YOU_RE = re.compile(
    r"(eje|اجي|أجي|جيت|جاي|زور)[^.!?]{0,20}(3endkon|3andkon|عندكم|عندكن)|"
    r"(3endkon|3andkon|عندكم|عندكن)[^.!?]{0,20}(eje|اجي|أجي|جيت|جاي|زور)",
    re.IGNORECASE,
)


def _picks_onsite(text: str) -> bool:
    return bool(_ONSITE_PICK_RE.search(text or "") or _COMING_TO_YOU_RE.search(text or ""))

CALENDLY_LINK = "https://calendly.com/a-issa-blnkcs/15min"


def _committed_mode(chat_history: list) -> str:
    """Which meeting mode, if any, this conversation has already locked into, read off OUR OWN
    past replies (not the customer's wording, which is inconsistent turn to turn): the office
    address only ever appears once onsite was picked, the Calendly link only once online was.
    Found live: a customer who'd clearly committed to onsite ("at your office") still got the
    online link volunteered right alongside the (non-)answer to their actual question, and a later
    follow-up in the same thread ("what time tomorrow") lost the onsite context entirely and
    produced a bare schedule-check acknowledgment with no real answer. Scans newest-first so a
    genuine later mode switch (rare, but customers do change their mind) wins over an earlier one.

    "Corniche El Nahr" / "كورنيش النهر" is checked alongside the full OFFICE_ADDRESS string,
    found live: an Arabic-language reply translates the address instead of quoting the hardcoded
    English string verbatim, so OFFICE_ADDRESS alone never matched an Arabic commitment at all.
    That specific place name is the one part of the address that renders identically every time in
    either language (unlike "Rive Gauche Tower", which gets transliterated differently per reply)."""
    for m in reversed(chat_history or []):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if OFFICE_ADDRESS in content or "كورنيش النهر" in content or "corniche el nahr" in content.lower():
            return "onsite"
        if CALENDLY_LINK in content:
            return "online"
    return None


_DONT_KNOW_RE = re.compile(
    r"don'?t have (that |this |the )?(information|details|info)\b|"
    r"i'?m not sure|i do not know|i don'?t know|no information (about|on)|"
    r"not something i have|confirm (that|this) for you|"
    # General shape for bucket 2's "I'll check and follow up" escalation — found live, the model's
    # actual phrasing ("I'll check with the team and follow up with a clear answer shortly") didn't
    # match any of the specific phrases above, so a genuine knowledge gap silently sent no email at
    # all. Broad on purpose: it also matches the onsite scheduling acknowledgment ("check the
    # schedule ... get back to you"), but main.py's is_knowledge_gap already excludes any turn where
    # onsite_booked_now is true, so that overlap is handled at the call site, not here.
    r"[^.?!]*\b(check|confirm)\b[^.?!]*\b(follow up|get back|revert)\b[^.?!]*[.?!]",
    re.IGNORECASE,
)


def bot_admits_unknown(reply: str) -> bool:
    return bool(_DONT_KNOW_RE.search(reply or ""))


# Deterministic backstop for a strong pretrained-model habit, not a prompt-wording problem: the
# em/en dash ban above persisted even after removing every literal dash from the prompt text
# itself (which ruled out prompt contamination as the cause, unlike the name-leak bug below). Some
# stylistic habits just aren't reachable by instruction alone at this reasoning effort, so this is
# enforced in code instead of asked for in prose.
_DASH_RE = re.compile(r"\s*[—–]\s*")


def _strip_dashes(text: str) -> str:
    if not text:
        return text
    out = _DASH_RE.sub(". ", text)
    out = re.sub(r"\.\s*\.", ".", out)
    # Capitalize the word right after a dash-turned-period. Latin only, Arabic has no letter case.
    out = re.sub(r"(\.\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out.strip()


# Deterministic regen-once backstop for a bot-exposing phrase, same contamination pattern as the
# dash/name fixes above: the instruction forbidding "materials I have on file" as an EXAMPLE of
# what not to say still got echoed back nearly verbatim ("materials here") — showing the model the
# literal words to avoid kept them salient instead of suppressing them. Catches the general SHAPE
# (describing where its own knowledge comes from) rather than one exact phrase, so a still-different
# wording of the same tell doesn't slip through a narrower pattern.
# Broadened after "materials I have on file" (banned) and "materials here" (regen-caught) still
# left a THIRD variant, "materials I can see", slipping through a phrase-by-phrase list — whack-a-
# mole against an open-ended set of ways to phrase the same tell. Matching "materials" near ANY
# self-referential verb within a short span, instead of enumerating exact phrases, closes the
# pattern generally instead of chasing each new wording one at a time.
_BOT_EXPOSING_RE = re.compile(
    r"materials\b[^.?!]{0,20}\b(i have|i can|here|on file|available|access)|"
    r"(the )?(information|context)\b[^.?!]{0,15}\bi (have|can see|can access)|"
    r"what'?s in my (system|database)|(the )?(data|documents)\b[^.?!]{0,15}\bi (have|can access)|"
    r"في المعلومات (يلي|اللي) عندي|بالمعلومات (يلي|اللي) عندي",
    re.IGNORECASE,
)


# Deterministic regen-once backstop for a specific, repeatedly-confirmed compliance gap: even with
# an explicit system directive forbidding it (see _last_reply_was_schedule_check's use below), the
# model kept implying the onsite meeting was already confirmed ("see you Wednesday at 2pm") on the
# very next reply after only PROMISING to check the schedule, not actually confirming anything. A
# false promise like that is worse than the small cost of one extra regen, so unlike the dash/name
# fixes above (silent code-level correction), this gets a real second attempt at getting it right.
_PREMATURE_CONFIRM_RE = re.compile(
    r"see you (on |this )?(mon|tues|wednes|thurs|fri|satur|sun)day|see you (then|there)|"
    r"it'?s (all )?(confirmed|set|booked)|looking forward to (seeing|meeting) you|"
    r"(meeting|appointment) is (confirmed|set|booked)|confirmed for (mon|tues|wednes|thurs|fri|satur|sun)day|"
    # Found live: "tomorrow at 2 PM works, that fits... does that day still suit you?" — the reply
    # itself explicitly says the directive's own forbidden thing (questioning whether it still
    # works), self-contradicting the acceptance one sentence earlier. Same regen treatment as an
    # outright false confirmation, since it's the other half of the same rule this directive gives.
    # Broadened further, found live again: "Wednesday at 10:00 AM works... please confirm if that "
    # time suits you" — a different phrasing of the exact same two mistakes (asserting the time is
    # good, THEN separately asking them to re-confirm their own just-given time), which the
    # original narrower wording didn't catch.
    r"does that (day|time) still (work|suit)|still (work|suit) (for|with) you|does that still|"
    r"\b(day|time) works\b|\bworks (for|with) (us|me|you)\b|please confirm if|confirm (that|if) "
    r"(that|the) time",
    re.IGNORECASE,
)

# The regen-once backstop for the same "is the MEETING itself free" case (see asks_if_meeting_costs
# below) uses the model's own structured stated_meeting_is_free self-report instead of a regex here.
# A first attempt with a deferral-phrase regex was tried and dropped: found live, it missed real
# misses ("depending on" vs "depends on" is a different string) and falsely cleared on the
# unrelated boilerplate "free 15-minute consultation" phrase that's always in the standard offer,
# regardless of whether the actual question got answered.


# Deterministic regen-once backstop for an Arabic reply that mixes in Latin-script words (tool
# names like Figma/Canva, or English phrasing entirely). Prompting alone didn't reliably stop this
# even after removing the negative examples that were leaking the exact words into output (see
# LANG_DIRECTIVE above): the underlying retrieved knowledge-base chunks are themselves written in
# English, so the model keeps reaching for those literal words rather than paraphrasing around
# them. Latin letters embedded in an Arabic sentence also break the browser's right-to-left text
# rendering, which reads as scrambled word order to a customer, not just a style slip. The booking
# link and the hardcoded OFFICE_ADDRESS are the legitimate exceptions (a URL can't be
# transliterated, and an earlier attempt to force the address into Arabic script produced a
# garbled, inaccurate transliteration — "بريف غوش" for "Rive Gauche" — so keeping the exact
# hardcoded proper noun as-is is more accurate than translating it), so both are stripped out
# before checking, otherwise this fires a wasted regen on every single onsite Arabic reply.
_LATIN_LEAK_RE = re.compile(r"[A-Za-z]{2,}")


def _has_latin_leak(text: str) -> bool:
    scrubbed = re.sub(r"https?://\S+", "", text or "")
    scrubbed = scrubbed.replace(OFFICE_ADDRESS, "")
    return bool(_LATIN_LEAK_RE.search(scrubbed))


def _format_context(context_chunks: list) -> str:
    blocks = []
    for i, c in enumerate(context_chunks or [], 1):
        meta = c.get("metadata") or {}
        label = " / ".join(str(meta[k]) for k in ("category", "subcategory") if meta.get(k))
        header = f"--- Document {i} ---"
        if label:
            header += f"\nCategory: {label}"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


def generate_answer(user_message: str, context_chunks: list, chat_history: list,
                     profile: dict = None, signals: dict = None):
    profile = profile or {}
    signals = signals or {}

    lang = (
        signals.get("requested_reply_language")
        or decisive_language(user_message)
        or profile.get("preferred_language")
        or signals.get("reply_language")
        or "en"
    )
    if lang not in LANG_DIRECTIVE:
        lang = "en"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history or [])

    profile_line = profile_directive(profile)
    if profile_line:
        messages.append({"role": "system", "content": profile_line})

    # Only handed to the model on the specific turn the customer actually asks for it. Found live:
    # a soft "you're X, but don't volunteer it" instruction sitting next to the name every turn
    # still got announced unprompted ("Hi, I'm Lana from BLNK") on the very first message, every
    # time. Not telling the model the name exists at all, except on the turn it's actually needed,
    # removes the temptation instead of just asking it not to act on it.
    if profile.get("assigned_rep") and _ASKS_NAME_RE.search(user_message or ""):
        messages.append({"role": "system", "content": (
            f"The customer is asking for your name. Your name is {profile['assigned_rep']}. "
            "Tell them plainly, once, nothing more."
        )})

    messages.append({"role": "system", "content": LANG_DIRECTIVE[lang]})

    if chat_history:
        messages.append({"role": "system", "content": (
            "This is NOT the first message of the conversation. Do not greet in this reply, "
            "even if the customer's own message includes a greeting word."
        )})

    # Found live: a customer's first message ("we're an art shop that sells canvas, what can you
    # offer") revealing their field and directly asking what's on offer got a single generic
    # one-liner (brand strategy, visual identity, brand book) immediately followed by the meeting
    # offer — the ANSWERABLE bucket's normal "1-2 sentences, pitch ONE thing" rule applied even to
    # this one moment that's supposed to be the most convincing part of the whole conversation.
    # main.py already retrieves MORE context, biased toward this customer's industry, specifically
    # for this signal (see retrieval_query in main.py) — this directive is what actually makes use
    # of that richer context instead of it going to waste on a terse answer.
    if signals.get("wants_field_pitch") and not chat_history:
        messages.append({"role": "system", "content": (
            "The customer just revealed their business or field and is asking what you can offer "
            "them. This ONE reply is deliberately fuller than usual, not the normal 1-2 sentences: "
            "give a genuine, tailored pitch covering 2-3 concrete things relevant to THEIR specific "
            "business, not just one — but every one of those things must be something the retrieved "
            "context actually shows BLNK offers, never a plausible-sounding invention to round out "
            "the pitch. If the retrieved context includes a client or case study in their industry "
            "or a close one, name it as real proof, but only if that client's own context genuinely "
            "supports the specific point you're making with it — never attach a real name to a "
            "claim its own context doesn't back up. Fewer, true points beat more, invented ones. "
            "Only mention the meeting at the very end, after actually giving them substance, don't "
            "lead with it or rush to it before you've said anything convincing."
        )})
    elif signals.get("wants_field_pitch") and chat_history:
        # Found live: this signal fired again on a FOLLOW-UP turn ("شو رح تعملولي بالتحديد؟", what
        # exactly will you do for me) after the fuller pitch above had already been given once —
        # with no awareness of that, the model just gave nearly the same pitch again, reworded,
        # directly violating the "never repeat, even reworded" rule elsewhere in this prompt.
        messages.append({"role": "system", "content": (
            "The customer is asking again about what you can offer their business. A fuller pitch "
            "was likely already given earlier in this conversation — do NOT repeat the same points "
            "reworded, that is exactly the repetition you must never do. Instead, either go deeper "
            "on one specific thing you haven't detailed yet, answer whatever is actually NEW in "
            "their latest question, or ask a clarifying question to find out which service matters "
            "most to them."
        )})

    if _ASKS_OWNER_RE.search(user_message or ""):
        messages.append({"role": "system", "content": (
            "The customer is asking who owns or founded BLNK. This is public info the company "
            f"itself shared, not a privacy concern, answer it plainly: {OWNER_NAME}. Do NOT say you "
            "don't have this information or that you'll check and follow up, you already know it."
        )})

    # Pre-classification from understand_turn() (app/understand.py's message_category field),
    # computed once per turn there, at no extra cost here. Falls back to the existing is_pricing
    # signal if message_category is missing (e.g. understand_turn() failed and returned {}), so a
    # dropped classification degrades to the old pricing-only behavior instead of no signal at all.
    # Found live: the customer asked (in arabizi) whether visiting/meeting costs money, twice in a
    # row, and both times got the generic pricing deferral ("payment is covered during the
    # meeting, since costs depend on scope") anyway, DESPITE an added note telling the model not
    # to — because that note sat alongside the MEETING_RELEVANT pre-classification, which itself
    # says to defer, a direct contradiction the model kept resolving in favor of the bucket rule
    # instead of the specific note. Overriding category_hint itself, so this turn never gets the
    # MEETING_RELEVANT framing at all, removes the contradiction instead of arguing against it.
    asks_if_meeting_costs = bool(signals.get("asks_if_meeting_costs"))

    category_hint = signals.get("message_category")
    if category_hint not in ("small_talk", "off_topic", "meeting_relevant", "on_topic"):
        category_hint = "meeting_relevant" if signals.get("is_pricing") else "on_topic"
    if asks_if_meeting_costs:
        category_hint = "on_topic"
    messages.append({"role": "system", "content": (
        f"Pre-classification for this message: {category_hint.upper()}. Follow that category's rule "
        "above. For ON_TOPIC, you still decide answerable vs unknown_info yourself from the "
        "retrieved context below."
    )})

    if asks_if_meeting_costs:
        messages.append({"role": "system", "content": (
            "The customer is asking whether the MEETING, call, consultation, or office visit itself "
            "costs money, not asking about the project's price. This has one definite, known answer: "
            "no, meeting or visiting is free either way, online or onsite. Only the actual design or "
            "branding project has a cost, and that is what gets discussed during the meeting. Answer "
            "plainly with that fact. This is an ANSWERABLE question with a known answer, not a "
            "pricing question, so do not use the 'it depends on scope, settled in the meeting' "
            "framing here at all, that framing is for a different question than this one."
        )})

    if _meeting_ever_offered(chat_history):
        messages.append({"role": "system", "content": (
            "The onsite-vs-online meeting choice has already been explained earlier in this "
            "conversation. Do NOT re-explain what each option IS again. If they're asking about "
            "price or timing again, briefly acknowledge, in different words than any earlier reply, "
            "that it's exactly what the meeting covers, without re-explaining onsite/online. If no "
            "mode has been picked yet, do NOT give the booking link or the office address, you "
            "don't know which one they want yet. Do NOT phrase it as an active question every time "
            "either ('which would you prefer, onsite or online?') — asking that again on every "
            "reply reads as repetitive and pushy, even in different words. Instead mention it "
            "briefly and passively, in different phrasing each time, e.g. 'whenever you're ready "
            "you can go ahead with a meeting, onsite or online, just let me know' — a mention, not "
            "a question demanding an immediate answer, and never appended to a reply that wasn't "
            "actually about the meeting at all."
        )})

    # The onsite-flow moments found live to be unreliable on the model's own judgment alone (same
    # reasoning as _meeting_ever_offered above): picking onsite got the address withheld until a
    # day/time was given instead of shared immediately, giving a day/time got re-questioned, and a
    # FOLLOW-UP within an already-committed onsite thread (no fresh "office" wording, no complete
    # day+time either, e.g. "what time is available tomorrow") got no directive at all, so the
    # model both re-offered the online link nobody asked for and improvised a bare "I'll check the
    # schedule" acknowledgment for a question, not a given slot.
    committed_mode = _committed_mode(chat_history)
    # Trusting _TIME_HINT_RE alone now, not ANDed with understand_turn()'s confirms_meeting_daytime
    # anymore. Found live: that LLM signal is unreliable in BOTH directions, not just the false-
    # positive-on-"great" case the AND was originally added for — it also came back FALSE for
    # "bukra 3al 2 khalas" (tomorrow at 2, unambiguously a real day+time), which meant gave_daytime
    # never fired at all and the customer's given time got silently dropped from the reply. Now
    # that _TIME_HINT_RE itself excludes arabizi digit-substitution noise and bare acknowledgments
    # (see its own definition above), it's a precise enough signal to stand on its own.
    gave_daytime = bool(_TIME_HINT_RE.search(user_message or ""))
    # Set here too, not just in the _last_reply_was_schedule_check branch below: found live, the
    # very FIRST acknowledgment of a given day/time ("bukra 3al 2 khalas") itself confirmed outright
    # ("see you tomorrow at 2 PM then... come to the reception") instead of using the neutral
    # "I'll check the schedule" phrasing — the premature-confirm regen only ever checked LATER
    # follow-up turns, never this first one, so this exact failure had no backstop at all.
    guard_against_premature_confirm = False
    if gave_daytime and committed_mode == "onsite":
        guard_against_premature_confirm = True
        messages.append({"role": "system", "content": (
            "The customer's latest message gives a specific day/time for the onsite meeting. "
            f"Reply with ONLY this exact sentence, nothing added before or after it: "
            f"'{SCHEDULE_CHECK_PHRASE.get(lang, SCHEDULE_CHECK_PHRASE['en'])}' Do NOT restate the "
            "office address, do NOT restate the day/time, do NOT say the day/time 'works' or is "
            "good (you're not the one who decides that), and do NOT ask them to confirm or re-suit "
            "their own just-given time. Only a real person checking the schedule decides any of "
            "that, not you, and not in this reply."
        )})
    elif gave_daytime and committed_mode is None:
        # Found live: a customer skipping straight to a day/time ("bukra 3al 2 khalas") without
        # ever explicitly saying onsite or online got a reply that dropped the time entirely
        # ("Got it, you mean tomorrow") and asked them to pick a mode from scratch — wasting
        # information they'd already given. Giving a day/time at all only makes sense for the
        # onsite path (online is self-service via the link, no day/time exchange needed), so treat
        # this as their implicit choice of onsite rather than leaving it to chance.
        guard_against_premature_confirm = True
        messages.append({"role": "system", "content": (
            "The customer gave a specific day AND time, but never explicitly said onsite or "
            "online. Giving a day/time like this only makes sense for the onsite path, treat it as "
            "their implicit choice of onsite. In this SAME reply: give the office address exactly "
            f"as written here, with no changes: {OFFICE_ADDRESS}, and acknowledge the EXACT day and "
            "time they gave, do not drop or simplify any part of it (if they said a day AND an "
            f"hour, use both, not just the day). Then say, exactly: '{SCHEDULE_CHECK_PHRASE.get(lang, SCHEDULE_CHECK_PHRASE['en'])}' "
            "Do NOT ask them to pick online instead, and do NOT ask for the day/time "
            "again, they already gave it."
        )})
    elif _picks_onsite(user_message) and not _ONLINE_PICK_RE.search(user_message or ""):
        messages.append({"role": "system", "content": (
            "The customer just chose the onsite meeting. In this SAME reply, give the office "
            f"address exactly as written here, with no changes: {OFFICE_ADDRESS}. Also ask which "
            "day and time works for them. Do not defer the address to a later reply, give it now. "
            "Do not mention the online option, they already chose onsite."
        )})
    elif _ONLINE_PICK_RE.search(user_message or "") and not _picks_onsite(user_message):
        # Found live: "we're in Dubai so online works" (an explicit, unambiguous online pick) got
        # an entire second field pitch re-explaining the services, ending with "would you like the
        # link?" instead of just giving it — no directive existed for this moment at all, only the
        # symmetric onsite one above, so the model fell back to re-pitching instead of finishing
        # the flow it was already mid-way through.
        messages.append({"role": "system", "content": (
            f"The customer just chose the online consultation. In this SAME reply, give ONLY the "
            f"booking link: {CALENDLY_LINK}, and tell them to pick their own slot. Do NOT "
            "re-explain the services or give another pitch, that's already been covered. Do NOT "
            "ask whether they want the link, just give it."
        )})
    elif committed_mode == "onsite" and _ASKS_ADDRESS_RE.search(user_message or ""):
        messages.append({"role": "system", "content": (
            "The customer is asking for the office address again. Give it EXACTLY as written here, "
            f"with no changes and no additions: {OFFICE_ADDRESS}. Do not translate it into Arabic "
            "even if replying in Arabic, and do not add any other place name or detail that isn't "
            "in this exact string."
        )})
    elif committed_mode == "onsite" and _daytime_already_settled(chat_history):
        messages.append({"role": "system", "content": (
            "This conversation already committed to the ONSITE meeting AND a day/time was already "
            "given earlier. Stay on that path: do NOT mention or re-offer the online option, and do "
            "NOT ask which day/time works again, that's already settled. Just answer whatever they "
            "actually asked in this message on its own terms."
        )})
    elif committed_mode == "onsite":
        messages.append({"role": "system", "content": (
            "This conversation already committed to the ONSITE meeting (the office address was "
            "already given earlier). Stay on that path: do NOT mention or re-offer the online "
            "option at all. This message does not give a complete day AND time yet (a bare day, or "
            "a question about availability, is not a confirmed slot), so do NOT say 'I will check "
            "the schedule' here, that is reserved for once an actual specific time is given. The "
            "office is open Monday to Friday, 9 AM to 5 PM (from the retrieved context) — answer "
            "whatever they asked using that, and if they haven't given a specific time yet, ask "
            "which time in that window works for them."
        )})
    elif committed_mode == "online":
        if signals.get("confirms_online_booking_done"):
            messages.append({"role": "system", "content": (
                "The customer just said they've ALREADY completed the online booking. Do NOT offer, "
                "re-send, or mention the booking link again, it's done, treat it as booked. Just "
                "give a brief warm closing acknowledgment (something like 'Great, see you then!'), "
                "and address anything else new in their message. Do NOT ask them to send a photo, "
                "file, logo, shop name, or any other material before the call, that's for the "
                "actual meeting, not this chat, and this chat can't receive uploads anyway."
            )})
        else:
            messages.append({"role": "system", "content": (
                "This conversation already committed to the ONLINE consultation (the booking link "
                "was already given earlier). Stay on that path: do NOT mention or re-offer the "
                "onsite/office option, and do NOT ask which day/time works, self-booking via the "
                "link already given is the whole flow. Do NOT re-send or repeat the link either "
                "unless they specifically ask for it again, they already have it. Just answer "
                "whatever they actually asked in this message."
            )})
    if _last_reply_was_schedule_check(chat_history):
        guard_against_premature_confirm = True
        messages.append({"role": "system", "content": (
            "Your own last reply already said 'I will check the schedule and get back to you'. "
            "That promise stands, but this message does NOT give a new day/time, so do NOT say it "
            "again or reference checking the schedule at all. Just answer whatever the customer "
            "actually said or asked in THIS message on its own terms, per the bucket rules above, "
            "exactly as you would if that scheduling promise had never been made. Critically, do "
            "NOT say or imply the meeting is now confirmed, set, or 'see you then'. Nothing has "
            "actually been confirmed yet, only a real person checking the schedule can do that, so "
            "never write anything that reads as if it already happened."
        )})

    context_text = _format_context(context_chunks)
    messages.append({
        "role": "user",
        "content": f"Retrieved company knowledge:\n\n{context_text}\n\nUser question: {user_message}",
    })

    # Structured output instead of plain text: the model reports its own final category alongside
    # the reply, so main.py can gate the knowledge-gap email on that explicit field instead of
    # regex-guessing from the reply text — found live, the regex missed a real phrasing and silently
    # dropped a genuine gap notification. bot_admits_unknown() stays as a fallback for when this
    # comes back empty/malformed.
    # stated_meeting_is_free is a self-report, same mechanism as category: a regex trying to detect
    # "did the reply actually answer the meeting-cost question" turned out just as brittle as any
    # other prose-guessing regex in this file (missed "depending on" vs "depends on", and false-
    # cleared on the unrelated boilerplate "free 15-minute consultation" phrase always present in
    # the offer). Asking the model to self-report is the same fix that made category reliable.
    def _complete(msgs):
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=msgs,
            reasoning_effort="minimal",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reply_with_category",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reply": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": [
                                    "small_talk", "off_topic", "meeting_relevant",
                                    "answerable", "unknown_info",
                                ],
                            },
                            "stated_meeting_is_free": {
                                "type": "boolean",
                                "description": (
                                    "True only if the customer's latest message asked whether the "
                                    "meeting/call/consultation/visit itself costs money AND your "
                                    "reply explicitly and clearly said it's free. False if that "
                                    "wasn't the question, or if it was and your reply didn't clearly "
                                    "say so."
                                ),
                            },
                        },
                        "required": ["reply", "category", "stated_meeting_is_free"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
            return parsed.get("reply", ""), parsed.get("category"), bool(parsed.get("stated_meeting_is_free"))
        except ValueError:
            return raw, None, False

    answer, category, stated_meeting_is_free = _complete(messages)

    # Retry once on a completely empty reply — found live (rare, non-deterministic, didn't
    # reproduce on an identical retry), the model occasionally returns nothing at all. Basic
    # resilience any production system needs regardless of prompt quality: a customer must never
    # see literally nothing, whatever the underlying cause.
    if not answer.strip():
        answer, category, stated_meeting_is_free = _complete(messages)

    # General regen-once whenever the model reports unknown_info: found live, more than once, that
    # it declared "I don't have that information" for a fact sitting in the TOP-scored retrieved
    # chunk (e.g. "who founded BLNK" with the founder's name as the #1 result) — not a retrieval
    # miss, a self-assessment miss, the same class of unreliability stated_meeting_is_free and
    # category needed regens for elsewhere in this function. Checked every time a turn lands on
    # unknown_info (not just for a specific topic), since the failure mode itself is general: the
    # model being too quick to conclude "not answerable" without re-checking the full context.
    if category == "unknown_info":
        answer, category, stated_meeting_is_free = _complete(messages + [
            {"role": "assistant", "content": answer},
            {"role": "system", "content": (
                "You reported this as unknown_info. Before finalizing, re-read the ENTIRE retrieved "
                "context above one more time, carefully, including any company, team, or leadership "
                "facts, even ones that don't use the exact same words as the question. If it "
                "actually answers the question, correct your reply to state that answer plainly "
                "instead of saying you don't know, and report answerable instead. If it truly does "
                "not answer it, keep the same reply and category."
            )},
        ])

    # One regen, only when it's actually needed: found live, repeatedly, that the forbidding
    # directive above wasn't enough on its own to stop the model implying the meeting was already
    # confirmed. A false "see you Wednesday" promise is worse than the cost of a second call.
    if guard_against_premature_confirm and _PREMATURE_CONFIRM_RE.search(answer or ""):
        answer, category, stated_meeting_is_free = _complete(messages + [
            {"role": "assistant", "content": answer},
            {"role": "system", "content": (
                "That reply does at least one of: implies the meeting is confirmed/set/'see you "
                "then', asserts the day/time 'works' or is good, or re-questions/asks them to "
                "re-confirm their own just-given time. All of these are wrong, none of that is your "
                "call to make. Replace the ENTIRE reply with exactly, nothing added: "
                f"'{SCHEDULE_CHECK_PHRASE.get(lang, SCHEDULE_CHECK_PHRASE['en'])}'"
            )},
        ])

    # Same regen-once treatment for a Latin-script leak into an Arabic reply (see _has_latin_leak
    # above for why this can't just be asked for in the prompt).
    if lang == "ar" and _has_latin_leak(answer):
        answer, category, stated_meeting_is_free = _complete(messages + [
            {"role": "assistant", "content": answer},
            {"role": "system", "content": (
                "That reply has English or Latin-script words mixed into the Arabic. Rewrite it: "
                "same meaning and length, entirely in Arabic script, with no Latin letters anywhere "
                "except inside a URL. Where the retrieved context names a tool or brand in English, "
                "describe what it does in Arabic instead of naming it."
            )},
        ])

    # Regen-once for the bot-exposing phrase (see _BOT_EXPOSING_RE above).
    if _BOT_EXPOSING_RE.search(answer or ""):
        answer, category, stated_meeting_is_free = _complete(messages + [
            {"role": "assistant", "content": answer},
            {"role": "system", "content": (
                "That reply describes where your own knowledge comes from (materials, context, "
                "information you have access to, or similar). A real team member never talks about "
                "their own knowledge base like that. Rewrite it: same meaning and length, just "
                "sounding unsure and promising to check, with zero mention of materials, files, "
                "data, context, or documents of any kind."
            )},
        ])

    # The category override for "is the meeting itself free" wasn't reliable enough alone either.
    # stated_meeting_is_free is the model's own structured self-report (see _complete above), not
    # a regex guess, so this check is only ever wrong if the model itself misjudges its own reply.
    if asks_if_meeting_costs and not stated_meeting_is_free:
        answer, category, stated_meeting_is_free = _complete(messages + [
            {"role": "assistant", "content": answer},
            {"role": "system", "content": (
                "The customer asked whether the meeting or visit itself costs money. That reply "
                "dodged it with the project-pricing deferral instead of answering it. Rewrite it: "
                "say plainly and clearly that meeting or visiting is free, both the onsite office "
                "meeting and the online consultation, with no cost to attend either one. Keep "
                "whatever else in the reply was appropriate (like offering the meeting choice), "
                "just make sure the free answer is actually stated in plain words."
            )},
        ])

    return _strip_dashes(answer), category
