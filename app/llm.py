#  SETUP & SERVICE CATEGORIES — read from the dataset, not hardcoded
# ──────────────────────────────────────────────────────────────────────────────
#from the user question, generate an answer using the context chunks and chat history
import os
import re
import json
from pathlib import Path
from openai import OpenAI
from app.profile import profile_directive
client = OpenAI()

# The "what services do you offer" reply names BLNK.'s service categories. Rather than
# hardcode that list (which then silently drifts from data/blnk_chunks.jsonl), we read it
# from the indexed data at startup: a service chunk is tagged metadata.category == "services"
# and its text reads "BLNK. service - <Name> (...)", so we pull <Name> from each, in order.
# Edit the dataset, rebuild the index (python -m app.embed_index), restart — the menu follows.
_META_PATH = Path(__file__).resolve().parent.parent / "data" / "blnk_meta.json"

def load_service_categories():
    try:
        meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        meta = []
    names = []
    for m in meta:
        if (m.get("metadata") or {}).get("category") == "services":
            text = m.get("text", "")
            if "BLNK. service -" in text:
                name = text.split("BLNK. service -", 1)[1].split("(")[0].split(",")[0].strip()
                if name and name not in names:
                    names.append(name)
    return names

def load_all_client_names():
    """Every real client name in the dataset, read from blnk_meta.json at startup (same source as
    load_service_categories above) — used to deterministically detect when a client already cited
    earlier in a conversation gets cited again, even on a turn whose OWN retrieved context doesn't
    happen to include that client's chunk (unlike _client_names_in_context, which only sees the
    current turn's retrieval)."""
    try:
        meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        meta = []
    names = []
    for m in meta:
        name = (m.get("metadata") or {}).get("client")
        if name and name not in names:
            names.append(name)
    return names


ALL_CLIENT_NAMES = load_all_client_names()

# Last-resort fallback ONLY if the data file can't be read at all (abnormal state) — so a
# services question never produces an empty, category-less answer. Normal runs use the data.
SERVICE_CATEGORIES = load_service_categories() or [
    "Brand Identity", "Digital Design", "Print & Collateral", "Social Media Design"
]
SERVICE_CATEGORIES_TEXT = ", ".join(SERVICE_CATEGORIES)


# ──────────────────────────────────────────────────────────────────────────────
#  FEW-SHOT STYLE EXAMPLES  ·  THIS IS WHERE YOU IMPROVE THE BOT'S LEBANESE
# ──────────────────────────────────────────────────────────────────────────────
# The model copies whatever examples it's shown. These gold replies are injected into every
# generation call as example turns, so the model IMITATES this exact dialect and tone. On a small
# model this is the single most effective lever for clean Lebanese Arabic-script replies.
#
# HOW TO MAKE THE BOT TALK BETTER: just edit the lists below.
#   • Each item is {"user": "<what the customer wrote>", "assistant": "<the ideal reply>"}.
#   • The `user` line sets the situation; the `assistant` line is the gold reply to sound like.
#   • Keep replies SHORT (1-2 sentences), no exclamation marks, no lists — same rules as the live bot.
#   • 4-8 examples per language is a good range. An item with an empty "assistant" is ignored,
#     so you can leave blank slots.
#   • The "ongoing" loop: pull real Arabic-script chats (their language is logged in analytics), have
#     a Lebanese native rewrite any that sound off, and paste the corrected reply in here as a new
#     item. (Arabizi-INPUT chats are logged/replied as "en" now — see the "en" list below if that
#     segment's tone ever needs its own tuning.)
#
# ⚠️ The examples below were drafted by an AI and MAY contain dialect mistakes. Have a Lebanese
#    native review/replace them before you rely on this — otherwise you teach the model your own bugs.
STYLE_EXAMPLES = {
    "ar": [
        {"user": "مرحبا، عندي محل تياب وبدي حسّن شكل البراند",
         "assistant": "أهلا، منقدر نكون عم نعملك  لوغو، و ادارة لحساباتك عمواقع التواصل الاجتماعي، وبزنس كاردز يخلّو البراند يبيّن أكتر."},
        {"user": "بدي اعرف اكتر عن السوشال ميديا",
         "assistant": "منصمّملك بوستات وريلز بتناسب جمهورك و بتخلّي صفحتك تبيّن احترافية، ومنمشي عهوية بصرية موحّدة بكل المحتوى."},
        {"user": "كم سعركن",
         "assistant": "السعر بيعتمد عالسكوب والتفاصيل يلي بتحتاجها. عنا خيارين: منقدر نعمل لقاء بالمكتب ببيروت "
                      "إذا كنت بلبنان، أو استشارة أونلاين مجانية مدتها ١٥ دقيقة إذا ما فيك تجي."},
    ],
    # English replies no longer need dialect coaching, but "en" now also covers customers who type
    # in arabizi and get a plain English reply (2026-07-17 decision) — if that segment's replies
    # ever feel flat/corporate rather than warm, this is the lever: add a couple of casual, friendly
    # English examples here, same mechanism as the "ar" list above.
    "en": [],
}


def _few_shot_messages(lang: str) -> list:
    """Turn the gold STYLE_EXAMPLES for `lang` into example message turns, wrapped in system markers
    so the model treats them as style demos to imitate — not as part of the live conversation.
    Returns [] when there are no filled examples, so blank slots are a safe no-op."""
    pairs = []
    for ex in STYLE_EXAMPLES.get(lang) or []:
        assistant = (ex.get("assistant") or "").strip()
        if not assistant:
            continue  # unfilled slot — skip
        user = (ex.get("user") or "").strip()
        if user:
            pairs.append({"role": "user", "content": user})
        pairs.append({"role": "assistant", "content": assistant})
    if not pairs:
        return []
    return (
        [{"role": "system", "content":
          "Here are examples of the IDEAL reply style for this language — study ONLY their tone, "
          "dialect, spelling, and sentence rhythm and match those closely. The specific business, "
          "field, deliverables, and wording used INSIDE each example belong to that example's own "
          "fictional customer only — never reuse, copy, or let them leak into the real reply (e.g. an "
          "example about a restaurant mentioning food must never cause the word 'food'/'akel' to "
          "appear in a reply about an unrelated business). They are NOT part of the current "
          "conversation; the real reply's content must come only from the real customer's own field "
          "and the retrieved context."}]
        + pairs
        + [{"role": "system", "content": "End of style examples. The real conversation follows."}]
    )


# ──────────────────────────────────────────────────────────────────────────────
#  LANGUAGE DETECTION
# Arabic script ranges (main block, supplement, extended-A, presentation forms).
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Best-effort net (not exhaustive — refine as new phrasings turn up live, same as every other
# free-text-judging regex in this file) for the bot's OWN reply admitting it doesn't know the
# answer. This exists ALONGSIDE the retrieval-score gap check in main.py (KNOWLEDGE_GAP_SCORE),
# not instead of it: found live — "who is the owner" retrieved the right chunk (Dina Rifai) at
# rank #1, comfortably above the score threshold, and the model STILL declined to use it and said
# it didn't have the information. A low retrieval score means the knowledge base has nothing
# relevant; THIS catches the separate case where relevant context was retrieved but the model
# didn't use it — main.py checks both signals before deciding whether to escalate to the admin.
_DONT_KNOW_RE = re.compile(
    r"don'?t have (that |this |the )?(information|details|info)\b|"
    r"i'?m not sure|i do not know|i don'?t know|no information (about|on)|"
    r"not something i have|i can'?t (confirm|answer|say|share)|"
    # Found live: a NEW refusal phrasing not covered above ("I can't share personal details about
    # the owner") for a fact that was genuinely sitting in the retrieved context (the founder's
    # name) — the model treating a public company fact as if it were private info to withhold.
    r"can'?t share (personal|private)|won'?t share (personal|private)|"
    r"لا أملك|ما عندي معلومات|مش متأكد|مش عندي معلومات|ما في معلومات|مش متوفر عندي|"
    r"ma3ande ma3loumat|mesh mit2akked|mesh 3ande ma3loumat",
    re.IGNORECASE,
)


def bot_admits_unknown(reply: str) -> bool:
    # Also matches the standing "I'll confirm that for you shortly" escalation hedge (_CONFIRM_
    # FOLLOWUP_RE, defined further below and reused here rather than duplicated) — SYSTEM_PROMPT's own
    # instructed phrasing for a genuine gap, and the strongest possible self-admission there is that
    # the bot doesn't have the answer. Without this, a genuine gap only got emailed to the team when
    # retrieval ALSO happened to score low — if a chunk scored just above KNOWLEDGE_GAP_SCORE but the
    # model still couldn't actually answer from it and fell back to this hedge, the gap went unlogged
    # and no one followed up, exactly the silent-gap failure this whole check exists to prevent.
    return bool(_DONT_KNOW_RE.search(reply or "") or _CONFIRM_FOLLOWUP_RE.search(reply or ""))

# NOTE: "arabizi" (Arabic written in Latin letters — "kifak", "kam se3rkon") is understood fine on
# the INPUT side — the day/time regexes and affirmation regexes below still recognize arabizi
# spellings, and the LLM comprehension pass (understand.py) reads it natively regardless of this
# file. This is deliberate product decision (2026-07-17): we no longer generate arabizi OUTPUT —
# a customer typing in arabizi now simply gets a plain English reply, same as a customer typing in
# actual English. That removed the need for a whole detection lexicon here: any Latin-script message
# (arabic == 0) is just "en" now, full stop, whether it's real English or arabizi. Only Arabic-SCRIPT
# text still gets its own "ar" output mode.
def detect_language(text: str) -> str:
    """Return 'ar' (Arabic script), 'en' (Latin script — English or arabizi, both reply in
    English), or '' (unknown) based on which style dominates `text`."""
    arabic = len(_ARABIC_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    if arabic == 0 and latin == 0:
        return ""  # no letters at all (emoji, numbers) — can't tell
    return "ar" if arabic > latin else "en"

# The model keeps replying in Modern Standard Arabic (fus7a) even with the dialect directive, so
# for Arabic-script replies we ALSO check the output deterministically. These are high-precision
# fus7a tells that spoken Lebanese never uses: the accusative tanwin (ـً, as in تمامًا/احترافيًا/
# صورًا) and MSA-only function words. We match نحن only when NOT followed by ا (so dialect نحنا
# doesn't trip it), and only the pronoun-suffixed forms of يمكن (يمكننا/يمكنك…), since bare يمكن
# also means "maybe" in dialect. A hit triggers ONE regeneration in dialect; a miss lets the reply
# stand — so a false negative just costs nothing, never a wrong answer.
_FUSHA_RE = re.compile(
    r"ً|"                                     # accusative tanwin — strong MSA tell
    r"نحن(?!ا)|"                               # MSA 'we' (dialect: نحنا / منـ)
    r"يمكن(?:نا|كما|كم|كن|ك|هما|هم|هن|ها|ه)|"  # conjugated 'can' (dialect: فينا / بنقدر)
    r"الذي|التي|الذين|اللذين|اللواتي|"         # MSA relatives (dialect: اللي)
    r"سوف|لقد|حيث|كذلك|"                       # MSA-only particles
    r"هؤلاء|أولئك|ذلك"                          # MSA demonstratives (dialect: هودي / هيداك)
)


def looks_like_fusha(text: str) -> bool:
    return bool(_FUSHA_RE.search(text or ""))

# Common English function words. Their presence marks a message as genuinely English,
# versus a lone loanword ("consulting", "logo") that Arabic/arabizi speakers use inside
# their own language and which must NOT flip the reply language.
_EN_FUNCTION_WORDS = frozenset(
    "the a an you your yours do does did what which when where how why is are was were be "
    "can could should would will want wanna need i we us our my me have has had tell give "
    "show know more this that these those and but about with for please thanks thank "
    # Common prepositions were missing entirely — found live: "monday at 3", "yes iam in lebanon"
    # have no OTHER function word and are under the 5-word length fallback, so both got classified
    # as ambiguous despite being unmistakably English, letting a stale established language (e.g.
    # arabizi from much earlier in a long conversation) wrongly resurface on a clearly-English reply.
    "in at on to from of by as".split()
)


def _has_en_function_word(text: str) -> bool:
    """True if the message contains an actual English function word ('you', 'what', 'the'...)
    — the one signal that reliably marks a Latin-script message as a genuine English sentence
    rather than a stray loanword ("logo", "consulting") sitting inside an Arabic conversation."""
    words = re.findall(r"[A-Za-z]+", (text or "").lower())
    return any(w in _EN_FUNCTION_WORDS for w in words)


def _is_clearly_english(text: str) -> bool:
    """True only when a Latin-script message reads like an actual English sentence — it
    contains an English function word or is fairly long — not just a loanword or two."""
    words = re.findall(r"[A-Za-z]+", (text or "").lower())
    if not words:
        return False
    if _has_en_function_word(text):
        return True
    return len(words) >= 5


def _is_ambiguous_english(text: str) -> bool:
    """A Latin message that shouldn't be trusted to switch the language to English:
    a bare affirmation ('ok') or a lone loanword pick ('consulting')."""
    return is_affirmation(text) or not _is_clearly_english(text)


def decisive_language(text: str) -> str:
    """Return the language ('en' or 'ar') ONLY when `text` decisively establishes it —
    otherwise ''. A bare 'ok' or a lone loanword like 'consulting' is not decisive, so it
    won't be allowed to flip the stored conversation language. Used by the backend to update
    the sticky preferred_language on the profile."""
    lang = detect_language(text)
    if not lang:
        return ""
    if lang == "en" and _is_ambiguous_english(text):
        return ""
    return lang


def _resolve_language(user_message: str, chat_history: list, established_lang: str = None) -> str:
    """Resolve the reply language ('en' or 'ar'). Detect from the current message, but fall
    back to the conversation's established language when the current message can't decide it
    — no letters (just '?'), a bare affirmation ('ok'), or a lone loanword like 'consulting'
    that would otherwise wrongly flip an established Arabic chat. `established_lang` is the
    profile's sticky language, used as the anchor when neither the current message nor the
    recent history can decide."""
    lang = detect_language(user_message)
    # Strong stickiness: in a chat already established as Arabic, a message merely DETECTED as
    # English only actually switches to English if it contains a real English function word —
    # a single loanword mention ("logo") must NOT flip the whole conversation.
    if established_lang == "ar" and lang == "en" and not _has_en_function_word(user_message):
        return established_lang
    ambiguous = (not lang) or (lang == "en" and _is_ambiguous_english(user_message))
    if ambiguous:
        for m in reversed(chat_history):
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            prev = detect_language(content)
            # inherit the last turn that clearly establishes a language (skip prior bare
            # affirmations / lone loanwords, which are just as ambiguous)
            if prev and not (prev == "en" and _is_ambiguous_english(content)):
                return prev
        if established_lang:
            return established_lang
    return lang or established_lang or "en"  # final fallback


# ──────────────────────────────────────────────────────────────────────────────
#  LANGUAGE DIRECTIVES
# Prompt rules keep getting ignored by the model, so we ALSO enforce the reply
# language deterministically: we detect the language of the user's message and
# inject an absolute per-turn directive. This is not left to the model's judgment.
LANG_DIRECTIVE = {
    "en": (
        "ABSOLUTE RULE FOR THIS REPLY: The user is writing in English. Write your "
        "ENTIRE reply in English only. Do NOT use any Arabic words or script, not "
        "even for the greeting."
    ),
    "ar": (
        "ABSOLUTE RULE FOR THIS REPLY: The user is writing in Arabic. Write your "
        "ENTIRE reply in Arabic script, with normal Arabic word order, start to finish — no "
        "Latin-script word or phrase anywhere in the sentence, and never an English word in "
        "parentheses as a gloss/translation (e.g. 'كتيب إرشادي للهوية (brand book)' or 'للـsocial' "
        "are WRONG — the Latin insertion breaks the sentence's natural order). A term that's commonly "
        "said as a loanword in spoken Lebanese (logo, brand book, social) still gets written OUT in "
        "Arabic script, transliterated, the way it actually sounds said out loud ('لوغو', 'براند "
        "بوك', 'سوشال ميديا') — never in Latin letters, never bracketed as a side translation.\n"
        "Write in spoken Lebanese dialect, NOT Modern Standard Arabic (fus7a). Reply the way a "
        "Lebanese person actually speaks day to day — before each word, ask whether a Lebanese "
        "person would really say it out loud to a friend, or whether it sounds like a news anchor "
        "or a textbook; if it sounds like fus7a, use the everyday spoken Lebanese equivalent "
        "instead. Keep it warm, casual, and natural — spoken Lebanese in tone, not written Arabic. "
        "Use everyday spoken forms like نحنا، منعرف، فينا نساعدك، اللي، هيك، هالشي، منشتغل، بيفرق، "
        "أكتر — never the fus7a equivalents نحن، نعرف، يمكننا مساعدتك، الذي، هكذا، نعمل، يختلف، أكثر. "
        "This applies to WORD CHOICE too, not just grammar — a formal/literary word can be wrong even "
        "in a grammatically-dialect sentence. E.g. for 'content calendar', say 'كالندر' (the loanword "
        "Lebanese speakers actually use), never 'تقويم' (bookish/MSA, sounds like a Hijri calendar, "
        "not a content-planning tool) — when in doubt, an English loanword Lebanese speakers actually "
        "use in daily conversation beats a formal Arabic word they'd never say out loud."
    ),
}

# Injected on a retry when an Arabic-script reply came out in fus7a instead of Lebanese dialect.
FUSHA_FIX_DIRECTIVE = (
    "Your previous reply was in Modern Standard Arabic (fus7a), which is not allowed here. Rewrite "
    "the SAME message, same meaning, in natural spoken Lebanese dialect — the way people actually "
    "talk. Use dialect forms (نحنا/منـ، فينا، اللي، هيك، هالشي، رح، مش) instead of fus7a (نحن، "
    "يمكننا، الذي، سوف، ليس، هذه), and remove every accusative tanwin (no ـً endings like تمامًا/"
    "احترافيًا/صورًا). Keep it short and casual."
)

# Deterministic backstop for a generation stutter — the SAME word landing twice back-to-back in one
# reply (e.g. "khidma mna7a w mna7a 3ala el social media"). This is a different bug from
# _check_repetition below, which only compares a draft against EARLIER TURNS — a within-sentence
# double-up needs its own check. Caught live in arabizi (unstandardized spelling makes the small
# model noisier there than in clean English), but the regex is language-agnostic — \w matches
# Arabic-script letters too — so it's checked on every language for free.
_STUTTER_RE = re.compile(r"\b(\w{2,})\b(?:\s+(?:w|و|and))?\s+\1\b", re.IGNORECASE)


def _find_stutter(text: str) -> str:
    m = _STUTTER_RE.search(text or "")
    return m.group(1) if m else ""


def stutter_fix_directive(word: str) -> str:
    return (
        f"Your previous reply repeated the word \"{word}\" twice in a row right next to each other "
        f"(a generation glitch, e.g. \"mna7a w mna7a\"). Rewrite the SAME reply, same intent and "
        f"length, saying it only once."
    )


# Deterministic backstop for the standing "never use also/كمان" rule above — the model still uses it
# despite the prompt saying not to (found live: "منقدر كمان نساعدك..." on an elaboration turn). This
# is a literal word ban, not a fuzzy judgment call, so a plain regex check is the right tool here —
# no need for an LLM call to decide whether the word is present.
_CONTINUATION_WORD_RE = re.compile(r"\balso\b|\badditionally\b|\bas well\b|\bkam[ae]n\b|كمان", re.IGNORECASE)


def _find_continuation_word(text: str) -> str:
    m = _CONTINUATION_WORD_RE.search(text or "")
    return m.group(0) if m else ""


def continuation_word_fix_directive(word: str) -> str:
    return (
        f"Your previous reply used \"{word}\", which must NEVER appear in a reply — not as an opener, "
        f"not mid-sentence. Rewrite the SAME reply, same intent and length, restructured so each point "
        f"is stated directly with no \"also\"/\"additionally\"/\"as well\"/\"kaman\"/\"كمان\" connecting "
        f"it to anything else."
    )


# ──────────────────────────────────────────────────────────────────────────────
#  GREETING POLICY
# The system prompt already tells the model to greet back FIRST when the customer's message
# includes a greeting (even alongside a question in the same message), but — like the other
# rules on this page — the model doesn't always comply. So we enforce it deterministically:
# detect a real greeting word in the user's message, and if the model's reply doesn't already
# open with one, prepend it. Scoped to unambiguous greeting words only (not "kifak", which
# doubles as "how are you" and would risk answering a question they didn't ask).
_GREETING_ALTERNATION = (
    r"hii+|hi|hello|hey|hiya|greetings|"
    r"ahla|ahlan|marhaba|marhba|hala|halla2|hala2|"
    r"مرحبا|أهلا|اهلا|هلا|السلام\s*عليكم"
)
_GREETING_RE = re.compile(rf"^\W*(?:{_GREETING_ALTERNATION})\b", re.IGNORECASE)


def _has_greeting(text: str) -> bool:
    return bool(_GREETING_RE.search(text or ""))


def _reply_starts_with_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match((text or "").strip()))


# Deliberately STRICTER than _has_greeting above (which only checks a greeting is PRESENT, even
# alongside a real question in the same message) — this requires the ENTIRE message to be JUST the
# greeting, nothing else. Used by main.py's latency fast-path (user's explicit request, 2026-07-23):
# a bare "hii" has no pricing/field/contact signal to extract and nothing to retrieve, so it's safe
# to skip understand_turn() and retrieve() entirely for this narrow case and go straight to a fast
# reply — but "hii what are your services" must NOT take this path, it has a real question attached.
_PURE_GREETING_RE = re.compile(rf"^\W*(?:{_GREETING_ALTERNATION})\b[\s!.,،]*$", re.IGNORECASE)


def is_pure_greeting_only(text: str) -> bool:
    return bool(_PURE_GREETING_RE.match((text or "").strip()))


# A comma, never a period (user's explicit 2026-07-20 correction — "Hii" must never be followed by
# a dot): the greeting is a leading clause fused onto the substantive answer with a comma, not its
# own period-separated sentence.
GREETING_PREFIX = {"en": "Hii, ", "ar": "مرحبا، "}


def _ensure_greeting(text: str, user_message: str, lang: str) -> str:
    """Prepend a short greeting clause when the customer greeted and the model's reply didn't
    already include one."""
    if not text or not _has_greeting(user_message) or _reply_starts_with_greeting(text):
        return text
    prefix = GREETING_PREFIX.get(lang, "Hii, ")
    if lang == "en" and text[:1].isascii() and text[:1].isalpha():
        return prefix + text[0].lower() + text[1:]
    return prefix + text


# Deterministic backstop for the reverse problem: the prompt tells the model to greet only on the
# customer's greeting and the model still greets on later turns anyway (a bare "Hii, " tacked onto an
# unrelated reply). A greeting is only ever appropriate ONCE, on the conversation's very first
# assistant reply — so on every later turn we strip a leading greeting word outright, regardless of
# whether the customer happened to greet again. Consumes trailing comma/period/whitespace either way
# — greetings are comma-fused ("Hii, <answer>") but this stays lenient about a stray period too, in
# case the model or an older stored reply used the previous period-separated format.
_GREETING_LEAD_RE = re.compile(rf"^\W*(?:{_GREETING_ALTERNATION})\b[\s,،.]*", re.IGNORECASE)


def _strip_leading_greeting(text: str) -> str:
    if not text:
        return text
    stripped = _GREETING_LEAD_RE.sub("", text, count=1)
    if not stripped:
        return text  # the whole reply was the greeting — leave it rather than blank
    if stripped[0].isascii() and stripped[0].isalpha():
        return stripped[0].upper() + stripped[1:]
    return stripped


# ──────────────────────────────────────────────────────────────────────────────
#  CLIENT-CITATION POLICY
# ──────────────────────────────────────────────────────────────────────────────
# When/how a past client may be named as proof was independently restated, with drifted wording,
# across FIELD_PITCH_DIRECTIVE, CLIENT_PROOF_DIRECTIVE, ELABORATION_DIRECTIVE, PRICING_DIRECTIVE,
# CALL_FLOW_DIRECTIVE, and both LLM verification prompts below. Extracted here as the single source
# of truth each of those references, instead of six independently-editable copies of the same rule.

# A shared broad parent category is NOT a genuine match — found live: "THS (Tapis Hussein Kabalan)",
# a carpet/textile retailer, got cited for an eyeglasses store because both are loosely "retail." The
# specific type of product/business has to genuinely be alike, not just the parent category. Used
# everywhere CLIENT_CITATION_SOURCE or CLIENT_MATCH_CRITERION is used, so both generation-time
# instructions and verification-time checks apply the same, stricter standard.
CLIENT_MATCH_SPECIFICITY_NOTE = (
    "A shared broad parent category is NOT enough on its own — if both are loosely 'retail' or both "
    "'services' but the actual specific type of product/business differs (e.g. a carpet store vs. an "
    "eyeglasses store, a restaurant vs. a café, real estate vs. construction), that does NOT count as "
    "a genuine match; the specific business type itself has to genuinely be alike."
)

# Which client may be cited, and how strictly it must be sourced. Used by FIELD_PITCH_DIRECTIVE
# (MUST cite if a match exists) and CLIENT_PROOF_DIRECTIVE (MAY cite if a match exists) — the
# must/may distinction stays in each directive's own lead-in sentence; this fragment is only the
# sourcing constraint both share.
CLIENT_CITATION_SOURCE = (
    "only a name that literally appears in the retrieved context — never a vague unnamed stand-in "
    "like 'a local café' or 'a client we worked with'. If you're not sure a client is a true match, "
    "or none in the context really fits their field, don't cite one — an uncertain, wrong-field, or "
    "vague name is worse than none, and never invent one. " + CLIENT_MATCH_SPECIFICITY_NOTE
)

# How often a specific named client may appear across a conversation. Used by CLIENT_PROOF_DIRECTIVE
# and ELABORATION_DIRECTIVE.
CLIENT_CITATION_FREQUENCY = (
    "A SPECIFIC named client gets cited as proof AT MOST ONCE across the whole conversation — if you "
    "(or an earlier reply) already named one, do NOT name another one this turn, even a different "
    "genuine match: repeating a named client every turn reads as spammy, not credible. In that case "
    "either skip proof entirely, or at most say — with no name attached — that BLNK has worked with "
    "companies similar to their domain before. That vague fallback line is ITSELF subject to the same "
    "once-per-conversation limit — if you (or an earlier reply) already used it, do NOT use it again "
    "either, even reworded: at that point just skip proof entirely this turn, nothing about past "
    "clients at all."
)

# Deterministic detection of the vague no-name fallback line itself — matches OUR OWN controlled
# phrasing (both CLIENT_CITATION_FREQUENCY and repeated_client_fix_directive dictate this exact
# family of wording), same reasoning as _CALL_OFFER_RE/_OFFICE_ADDRESS_RE matching our own text
# rather than the customer's free-form input. Found live (2026-07-21): "what about logo" then "logo
# prices," two turns in a row, both fell back to this line — the soft "don't repeat it either"
# instruction above didn't hold, so this gets the same final-safety-net strip as everywhere else.
_VAGUE_CLIENT_FILLER_RE = re.compile(
    r"worked with (companies|businesses|clients|brands) (in|similar to|like)\b[^.?!]{0,40}"
    r"(domain|field|industry|yours|theirs)",
    re.IGNORECASE,
)


def _vague_client_filler_already_used(chat_history: list) -> bool:
    return any(
        m.get("role") == "assistant" and _VAGUE_CLIENT_FILLER_RE.search(m.get("content", ""))
        for m in chat_history or []
    )


def client_citation_suppressed(turn_type: str) -> str:
    """The 'no client at all this turn' rule, parameterized by turn type. Used by PRICING_DIRECTIVE
    ("a pricing/meeting turn") and CALL_FLOW_DIRECTIVE ("the meeting-logistics flow")."""
    return (
        f"Do NOT name or cite any past client in this reply, named or unnamed — that belongs to a "
        f"pitch/proof turn, not {turn_type}."
    )


# The "genuine match" definition shared by both LLM verification prompts below.
CLIENT_MATCH_CRITERION = "own business genuinely matches the customer's stated field"

# Deterministic backstop for pricing/meeting-logistics turns: PRICING_DIRECTIVE and
# CALL_FLOW_DIRECTIVE both now explicitly say not to cite any client, and the repetition check also
# flags naming a second client — but a client name STILL leaked into a pricing reply in testing
# (LLM-based instruction-following and LLM-based detection both missing it independently). Detecting
# THIS case doesn't need an LLM call at all: the real client names are already sitting right there in
# metadata.client on the retrieved chunks, so a plain substring check is both cheaper and more
# reliable than asking a model to notice its own mistake.
def _client_names_in_context(context_chunks: list) -> list:
    names = []
    for c in context_chunks or []:
        name = (c.get("metadata") or {}).get("client")
        if name and name not in names:
            names.append(name)
    return names


def _first_client_mentioned(text: str, names: list) -> str:
    for name in names or []:
        if name and name in (text or ""):
            return name
    return ""


def _strip_client_sentence(text: str, client_name: str) -> str:
    """Remove any sentence mentioning `client_name` — the deterministic last resort when a
    wrong-field client citation (see _check_industry_mismatch) survives a regen attempt. Found
    live: "MediDate" (an F&B packaging case study) kept getting cited for a flower/perfume shop
    despite the mismatch check correctly flagging it — the regen attempt just wasn't reliable
    enough on its own, same class of issue as the other checks in this file that needed a re-check
    + deterministic remedy instead of trusting one regen. Dropping the sentence is safer than a
    third regen gamble: a slightly shorter reply beats a confidently wrong example reaching the
    customer, and this never blanks the whole reply if that sentence was the only content."""
    if not text or not client_name:
        return text
    sents = _split_sentences(text)
    kept = [s for s in sents if client_name not in s]
    if not kept:
        return text
    return " ".join(kept).strip()


def no_client_this_turn_fix_directive(client_name: str) -> str:
    return (
        f"This is a pricing or meeting-logistics turn, which must NOT cite any past client — but your "
        f"previous reply named \"{client_name}\". Rewrite the SAME reply, same intent and length, with "
        f"every mention of {client_name} (and any other client name) removed entirely."
    )


# Deterministic backstop for the "cite a client at most once per conversation" rule — CLIENT_PROOF_
# DIRECTIVE and ELABORATION_DIRECTIVE both state it, and the LLM repetition check also flags it, but
# a client (even the SAME one, not just a different one) still got cited on 3 consecutive elaboration
# turns in testing. Detecting a REPEATED citation doesn't need an LLM call: it's a plain substring
# check against every prior assistant turn, using the full client roster (ALL_CLIENT_NAMES) rather
# than just this turn's retrieved context — a client cited 2 turns ago may not be in THIS turn's
# retrieval anymore, so checking only the current context (like _client_names_in_context) would miss it.
def _already_cited_client(chat_history: list) -> str:
    for m in chat_history or []:
        if m.get("role") == "assistant":
            found = _first_client_mentioned(m.get("content", ""), ALL_CLIENT_NAMES)
            if found:
                return found
    return ""


def repeated_client_fix_directive(earlier_client: str, new_mention: str, vague_filler_already_used: bool = False) -> str:
    same = earlier_client == new_mention
    fallback = (
        "drop the proof entirely — do NOT fall back to the generic 'worked with companies in their "
        "domain before' line either, that was already used earlier in this conversation and repeating "
        "it is just as stale as repeating the name"
        if vague_filler_already_used else
        "either drop the proof entirely, or at most reference generically that BLNK has worked with "
        "companies in their domain before, with no name attached"
    )
    return (
        (f"Your previous reply named \"{new_mention}\" again, but you (or an earlier reply) already "
         f"cited {earlier_client} earlier in this conversation"
         if same else
         f"Your previous reply named \"{new_mention}\", but you (or an earlier reply) already cited "
         f"a DIFFERENT client, {earlier_client}, earlier in this conversation") +
        f" — a specific client is cited at most ONCE per conversation, period, whether it's the same "
        f"name again or a new one. Rewrite the SAME reply, same intent and length, with every client "
        f"name removed — " + fallback + "."
    )


# ──────────────────────────────────────────────────────────────────────────────
#  BOOKING CONSTANTS
# Whenever we suggest a meeting, we now offer TWO paths, and let the customer self-select between
# them rather than us guessing their location: an onsite meeting at BLNK's office (relevant if
# they're in Lebanon), or the free 15-minute online consultation via this self-service Calendly link
# (the old "ask which day works / I'll check the schedule and get back to you" flow, kept ONLY for the
# onsite path — see CALL_FLOW_DIRECTIVE — since there is no self-booking calendar for a physical
# office visit). The link must always be pasted exactly and unaltered.
CALENDLY_LINK = "https://calendly.com/a-issa-blnkcs/15min"

# Deterministic restore text for the "a later regen silently dropped the link" safety net near the
# end of generate_answer — never invents new framing, just guarantees the actual clickable link
# reaches the customer regardless of what happened to the surrounding sentence upstream.
BOOKING_INSTRUCTION_LINK_FALLBACK = {
    "en": f"You can pick a slot that works for you here: {CALENDLY_LINK}",
    "ar": f"فيك تختار الوقت يلي بيناسبك من هون: {CALENDLY_LINK}",
}

BOOKING_INSTRUCTION = (
    "Offer them a meeting, phrased as a CHOICE, and STOP there in this reply — do NOT send the online "
    "booking link and do NOT mention an office location or address yet, either one. Just mention that "
    "if they're based in Lebanon you can schedule an onsite meeting at BLNK's office, and otherwise "
    "there's a free 15-minute online consultation, and let them tell you which. Sending the link (for "
    "online) or the address and asking their convenient time (for onsite) only happens on the NEXT "
    "reply, once they've actually said which one — that hand-off is handled by a dedicated turn-by-turn "
    "directive, not this one, so don't pre-empt it here."
)

# Deterministic last resort for the pricing-choice check below: after 2 verified regen attempts,
# the model can still land on a vague "we can discuss this during a meeting" that never actually
# names the onsite/online choice (observed live — a strong local attractor for some inputs that
# survives repeated regeneration). Rather than keep paying for more unreliable attempts, guarantee
# the choice reaches the customer by appending it in code — this IS a conversion-critical moment.
BOOKING_CHOICE_FALLBACK = {
    "en": " If you're based in Lebanon, we can schedule an onsite meeting at BLNK's office, or set up a free 15-minute online consultation.",
    "ar": " عنا خيارين: منقدر نعمل لقاء بالمكتب ببيروت إذا كنت بلبنان، أو استشارة أونلاين مجانية مدتها 15 دقيقة إذا ما فيك تجي.",
}

# Deterministic last resort for a repeat contact_request turn (see BOOKING_CHOICE_REPEAT_NOTE and the
# final-safety-net strip in generate_answer) — a short, differently-worded reference back to the
# meeting rather than the full onsite/online explanation, only reached if the mid-pipeline regen
# didn't hold and stripping the restated choice left nothing usable behind.
BOOKING_CHOICE_REPEAT_FALLBACK = {
    "en": "You can go ahead with the meeting option I mentioned before, whenever works for you.",
    "ar": "فيك تعتمد على خيار الاجتماع يلي ذكرته قبل، بأي وقت بيناسبك.",
}

# Same shape, for a FIRST-time contact_request turn (nothing "mentioned before" to point back to) —
# presents the real choice instead of assuming onsite. See the "don't assume onsite" guarantee's own
# comment for the bug this backstops.
NEUTRAL_MEETING_CHOICE_FALLBACK = {
    "en": ("We can set up a meeting to help with that — onsite at BLNK's office if you're based in "
           "Lebanon, or a free 15-minute online consultation otherwise."),
    "ar": ("فينا نحدد موعد لمساعدتك بهالشي، حضوري بمكتب BLNK إذا كنت بلبنان، أو استشارة أونلاين مجانية "
           "مدتها ١٥ دقيقة إذا لأ."),
}

# Deterministic last resort for the field-pitch client citation: FIELD_PITCH_DIRECTIVE already
# requires naming a genuinely-matching past client in this SAME reply — this is the key moment to
# convince the customer, so it's as conversion-critical as the pricing choice above — but the model
# still skipped the citation entirely even after one regen attempt (found live: a real-estate pitch
# never mentioned Sakani despite it being a genuine, retrieved real-estate case study). Rather than
# keep paying for more unreliable regeneration, guarantee the name reaches the customer by appending
# it in code once the regen has already been tried and still came up short.
FIELD_PITCH_CLIENT_FALLBACK = {
    "en": " We did exactly this for {client}.",
    "ar": " متل ما عملنا مع {client}.",
}


def field_pitch_client_fallback(client_name: str, lang: str) -> str:
    template = FIELD_PITCH_CLIENT_FALLBACK.get(lang, FIELD_PITCH_CLIENT_FALLBACK["en"])
    return template.format(client=client_name)

# Deterministic last resort for the day/time-handling guarantee below: the mid-pipeline regen
# attempt proved unreliable specifically for the "sobh/masa misread as a greeting" failure mode
# (verified live: only 1 of 3 regenerated replies actually fixed it — the model kept completing the
# greeting call-and-response instead of reading the time). Rather than keep gambling on retries,
# replace the whole reply with a safe, guaranteed-correct acknowledgment as the final safety net.
DAYTIME_ACK_FALLBACK = {
    "en": "Got it — I'll check the schedule and get back to you shortly to confirm.",
    "ar": "تمام، رح شيك على الجدول وارجعلك قريباً لأأكدلك.",
}


# ──────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# The system prompt is intentionally kept short and stable. Everything that can be
# decided deterministically in code — reply language, whether the 15-min call is
# allowed yet, whether this is a pricing turn — is NOT reasoned about here; the
# backend computes it and injects a focused per-turn directive (see below). That
# keeps this prompt from repeating itself and from carrying rules that only apply
# on some turns.
SYSTEM_PROMPT = """You're a real BLNK team member chatting with a potential customer — warm and
human, never a corporate bot. Answer using ONLY the provided context and conversation history; never
invent clients, prices, numbers, services, or facts. A team/leadership fact from the context (e.g. who
owns or founded BLNK) is public info the company itself shared in this knowledge base — answer it
plainly, that's not a privacy concern, treat it like any other company fact. If you lack what you need
for a full answer, first try a genuinely useful, honest GENERAL answer from whatever IS in the context
(a related fact, a nearby service) rather than nothing — partial beats a flat refusal. Only when
there's truly nothing relevant: don't flatly say "I don't have that information" and stop there, and
don't guess or invent one — say briefly and warmly that you'll confirm it and follow up (a teammate
really is notified behind the scenes, so it's a genuine promise, not a brush-off). If the meeting
hasn't been offered anywhere yet in this conversation, you may offer it now as a choice — onsite at
BLNK's office if they're in Lebanon, otherwise a free 15-minute online consultation — WITHOUT the link
or address in that same reply; the hand-off happens once they say which they prefer (see the dedicated
turn-by-turn directive). If it was ALREADY offered earlier, do NOT re-offer it — "I'll confirm and
follow up" is enough on its own. Write plain, natural sentences — no markdown, no lists, no
exclamation marks, and NEVER a semicolon (;) — split into two sentences instead, or use a comma/dash
if the two halves genuinely belong together. Never say or hint you're an AI; if asked, stay warmly in
role.

Greeting: if they greeted you — even alongside a question in the same message — greet back FIRST with
just a short greeting word ("مرحبا" in Arabic script, "Hii" in English/arabizi), fused with a COMMA as
a leading clause onto the answer, NEVER a period ("Hii, <answer>", never "Hii. <answer>" as two
sentences). If they did NOT greet, skip the greeting and answer directly — never manufacture one. A
greeting is JUST the greeting word: don't also answer an unasked "how are you," and don't thank them
for their message. If their ENTIRE message is only a greeting with nothing else, never just echo the
same word back — add a short, natural invitation to say how you can help, so the conversation moves
forward. Never open the clause right after the greeting with "BLNK" as the subject (reads like BLNK is
the customer's own name) — use "we" instead, and work the company name in later if at all.

Accuracy over completeness (this GOVERNS everything): only state a fact, client, service detail, or
example when the context clearly supports it AND it genuinely fits THIS customer and what they asked.
When unsure, leave it out — a shorter, certain reply (or offering the consultation) always beats
something that might be wrong. Never stretch, approximate, or guess.

Length: match their energy and keep it short — 1-2 sentences, never a paragraph. Pitch ONE thing —
the single most convincing point for their business — and stop; don't stack multiple services or
examples. (Exception: the FIRST reply after they reveal their field is a deliberately fuller pitch
across everything relevant — see the dedicated directive; this rule governs every other reply.)

Field & proof: use only the industry they actually gave you, in their exact words — never assume it or
broaden it into a vaguer category ("تياب"/"tyeb" = clothing, not food; "chocolate" stays chocolate,
not "sweets"). Once known, tailor to it and drop unrelated industry words. If they pick a service and
their field is still unknown, ask once; if already known, never ask again.
(Whether/how to cite a past client is a per-turn directive's call — the accuracy rule above still
governs it: never a wrong-field or invented example.)

Pricing (any wording, any language): don't raise it unless asked, and never give exact numbers or
ranges — nothing overrides that. (The rest of a pricing reply's phrasing is a per-turn directive's
call.)

Sound human, not a bot:
- Read every message as part of the conversation, not on its own. A short/topic-free/ambiguous latest
  message (a bare "yes"/"ok"/"نعم", "can i know approximately") almost always continues whatever
  topic was most recently active — answer THAT, don't guess a fresh interpretation from the bare words
  alone (a follow-up right after a cost question is still about cost, not timing, even without the
  word "cost"). A bare "yes"/"ok"/"نعم" refers to what you just offered — continue it, never ask "yes
  to what?".
- Make your point and stop. Don't tack a question or an "if you want, I can..." offer onto the end out
  of habit — say the useful thing outright. Ask only when you genuinely need the answer (their field,
  or which day). Never end with a motivational/reassuring closing line or a design-truism — if it
  would read just as true pasted onto a reply for a totally different business, that's generic filler:
  cut it, end on the concrete substance instead.
- Don't restate or echo back what they just told you before answering (no "I understand you own a
  clothing shop") — they already know what they said, skip the recap. This includes turning it into a
  confirming QUESTION: if they already stated their field/business in THIS message, don't ask it back
  to confirm — treat it as established and answer their actual question directly, in the same reply.
- Never use "also"/"additionally"/"as well"/"kaman"/"كمان" anywhere in a reply, full stop.
- Once you've referenced their field/industry earlier in THIS conversation, don't re-open every later
  reply with the same industry-labeled preamble ("For your real estate company...") — that's a
  template tell. Just answer directly; the tailoring should live in the specifics, not a repeated
  opening label.
- Don't congratulate or gush when they simply mention their business or a normal fact — be warm but
  grounded, react like a normal person.
- Never repeat an idea, list, or recommendation you've already given, even reworded — add something
  genuinely new or give a confident recommendation instead.
- Describing services: phrase what BLNK does as OFFERING/PROVIDING it, never merely "having" it —
  English "we offer"/"we provide"; Arabic "نقدّم"/"منقدّم" as the actual verb, not "عنا"/"عندنا".
- Sell with confidence and BE CONCRETE: never pitch a service generically — paint what it actually
  does for THEIR specific business, vividly and relevantly (a restaurant means appetizing food photos
  and reels of their dishes, not just "reels and post templates"). Lead with the single most impactful
  thing rather than listing options. (Only suggest services that exist in the context.)
- Don't understand them? Ask them to rephrase (don't guess or repeat); if still unclear, suggest a
  quick call.
- Call: only offer a meeting when they explicitly ask about pricing/cost, how to get in touch or move
  forward, or scheduling/timing — never just because the conversation has continued for a while or they
  showed general interest/curiosity (a "how could you help me" / "tell me more" question is NOT on its
  own a reason to offer it). When one of those three does apply, offer a meeting phrased as a choice and
  STOP there. Only the FIRST time: once mentioned anywhere earlier in this conversation, do NOT bring it
  up again on your own — it must never be suggested more than once unprompted, in any scenario (dedicated
  directives handle the narrow exceptions). Mention that if they're in Lebanon you can schedule an
  onsite meeting at BLNK's office, otherwise there's a free 15-minute online consultation — but do NOT
  send the link or address in this same reply; wait for their preference. Once they choose online,
  THEN share the link so they pick their own slot: https://calendly.com/a-issa-blnkcs/15min , never
  asking which day/time suits them or saying you'll check a schedule — self-booking via the link is
  the whole flow. The onsite option has its own process once chosen (a dedicated directive, not this
  standing rule)."""


# ──────────────────────────────────────────────────────────────────────────────
#  PER-TURN DIRECTIVE DEFINITIONS
# The general prompt asks the model not to suggest the call early, but a standing
# rule is weak on the first turns and the model keeps offering it anyway. So we
# also hard-gate it: until this many assistant replies have happened, we inject an
# absolute, per-turn directive that forbids mentioning the call at all.
EARLY_TURN_THRESHOLD = int(os.environ.get("CALL_GATE_TURNS", "3"))

NO_CALL_DIRECTIVE = (
    "IMPORTANT: It is still early in this conversation. In THIS reply you must NOT "
    "suggest, mention, hint at, or offer the 15-minute call, a meeting, a booking, "
    "or 'having someone follow up on a call'. Simply answer the user's question and "
    "keep the conversation going. The ONLY exception: if you genuinely cannot answer "
    "their question because the needed information isn't available to you, you MAY "
    "invite them to the free 15-minute consultation as the way to get a proper answer. "
    "Otherwise, do not mention the call at all."
)

# Injected when the customer's latest message is JUST a bare closing acknowledgment/thanks ("great
# thank you", "okay thanks", "tamam shukran") with no question, no new field/request, and no active
# call-flow reply pending — i.e. nothing left for this turn to actually do. Without this, nothing in
# the standing SYSTEM_PROMPT tells the model to stop: the "sound human" bullets only cover not tacking
# a QUESTION onto the end, and the "a bare yes/ok continues what you offered" bullet is written for a
# customer accepting an open offer, not for a customer signaling they're done — so on a plain "great
# thank you" the model kept treating it as license to add another sentence of service pitch (found
# live: "We focus on creating designs that..." / "Our team uses tools like Figma..." after the
# customer had already said thanks). This directive is checked in EXCLUSIVE_DIRECTIVES right after
# call_flow (so an in-flow bare ack like "okay" while scheduling still gets CALL_FLOW_DIRECTIVE's own,
# more specific handling) and is also excluded from CLIENT_PROOF_DIRECTIVE firing (see
# ADDITIVE_DIRECTIVES) — a closing "thanks" is not the moment to name-drop a client either.
BARE_ACK_DIRECTIVE = (
    "The customer's latest message is just a short closing acknowledgment or thanks — nothing new, no "
    "question, nothing left open. Reply with ONLY a brief acknowledgment (e.g. 'You're welcome' / "
    "'العفو', or just a short warm word) and STOP there. Do NOT add any new information, do NOT pitch "
    "or explain a service, do NOT bring up a client example, and do NOT ask a follow-up question — the "
    "conversation is fine ending here, don't manufacture a reason to keep talking."
)

# Deterministic backstop for BARE_ACK_DIRECTIVE's own explicit "do NOT ask a follow-up question"
# instruction — found live it still gets ignored about half the time (e.g. a bare "nice"/"cool" got
# "...feel free to ask" tacked on anyway). Same non-compliance pattern as everywhere else in this
# file: strip the trailing invitation clause rather than trust the instruction alone.
_BARE_ACK_INVITATION_RE = re.compile(
    r"[^.?!]*\b(feel free to ask|let me know if|if there'?s (anything|something)|"
    r"if you (have|need) any|don'?t hesitate to)\b[^.?!]*[.?!]",
    re.IGNORECASE,
)

# How to answer a pricing question lives in the system prompt (the model recognizes
# pricing intent in any phrasing far better than a regex can). This regex is only a
# best-effort signal for ONE deterministic side effect: pricing is the case where
# suggesting the meeting is right even early on (an exact quote can only be settled
# on the call), so a match lifts the early-turn call gate. A miss just means the
# model follows its normal "don't push the call early" behavior — nothing breaks.
_PRICE_RE = re.compile(
    # \b word boundaries on every plain English term (2026-07-26): found live — "fee" with no
    # boundary matched as a bare substring inside "feek" (arabizi "can you"/"in you"), so "Keef feek
    # ts3dne" ("how can you help me") false-positived as a pricing question purely because "fee"
    # happens to appear inside an unrelated arabizi word. Any short common English word in this list
    # (cost, rate, charge...) has the exact same risk of hiding inside a longer English or arabizi
    # word — this is a general regex-construction fix, not a patch for "feek" specifically.
    r"\bprice\b|\bpricing\b|\bcost\b|\bcosts\b|\bbudget\b|\bquote\b|\bexpensive\b|\bafford\b|"
    r"\brate\b|\brates\b|"
    r"\bfee\b|\bfees\b|\bcharge\b|\bcharges\b|\bpackage\b|\bpackages\b|how much|"
    # Found live: "can i know the approximate range before" (insisting on a pricing follow-up) had
    # no dedicated match — bare "range" is deliberately NOT added on its own (too broad, would
    # false-positive on "range of services"), only these pricing-specific phrasings.
    r"approximate range|price range|ballpark|rough estimate|rough number|rough figure|"
    r"سعر|أسعار|اسعار|تسعير|تكلف|تكلفة|كلفة|بكم|قديش|قدّيش|ميزاني|"
    r"رسوم|باقة|باقات|كم بدو|كم يكلف|كم رح|كم راح|عرض سعر|رقم تقريبي|تقريبا قديش|"
    # arabizi (Arabic in Latin letters) — incl. plain-Latin spellings with no 3/7/2
    r"se3r|si3r|se3er|si3er|as3ar|as3aar|asaar|as2ar|s3r|s3ar|"
    # ad{1,2}esh (not a fixed "addesh") deliberately covers BOTH "adesh" and "addesh" in one pattern
    # — found live "adesh" (single d) slipped through a list that only had the double-d spelling.
    r"bikam|bkam|ad{1,2}esh|adaysh|2ad{1,2}esh|kad{1,2}esh|ad{1,2}eh|2ad{1,2}eh|aade|"
    r"taman|taklofe|taklfe|taklefe|taklife|kelfe|kilfe",
    re.IGNORECASE,
)


def is_pricing_question(text: str) -> bool:
    return bool(_PRICE_RE.search(text or ""))


# _PRICE_RE/_TIMING_RE are deliberately trusted UNCONDITIONALLY over the LLM's own is_pricing signal
# (see the "OR the regex in unconditionally" comment where pricing gets computed) — but that design
# breaks down exactly here: a bare substring match can't tell "how much does it cost" from "I did NOT
# ask about cost" apart, since both literally contain "cost". Found live: "i did not ask for cost" (a
# customer correcting the bot after a pricing tangent) still matched _PRICE_RE on "cost" and forced
# pricing=True right back on, so the correction itself got answered as a fresh pricing question —
# doubling down on the exact misunderstanding the customer was pointing out. Scoped narrowly (negation
# word within a few words of the price/timing term) so it only catches this specific shape, not a
# genuine compound like "not the online option, but how much does it cost".
_NEGATED_PRICE_OR_TIMING_RE = re.compile(
    r"\b(not|n't|never|didn'?t|don'?t|doesn'?t|ما|مش|مو)\b[^.?!,]{0,25}"
    r"\b(price|pricing|cost|costs|how\s*long|timing|timeline|سعر|أسعار|اسعار|تكلف|تكلفة|كلفة)\b",
    re.IGNORECASE,
)


# User's explicit product decision: a project-timing/duration question should be treated exactly
# the same as a pricing question — same directive, same guarantees (no exact number, explain what it
# depends on, offer the meeting), never a parallel system to independently maintain. Folded into the
# SAME `pricing` flag wherever it's computed (see is_pricing_question's one call site below) rather
# than given its own EXCLUSIVE_DIRECTIVES slot, so it gets PRICING_DIRECTIVE's strip/repeat-
# suppression/compound-intent handling for free instead of needing all of that re-built in parallel.
_TIMING_RE = re.compile(
    r"how long|how much time|turnaround|time.?frame|timeline|timelines|duration|"
    # Found live: "what about the time it takes" — a natural paraphrase distinct from every pattern
    # above (no "how long", no "turnaround", no "timeline") — matched neither this regex nor the
    # pricing one, so the turn fell through both deterministic checks entirely.
    r"time it (takes|needs)|"
    r"how many (days|weeks|months)|(days|weeks|months) (to|for) (complete|finish|deliver)|"
    r"كم مدة|قديش بتاخد|قديش بياخد|قديش وقت|شو المدة|مين مدة|"
    r"adde (badd?[ao]?|bad[ao]?)\s*(wa2et|w2et|wa2t|w2t|mo?ud[ae])|"
    # Found live TWICE: "shu howe l waet l bhtejo" ("what IS the time it needs") uses a different
    # sentence structure than "adde badda wa2et" above (a question word ... "waet", not specifically
    # "adde badda") — then "w adesh waet ento b haje" hit a second gap, an untried spelling of
    # "قديش" ("adesh", no leading 2/q) that a narrower question-word list missed. Reusing the SAME
    # full "qaddeesh" spelling-variant list _PRICE_RE already had to build out for the price context
    # (bikam/bkam/addesh/adaysh/2addesh/kaddesh/addeh/2addeh/aade) rather than maintaining a second,
    # narrower copy here that will keep independently missing spellings the other one already covers.
    r"(shu|kam|adde|bikam|bkam|ad{1,2}esh|adaysh|2ad{1,2}esh|kad{1,2}esh|ad{1,2}eh|2ad{1,2}eh|aade|"
    r"qadeesh|addaysh)"
    r".{0,20}(wa2et|w2et|waet|wa2t)|"
    r"kam\s*(you?m|yom|shahar|osboo3)|mouda|mo3ad al?[- ]?tasleem",
    re.IGNORECASE,
)


def is_timing_question(text: str) -> bool:
    return bool(_TIMING_RE.search(text or ""))


# User's explicit product decision (2026-07-24): price and timing are tracked as two SEPARATE
# dimensions, each getting the scope-dependency explanation exactly ONCE — the first time THAT
# dimension is asked about (even if the other one was already explained earlier and the meeting was
# already offered), then every later ask about that SAME dimension collapses to the one-line meeting
# redirect. E.g.: price asked -> explained + meeting offered. Timing asked for the FIRST time later
# -> explained too (no meeting re-offer, it's already on the table) — NOT the redirect, because timing
# itself has never been explained yet. Timing asked AGAIN after that -> now the redirect applies.
# These two regexes detect, from a PAST assistant reply, which dimension it actually explained — cost/
# price framing vs. how-long/timing framing — so today's turn can look up "has MY dimension already
# been covered," not just "has pricing-in-general ever come up."
# "Depends"/"determined" side, EN + AR (Arabic replies are freely translated per-turn, not a fixed
# template like the English directive text, so this can't be as exhaustive as the English half — best-
# effort coverage of the phrasings actually seen live, not a guarantee).
_DEPEND_WORD = r"depend\w*|يعتمد|بيعتمد|بتعتمد|معتمد|بتحدد|بيتحدد|بتتحدد|بيتقرر|بتتقرر"
_COST_SCOPE_EXPLAINED_RE = re.compile(
    rf"\b(cost|price|pricing|سعر|أسعار|اسعار|تكلف|تكلفة|كلفة)\b[^.؟!]{{0,60}}({_DEPEND_WORD})|"
    rf"({_DEPEND_WORD})[^.؟!]{{0,60}}\b(cost|price|pricing|سعر|أسعار|اسعار|تكلف|تكلفة|كلفة)\b",
    re.IGNORECASE,
)
_TIMING_SCOPE_EXPLAINED_RE = re.compile(
    rf"\b(how long|timeline|timing|duration|takes|مدة|وقت|بياخد|بتاخد)\b[^.؟!]{{0,60}}({_DEPEND_WORD})|"
    rf"({_DEPEND_WORD})[^.؟!]{{0,60}}\b(how long|timeline|timing|duration|takes|مدة|وقت|بياخد|بتاخد)\b",
    re.IGNORECASE,
)


def _topic_scope_already_explained(chat_history: list, is_timing_topic: bool) -> bool:
    rx = _TIMING_SCOPE_EXPLAINED_RE if is_timing_topic else _COST_SCOPE_EXPLAINED_RE
    return any(
        m.get("role") == "assistant" and rx.search(m.get("content", ""))
        for m in chat_history or []
    )


# The "explains WHY it varies" half of a scope-dependency acknowledgment (e.g. "it depends on the
# scope of what you need") — distinct from _PRICE_RE/_CALL_OFFER_RE, which match the QUESTION or the
# meeting-offer, not this specific answer shape. Factored out to module level (was inline inside
# _is_pricing_or_booking_sentence below) so the compound-intent "pricing ack owed" backstop can reuse
# the exact same pattern instead of maintaining a second copy that could drift.
_SCOPE_DEPENDENCY_RE = re.compile(
    r"\bscope\b|\bdepend\w*\b|\bvar(?:y|ies|ying)\b|\bspecifics?\b|\brequirements?\b|"
    r"\bestimate\b|\bsize of\b|\btakes?\b|\btimeline\b|"
    r"يعتمد|بيعتمد|بتعتمد|معتمد|بتحدد|بيتحدد|بتتحدد|بيتقرر|بتتقرر|"
    r"بيختلف|بتختلف|تفاصيل|متطلبات|حجم\s*(المشروع|مشروعك|شركتك)?",
    re.IGNORECASE,
)


def _has_pricing_ack(text: str) -> bool:
    """Stricter than a bare _SCOPE_DEPENDENCY_RE match: requires an actual price/cost/timing word
    ALONGSIDE the dependency language, not just one or the other. Found live: a field-pitch reply
    ending in "we can discuss the specifics of your project during our meeting" matched
    _SCOPE_DEPENDENCY_RE on "specifics" alone and was wrongly counted as a genuine pricing
    acknowledgment — the sentence never actually mentioned price, cost, or timing at all, only a
    generic "specifics"/"meeting" that happened to overlap the regex. Used by the compound-intent
    pricing-ack backstops, where that false positive let the pricing question go unanswered."""
    text = text or ""
    return bool((_PRICE_RE.search(text) or is_timing_question(text)) and _SCOPE_DEPENDENCY_RE.search(text))


# Shared by PRICING_DIRECTIVE and CALL_FLOW_DIRECTIVE — the two turn types that are purely
# transactional/logistical (confirming a detail, giving an address or link, asking one specific
# question), where the model kept padding the reply with an extra clause of pitch/marketing content
# despite the specific instructions on those turns ("we focus on creating cohesive brand identities
# that resonate with...", tacked onto a booking-link handoff or a time confirmation). This was
# originally a standing SYSTEM_PROMPT rule, but it only ever matters on these two turn types — moved
# here so it's injected only when actually relevant, instead of costing tokens on every single turn.
STAY_ON_TASK_RULE = (
    "This reply's job is purely transactional or logistical — do ONLY that job and stop, in ONE "
    "sentence when possible. Never pad it with so much as one extra clause of pitch, "
    "service-explanation, or marketing content, no matter how small or natural it feels (\"we focus "
    "on creating cohesive brand identities that resonate with...\", \"we can discuss how to enhance "
    "your branding\", re-explaining the process, re-selling a service, hyping up what comes next). "
    "This is absolute: this turn is NEVER a selling turn, even when sharing the booking link itself "
    "— hand over the link and stop, don't sell on top of it."
)

# Injected only when the regex flags a pricing turn, to reinforce the system prompt's
# pricing rule at the point it matters and to explicitly OK the call this early.
PRICING_DIRECTIVE = (
    "This looks like a price/cost question OR a project-timing/duration question ('how long does it "
    "take', 'adde badda wa2et') — treat both exactly the same way, per this one directive. Give NO "
    "numbers, ranges, or day/week counts. Do NOT explain the process, deliverables, what a service "
    "includes, or any other service/process detail here, even if it seems related — this reply has "
    "exactly ONE job: acknowledge scope-dependency (of price OR timing, whichever was actually asked) "
    "and offer the meeting, nothing else. That scope-dependency line specifically must NOT be a bare "
    "'it depends on the scope' — that alone reads as a brush-off. State what actually makes it vary: "
    "the scope of what they need, the specific services/deliverables involved, AND the size of their "
    "business, all three named explicitly in the sentence — e.g. for a price question: 'The cost "
    "depends on the scope of what you need, which specific services are involved, and the size of "
    "your business.' For a timing question, say the same thing about timing instead: 'How long it "
    "takes depends on the scope of what you need, which specific services are involved, and the size "
    "of your business.' Match that level of concreteness (own wording is fine, dropping any of the "
    "three factors is not) — this is still only explaining WHY it varies, never a number, a day/week "
    "count, or a description of what a service includes. Then " + BOOKING_INSTRUCTION +
    " It is appropriate to offer the meeting now. Do NOT end vaguely with 'we can talk more about "
    "it' / 'n2dar n7ki' — offering the meeting is the next step. " +
    client_citation_suppressed("a pricing/meeting turn") + " " + STAY_ON_TASK_RULE
)

# Deterministic last-resort text for a repeat pricing question (see PRICING_REPEAT_NO_MEETING_NOTE
# and the final-safety-net strip in generate_answer) — only reached if stripping every meeting-
# mentioning or "depends on..." sentence left nothing behind at all, so the customer isn't sent a
# blank reply. Deliberately does NOT use "depends on..." itself (2026-07-24) — every caller of this
# fallback is a REPEAT scenario where that explanation was already given once and must not recur, so
# the fallback has to be the same one-line redirect everything else in this simplification converges on.
PRICING_SCOPE_FALLBACK = {
    "en": "Pricing and timing specifics are always discussed during the meeting.",
    "ar": "بنحكي بالتفصيل عن السعر والوقت خلال الاجتماع.",
}

# Found live TWICE with two different pairings — services+pricing ("feye aarf aktr shu aandkon
# khadamet w shu l asaar"), then later service-detail+pricing ("i wanna know more abt brand identity"
# + "and your pricings if possible") — same root cause both times: only ONE exclusive directive can
# be the reply's primary focus (EXCLUSIVE_DIRECTIVES fires just the first match), so whichever OTHER
# genuinely-asked part is lower priority gets silently dropped. Rather than hardcode a new additive
# note per pairing (which just repeats this same patch for every future combination the customer
# invents — services+more_detail, contact+field_pitch, etc.), this generically detects ANY
# lower-priority ask that's ALSO true this turn and tells the model to cover it too, in one line each.
# See _dropped_exclusive_asks / compound_intent_note below, wired in from generate_answer (needs
# per-turn `signals`/`profile` values like the actual service/industry name, so it can't be a static
# ADDITIVE_DIRECTIVES entry like CLIENT_PROOF_DIRECTIVE — those only see turn-invariant flags).
# Every builder takes (signals, profile, meeting_ever_mentioned) — the third arg is only actually
# used by "pricing"/"contact_request" (see 2026-07-21 note below), but kept uniform across all five
# so the generic call site doesn't need to special-case which ones care about it.
SECONDARY_ASK_BUILDERS = {
    # Found live (2026-07-21): this used to unconditionally say "offer the meeting as the next
    # step," even when field_pitch (checked earlier than pricing/contact_request in EXCLUSIVE_
    # DIRECTIVES) was the actual primary directive and the meeting had ALREADY been offered earlier
    # in the conversation — the exact "suggested more than once" bug the user was explicit must never
    # happen, just reached through the compound-intent path instead of pricing/contact_request
    # winning outright. Branches on meeting_ever_mentioned the same way PRICING_REPEAT_NO_MEETING_
    # NOTE/BOOKING_CHOICE_REPEAT_NOTE do for the primary-directive case.
    "pricing": lambda signals, profile, meeting_ever_mentioned: (
        "they ALSO asked about pricing/cost OR project timing/duration (read their message to tell "
        "which one it actually was) — " + (
            "the scope-dependency explanation was already given earlier in this conversation, so do "
            "NOT recite it again in any form; just add ONE clear, short line that pricing and timing "
            "specifics are always discussed during the meeting — that's it, do NOT re-explain or "
            "re-offer the onsite-vs-online choice itself, it was already given earlier"
            if meeting_ever_mentioned else
            "note briefly that IT (the cost, if they asked about cost; how long it takes, if they "
            "asked about timing — don't answer the one they didn't ask) depends on the scope of what "
            "they need, and still offer the meeting as the next step for a real number"
        )
    ),
    "contact_request": lambda signals, profile, meeting_ever_mentioned: (
        "they ALSO explicitly asked how to get in touch or move forward — " + (
            "briefly point them back to the meeting option already mentioned earlier in this "
            "conversation, in ONE short, DIFFERENTLY-worded clause (e.g. 'you can check out the "
            "meeting options I mentioned before') — do NOT restate the full onsite/online explanation "
            "again"
            if meeting_ever_mentioned else
            "mention scheduling the meeting as the next step"
        )
    ),
    "field_pitch": lambda signals, profile, meeting_ever_mentioned: (
        "they ALSO revealed their business field"
        + (
            f" ({signals.get('industry') or profile.get('industry')})"
            if (signals.get("industry") or profile.get("industry")) else ""
        )
        + " — briefly note which BLNK service categories are most relevant to it"
    ),
    "services_overview": lambda signals, profile, meeting_ever_mentioned: (
        "they ALSO asked broadly what services BLNK offers — name the categories in one short line "
        "(" + SERVICE_CATEGORIES_TEXT + ")"
    ),
    "more_detail": lambda signals, profile, meeting_ever_mentioned: (
        "they ALSO asked to learn more about "
        + (signals.get("interested_service") or profile.get("interested_service") or "a specific service")
        + " — explain concretely what it practically includes: name the ACTUAL deliverables/"
          "components from the retrieved context (e.g. for Brand Identity: logo design, brand book & "
          "guidelines, visual systems, brand strategy — not a vague 'helps establish your brand's "
          "personality' description that never says what's actually delivered) and how it helps "
          "their business. Found live: compressed into a single generic sentence, this kept landing "
          "as vague filler instead of a real answer — it has extra room in this reply specifically "
          "for this (see the sentence-cap headroom below), so use it. Don't invent specifics beyond "
          "what the context actually supports"
    ),
}


def _dropped_exclusive_asks(flags: dict) -> list:
    """Names of EXCLUSIVE_DIRECTIVES entries whose condition is true but which did NOT win — since
    select_directives only fires the first match, these are the genuinely-asked parts that would
    otherwise be silently dropped this turn."""
    winner_seen = False
    dropped = []
    for name, condition, _ in EXCLUSIVE_DIRECTIVES:
        if condition(flags):
            if not winner_seen:
                winner_seen = True
            else:
                dropped.append(name)
    return dropped


def _primary_exclusive_ask(flags: dict) -> str | None:
    """Name of the EXCLUSIVE_DIRECTIVES entry that actually wins this turn (the first true
    condition), or None if none apply. Used by compound_intent_note to tell whether the primary
    directive is one that forbids mentioning the meeting (field_pitch/field_pitch_unclear) — see its
    2026-07-21 note on the pricing-as-secondary-ask contradiction this resolves."""
    for name, condition, _ in EXCLUSIVE_DIRECTIVES:
        if condition(flags):
            return name
    return None


def compound_intent_note(flags: dict, signals: dict, profile: dict, meeting_ever_mentioned: bool = False) -> str | None:
    """When the customer's message contains MULTIPLE distinct asks at once, only one exclusive
    directive can be the reply's primary focus — but every other genuinely-asked part still needs an
    answer. Returns an instruction covering all of them, or None if nothing was dropped. User's
    explicit framing (2026-07-20): this is NOT "answer the last thing asked" or "list each part as
    its own disconnected statement" — the customer sent these as ONE related thought, so the reply
    must read as ONE connected answer to that whole thought, not a checklist of separate fragments
    bolted together. Excluded entirely when `call_flow` is the winner (flags["offered_call"]):
    CALL_FLOW_DIRECTIVE is a structural, exactly-ONE/TWO-jobs logistics directive with its own strict
    "nothing else" guarantees (STAY_ON_TASK_RULE, off-task-pitch check, premature-rebooking check) —
    found live that layering a compound-intent note on top of it re-pitches unrelated service content
    and can even duplicate the company-info question, exactly the kind of extra content those other
    guarantees exist to prevent. Only applies when a content-answering directive (pricing, contact,
    field_pitch, services_overview, more_detail) is the primary focus.

    meeting_ever_mentioned (2026-07-21 addition): found live — field_pitch now often wins the primary
    slot (see the EXCLUSIVE_DIRECTIVES reordering note), which left pricing/contact_request only
    reachable here as a SECONDARY ask; the pricing/contact_request builders unconditionally said
    "offer the meeting as the next step" even when it had already been offered earlier in the
    conversation — the compound-intent path was a back door around the pricing_repeat/contact_repeat
    suppression, reintroducing the exact "suggested more than once" bug through a route that
    suppression doesn't cover. Threaded down to SECONDARY_ASK_BUILDERS so those two builders can
    adjust their wording the same way the primary-directive path already does."""
    # NOTE: this checks whether call_flow actually WON (per the docstring above), not the raw
    # offered_call flag — field_pitch is checked before call_flow in EXCLUSIVE_DIRECTIVES and,
    # per the field_pitch_won note elsewhere in this file, routinely wins the primary slot even while
    # offered_call is true. Checking the raw flag instead of the actual winner silently disabled this
    # entire function on every turn after the meeting had ever been offered — exactly the scenario the
    # meeting_ever_mentioned paragraph above was written to handle, so it never actually ran.
    if _primary_exclusive_ask(flags) == "call_flow":
        return None
    dropped = [n for n in _dropped_exclusive_asks(flags) if n in SECONDARY_ASK_BUILDERS]
    if not dropped:
        return None
    asks = [SECONDARY_ASK_BUILDERS[n](signals, profile, meeting_ever_mentioned) for n in dropped]
    also_list = "; ".join(asks)
    # Found live (2026-07-21): field_pitch/field_pitch_unclear as primary directly CONTRADICTS the
    # pricing secondary-ask builder whenever pricing is dropped and the meeting hasn't been offered
    # yet — the primary directive says "do not push the call," the secondary ask says "offer the
    # meeting for a real number." The model resolved this by dropping BOTH the pricing acknowledgment
    # AND the meeting offer, silently ignoring the compound ask entirely rather than picking one.
    # Spelling out that the override applies (rather than leaving the two directives to visibly
    # disagree) fixes that.
    override_note = ""
    primary = _primary_exclusive_ask(flags)
    if primary in ("field_pitch", "field_pitch_unclear") and "pricing" in dropped and not meeting_ever_mentioned:
        override_note = (
            " Note: the field-pitch instruction above told you not to push the meeting — that "
            "restriction does NOT apply here, because the customer ALSO explicitly asked about "
            "pricing, and that part of the reply DOES need the meeting offer (per the pricing note "
            "just given). The two are not in conflict: cover the field pitch without the call, then "
            "naturally lead into the pricing acknowledgment and the meeting offer for that part."
        )
    # Found live, same shape of conflict as above but the other direction: pricing (and contact_
    # request) as primary both carry STAY_ON_TASK_RULE's ABSOLUTE "never pad with pitch/service-
    # explanation content" instruction — which then directly fights the secondary-ask note below
    # whenever the dropped ask is content-shaped (services_overview, field_pitch, more_detail), all
    # of which ask for exactly that kind of content. The model resolved it by obeying the "absolute"
    # framing and silently dropped the secondary ask entirely (e.g. "pricings... and how you could
    # help my business" — the pricing acknowledgment came through, the service categories the
    # customer also asked for never did). Spelling out the exception the same way the branch above
    # does fixes it.
    elif primary in ("pricing", "contact_request") and any(
        d in dropped for d in ("services_overview", "field_pitch", "field_pitch_unclear", "more_detail")
    ):
        override_note = (
            " Note: the instruction above said never to pad this reply with pitch or service-"
            "explanation content — that's about not volunteering EXTRA content they didn't ask for. "
            "It does NOT forbid the ONE addition just described above, which the customer "
            "THEMSELVES explicitly also asked for in this same message — that one piece is required, "
            "not an exception you should skip to stay safe."
        )
    return (
        "IMPORTANT: the customer's message actually asked MULTIPLE related things at once, not just "
        "one — treat all of it as ONE combined question, not a series of separate, unrelated asks. "
        "The directive above covers only part of it; on top of that, " + also_list + "." + override_note +
        " Write ONE "
        "single flowing reply that naturally covers everything they asked, in the order that reads "
        "most naturally — do NOT answer only the primary part and tack the rest on as a disconnected "
        "checklist or numbered list, and do NOT answer only the last thing they said while ignoring "
        "the rest. Still finish with whatever the primary directive above requires (e.g. the meeting "
        "offer) — don't drop that either."
    )

# Injected when the customer explicitly asks how to get in touch, reach BLNK, or move forward —
# an unambiguous signal they want next steps NOW, not more selling. Modeled on PRICING_DIRECTIVE:
# found live that without a dedicated directive, this turn falls through the cracks (it's neither
# `pricing` nor `offered_call`, since WE didn't offer the call last turn — the CUSTOMER is initiating
# contact themselves) and the model pads the reply with a full extra sentence of re-pitched content
# ("we can create engaging social media content...", already said moments earlier) before finally
# getting to the meeting offer.
CONTACT_REQUEST_DIRECTIVE = (
    "The customer is explicitly asking how to get in touch, reach you, or move forward — they want "
    "next steps now, not more selling. This reply has exactly ONE job: do NOT recap, restate, or "
    "re-pitch anything already discussed (a service, a benefit, a past client — not even one short "
    "reminder clause of it) before answering; go STRAIGHT to next steps. " + BOOKING_INSTRUCTION + " " +
    client_citation_suppressed("a contact/next-steps turn") + " " + STAY_ON_TASK_RULE
)

# Injected when understand_turn thinks the customer wants a field pitch but couldn't actually pin
# down what their field/industry IS (signals.industry came back null) — found live (2026-07-21):
# "we are c.o company, wanna know how you can help" got wants_field_pitch=true with industry=null,
# an internally inconsistent read from understand_turn itself. FIELD_PITCH_DIRECTIVE below already
# has a soft "if genuinely unsure, ask instead of guessing" preamble, but relying on the model to
# self-judge its own confidence inside an otherwise expansive, multi-paragraph pitch directive proved
# exactly as unreliable as every other soft self-check in this file — it went ahead with a full
# multi-category pitch AND cited a client (unfounded — there's no known field to judge relevance
# against) AND pushed the meeting (which the very same directive explicitly forbids). This is the
# deterministic fix: gate on industry_known (a real code-level fact, not the model's self-assessment)
# so the ambiguous case gets its OWN short, single-purpose directive instead of ever reaching the
# big one. See EXCLUSIVE_DIRECTIVES below for how the two are split.
FIELD_PITCH_UNCLEAR_DIRECTIVE = (
    "The customer just mentioned their business, but it's genuinely UNCLEAR from what they wrote "
    "exactly what field/industry that is (an unfamiliar word, an abbreviation, something ambiguous). "
    "Do NOT guess a field and pitch toward it, do NOT give a generic multi-category pitch, and do NOT "
    "cite a past client — you can't judge whether one is genuinely relevant without knowing the "
    "field, and a wrong guess is worse than asking. This reply has exactly ONE job: warmly and "
    "briefly ask them to confirm or spell out their field/business in plain terms, and stop there. Do "
    "NOT offer, suggest, or mention the meeting/call in this reply — they haven't asked for that, and "
    "it isn't helpful before you even know what they do."
)

# Injected when the customer has just revealed their field and wants to know how BLNK can help.
# This is the key moment to win them over, so we deliberately override the short-reply limit and
# push a fuller, proof-backed, tailored pitch — using ONLY what the retrieved context supports.
FIELD_PITCH_DIRECTIVE = (
    "The customer has just told you their field/business. FIRST: are you genuinely confident, from "
    "what they actually wrote, exactly what field/business that is? If there's real doubt — an "
    "unfamiliar word, an unclear translation, anything you're not sure you're reading correctly — do "
    "NOT guess a field and pitch toward it; that risks a confidently wrong answer, which is worse than "
    "asking. Instead, briefly ask them to confirm or spell out their field/business in plain terms, "
    "and stop there — nothing else in this directive applies until you actually know it. Otherwise, if "
    "you ARE confident what their field is: this is the moment to convince them — go "
    "broader than usual here: cover EVERY one of BLNK's service categories (" + SERVICE_CATEGORIES_TEXT +
    ") that would genuinely benefit a business like theirs — from a logo through to social media, "
    "whichever of those genuinely apply — not just a single deliverable. This OVERRIDES the standing "
    "'pitch one thing and stop' rule for this ONE reply only; every other reply still follows it. Only "
    "include a category that's a genuine fit for THIS specific field, grounded in the retrieved context "
    "— never pad in a category that doesn't actually apply just to fill out the list. For EACH category "
    "you include, give a brief phrase or short clause of what it specifically means for THIS business — "
    "NOT a full descriptive sentence per category, that reads long; tight and punchy instead (not a "
    "generic bullet with no connection to their field either) — e.g. for a clothing store: the logo/"
    "brand identity that sets their look apart, product photography that makes pieces look worth buying, "
    "social media content that keeps customers engaged. Connect these into ONE flowing, natural pitch — "
    "aim for about 3 short sentences total covering everything relevant, not one sentence per category "
    "stacked into a longer paragraph — not a dry mechanical list ('we do X, we do Y, we do Z') either. "
    "Keep the exact word the customer used for their field or product — if they said 'chocolate', talk "
    "about chocolate; never broaden it into a vaguer category (e.g. 'halaweyat'/'sweets'). Skip the "
    "preamble about understanding their field (no 'we know how important X is' / 'منعرف أدّيه مهم...'). "
    "In this SAME first reply, name ONE past client as proof, woven into the pitch rather than tacked on "
    "as its own extra sentence — but ONLY a real client from the retrieved context whose own business "
    "genuinely matches the customer's field, using their actual name as it appears in the context (e.g. "
    "'Kor and More') — " + CLIENT_CITATION_SOURCE + " "
    "Don't hold a valid client example back for a later turn. Do not push the call in this reply. No "
    "exclamation marks, no markdown, no lists, and never use 'also'/'additionally'/'كمان'/'kamen' to "
    "connect the categories — state each one directly."
)

# Injected when the customer asks to know MORE about something already on the table ("aarf aktar
# tafasel", "tell me more"). The point is to go DEEPER on the same service, not to pivot to a new
# one — which is the opposite of what the anti-repeat rule might otherwise nudge the model toward.
ELABORATION_DIRECTIVE = (
    "The customer wants real detail about ONE specific named service they just asked about, not a "
    "sales pitch or a menu of everything BLNK does. Two cases, handle whichever actually applies: "
    "(a) if this exact service/deliverable was ALREADY discussed earlier in this conversation, go "
    "DEEPER — add concrete specifics you haven't already said, don't just restate the same sentence; "
    "(b) if this is the FIRST time this specific service has come up, there is nothing earlier to "
    "build on — just explain it clearly and concretely from scratch: what it practically includes "
    "and how it helps their business. Either way, keep it to ONE short sentence of real substance. " +
    CLIENT_CITATION_FREQUENCY + " "
    "Do NOT introduce a different service, deliverable, or offering than the one they actually asked "
    "about, even something technically related (e.g. if they asked about logo design, do NOT pivot to "
    "business cards or website design instead) — answer the one they named, nothing else. "
    "Do NOT offer, suggest, or push the meeting/call in this reply — they only asked a general "
    "question about a service, not to get in touch, ask about pricing, or discuss how it applies to "
    "their own specific field; those are the moments that call for the meeting, this isn't one of "
    "them. Just answer their question with real detail and stop there."
)

# Injected when the customer asks BROADLY what BLNK offers ("shu btaadmo khadamet", "what do you
# do"). This is a menu request, not a deep-dive — it must beat the elaboration rule so we don't
# dump all the detail of one service when they just want to see the categories and pick one.
SERVICES_OVERVIEW_DIRECTIVE = (
    "The customer is asking broadly WHICH services BLNK offers — not about one specific service. Give a "
    "ONE-LINE overview naming the categories (" + SERVICE_CATEGORIES_TEXT + "), then end with a real, "
    "complete question inviting them to say more about what they need — e.g. 'what are you interested "
    "in?'. Do NOT phrase it as 'which one' in any form (not 'which one?', not 'which one would you "
    "like to hear more about?') — NEVER a bare fragment like 'which one?' or 'which one.' on its own "
    "either, and do NOT use 'would you like' anywhere in that closing question (a separate cleanup "
    "step removes reflexive 'would you like...' offers and will otherwise mangle it). Do NOT "
    "deep-dive or detail a single service here — keep it short and let them choose. No markdown, no "
    "exclamation marks."
)

# Distinct from SERVICES_OVERVIEW_DIRECTIVE (found live: "what industries have you worked with
# before" got answered with the SERVICE categories — Brand Identity, Digital Design, etc. — instead
# of naming past-client industries, since nothing distinguished "what do you offer" from "who have
# you worked with"). This one names actual INDUSTRIES/sectors from the retrieved context and past
# client list, never service categories.
CLIENT_INDUSTRIES_DIRECTIVE = (
    "The customer is asking broadly which INDUSTRIES or types of businesses BLNK has worked with — "
    "not which services BLNK offers. Answer with actual industries/sectors from the retrieved context "
    "and past clients (e.g. restaurants, real estate, beauty/salons, F&B, retail — whatever genuinely "
    "appears in the context), naming a client or two as concrete proof if the context supports it. Do "
    "NOT answer with the service category list (Brand Identity, Digital Design, Print & Collateral, "
    "Social Media Design) — that answers a different question they did not ask. If the context "
    "doesn't clearly show which industries, say so honestly rather than guessing, per the standing "
    "accuracy rule. Keep it short — one or two sentences."
)

# General catch-all for a message that doesn't fit ANY of the anticipated sales categories above
# (pricing/field_pitch/more_detail/services_overview/contact/bare_ack) — a customer saying they don't
# understand the reply language, a complaint, an off-topic question, a request for a human, or any
# other kind of message this schema never enumerated. Root cause this fixes: with no directive for
# these turns, the pipeline silently fell through to nothing — the model just kept generating from
# retrieved context + history, which for a topic-anchored retrieval query meant continuing whatever
# was already being discussed and ignoring what the customer actually just said (found live: a
# customer writing "ma ba3rif english"/"ma fahmene 3alaykon" — a language complaint, not a services
# question — got answered with more generic social-media-package text on every single turn). The
# actual paraphrase of what the customer said (signals["customer_intent"]) is injected as its own
# system message right after this one (see generate_answer) rather than baked into this constant,
# since this text is static/testable via select_directives while the paraphrase is per-turn.
UNADDRESSED_INTENT_DIRECTIVE = (
    "The customer's latest message doesn't match any of this agency's standard sales categories "
    "(pricing, a service pitch, wanting more detail on a service, a services overview, "
    "contact/booking, or a bare acknowledgment). That does NOT mean something is wrong with the "
    "message — it's just as likely a perfectly normal, on-topic question this schema didn't "
    "specifically anticipate (e.g. asking to see a portfolio piece just mentioned, a factual question "
    "about the company or process) as it is confusion, a complaint, or an off-topic request. The next "
    "system message describes, in plain language, what the customer actually said or meant this turn "
    "— read it and respond to THAT directly and specifically, in whatever tone actually fits it (plain "
    "and factual for an ordinary question, empathetic only if it's genuinely a complaint/confusion). "
    "Do NOT ignore it and default to continuing whatever topic was being discussed before (especially "
    "the meeting/booking, if that's not what this message is actually about), and do NOT repeat a "
    "generic pitch or service description that has nothing to do with what they just said. If it's a "
    "complaint about not understanding the reply, acknowledge it and address it directly (e.g. offer "
    "to reply in Arabic instead if that's the issue). If you genuinely don't have enough information "
    "to address what they said, say so honestly rather than guessing (per the standing accuracy "
    "rule), and offer the free consultation if that fits."
)

# Deterministic backstop for SERVICES_OVERVIEW_DIRECTIVE's "never say 'which one'" rule above — found
# live: despite the directive explicitly banning it, the model still ended a services-overview reply
# with the bare fragment "Which one." (not even phrased as a real question). This is a literal phrase
# ban, not a fuzzy judgment call, so a plain regex is the right tool, same pattern as the "also"/"كمان"
# ban (_CONTINUATION_WORD_RE).
_WHICH_ONE_RE = re.compile(r"\bwhich one\b", re.IGNORECASE)


def _find_which_one(text: str) -> bool:
    return bool(_WHICH_ONE_RE.search(text or ""))


# Found live, in a SECOND pass at this bug: the regen fixing "which one" can itself produce a WORSE
# broken fragment ("...Social Media Design. What.") — a genuinely truncated non-question, not the
# banned phrase but just as bad. `_find_which_one` alone can't catch this since "What." doesn't
# contain "which one" — so the real check needs to be broader: is the closing sentence a real,
# complete question at all, not just "is the banned phrase absent". A bare 1-2 word non-question
# ending is exactly the class of defect this whole guarantee exists to prevent.
# TRACED THE ACTUAL ROOT CAUSE: it wasn't a fresh model quirk — the perfectly fine model draft "What
# would you like to know more about?" was itself getting silently mangled DOWNSTREAM by
# `_trim_bot_tail`'s `_OFFER_TAIL_RE` step, which strips reflexive "would you like..." offers and
# doesn't know this particular "would you like" is the mandated closing question, not an offer. So
# this check ALSO flags anything `_OFFER_TAIL_RE` would later strip — catching it here, before that
# happens, rather than only after the damage is already done.
def _services_overview_ending_bad(text: str) -> bool:
    sents = _split_sentences(text or "")
    if not sents:
        return True
    last = sents[-1].strip()
    if _WHICH_ONE_RE.search(last):
        return True
    if _OFFER_TAIL_RE.search(last):
        return True
    if not last.rstrip().endswith(("?", "؟")):
        return True
    return len(last.split()) <= 2


def which_one_fix_directive() -> str:
    return (
        "Your previous reply's closing question was either the banned 'which one' phrasing, a bare "
        "incomplete fragment, or used 'would you like' (a separate cleanup step strips that phrase "
        "and will mangle the sentence). Rewrite the SAME reply, same intent and length, ending "
        "instead with a real, COMPLETE question that avoids all of that — e.g. 'what are you "
        "interested in?' — never a 1-2 word fragment."
    )


# Final deterministic replacement if the regen above still doesn't fix it — never let 'which one' (or
# a bare fragment missing even a question mark) reach the customer. Appended as a full replacement of
# the trailing question rather than just stripped, since SERVICES_OVERVIEW_DIRECTIVE always expects
# this reply to end with SOME invitation to continue, not silence.
WHICH_ONE_FALLBACK = {
    "en": "What are you interested in?",
    "ar": "شو يلي حابب تعرف أكتر عنه؟",
}


def _replace_which_one_ending(text: str, lang: str) -> str:
    """Deterministically swap out the LAST sentence for the safe fallback question when it still
    contains 'which one' after the regen attempt — guarantees the banned phrase never reaches the
    customer, regardless of how the model phrases the rest of that sentence."""
    sents = _split_sentences(text or "")
    if not sents:
        return text
    sents[-1] = WHICH_ONE_FALLBACK.get(lang, WHICH_ONE_FALLBACK["en"])
    return " ".join(sents)

# The lighter, always-on cousin of FIELD_PITCH_DIRECTIVE: on any ordinary turn where we already
# know the customer's field, back a point up with a REAL past client in that field — but only one
# that actually shows up in the retrieved context. This is a single name-drop for credibility, not
# an extra sentence of pitch, and it never fires on the pitch turn (which already does this) or on
# the pricing/call-flow turns (which have one dedicated job).
CLIENT_PROOF_DIRECTIVE = (
    "You already know the customer's field/business. " + CLIENT_CITATION_FREQUENCY + " "
    "If NO client has been named yet in this conversation, you may name ONE past client as proof, but "
    "ONLY when that client's business genuinely matches the customer's field, and " + CLIENT_CITATION_SOURCE + " "
    "Keep the reply short; this is one name-drop woven in, not extra pitch. "
    "If the customer's message is a DIRECT question asking whether you've worked with businesses like "
    "theirs before (e.g. 'have you worked with restaurants', 'did you do this for a real estate "
    "company before', 'do you have experience with X') — this is not a generic conversational turn, "
    "it's a direct question that deserves a direct, confident answer: if a genuinely matching client "
    "exists in the retrieved context, say yes and name them plainly, don't hedge or talk around it "
    "with vague generalities instead of actually answering. If none genuinely matches, say so honestly "
    "(e.g. that this would be a first for that specific type of business) rather than deflecting into "
    "unrelated claims of experience."
)

# When OUR previous message suggested the call OR asked for the user's availability,
# the user's reply is part of the call flow. A bare "yes" or "monday" carries no topic,
# so without steering the model can wander off (retrieve the website) or improvise a
# booking ("Monday works, what time?"). BLNK.'s own phrasing is controlled, so matching
# is reliable. We include the availability-asking phrasing so the flow is still detected
# on the turn AFTER the call was first offered.
_CALL_OFFER_RE = re.compile(
    r"15[- ]?minute|15[- ]?min|quick call|a call|the call|\bcall\b|\bzoom\b|meeting|"
    r"calendly|consultation|book (?:a|your)|what day|which day|what time|which time|works best|"
    r"check the schedule|office|onsite|on-site|in[- ]person|"
    r"مكالمة|مكالمه|اتصال|جلسة|كول|زوم|ميتنغ|ربع ساعة|استشارة|كونسلتيشن|كاليندلي|احجز|احجزي|"
    r"أي يوم|اي يوم|أي وقت|اي وقت|بناسبك|شيك ع الجدول|الجدول|مكتب|مكتبنا|بمكتبنا|بالمكتب|حضوري|"
    # The office address itself is a stable, known fact — the model sometimes gives it without
    # also saying "office"/"مكتب" nearby (e.g. just the raw address), which then made this regex
    # miss that a meeting had been offered at all, so CALL_FLOW_DIRECTIVE never fired on the next
    # turn and the conversation went incoherent (found live). Matching the address components
    # directly closes that gap regardless of how the surrounding sentence is phrased.
    r"Rive Gauche|ريف غوش|Corniche El Nahr|كورنيش النهر",
    re.IGNORECASE,
)

# The onsite-side and online-side halves of the meeting choice, split out so a check can verify
# BOTH are actually present (i.e. the choice was really offered) rather than just any one
# meeting-related word — see the pricing vague-offer check in generate_answer.
_ONSITE_OFFER_TERMS_RE = re.compile(
    # Lebanon/Beirut mentions added (found live): CALL_FLOW_DIRECTIVE's own onsite-confirm branch
    # explicitly treats "saying they're in Lebanon" as choosing onsite (the office is only relevant
    # there), but this detection regex didn't recognize it — so "eh ana b lebnen" ("yeah I'm in
    # Lebanon") looked like "no mode chosen yet" to call_flow_mode_pending, forcing a re-ask/re-
    # explain of both options instead of going straight to the onsite branch (address + day/time).
    r"onsite|on-site|\boffice\b|in[- ]person|مكتب|مكتبنا|بمكتبنا|بالمكتب|حضوري|"
    r"lebanon|lebnan|lebnen|lubnan|lubnen|بيروت|beirut|لبنان|"
    # Found live right after the Lebanon fix above: "eh beje aal maktab" ("yeah I want it at the
    # office") — the Arabic-script مكتب was already covered but not its arabizi transliteration.
    r"maktab\w*",
    re.IGNORECASE,
)
_ONLINE_OFFER_TERMS_RE = re.compile(
    r"online|\bzoom\b|calendly|consultation|15[- ]?min|زوم|اونلاين|أونلاين|استشارة|كونسلتيشن|كاليندلي",
    re.IGNORECASE,
)

# User's explicit product decision (2026-07-23): CALL_FLOW_DIRECTIVE's onsite branch used to treat
# "directly asking where you're located" exactly the same as an actual commitment ("eh ana b
# lebnen", "let's meet in person") — both got the address AND an immediate day/time push in the
# same reply. Found live this reads as pushy when the customer was JUST asking a location question
# out of curiosity, not committing to schedule anything yet. Split: a PURE location question (a
# question word present, no separate commitment word) gets the address and STOPS there; an actual
# commitment still gets the full address + day/time treatment, unchanged.
_LOCATION_QUESTION_WORD_RE = re.compile(
    r"\bwhere\b|\blocation\b|\baddress\b|\bwen\b|وين|أين|اين|فين|"
    # "were" is a common typo for "where" — NOT matched bare (too common/risky a word on its own,
    # e.g. "we were happy with it"), only combined with "located" nearby.
    r"\bwere\b.{0,15}\blocated\b",
    re.IGNORECASE,
)
# The subset of _ONSITE_OFFER_TERMS_RE that signals an actual COMMITMENT rather than just naming
# the place being asked about — "maktab"/"office"/"مكتب" alone don't count (asking "where's the
# office" inherently mentions the office), but explicitly saying "onsite"/"in person"/naming
# Lebanon/Beirut does.
_ONSITE_COMMITMENT_RE = re.compile(
    r"onsite|on-site|in[- ]person|حضوري|"
    r"lebanon|lebnan|lebnen|lubnan|lubnen|بيروت|beirut|لبنان",
    re.IGNORECASE,
)


def is_pure_location_question(text: str) -> bool:
    t = text or ""
    return bool(_LOCATION_QUESTION_WORD_RE.search(t) and not _ONSITE_COMMITMENT_RE.search(t))


# User's explicit product decision (2026-07-23): when the customer asks us to SEND/SHARE the
# location (implying they want a droppable pin/map link to navigate by) rather than just asking
# where the office is, give them the WhatsApp number and tell them to contact that number for the
# location — not the raw address text on its own.
_SEND_LOCATION_RE = re.compile(
    r"send\s*(me\s*)?(the\s*|your\s*)?(location|pin|loc\b)|share\s*(the\s*|your\s*)?location|"
    r"ابعتلي|ابعتيلي|بعتلي|بعتيلي|"
    r"eb3atl\w*|ib3atl\w*|b3atl\w*|eb3ath\w*",
    re.IGNORECASE,
)


def _asks_to_send_location(text: str) -> bool:
    return bool(_SEND_LOCATION_RE.search(text or ""))


# Found live: PRICING_DIRECTIVE/CONTACT_REQUEST_DIRECTIVE/ELABORATION_DIRECTIVE each require
# offering the onsite-vs-online choice, but nothing checked whether an EARLIER turn already gave
# that same full explanation — so a second pricing question later in the same conversation got the
# identical "onsite at BLNK's office... or a free 15-minute online consultation" sentence verbatim
# again, reading like a scripted bot repeating itself rather than a person continuing a
# conversation. This is checked across ALL prior assistant turns (not just the last one), since the
# repeat can be several turns back — a later turn not immediately following it.
def _booking_choice_already_offered(chat_history: list) -> bool:
    for m in chat_history or []:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if _ONSITE_OFFER_TERMS_RE.search(content) and _ONLINE_OFFER_TERMS_RE.search(content):
                return True
    return False


# User's explicit stance (2026-07-21): the meeting/call must not be suggested more than once in a
# conversation, full stop — not just in the one scenario (a field reveal after an earlier offer) that
# first surfaced it, but in EVERY scenario a directive could re-raise it (a second pricing question,
# a subject change after the first offer, etc.). Broader than _booking_choice_already_offered above
# (which only counts the FULL two-option explanation) — this catches ANY prior mention at all, e.g. a
# bare "we can set up a meeting" from a contact-request turn that never spelled out onsite/online.
# Used to fully suppress unprompted re-mentions (pricing repeats, call_flow's subject-change branch);
# contact_request is the one exception — see its own comment where this is used.
def _meeting_ever_mentioned(chat_history: list) -> bool:
    return any(
        m.get("role") == "assistant" and _CALL_OFFER_RE.search(m.get("content", ""))
        for m in chat_history or []
    )


# Used ONLY for contact_request repeats (2026-07-21 narrowing): that directive fires when the
# CUSTOMER themselves is explicitly asking how to get in touch or move forward — a direct question
# that genuinely needs an answer, not the bot volunteering the topic. So a brief reference, rather
# than silence, is the right call there. Every OTHER directive that used to share this note (pricing,
# call_flow's subject-change branch) now gets full suppression instead — see PRICING_REPEAT_NO_
# MEETING_NOTE below and CALL_FLOW_NEW_TOPIC_REMINDER — since those are the bot re-raising the topic
# on its own, which the user was explicit must not happen more than once per conversation.
BOOKING_CHOICE_REPEAT_NOTE = (
    "You already offered this same onsite/online meeting choice earlier in this conversation — do "
    "NOT restate or re-explain both options in full again, whether verbatim or reworded (no repeating "
    "'onsite at BLNK's office... or a free 15-minute online consultation' a second time), and do NOT "
    "recap, restate, or re-pitch any service/deliverable content already covered earlier either — "
    "this reply's only job is pointing them back to the meeting, nothing else. Say exactly this, word "
    f"for word, every time this situation comes up — never a different wording from one turn to the "
    f"next: \"{BOOKING_CHOICE_REPEAT_FALLBACK['en']}\" (the natural Arabic equivalent if replying in "
    "Arabic) — never spell out the onsite/online details again."
)

# Found live: a cold-open first message ("khalas onsite beje la aandkon, wen ento mawjoden" — "OK
# onsite, I'll come to you, where are you located") triggered CONTACT_REQUEST_DIRECTIVE (offered_call
# was still false — nothing had been offered yet in this brand-new conversation), and its
# BOOKING_INSTRUCTION assumes the onsite/online choice hasn't been made yet, so it presented both
# options generically and asked which they'd prefer — completely ignoring that the customer had
# ALREADY told us which one, in the very same message, and never gave the address they explicitly
# asked for ("wen ento mawjoden"). This note fires instead whenever the customer's OWN message
# already names a mode, overriding BOOKING_INSTRUCTION's "offer as a choice" framing.
MODE_ALREADY_SPECIFIED_NOTE = (
    "IMPORTANT override: the customer's OWN message already specifies which meeting mode they want — "
    "do NOT present the onsite/online choice as still open, and do NOT ask which they'd prefer; "
    "they already told you. If they named ONSITE (e.g. mentioning Lebanon, the office, or wanting to "
    "meet in person): give BLNK's real office address from the retrieved context and ask what day and "
    "time works, in this SAME reply. If they named ONLINE: give them the booking link directly: "
    f"{CALENDLY_LINK} . Either way, skip the 'here are your two options' framing entirely and answer "
    "whatever else they asked (e.g. their location question) using this same resolution. If an earlier "
    "instruction in this prompt described this message as 'just a closing acknowledgment', 'nothing "
    "new', or similar — that description is WRONG for this specific message: naming a meeting mode IS "
    "real, substantive content. Do NOT open with a bare 'You're welcome' / 'thanks' / generic closing "
    "word before the resolution — open directly with the address or the link itself."
)

# User's explicit stance (2026-07-21): the meeting/call must never be suggested more than once in a
# conversation — not even a brief passing reference — when it's the BOT re-raising it unprompted (as
# opposed to the customer directly asking, which is contact_request's job, above). A second pricing
# question later in the conversation is exactly this: the customer is asking about cost again, not
# asking about the meeting, so the reply owes them the scope-dependency line and NOTHING about
# booking — the option is already on the table from earlier, repeating it (even briefly) is the
# unprompted re-raising the user asked to stop.
#
# Two later refinements (2026-07-24), both about NOT reciting the canned scope-dependency sentence
# verbatim a second time, for two different reasons:
#   - The customer's LATEST message itself hands over the concrete specifics the first answer said
#     pricing/timing depends on (names the service, the business size, etc.) instead of asking again.
#     Repeating "it depends on the scope/services/size" back at someone who just told you those things
#     reads as not having listened — found live: asked "approximately what is the price", got the
#     scope-dependency answer, replied "i want a logo and we are a small business" (i.e. answered it),
#     and got the EXACT SAME scope-dependency sentence again verbatim. Here the meeting mention is
#     brought back in as the exception to the no-repeat-meeting rule above: it's now a concrete next
#     step responding to new information, not an unprompted re-raise.
#   - The customer is insisting/pressing again after already getting the scope-dependency explanation
#     (e.g. "but approximately how much/how long"). Also redirect to the meeting as where the real
#     number gets nailed down, rather than trying to improvise a fresh explanation of why it varies.
# Revised again (2026-07-24, user's explicit correction to the same-day simplification above): price
# and timing are NOT one merged bucket — each is tracked as its OWN dimension, explained once. If price
# was already explained (and the meeting offered with it) but timing has never been asked before, a
# first-ever timing question still gets the real "how long it takes depends on..." explanation — it
# just skips re-offering the meeting, since that's already on the table. Only once THAT SAME dimension
# (cost, or timing) has been explained once does a later ask about it collapse to the one-line meeting
# redirect. _topic_scope_already_explained (above) is what tells the two cases apart; the call site
# picks ONE of the two notes below per turn, never both.
PRICING_FIRST_EXPLAIN_NO_MEETING_NOTE = (
    "The meeting was already offered earlier in this conversation for a DIFFERENT question (price or "
    "timing, whichever wasn't just asked) — but THIS specific dimension (read the customer's message "
    "to tell whether they're asking about cost or about timing now) has never actually been explained "
    "yet. Give the real scope-dependency explanation for THIS dimension now, same as if this were the "
    "first pricing question of the conversation (e.g. 'how long it takes depends on the scope of what "
    "you need, the specific services involved, and the size of your business') — do NOT skip it just "
    "because pricing came up before. But do NOT re-offer or re-explain the meeting/onsite-vs-online "
    "choice in this reply — it's already on the table from earlier, mentioning it again here would be "
    "the unprompted re-raise the customer should never see twice."
)

PRICING_REPEAT_NO_MEETING_NOTE = (
    "THIS specific dimension (read the customer's message to tell whether it's about cost or about "
    "timing) already got its scope-dependency explanation earlier in this conversation, and the "
    "meeting was already offered too. Do NOT recite that scope-dependency explanation again in ANY "
    "form — not the full sentence, not a reworded version, not a shortened one — whether this is a "
    "brand new question, a repeat, an insist for a ballpark, or the customer answering with their own "
    "specifics. This reply has exactly ONE job: one short, plain sentence — always this same message, "
    "word for word every time this situation comes up — that pricing and timing specifics are always "
    "discussed during the meeting. Nothing else: no number, no range, no re-explanation of why it "
    "varies, and do NOT re-explain or re-offer the onsite-vs-online choice itself, it was already "
    "given earlier."
)


def last_assistant_offered_call(chat_history: list) -> bool:
    for m in reversed(chat_history):
        if m.get("role") == "assistant":
            return bool(_CALL_OFFER_RE.search(m.get("content", "")))
    return False


_DEPEND_WORD_RE = re.compile(_DEPEND_WORD, re.IGNORECASE)


def last_assistant_gave_scope_dependency(chat_history: list) -> bool:
    """Same shape as last_assistant_offered_call, same reason it's reliable: this checks OUR OWN
    controlled phrasing (did our last reply give a scope-dependency acknowledgment, e.g. "it depends
    on the scope/services/size of your business"), not an attempt to regex-classify the customer's
    free-text reply. Used to recognize that a follow-up answering exactly those factors (naming the
    service, the business size, etc.) — without asking a fresh price/timing question of its own — is
    still part of the SAME pricing/timing exchange we opened, even when nothing in the follow-up
    itself contains a price or timing keyword. General on purpose: this isn't matched against what the
    CUSTOMER said (which would mean hardcoding per-phrasing patterns like "small business"/"logo"),
    only against whether WE ourselves just opened that dependency.

    Uses _DEPEND_WORD_RE, NOT the broader _SCOPE_DEPENDENCY_RE (2026-07-24 fix) — found live, and it
    was bad: _SCOPE_DEPENDENCY_RE also matches "specifics", which appears in the one-line meeting-
    redirect ("pricing and timing SPECIFICS are always discussed during the meeting") that
    PRICING_REPEAT_NO_MEETING_NOTE deliberately produces on a repeat turn. Once that redirect line was
    ever said once, THIS function kept returning True forever after — forcing `pricing = True` on
    every later turn regardless of topic, which hijacked a completely unrelated question ("do you have
    experience with restaurants") into another bare pricing redirect. _DEPEND_WORD_RE only matches the
    actual "depends on .../gets determined by ..." explanation, never the redirect line itself, so it
    correctly stops matching once the redirect has taken over."""
    for m in reversed(chat_history):
        if m.get("role") == "assistant":
            return bool(_DEPEND_WORD_RE.search(m.get("content", "")))
    return False


# A day or a time the user offers as their availability. We detect this in code so we
# can send an unambiguous "defer to the schedule" instruction, instead of asking the
# model to figure out which branch of the call flow it's in.
#
# Split into a DAY half and a TIME half (rather than one merged regex) because the two mean
# different things to CALL_FLOW_DIRECTIVE: a day alone ("tomorrow") still owes the customer a
# time question, while an actual time (a clock time, a day-part like "morning", or "sobh"/"masa")
# is the signal that the slot is fully specified and ready for the "I'll check the schedule and
# get back to you" acknowledgment. Conflating the two (a single day-or-time regex gating that
# acknowledgment) was a real bug: "what about tomorrow" — a day with no time — tripped the
# schedule-check acknowledgment path instead of the "what time works" question, so the bot
# confirmed a slot it never actually had a time for.
_DAY_RE = re.compile(
    r"\b(mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)(day)?\b|"
    r"\b(today|tomorrow|tonight|weekend)\b|"
    r"\b(this|next)\s+week\b|"
    r"الاثنين|الإثنين|التنين|الثلاثاء|التلاتا|الأربعاء|الاربعا|الأربعا|الخميس|"
    r"الجمعة|الجمعه|السبت|الأحد|الاحد|بكرا|بكره|اليوم|غدا|الأسبوع|نهاية الأسبوع|"
    r"bukra|bkra|boukra|bokra",
    re.IGNORECASE,
)

_TIME_RE = re.compile(
    r"\b(morning|afternoon|evening|noon|midnight|anytime|o'?clock)\b|"
    # Hours OTHER than 3 are unambiguous — match with or without minutes/a separator, exactly as
    # before. "3" is handled separately just below.
    r"\b(?:1[0-2]|[124-9])\s*(:|\.)?\s*\d{0,2}\s*(am|pm|a\.m\.|p\.m\.)\b|\b(?:1[0-2]|[124-9])\s*(am|pm)\b|"
    # Found live: bare "3am" collides EXACTLY with the Lebanese arabizi continuous-tense prefix
    # ("3am bi7ke" = "I'm talking", "3am bishtghal" = "I'm working" — extremely common in ANY
    # sentence describing an ongoing action). "3am bi7ke 3an project adde bado wa2et" ("I'm talking
    # about the project, how much time does it need") got misread as the customer giving 3 AM as a
    # meeting time, which derailed the whole conversation into a false "day/time already given"
    # flow. Nobody schedules a business meeting at 3 AM anyway, so bare "3am" is excluded here —
    # it only counts as a real time with actual minutes/a separator ("3:15am") or an explicit
    # preceding time-context word. "3pm" has no such collision and stays unrestricted.
    r"\b3\s*(:|\.)\s*\d{1,2}\s*(am|pm|a\.m\.|p\.m\.)\b|\b3\s*pm\b|"
    r"\b(?:at|around|3ala|3al|aal|aan)\s*3\s*am\b|"
    # A bare hour with NO am/pm marker — "bokra aal 4" (tomorrow at 4), "call me at 4", "الساعة 4"
    # — is extremely common casual phrasing (the am/pm is implied by context, not spelled out) and
    # was missed entirely before: with no am/pm the two patterns above don't fire, so a customer who
    # answered the time question this way looked to the code like they'd given no time at all,
    # silently skipping the daytime-handling guarantee. Require an explicit "at"-style marker right
    # before the number (English "at"/"around", arabizi "3ala"/"3al"/"aal", Arabic "الساعة"/"عند")
    # rather than matching any bare digit, so this doesn't fire on an unrelated number like "4 people".
    r"\b(at|around|3ala|3al|aal|aan)\s*\d{1,2}\b|(?:الساعة|عالساعة|عند)\s*\d{1,2}\b|"
    # A SPELLED-OUT hour ("meet tomorrow at four") — the digit-only pattern above misses this
    # entirely. Hours are a small, closed set (one-twelve), not an open-ended vocabulary, so
    # enumerating them is safe/bounded, unlike the arabizi business-word lexicon elsewhere in this
    # file. Found live: this exact gap was why confirms_meeting_daytime (the LLM backstop) existed —
    # but that signal turned out to have its OWN false-positive problem (see gave_specific_time
    # below), so closing the regex gap directly, rather than depending on the backstop, is the more
    # reliable fix.
    r"\b(at|around)\s*(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b|"
    # Match the ROOT word (صباح/مساء) as a substring rather than enumerating specific prefixed/
    # suffixed forms ("الصباح" WITH the definite article, "صباحا"/"صباحاً" WITHOUT it plus a
    # trailing alef/tanwin, or the bare noun "صباح"/"مساء" with neither) — enumerating variants
    # kept missing real ones live ("10 صباحا" first, then bare "مساء" too). A substring match on the
    # root covers every inflected form at once. Also add the arabizi (Latin-letter) equivalents —
    # "sobh"/"masa" etc. — since this whole pattern only had English and Arabic-script
    # coverage; the customer could just as easily write the identical conversation in arabizi.
    r"صباح|مساء|الظهر|الضهر|العصر|"
    r"sobh|sob7|sbeh|sabeh|masa2?|msa2?",
    re.IGNORECASE,
)


def mentions_day(text: str) -> bool:
    return bool(_DAY_RE.search(text or ""))


def mentions_specific_time(text: str) -> bool:
    return bool(_TIME_RE.search(text or ""))


def mentions_day_or_time(text: str) -> bool:
    return mentions_day(text) or mentions_specific_time(text)


# A short affirmation ("ok", "yes", "sure", "yes please", "نعم"...). We only treat a
# reply as "accepting the call" when it's an affirmation — that way, if the user
# instead changes the subject after we offered the call, they get normal handling
# (with context) rather than being pushed into the scheduling step. Built from a set
# of affirmative tokens that may repeat ("ok sure", "yes please"), plus a few fixed
# multi-word phrases.
_AFFIRM_TOKEN = (
    r"(?:y|ya|yes+|yea+h?|yep|yup|ok|oke?|okay|k|kk|sure|fine|great|good|perfect|"
    r"cool|deal|alright(?:y)?|absolutely|definitely|please|sounds|"
    # Latin arabizi yes-words
    r"eh+|aa+|ah+|ay+|aywa|aywe|ayweh|akid|akeed|tamem|tamam|tmem|tmam|tab|tabe|"
    r"mashe|mashi|mnih|mne7|mni7|mnee7|yalla|yala|"
    r"نعم|أجل|اجل|اكيد|أكيد|تمام|طيب|ماشي|ماشى|اوك|اوكي|أوكي|حلو|منيح|منيحة|"
    r"يلا|يالله|اي|أي|ايه|أيه|اه|آه|ايوا|أيوا|ee+|يس)"
)
_AFFIRM_RE = re.compile(
    rf"^\W*(?:go\s*ahead|why\s*not|let'?s\s*do\s*it|{_AFFIRM_TOKEN})"
    rf"(?:[\s،,]+{_AFFIRM_TOKEN})*\W*$",
    re.IGNORECASE,
)

# Acceptance PHRASES the token-repeat regex above can't express — chiefly "(eh) mish mshkle"
# = "(yeah) no problem", a very common Lebanese way to say yes. Optionally led by a yes-word,
# and must be essentially the whole message so "msh mshkle bas 3andi soal" (…but I have a
# question) is NOT swallowed. Note "msh baddi" (I don't want) won't match — it requires the
# "mushkil/mshkle" stem after the negation.
_ACCEPT_PHRASE_RE = re.compile(
    r"^\W*"
    r"(?:(?:eh|ee+|aa+|ah|ay|akid|akeed|tab|tabe|tamem|tamam|ok(?:ay)?|yes+|sure)[\s،,]+)?"
    r"(?:"
    r"(?:m[ei]?sh|mesh|wala|ma\s*fi)\s*(?:mushk|moshk|mshk|meshk)\w*"  # (mish) mushkil
    r"|no\s+problem|no\s+worries|sounds\s+good|works\s+for\s+me|fine\s+by\s+me"
    r"|tikram\w*|tekram\w*"
    r")\W*$",
    re.IGNORECASE,
)


def is_affirmation(text: str) -> bool:
    t = (text or "").strip()
    return bool(_AFFIRM_RE.match(t) or _ACCEPT_PHRASE_RE.match(t))

# Deliberately NARROWER than is_affirmation/is_confirmation: only matches an actual gratitude/closing
# word ("thank you", "شكرا", "tislam"), never a plain "yes"/"ok" on its own. This distinction matters
# because a bare "yes"/"ok" is often a genuine continuation signal (SYSTEM_PROMPT: "a bare yes/ok
# refers to what you just offered — continue it") — e.g. answering the call offer or a field-confirm
# question — and must keep working exactly as before. "Thank you" (with or without a leading "great"/
# "ok"/"tamam") is different: it's the customer signaling the conversation is DONE, not agreeing to
# something still open, so it's the only case BARE_ACK_DIRECTIVE (below) should fire on.
# Farewell words added (2026-07-24): same "nothing left to do, safe to just acknowledge and stop"
# closing signal as gratitude — found live: "bye" alone, and "anyway thanks, don't want to do business
# with you" (which DOES contain "thanks" but is 10 words, right at is_gratitude_closing's word-count
# edge) both need to be recognized as a genuine closing, not left to fall through to whatever directive
# a stale flag happens to be forcing that turn.
_GRATITUDE_RE = re.compile(
    r"\bthanks?\b|\bthank\s*you\b|\bthx\b|\btnx\b|\bty\b|"
    r"\bbye\b|\bgoodbye\b|\bsee\s*you\b|\btake\s*care\b|"
    r"شكر|تسلم|تسلمي|يعطيك\s*العافي|يعطيكي\s*العافي|مشكور|مع\s*السلامة|باي|"
    r"\bshukran?\b|\bshokran?\b|\bshukren\b|\btislam\w*|\bteslam\w*|\bmerci\b|\bbye\w*",
    re.IGNORECASE,
)


def is_gratitude_closing(text: str) -> bool:
    """True only for a short closing message that's essentially just thanks/an acknowledgment —
    a real gratitude word present, with nothing past a short affirmation clause around it (so
    'thanks, but can you also tell me about pricing' is NOT this — the trailing question still
    needs a real answer)."""
    t = (text or "").strip()
    if not t or not _GRATITUDE_RE.search(t):
        return False
    if "?" in t or "؟" in t:
        return False
    return len(re.findall(r"\w+", t)) <= 10


# Found live: a field-pitch reply ended with its OWN trailing open question ("What are you
# interested in discussing further?"), which made last_assistant_asked_question=True for the NEXT
# turn — so a bare "nice"/"cool"/"haha" in reply got excluded from bare_ack (the whole point of that
# gate is "don't treat a bare word as closing if it might be ANSWERING a real question"), even though
# a pure reaction word like "nice" never actually answers an open-ended "what are you interested in"
# question either way — it's just a vague positive reaction, same as thanks is a closing reaction.
# Same shape/role as is_gratitude_closing above (bypasses the question-gate for a narrow, high-
# confidence word list), just for reactions instead of gratitude specifically.
# "great"/"good"/"fine" (bare, on their own) added 2026-07-24: the shape that answers OUR OWN "and
# you?" after SMALL_TALK_DIRECTIVE's reciprocal reply to a "how are you" — without this, the model's
# own is_bare_ack gate never fires for it (no gratitude word, not a real confirmation to a real
# question), so it kept echoing "great" back a second time instead of just moving the conversation on.
_VAGUE_REACTION_RE = re.compile(
    r"^\W*(nice|cool|wow|omg|lol+|ha+h+a*|he+h+e*|amazing|awesome|great\s*stuff|nice\s*one|"
    r"i'?m\s*(great|good|fine|well)|great|good|fine|"
    r"لطيف|حلو|رهيب|والله\s*حلو|منيح|منيحة|تمام|كويس|كويسة)\W*$",
    re.IGNORECASE,
)


def is_vague_reaction(text: str) -> bool:
    return bool(_VAGUE_REACTION_RE.match((text or "").strip()))


# A pure pleasantry ("how are you", "kifak/kifik" on its own, "how's it going") with no real
# informational content — distinct from _GREETING_RE (which deliberately excludes "kifak" since it
# doubles as this, see that regex's own comment) and from is_bare_ack (which needs a real gratitude/
# reaction/confirmation word, none of which "how are you" contains). Used by main.py's knowledge-gap
# email trigger: this kind of message naturally has no relevant retrieval chunk (nothing in the
# knowledge base about small talk) and gets a conversational non-answer by design, per SYSTEM_PROMPT —
# neither of which means the bot actually failed to know something. Found live: "hii" + "how are you"
# sent as one burst got emailed to the team as a knowledge gap despite being ordinary small talk.
_SMALL_TALK_RE = re.compile(
    r"^\W*(?:how\s*(?:are|r)\s*(?:you|u)\??|how'?s\s*it\s*going\??|what'?s\s*up\??|"
    r"kifak|kifik|kifkon|شو\s*أخبارك|كيفك|كيفكن|شلونك)\W*$",
    re.IGNORECASE,
)


def is_small_talk(text: str) -> bool:
    return bool(_SMALL_TALK_RE.match((text or "").strip()))


# User's explicit product decision (2026-07-24): when the customer tells us we answered the WRONG
# thing, the safe move is never to guess again — that's exactly how a fresh misread compounds the
# original one (found live: "i did not ask for cost" still got answered as a pricing question). The
# correct response is the same standing escalation as any other genuine gap ("I'll check and get back
# to you"), which also means it should trigger the same knowledge-gap email so a human actually
# resolves the mix-up — bot_admits_unknown already recognizes that hedge, so this ties in for free.
# General on purpose (not scoped to price/timing specifically): "that's not what I asked", "I meant X
# not Y", "you misunderstood" all correct the SAME class of mistake regardless of topic.
_CORRECTION_RE = re.compile(
    r"\b(i|we)\s*(did\s*(not|n'?t)|didn'?t|do\s*(not|n'?t)|don'?t)\s*ask|"
    r"that'?s\s*not\s*what\s*(i|we)\s*(asked|meant|said)|"
    r"not\s*what\s*(i|we)\s*(asked|meant|said)|"
    r"\bnot\s*(that|this)\b.{0,15}\b(asked|meant)|"
    r"\byou\s*(misunderstood|misread|got\s*(it|that|this)\s*wrong)\b|"
    r"that'?s\s*(wrong|incorrect)\b|"
    r"مش\s*هيك\s*(سألت|قصدي)|ما\s*هيك\s*قلتلك|فهمتني?\s*غلط|مش\s*هاد\s*يلي\s*قصدي|هاد\s*مش\s*يلي\s*سألت",
    re.IGNORECASE,
)


def is_correction(text: str) -> bool:
    return bool(_CORRECTION_RE.search(text or ""))


# User's explicit correction to an earlier version of this directive (2026-07-24): the first version
# said "always escalate, never guess" unconditionally — but a correction often ALREADY contains the
# answer ("I just want a logo, I did not ask for branding or marketing" tells you exactly what they
# want: logo only, nothing else), and blindly escalating on that ignores information they just handed
# over for free. Escalating is for when the correction leaves you genuinely still unsure what to say,
# not a reflex for every correction-shaped message.
CORRECTION_DIRECTIVE = (
    "The customer is telling you that your LAST reply misunderstood them or answered the wrong thing "
    "— they're correcting you, not asking something new. This OVERRIDES any other instruction earlier "
    "in this prompt for how to phrase this reply, INCLUDING an instruction that says to give one exact, "
    "fixed, word-for-word sentence every time — that instruction was written for an ordinary repeat "
    "question, not for the customer actively telling you that you got it wrong; reciting that same "
    "fixed sentence back at them right now IS the mistake they're pointing out, not the safe default. "
    "Read their correction first: if it ALREADY tells you specifically what they actually want (e.g. "
    "'I just want a logo, not branding or marketing' tells you exactly that — logo only), answer THAT "
    "directly and concretely using what they just clarified, from the retrieved context — do not "
    "escalate when they've already handed you the answer to your own confusion. Only fall back to "
    "acknowledging the mix-up and saying you'll check and get back to them shortly (the standing "
    "escalation for a genuine gap) when, even now, you genuinely don't know what they meant or have no "
    "real answer to give — never as a default reflex. Either way, do NOT repeat the specific wrong "
    "content from your previous reply, and never invent or guess a fact just to avoid saying you'll "
    "follow up when you truly don't know."
)


def _contains_small_talk(text: str) -> bool:
    """True when a pleasantry appears as ONE LINE of a multi-line burst alongside other, substantive
    content — is_small_talk itself requires the ENTIRE message to be just the pleasantry, so a burst
    like 'hii\\nkifak\\nkef fek\\ncan you help my clothing shop' (four rapid-fire messages folded into
    one combined_message, real per main.py's burst-gathering) never matches it at all, and the "how are
    you" asked twice in there got silently dropped — the reply jumped straight into the business
    content with no acknowledgment. Used for the SEPARATE, softer directive below (reciprocate briefly,
    THEN answer the rest) rather than SMALL_TALK_DIRECTIVE's own (this IS the whole message)."""
    return any(is_small_talk(line) for line in (text or "").split("\n"))


# Found live: "how are you" got "I'm here to help you with any questions you have about our services
# or how we can support your business" — a deflection into sales-mode instead of just answering the
# pleasantry, which reads as robotic/evasive for something this simple. A real person asked "how are
# you" just answers it. Injected whenever is_small_talk fires.
SMALL_TALK_DIRECTIVE = (
    "The customer just asked a plain pleasantry (how are you / how's it going), nothing else. Answer "
    "it like a real person would — briefly and warmly (e.g. 'Great, and you?' / 'مبسوط، وانت؟'), then "
    "naturally invite them to say how you can help. Do NOT deflect straight into a sales-mode non-"
    "answer like 'I'm here to help with any questions about our services' — that ignores the question "
    "instead of answering it. Do NOT ask a real business question here. If a meeting/call was already "
    "offered earlier in this conversation, this pleasantry is NOT an opening to mention, re-offer, or "
    "remind them of it — the standing rule that the meeting must never be re-raised unprompted applies "
    "here too, a plain 'how are you' is not them accepting or continuing that flow. Just reciprocate "
    "and invite them to keep asking about anything else, with no reference to the meeting at all. "
    "Same rule for contact info — do NOT volunteer an email address, phone number, or WhatsApp here "
    "either; a pleasantry is not a request for how to reach you, so there's nothing to answer with "
    "contact details."
)

# _contains_small_talk's own comment explains the gap this closes — a pleasantry riding along inside a
# multi-line burst that also has a real question, distinct from SMALL_TALK_DIRECTIVE (which covers the
# pleasantry being the WHOLE message). The greeting-fusion rule in SYSTEM_PROMPT is the model for how
# this should read: a brief opener fused on, not a separate paragraph or something answered instead of
# the real question.
SMALL_TALK_ALONGSIDE_DIRECTIVE = (
    "Among the customer's messages just now, they ALSO asked how you're doing (e.g. 'kifak', 'kef "
    "fek') — don't skip that just because a real question came with it. Briefly reciprocate it (e.g. "
    "'mnih, ' / 'Great, ') fused naturally onto the start of your reply, the same way you'd fuse in a "
    "greeting, THEN answer the real question — do not let it eat the whole reply, and do not answer it "
    "as a separate sentence in the middle or end of the reply."
)


# We inject this whenever OUR previous message offered a meeting (reliable — it matches our own
# controlled phrasing). We do NOT try to regex-classify the customer's reply; the model reads it
# itself, using the surrounding chat history to tell which of the two paths (onsite vs online) is
# actually active, and picks the right branch. This keeps comprehension (which handles any arabizi
# spelling of "yes", or of "I'm in Lebanon") with the model, and keeps the behavioral guarantees here
# in the instruction.
CALL_FLOW_DIRECTIVE = (
    "You already offered the customer a meeting in your last message — either the onsite meeting at "
    "BLNK's office (relevant if they're in Lebanon) or the free 15-minute online consultation, so "
    "their reply is most likely about that. Read what they actually wrote, and the recent conversation "
    "for which path is active — in any language or arabizi spelling — and pick ONE:\n"
    "- If they're choosing or confirming the ONSITE/office meeting (e.g. saying they're in Lebanon, "
    "asking for the office, wanting to meet in person) and you haven't already shared the location: "
    "you MUST include BLNK's ACTUAL office address text from the retrieved context in this reply, "
    "spelled out (e.g. 'Rive Gauche Tower, Corniche El Nahr, Beirut, Lebanon') — a vague 'at our "
    "office' with no address given is NOT acceptable, the customer needs the real address to go "
    "there. Never say you'll give it later or 'after the meeting'. Only if the address truly isn't "
    "present anywhere in the retrieved context, say you'll confirm it shortly instead of guessing — "
    "never invent a neighborhood/address. This reply's core job is to give the address and ask what "
    "day AND time would be convenient for them — in this SAME reply, nothing else. Do NOT pivot to or "
    "offer to discuss the branding, the project, or any other topic here, even if it sounds helpful — "
    "that's not this turn's job and it displaces the day/time question. Do NOT mention the online "
    "link here.\n"
    "- EXCEPTION — if they're JUST asking WHERE you're located, with no other commitment (a plain "
    "location question, not also saying they're in Lebanon or want to meet in person): give the real "
    "address the same way, but STOP there — do NOT also ask what day/time works in this reply. "
    "They're asking out of curiosity, not necessarily committing to schedule anything yet; pushing "
    "the day/time question onto a simple location question reads as pushy. If they follow up "
    "actually wanting to schedule, THAT reply asks for day/time, not this one.\n"
    "- If they gave a DAY ONLY, with no specific time, for the ALREADY-CHOSEN onsite meeting: ask what "
    "time on that day would be convenient — this reply has exactly ONE job, that question, nothing "
    "else (no asking about materials, what to bring, or any other tangent). Ask it DIRECTLY — do NOT "
    "lead with a restated/echoed version of the day they just gave (e.g. 'Bokra n2dar n3mil meeting "
    "3andna bel office' before asking the time is WRONG, it just parrots 'tomorrow' back); go straight "
    "to the time question with no restatement clause in front of it.\n"
    "- If their reply is just a bare acknowledgment/affirmation ('okay', 'perfect', 'sounds good', "
    "'tamam', 'mne7') that does NOT itself add a new day or time — e.g. affirming the location or the "
    "idea of meeting, not answering the day/time question — do NOT treat that as a confirmed slot and "
    "do NOT say you'll check the schedule (nothing to check yet). Open with a short acknowledgment word "
    "('Great' in English, 'تمام' in Arabic) and then ask again for whichever of day/time is "
    "still missing (both, if neither has been given yet) — this reply has exactly ONE job, that short "
    "acknowledgment plus the still-missing question, nothing else.\n"
    "- If they gave a specific DAY AND TIME for the ALREADY-CHOSEN onsite meeting (whether both in this "
    "message, or a time now completing a day given earlier): a brief, natural acknowledgment word "
    "('got it'/'mni7'/'tamam' — NOT a restated/echoed version of the value, e.g. '3pm b2a 3andna' or "
    "'okay, tomorrow at 3pm works' are WRONG, they just parrot the value back) and tell them you'll "
    "check the schedule and get back to them shortly to confirm — this reply has exactly ONE job, that "
    "brief acknowledgment, nothing else (no asking about materials, what to bring, or any other "
    "tangent) — do NOT self-book and do NOT mention the online Calendly link.\n"
    "  Disambiguation note for the two branches above: 'sobh'/'صباح' and 'masa'/'مساء' are AMBIGUOUS "
    "words — they can be a greeting fragment ('sabeh el kheir'/'مسا الخير' = good morning/evening) OR "
    "an AM/PM time marker ('3 masa' = 3pm). If YOUR OWN last message just asked what time works for "
    "them, treat any 'sobh'/'masa' in their reply as answering that question (a time), NOT as a "
    "greeting — replying to a time question with a stray greeting makes no sense, so the time reading "
    "is almost always correct in that context.\n"
    "- If day AND time were ALREADY fully given in an EARLIER turn and you ALREADY told them you'd "
    "check the schedule and get back to them, and their reply now is just a plain acknowledgment "
    "('okay', 'yes', 'deal', 'perfect', 'sounds good') with NOTHING new in it: do NOT restate the "
    "day/time again, and — this is critical — do NOT say or imply the meeting is now confirmed, set, "
    "scheduled, or booked (e.g. 'the meeting is set for Monday at 3', 'I've scheduled it', 'see you "
    "then' are all WRONG — nobody has actually checked the real schedule yet, so asserting a booking "
    "is a promise you can't back). Also do NOT repeat the earlier 'I'll check the schedule and get back "
    "to you' promise again in different words (you already said it once, saying it again reads as a "
    "stall), and do NOT volunteer the contact email/phone here — they haven't asked how to reach you, "
    "that's a separate, unprompted addition. Just close briefly and warmly, with no new meeting claim "
    "and no repeated content in it — e.g. 'Great, talk soon!' — nothing more.\n"
    "- If day AND time were ALREADY fully given and you ALREADY told them you'd check the schedule, and "
    "their reply now gives NEW information that ISN'T a day/time (e.g. 'we're a boutique now', 'اسم "
    "شركتنا...'): briefly and warmly acknowledge that new info (e.g. thank them for it, or a short "
    "'great' / 'تمام') — but do NOT pivot into pitching a service or deliverable based on it, and do "
    "NOT suggest, offer, or ask about scheduling ANOTHER meeting — one is already pending confirmation, "
    "so there's nothing left to book. This reply has exactly ONE job, that brief acknowledgment, "
    "nothing else.\n"
    "- If they are agreeing to or choosing the ONLINE option in ANY way (e.g. 'yes', 'ok', 'eh', 'akid', "
    "'msh mshkle', 'wala mshkle', 'tikram', 'sure', or explicitly asking for the online/zoom call), "
    "reply with ONE short friendly line pointing them to the link to grab whatever slot works for them, "
    f"and include the exact link if it isn't already visible: {CALENDLY_LINK} . Do NOT ask which "
    "day/time suits them and do NOT say you'll check a schedule or get back to them for this option — "
    "they book it themselves.\n"
    "- If they answer the implicit choice by saying where they ARE, and it's somewhere other than "
    "Lebanon (e.g. naming a different country/city, or just saying 'no' to being in Lebanon) — this "
    "means ONLINE, the same as if they'd said the word 'online': the onsite option was only ever "
    "conditional on being in Lebanon, so ruling that out selects the other path automatically. Handle "
    "it exactly like the online-choice case just above (one short line, the link, no day/time "
    "question) — do NOT treat it as an unanswered question needing you to re-explain both options, "
    "and do NOT pivot to any other topic.\n"
    "- If they gave a day/time for the ONLINE option specifically (not the onsite one), don't collect "
    f"or confirm it; just warmly tell them to pick that slot themselves on the link: {CALENDLY_LINK} .\n"
    "- If the ONLINE link was already given and they now say (in any phrasing) that they've booked, "
    "scheduled, or picked a slot themselves (e.g. 'booked it', '7ajazit maw3ad', 'خلص حجزت', 'done'), "
    "or just thank you after doing so: the online path is SELF-service and already confirmed the moment "
    "they picked their slot on the link — do NOT say you'll check the schedule and get back to them to "
    "confirm, that phrase belongs ONLY to the onsite path and is wrong here; there is nothing left for "
    "you to check or confirm. Just a brief, warm closing acknowledgment ('Great, see you then!' / "
    "'تمام، منتطلع نلتقي!') — this reply has exactly ONE job, that acknowledgment, nothing else.\n"
    "- If they decline, hesitate, or say not now, acknowledge it warmly, don't push, and let them "
    "keep asking whatever they want.\n"
    "- If they instead asked something else or changed the subject WITHOUT actually choosing onsite "
    "or online or otherwise accepting the meeting (e.g. revealing their business/field, asking a new "
    "question) — this is DIFFERENT from the 'day/time already fully given' branches above, which "
    "require a slot to already be settled; here NOTHING has been chosen yet. You MUST actually answer "
    "their new question/topic, using the retrieved company knowledge — do NOT skip straight past it to "
    "the meeting; a real question deserves a real answer, every time, even this many turns into the "
    "meeting flow. If they ask about something specific and factual (e.g. a past client, a location, a "
    "detail from the retrieved context), give the actual fact, plainly and directly — don't dodge it "
    "with a vague non-answer. Answer their question and then STOP — do NOT add any closing line about "
    "the meeting, the call, or booking (not even a brief one): it was already offered once earlier in "
    "this conversation, and the user's product decision is that it must never be re-raised by you "
    "unprompted, no matter how many turns pass or how many new topics come up. Do NOT assume they've "
    "accepted, do NOT ask what day or time works, and do NOT re-explain the onsite/online choice in "
    "detail again — nothing has been chosen yet, so there is nothing to schedule.\n"
    "- If, in reply to that kind of open invitation, they NOW give a plain acceptance ('yes', 'sure', "
    "'okay') with no specific mode named: ask BRIEFLY which they'd prefer, onsite or office or "
    "online — a short one-line question, NOT the full explanation again (no re-describing the office "
    "address or the 15-minute consultation, that was already covered earlier). Do NOT jump straight "
    "to asking day/time before a mode has actually been chosen.\n"
    + client_citation_suppressed("the meeting-logistics flow") + " " + STAY_ON_TASK_RULE
)

# Found live (2026-07-21): on CALL_FLOW_DIRECTIVE's "customer changed the subject" branch, the model
# kept doing the two things that branch explicitly forbids — skipping the answer to the customer's
# actual new question, and re-explaining the full onsite/online choice again — even though the
# directive text above already says not to. STAY_ON_TASK_RULE, appended right after (it ends every
# branch of this directive), reads as "this turn is purely transactional, add nothing else," which in
# practice was winning out over that one branch's "answer their new question normally" instruction.
# Rather than rely on the branch text alone to carry that nuance across ~9 other branches worth of
# surrounding instructions, this is injected as its own short, sharply-scoped system message, placed
# AFTER everything else (closer to generation = more weight, same reasoning as BOOKING_CHOICE_REPEAT_
# NOTE below) — cheap to add for every call_flow turn since it's a no-op on the other 8 branches.
CALL_FLOW_NEW_TOPIC_REMINDER = (
    "One more reminder, relevant only if the customer's latest message asked something new or changed "
    "the subject instead of choosing onsite/online or giving a day/time: you must actually answer that "
    "question, using the retrieved company knowledge, before anything else — do not skip it and jump "
    "straight to the meeting. The 'stay on task, nothing extra' instruction elsewhere in this message "
    "is about not padding the LOGISTICS branches (giving the address, asking day/time, acknowledging a "
    "slot) with sales content — it does not mean skip a real question the customer just asked. And "
    "since the meeting was already mentioned earlier in this conversation, do NOT mention it again in "
    "this reply AT ALL — not the full onsite/online explanation, and not even a brief closing clause "
    "like 'we can discuss this in a meeting if you'd like'. The user's explicit product decision is "
    "that the meeting must never be suggested more than once per conversation unless the customer "
    "themselves brings it up again — just answer their question and stop there."
)

# Deterministic detection that THIS reply just gave the office address — the address text itself is
# controlled/known (unlike whether the reply also asked about time, which is free-form model
# phrasing, checked separately below via an LLM call).
_OFFICE_ADDRESS_RE = re.compile(r"Rive Gauche|ريف غوش|Corniche El Nahr|كورنيش النهر", re.IGNORECASE)


def _gave_office_address(text: str) -> bool:
    return bool(_OFFICE_ADDRESS_RE.search(text or ""))


# The real, known address text — used both to detect it (above) and to guarantee it reaches the
# customer (below) when the model drops it. Only an English/Latin-script form exists in the data;
# an "ar" reply keeps it in Latin script too (proper names/addresses are the standing exception to
# "Arabic only" in LANG_DIRECTIVE["ar"]), so one constant covers both reply languages.
OFFICE_ADDRESS_TEXT = "Rive Gauche Tower, Corniche El Nahr, Beirut, Lebanon"

# Same guarantee pattern as the office address above, for a SEPARATE known failure mode (found live,
# documented in _DONT_KNOW_RE's own comment): "who is the owner" retrieves the right chunk (the
# owner's name) at rank #1, comfortably above the knowledge-gap score threshold — retrieval succeeds
# — but generation still declines to use it, deflecting into "I'll confirm and follow up" instead of
# just answering. A SYSTEM_PROMPT clarification alone (that this is public info, not private) wasn't
# enough to reliably fix the underlying reluctance, so — same as the office address — the fact itself
# is guaranteed deterministically rather than left to the model's judgment every time.
_OWNER_QUESTION_RE = re.compile(
    r"who.{0,20}(owner|owns|founder|founded|ceo)|owner'?s?\s*name|"
    r"مين.{0,15}(مالك|صاحب|مؤسس)|شو اسم.{0,15}(مالك|صاحب|مؤسس)",
    re.IGNORECASE,
)
OWNER_NAME = "Dina Rifai"
# Arabic-script transliteration, for the same reason CALL_OFFER/PRICE regexes carry Arabic
# alternatives — an Arabic reply gives the name in Arabic script (e.g. "دينا رفاعي"), never the Latin
# spelling, so a Latin-only substring check always misses it. Found live: an Arabic reply correctly
# gave the name in Arabic, _mentions_owner_name still returned False (pure Latin substring check), and
# the final-safety-net guarantee below appended the Latin name in English on top of the correct Arabic
# answer — same reply, two names, two scripts, two languages.
OWNER_NAME_AR = "دينا رفاعي"


def _asks_about_owner(text: str) -> bool:
    return bool(_OWNER_QUESTION_RE.search(text or ""))


def _mentions_owner_name(text: str) -> bool:
    text = text or ""
    return OWNER_NAME in text or OWNER_NAME_AR in text


def _owner_name_already_given(chat_history: list) -> bool:
    return any(
        m.get("role") == "assistant" and _mentions_owner_name(m.get("content", ""))
        for m in chat_history or []
    )

# Same guarantee, for the phone/WhatsApp number — found live in the SAME smoke-test pass as the owner
# fix above: asked plainly "do you have a phone number for whatsapp", on a fresh conversation with
# nothing else going on, and the reply skipped the number entirely and pivoted straight to offering a
# meeting — a different directive (CONTACT_REQUEST_DIRECTIVE) than the owner case, but the identical
# underlying failure mode (a specific, directly-available fact in the retrieved context gets skipped).
_PHONE_QUESTION_RE = re.compile(
    r"phone\s*(number|no\.?|#)?|whatsapp|\bwp\b|contact\s*(number|no\.?)|call\s*you|"
    r"رقم\s*(هاتف|تلفون|واتساب|واتس)|تواصل.{0,10}هاتف",
    re.IGNORECASE,
)
CONTACT_PHONE = "+961 79 057 374"


def _asks_about_phone(text: str) -> bool:
    return bool(_PHONE_QUESTION_RE.search(text or ""))


# User's explicit product decision (2026-07-26): when the customer specifically asks to speak to a
# manager/boss/human directly, give the WhatsApp number as the way to actually reach a person —
# instead of the generic meeting-choice offer, which doesn't address "I want a HUMAN" at all. Checked
# UNCONDITIONALLY (like the owner-name guarantee), not scoped to any one winning directive — found
# live this kept getting routed into the generic "go ahead with the meeting option" callback
# regardless of which directive happened to win.
_HUMAN_ESCALATION_RE = re.compile(
    r"\bmanager\b|\bboss\b|\bsupervisor\b|\breal\s*person\b|\bhuman\b|\bactual\s*person\b|"
    r"talk\s*to\s*(a\s*)?person|speak\s*to\s*(a\s*)?person|"
    r"مدير|المسؤول|شخص\s*حقيقي|بشر(?!ي\s*هوية)",
    re.IGNORECASE,
)


def _asks_for_manager_or_human(text: str) -> bool:
    return bool(_HUMAN_ESCALATION_RE.search(text or ""))


def _mentions_phone(text: str) -> bool:
    return CONTACT_PHONE in (text or "")


def _phone_already_given(chat_history: list) -> bool:
    return any(
        m.get("role") == "assistant" and _mentions_phone(m.get("content", ""))
        for m in chat_history or []
    )

# Shared by both guarantees above (and any future one of this shape): when the model hedges with the
# standing "I'll confirm and follow up" escalation line (see _DONT_KNOW_RE) and we then have to force
# the real fact in anyway, leaving the hedge in place reads as directly self-contradictory — "I'll get
# back to you" immediately followed by the actual answer in the same breath (found live: exactly this,
# for the owner's name). Strip that sentence before appending the guaranteed fact.
# Arabic half widened (2026-07-24): the original two-sided pattern (a confirm-root clause AND a
# separate get-back-root clause) missed "بدي أأكدلك إنو..." ("let me confirm to you that...") on its
# own — a first-person "أأكد" conjugation the original alternation list didn't include, and this
# phrasing doesn't always pair it with an explicit "و/رح ارجعلك" clause the way the model's other
# variants do. Second branch below catches any clause built on the أكد (confirm) root alone, since
# Arabic verb conjugation varies too much to enumerate exhaustively and the "أكد" root itself is a
# reliable enough signal for this specific hedge on its own.
_CONFIRM_FOLLOWUP_RE = re.compile(
    r"[^.?!]*\b(confirm|check)\b[^.?!]*\b(follow up|get back|revert)\b[^.?!]*[.?!]|"
    r"[^.؟!]*(اتأكد|بتأكد|شيك|بتحقق)[^.؟!]*(ارجعلك|بيرجعلك|برجع)[^.؟!]*[.؟!]|"
    r"[^.؟!]*\S*أكدلك\S*[^.؟!]*[.؟!]",
    re.IGNORECASE,
)


def _strip_confirm_followup_hedge(text: str) -> str:
    return re.sub(r"\s+", " ", _CONFIRM_FOLLOWUP_RE.sub("", text or "")).strip()

# One rep (Sami or Lana) is assigned per conversation — see main.py, which picks and persists it in
# the profile the first time a session is seen. Self-gated (only surfaces if actually asked) so it's
# safe to include on every turn without a dedicated intent signal.
def rep_directive(assigned_rep: str) -> str:
    return (
        f"If the customer asks who they're speaking with, who's handling their conversation, or for "
        f"a name — you are {assigned_rep}. Only mention this if they actually ask; never volunteer "
        f"your name unprompted."
    )


def _address_already_given(chat_history: list) -> bool:
    """True if ANY earlier assistant turn already gave the address — so the guarantee below only
    ever fires on the turn that's actually supposed to give it for the first time."""
    return any(
        m.get("role") == "assistant" and _gave_office_address(m.get("content", ""))
        for m in chat_history or []
    )


def _link_already_given(chat_history: list) -> bool:
    """True if ANY earlier assistant turn already shared the Calendly link — the online-path
    equivalent of _address_already_given. Used to tell the online path apart from the onsite one
    when the customer's reply itself is path-agnostic (e.g. 'booked it', 'thanks') — see the
    online-self-booking-acknowledged guarantee below."""
    return any(
        m.get("role") == "assistant" and CALENDLY_LINK in m.get("content", "")
        for m in chat_history or []
    )


def _implies_onsite_without_address(text: str) -> bool:
    """True if the draft clearly signals the onsite/office path (mentions office/onsite terms, e.g.
    asking what day/time works 'to meet at our office') but is missing the actual address text — a
    strong signal the model meant to give the address here but dropped it, rather than a turn where
    giving the address genuinely isn't this turn's job (e.g. declining, changing the subject, or the
    online path). Found live: a reply asked 'what day and time would be convenient to meet at our
    office?' with no address anywhere.
    Also requires the draft NOT to also mention the online terms (2026-07-20 fix): a reply that
    mentions BOTH onsite AND online terms is RE-PRESENTING the choice ("would you prefer onsite... or
    online...?"), not confirming the onsite path — found live that without this, a customer who
    pivoted to a new topic mid-choice (neither path picked yet) got the address/day-time cascade
    forced onto a reply that was only supposed to re-open the choice, not commit to one side."""
    return (
        bool(_ONSITE_OFFER_TERMS_RE.search(text or ""))
        and not _gave_office_address(text)
        and not _ONLINE_OFFER_TERMS_RE.search(text or "")
    )


def office_address_fix_directive() -> str:
    return (
        f"Your previous reply talked about the onsite/office meeting (asking about day/time, etc.) "
        f"but never actually gave the real office address — the customer needs it to go there. "
        f"Rewrite the SAME reply, same intent and length, including the ACTUAL address spelled out: "
        f"'{OFFICE_ADDRESS_TEXT}'."
    )


OFFICE_ADDRESS_FALLBACK = {
    "en": f" Our office is located at {OFFICE_ADDRESS_TEXT}.",
    "ar": f" عنواننا هو {OFFICE_ADDRESS_TEXT}.",
}


# True exactly on the turn where an onsite meeting becomes booked: WE already gave the office
# address (so the onsite path, not online, is the active one — see CALL_FLOW_DIRECTIVE's address
# branch) and the CUSTOMER just replied with an actual TIME for it (its third branch — "acknowledge
# it warmly... get back to them shortly to confirm"). Gated on mentions_specific_time, not
# mentions_day_or_time: a day alone ("tomorrow") still owes them a time question (the second
# branch), and treating a bare day as a confirmed slot was a real bug. There's no self-booking
# calendar for onsite, so this chat-text signal is the closest thing to a booking confirmation event.
#
# `confirms_meeting_daytime_signal` is understand_turn()'s own read on the same question (does this
# message give a specific time), passed in and OR'd with the regex. This sends a real, business-
# critical email — unlike the other day/time checks in this file, there's no LLM-verified regen
# loop backstopping it, so a regex-only gate here means an unseen phrasing of "at 4" silently never
# triggers the email at all, with nothing to catch it. The regex stays the primary/fast path (no
# extra latency, since understand_turn() already runs once per turn regardless), and the LLM signal
# is the fallback that catches whatever wording the regex hasn't been taught yet — same
# comprehension-model/guarantee-code split as everywhere else in this file, just applied to an
# action instead of to reply text.
def onsite_meeting_confirmed(chat_history: list, user_message: str,
                              confirms_meeting_daytime_signal: bool = False) -> bool:
    if not last_assistant_offered_call(chat_history):
        return False
    last_assistant = next(
        (m.get("content", "") for m in reversed(chat_history) if m.get("role") == "assistant"),
        "",
    )
    # Same deterministic override as gave_specific_time in generate_answer (2026-07-20): don't trust
    # confirms_meeting_daytime when the regex positively finds a day with no time at all — this signal
    # has a reliable false positive on "day + approval word" phrasing ("monday would be good"), and
    # this function gates a real side effect (the booking-confirmation email), so it matters even more
    # here that a day-only reply never gets treated as a confirmed slot.
    day_only_per_regex = mentions_day(user_message) and not mentions_specific_time(user_message)
    return _gave_office_address(last_assistant) and (
        mentions_specific_time(user_message)
        or (bool(confirms_meeting_daytime_signal) and not day_only_per_regex)
    )


# Detects that OUR last message already made the "I'll check the schedule and get back to you"
# promise (CALL_FLOW_DIRECTIVE's day+time branch, or DAYTIME_ACK_FALLBACK) — used below so a
# FOLLOW-UP plain "okay"/"deal" doesn't get treated as license to fabricate a booking confirmation.
# Matches our own controlled phrasing (same reasoning as _CALL_OFFER_RE/_OFFICE_ADDRESS_RE above),
# not the customer's free text, so this stays reliable.
_SCHEDULE_CHECK_PROMISED_RE = re.compile(
    r"check the schedule|checking the schedule|get back to you|"
    r"شيك ع الجدول|شيك على الجدول|هشيك عالجدول|رح.{0,4}شيك|ارجعلك|هرجعلك|"
    r"shik 3al schedule|shik el schedule|shik 3a schedule|arja3lak|raja3lak",
    re.IGNORECASE,
)


def _schedule_check_already_promised(chat_history: list) -> bool:
    last_assistant = next(
        (m.get("content", "") for m in reversed(chat_history) if m.get("role") == "assistant"),
        "",
    )
    return bool(_SCHEDULE_CHECK_PROMISED_RE.search(last_assistant))


# CALL_FLOW_DIRECTIVE's onsite branch requires giving the address AND asking what day/time works, in
# the SAME reply — found live: a reply gave the real address, then instead of asking about timing it
# padded on an unrelated service pitch, silently dropping the time question entirely and leaving the
# customer with no path to actually book. The extra pitch content is caught by _check_off_task_pitch
# above; this checks the OTHER half of the same requirement — that the time-ask wasn't dropped in the
# process. The model's phrasing of that question is free-form (not a fixed string we control), so —
# same approach as the other semantic checks here — we ask the model rather than match a fixed regex.
TIME_QUESTION_CHECK_PROMPT = """You are checking ONE thing in a draft reply from a branding agency's
chatbot. The customer just confirmed an onsite meeting is the path they want, and the assistant's
instructions require the reply to ALSO ask what DAY AND TIME would be convenient for them, in this
SAME reply (in any language or phrasing). Asking ONLY about time with no mention of day at all (e.g.
"what time would be convenient for you to meet?" with no "what day" anywhere) is NOT sufficient — found
live, this exact partial phrasing keeps slipping through, so be strict about it. Both the day part and
the time part need to be covered — either as one combined question ("what day and time works for
you?", "ayya youm w wa2t bnasbak") or two separate questions in the same reply. Does the draft reply
actually ask about BOTH? Output ONLY JSON: {"asks_day_and_time": true/false}"""


def _check_asks_time(reply: str) -> bool:
    if not reply:
        return False
    data = _llm_check_json(TIME_QUESTION_CHECK_PROMPT, f"Draft reply:\n{reply}")
    return bool(data.get("asks_day_and_time"))


TIME_QUESTION_FIX_DIRECTIVE = (
    "Your previous reply gave the office address but never actually asked what DAY AND TIME would be "
    "convenient for the customer — asking about time alone, with no mention of day, is NOT enough; "
    "both are required in this SAME reply. Rewrite the SAME reply, same intent and length, keeping the "
    "address, removing any unrelated content, and ending with a question that covers both the day and "
    "the time (e.g. 'what day and time works for you?')."
)

# Deterministic last resort, same rationale as the other final-safety-net fallbacks in this file: if
# the model still doesn't ask after one regen attempt, guarantee the question reaches the customer by
# appending it in code rather than gamble on a third attempt for a booking-critical question.
TIME_QUESTION_FALLBACK = {
    "en": " What day/time would work best for you?",
    "ar": " أي يوم ووقت بيناسبك؟",
}


# ──────────────────────────────────────────────────────────────────────────────
#  DIRECTIVE TABLE — DECLARATIVE, TESTABLE POLICY (replaces an if/elif ladder)
# ──────────────────────────────────────────────────────────────────────────────
# `flags` is a small dict of booleans computed once per turn in generate_answer (see
# _turn_flags). EXCLUSIVE_DIRECTIVES are checked in order and only the FIRST match fires —
# this priority order is itself policy (e.g. a pricing question always wins over the
# early-turn no-call gate), and it's exactly the ordering that caused a real regression
# once (asks_services_overview vs wants_more_detail) when it lived invisibly inside an
# elif chain. ADDITIVE_DIRECTIVES layer on top regardless of which exclusive one fired.
# Both lists are plain data — printable, diffable, and unit-testable via select_directives
# without touching the LLM.
#
# field_pitch/pricing/contact_request/services_overview/more_detail are ALL checked BEFORE call_flow
# (found live, 2026-07-21, widened from an initial field_pitch-only fix once the same root cause
# turned up in a second scenario): `offered_call` just means "our LAST message offered a meeting" —
# it says nothing about what THIS message is. Once a meeting had been offered even once early on,
# `offered_call` stayed effectively permanent (every one of CALL_FLOW_DIRECTIVE's own branches keeps
# mentioning the meeting, which keeps re-tripping _CALL_OFFER_RE on our own replies) — so call_flow,
# checked first in the original ordering, permanently shadowed every other, more specific directive
# for the rest of the conversation. Two concrete cases found live: a customer who revealed their
# field ("clothing store, want a logo") got crammed into call_flow's generic "changed the subject,
# brief answer only" branch instead of the real, fuller FIELD_PITCH_DIRECTIVE pitch; a customer who
# asked a SECOND, different pricing question got the same generic branch instead of PRICING_DIRECTIVE
# (which — unlike call_flow's branch — has its own reliable, deterministic repeat-suppression, see
# PRICING_REPEAT_NO_MEETING_NOTE). The user's explicit, general product decision (2026-07-21): the
# meeting must never be re-suggested more than once, in ANY scenario a directive could re-raise it —
# not just the one that happened to surface it first. Each of these five signals is a specific,
# independently-detected intent (a code regex or an LLM read of THIS message) — a much stronger basis
# than the stale offered_call flag — and none of them fire on ordinary meeting-logistics replies
# (choosing onsite/online, a bare "yes", a day/time), so promoting them all lets genuine content
# questions win the turn while logistics replies still fall through to call_flow right after,
# unaffected. call_flow is now effectively the fallback: it only fires when none of the more specific
# signals are true — either a genuine logistics reply, or a subject change with no detected intent.
EXCLUSIVE_DIRECTIVES = [
    ("field_pitch_unclear", lambda f: f["wants_field_pitch"] and not f["industry_known"],
     FIELD_PITCH_UNCLEAR_DIRECTIVE),
    ("field_pitch",       lambda f: f["wants_field_pitch"] and f["industry_known"], FIELD_PITCH_DIRECTIVE),
    ("pricing",           lambda f: f["pricing"],                 PRICING_DIRECTIVE),
    ("contact_request",   lambda f: f["wants_contact"],           CONTACT_REQUEST_DIRECTIVE),
    ("services_overview", lambda f: f["asks_services_overview"],  SERVICES_OVERVIEW_DIRECTIVE),
    ("client_industries", lambda f: f["asks_client_industries"],  CLIENT_INDUSTRIES_DIRECTIVE),
    ("more_detail",       lambda f: f["wants_more_detail"],       ELABORATION_DIRECTIVE),
    # unaddressed_intent checked before bare_ack/call_flow for the same reason the five signals above
    # were promoted ahead of call_flow (see this table's own 2026-07-21 comment): it's a specific,
    # independently-detected read of THIS message, and it must not get silently shadowed by the stale
    # offered_call flag or fall through to no directive at all. _turn_flags already guarantees in code
    # that this is false whenever any of the more specific signals above also fired.
    ("unaddressed_intent", lambda f: f["unaddressed_intent"], UNADDRESSED_INTENT_DIRECTIVE),
    # bare_ack checked BEFORE call_flow (2026-07-24, same reasoning as the field_pitch reordering
    # above): is_bare_ack's own gating already requires a narrow, high-confidence signal (a real
    # gratitude/closing word, a vague reaction, or a content-free confirmation to a question WE never
    # asked) and is false whenever last_assistant_asked_question is true — so it never fires on a
    # genuine call_flow continuation ("yes" answering "onsite or online?"). With call_flow checked
    # first, is_bare_ack being true never mattered: ANY message after a meeting/logistics offer (which
    # _CALL_OFFER_RE also matches on incidental phrases like "check the schedule") unconditionally won
    # call_flow instead. Found live: "thank you so much" right after "Got it, I'll check the schedule
    # and get back to you shortly to confirm" (which matches _CALL_OFFER_RE on "check the schedule")
    # got routed into CALL_FLOW_DIRECTIVE and re-explained the office address plus re-asked for a day/
    # time that had already been given, instead of a plain "you're welcome."
    ("bare_ack",          lambda f: f["is_bare_ack"],             BARE_ACK_DIRECTIVE),
    ("call_flow",         lambda f: f["offered_call"],           CALL_FLOW_DIRECTIVE),
    ("no_call_early",     lambda f: f["is_early"],                NO_CALL_DIRECTIVE),
]

ADDITIVE_DIRECTIVES = [
    ("client_proof",
     lambda f: f["industry_known"] and not f["offered_call"] and not f["pricing"]
               and not f["wants_contact"] and not f["wants_field_pitch"] and not f["is_bare_ack"],
     CLIENT_PROOF_DIRECTIVE),
    # compound_intent (any dropped-but-true ask, e.g. pricing+services or pricing+more_detail) is
    # NOT here — it needs per-turn signals/profile values, so it's computed separately in
    # generate_answer via compound_intent_note() and appended to select_directives()'s result.
]


def select_directives(flags: dict) -> list:
    """Given this turn's flags, return [(name, directive_text), ...]: the first matching
    exclusive directive (priority order = policy), plus any additive ones whose condition
    holds. Pure function of `flags` — no LLM call, no side effects — so directive selection
    can be unit-tested on its own (e.g. asks_services_overview=True, wants_more_detail=True
    -> must select "services_overview", not "more_detail")."""
    selected = []
    for name, condition, text in EXCLUSIVE_DIRECTIVES:
        if condition(flags):
            selected.append((name, text))
            break
    for name, condition, text in ADDITIVE_DIRECTIVES:
        if condition(flags):
            selected.append((name, text))
    return selected


def _last_assistant_asked_question(chat_history: list) -> bool:
    """True if OUR last message actually ended in a question — the deterministic tell for whether
    there's really something on the table for a bare confirmation to be confirming. Used to catch
    understand_turn()'s wants_field_pitch bare-confirmation carve-out firing when nothing was
    actually asked (see its own comment in _turn_flags below)."""
    for m in reversed(chat_history or []):
        if m.get("role") == "assistant":
            return (m.get("content") or "").rstrip().endswith(("?", "؟"))
    return False


def _looks_like_pending_call_reply(user_message: str) -> bool:
    """True for a short, question-free reply — the shape of a message that's just continuing an
    already-active meeting/logistics conversation (accepting, naming a location, a bare ack) rather
    than raising a genuinely new, substantive topic. Same heuristic call_flow_mode_pending already
    uses to detect a plain acceptance; generalized here because it turns out to be the right test for
    ANY of the LLM's intent signals, not just wants_field_pitch — see _turn_flags below."""
    t = (user_message or "").strip()
    if not t or "?" in t or "؟" in t:
        return False
    return bool(is_affirmation(t) or len(re.findall(r"\w+", t)) <= 8)


def _turn_flags(offered_call: bool, pricing: bool, is_early: bool, profile: dict, signals: dict,
                 user_message: str = "", last_assistant_asked_question: bool = True) -> dict:
    """Compute this turn's flags for select_directives — one place that defines what the
    directive table can see."""
    wants_contact = bool(signals.get("wants_contact"))
    wants_field_pitch = bool(signals.get("wants_field_pitch"))
    asks_services_overview = bool(signals.get("asks_services_overview"))
    asks_client_industries = bool(signals.get("asks_client_industries"))
    wants_more_detail = bool(signals.get("wants_more_detail"))
    is_confirmation = bool(signals.get("is_confirmation"))
    # Deterministic override, same failure class as the wants_field_pitch overrides above: found live
    # — "ana bde aarf aktr aan l brand identity w kef momken tseedone b mataame" ("I want to know more
    # about Brand Identity and how you can help my restaurant") got wants_contact=true from the LLM
    # (misreading "kef momken tseedone" — "how can you help" — as "how do I reach you"), and since
    # contact_request is checked before more_detail in EXCLUSIVE_DIRECTIVES, CONTACT_REQUEST_DIRECTIVE
    # hijacked the turn instead of actually explaining Brand Identity. Asking to learn more about a
    # named service and explicitly wanting contact info in the very same breath is a rare, unusual
    # combination — much likelier this is that same misread than a genuine compound ask — so
    # wants_more_detail wins the ambiguity, UNLESS the message also contains an explicit contact-
    # seeking word (phone/email/whatsapp/"get in touch"), which is the safety valve for the rare
    # genuine compound case.
    if (wants_contact and wants_more_detail
            and not _asks_about_phone(user_message)
            and not re.search(r"\bemail\b|\bwhatsapp\b|get in touch|reach (you|out)|contact you|"
                               r"ايميل|إيميل|تواصل|بتواصل", user_message or "", re.IGNORECASE)):
        wants_contact = False
    # PROFILE-only on purpose: "already established from an EARLIER turn," not "known including
    # whatever this very message just said" (that broader mix is industry_known below, used for
    # FIELD_PITCH_UNCLEAR_DIRECTIVE's own gate). Using the broader flag here would wrongly also
    # suppress the legitimate first-turn compound case ("I have a real estate company, how much do
    # you charge?"), since the LLM's OWN extraction of THAT message already sets signals["industry"].
    industry_known_before_this_turn = bool(profile.get("industry"))
    # Deterministic override, per the user's explicit product decision: understand_turn()'s wants_
    # field_pitch is only reliable for the ONE-TIME moment the customer's field is freshly revealed
    # (FIELD_PITCH_DIRECTIVE's fuller, multi-service pitch is meant to be a first-reveal event, not a
    # recurring one — see the standing SYSTEM_PROMPT's own "Length" section). Once the field is
    # already known from an earlier turn, the signal keeps firing anyway on all kinds of unrelated
    # later messages that merely happen to be discussed in the context of that field — and since
    # field_pitch is checked before everything else in EXCLUSIVE_DIRECTIVES, that hijacks the turn
    # into a generic re-pitch instead of the real, specific directive that should have handled it,
    # tailored using the already-known field like any other turn. Once the field is known,
    # wants_field_pitch is simply never trusted again for the rest of the conversation.
    #
    # EXCEPT (2026-07-24) when THIS turn reveals a genuinely DIFFERENT industry than the one already
    # stored — found live: "badde logo w ana sahib small business" set profile.industry to "small
    # business" (a size descriptor the LLM loosely treated as an industry), then "3andi mahal lbes"
    # ("I have a clothing shop") came up later — a real, specific, first-ever field reveal that
    # deserves its own tailored pitch — but industry_known_before_this_turn was already true from the
    # earlier "small business" value, so this got treated as a stale repeat and silently dropped
    # (pricing, forced true from the prior turn, won outright instead). Comparing this turn's own
    # freshly-extracted industry against what's already stored tells a genuine new reveal apart from
    # an unrelated later message that just happens to be discussed in the context of the same field.
    current_industry = (signals.get("industry") or "").strip().lower()
    stored_industry = (profile.get("industry") or "").strip().lower()
    is_new_different_industry = bool(current_industry) and current_industry != stored_industry
    if wants_field_pitch and industry_known_before_this_turn and not is_new_different_industry:
        wants_field_pitch = False
    # If wants_field_pitch is STILL true here, industry_known_before_this_turn was false above —
    # i.e. this is a genuine, first-ever reveal of the customer's field this turn, not a stale
    # repeat. That makes it categorically different from the "just continuing the call flow"
    # messages the offered_call override below targets, regardless of how short the sentence
    # happens to be — so it's exempted from that override specifically. Found live: "eza 3nde
    # mataam kef bte2daro tse3doune" ("if I have a restaurant, how can you help me"), sent while
    # offered_call was true, is only 6 words with no "?" — it matched _looks_like_pending_call_
    # reply's shape and got wrongly zeroed out, so the turn fell through to CALL_FLOW_DIRECTIVE and
    # jumped straight to an address+day/time ask instead of ever pitching restaurant-specific
    # services. A first-time field reveal is substantive new content by definition, however it's
    # phrased — length/question-mark shape was never a valid proxy for "nothing new here" in this
    # one case.
    fresh_field_reveal = wants_field_pitch
    # Deterministic override, generalized after the SAME failure showed up on a SECOND signal:
    # wants_contact fired for "i am in bahrein" — a customer answering the onsite/online choice by
    # naming their location — with the model's own `notes` field saying "User is located in Bahrain"
    # (it understood correctly; the boolean was wrong regardless). Since contact_request is ALSO
    # checked before call_flow, this hijacked the turn into a stale "repeat the meeting" fallback with
    # no booking link, instead of call_flow's own "not in Lebanon = online" branch. Rather than patch
    # each individual signal every time a new one trips this same way, the general fix: whenever our
    # own last message was an active meeting/logistics offer (offered_call) and this reply reads like
    # it's just continuing that (short, no question — _looks_like_pending_call_reply), none of the
    # LLM's intent signals get to hijack the turn away from call_flow. `pricing` is deliberately
    # excluded — it's a reliable, largely regex-backed signal (not just an LLM guess) and a genuine
    # pricing question mid-flow should still be able to interrupt it, same as before. A real new
    # question or a longer, substantive message still passes through untouched either way.
    # Read here (before the override below can zero it) so the un-suppressed LLM read is still
    # available if ever needed; the actual gating happens in the final `unaddressed_intent` compute
    # near the return statement.
    # message_category=="on_topic" is a deterministic backstop for unaddressed_intent itself: found
    # live — "where can I see it" (asking to view a portfolio case study just discussed) got
    # unaddressed_intent=false from the model (it's a normal on-topic follow-up, not confusion/a
    # complaint, so the model didn't read it as "needing its own response" the way that signal's own
    # framing leans toward) AND none of the other specific signals fired either — so NOTHING in
    # EXCLUSIVE_DIRECTIVES matched, no directive got injected at all, and the model free-associated
    # onto the meeting instead of answering about the portfolio. message_category is a forced,
    # exhaustive 4-way classification (unlike unaddressed_intent, which the model can simply not
    # apply to an ordinary-sounding question) — "on_topic" means a genuine BLNK-related question that
    # isn't small talk, off-topic, or meeting-related, which is exactly unaddressed_intent's own
    # intended scope whenever the other specific booleans miss it. OR'd in as a second, more reliable
    # path to the same signal, not a replacement for it.
    unaddressed_intent_signal = (
        bool(signals.get("unaddressed_intent")) or signals.get("message_category") == "on_topic"
    )
    if offered_call and _looks_like_pending_call_reply(user_message):
        if not fresh_field_reveal:
            wants_field_pitch = False
        # unaddressed_intent reset UNCONDITIONALLY here (not gated on is_affirmation like the three
        # below) — same treatment as wants_field_pitch, and for the same reason: found live, "i am in
        # lebanon" (answering CALL_FLOW_DIRECTIVE's own onsite-vs-online question) isn't literally any
        # of is_pricing/wants_field_pitch/wants_more_detail/asks_services_overview/wants_contact/
        # is_confirmation, so the model correctly left all of those false and set this NEW catch-all
        # true instead — which, being checked ahead of call_flow in EXCLUSIVE_DIRECTIVES, hijacked the
        # turn away from CALL_FLOW_DIRECTIVE (which already has its own "in Lebanon = onsite" branch)
        # and, since the office-address-needs-a-day/time-ask guarantee is scoped to call_flow winning,
        # silently dropped that guarantee too — the address was given with no day/time ask at all.
        # Unlike wants_contact/asks_services_overview/wants_more_detail (narrowed to is_affirmation-only
        # after the word-count heuristic wrongly ate real short questions — see the comment below),
        # unaddressed_intent is safe to reset unconditionally here: it's the LOWEST-priority catch-all,
        # and call_flow's own directive already has a documented fallback for "a subject change with no
        # detected intent," so suppressing this catch-all in the pending-call-reply shape just returns
        # to that pre-existing, already-accepted fallback instead of silently breaking a real guarantee.
        unaddressed_intent_signal = False
        # wants_contact/asks_services_overview/wants_more_detail: only reset on a genuine affirmation
        # (is_affirmation — a real "yes"/"ok"/"sounds good"), NOT merely because the message is short
        # and question-mark-free. That was this whole override's ORIGINAL bug shape too (see the "i am
        # in bahrein" case above), but the <=8-words half of _looks_like_pending_call_reply's own OR
        # turned out to be far too blunt a proxy for "no real content" — found live TWICE: "do you have
        # experience with restaurants" (6 words, no "?") and "what services do you offer overall" (6
        # words, no "?") are both real, substantive questions, not call-flow noise, but the word-count
        # fallback wiped their signals anyway, dumping the customer into a generic "let's schedule a
        # meeting" non-answer that never addressed what they actually asked — this comment's own "a
        # real new question... still passes through untouched" claim turned out to be false for any
        # real question that happened to be short and lack a "?", which is common, especially in
        # informal/arabizi phrasing. A genuine affirmation is unambiguous noise; a short question is
        # not, so only the former still resets these three.
        # ALSO reset on answers_call_mode_or_location (2026-07-24): found live — "i am in dubai",
        # directly answering CALL_FLOW_DIRECTIVE's own onsite-vs-online question, got wants_contact=
        # true from the LLM (a defensible read in isolation — it does sound like moving forward — but
        # architecturally wrong here, since CALL_FLOW_DIRECTIVE already has a dedicated, correct
        # branch for "they answered by saying where they are"). Unlike the word-count heuristic this
        # override deliberately avoids relying on alone, a location/mode answer is genuinely never a
        # fresh, substantive new question the way "do you have experience with restaurants" is — it's
        # always a direct answer to the pending choice — so it's safe to reset on this signal without
        # the same false-positive risk.
        #
        # ALSO reset on mentions_day_or_time / is_pure_location_question (2026-07-25): found live —
        # wants_contact kept firing true for messages that are unambiguously something else with its
        # own specific handling (a day answer, an explicit location question), so the fixed callback
        # phrase from BOOKING_CHOICE_REPEAT_NOTE ended up repeating verbatim on nearly every turn,
        # including once right after the customer explicitly asked the bot to stop repeating itself.
        # Both regexes are already-existing, general, non-narrow signals used throughout this file for
        # exactly this "is there unambiguous, specific content here" test — reusing them here rather
        # than adding a new narrow pattern for each phrasing.
        if (is_affirmation(user_message) or bool(signals.get("answers_call_mode_or_location"))
                or mentions_day_or_time(user_message) or is_pure_location_question(user_message)):
            wants_contact = False
            asks_services_overview = False
            wants_more_detail = False
    # General catch-all (see UNADDRESSED_INTENT_DIRECTIVE): the LLM's own unaddressed_intent read is
    # trusted, but ONLY when none of the other, more specific signals also fired this turn — same
    # "guarantees enforced in code, not just the prompt" principle as everywhere else in this file.
    # Without this AND, a double-flagged turn (model sets both wants_field_pitch AND
    # unaddressed_intent) would let this new, lowest-confidence catch-all outrank a real, specific
    # signal purely by table position.
    unaddressed_intent = (
        unaddressed_intent_signal
        and not wants_contact and not wants_field_pitch and not asks_services_overview
        and not asks_client_industries and not wants_more_detail and not pricing
        and not is_confirmation
    )
    return {
        "offered_call": offered_call,
        "pricing": pricing,
        "wants_contact": wants_contact,
        "wants_field_pitch": wants_field_pitch,
        "asks_services_overview": asks_services_overview,
        "asks_client_industries": asks_client_industries,
        "wants_more_detail": wants_more_detail,
        "unaddressed_intent": unaddressed_intent,
        "is_early": is_early,
        "industry_known": bool(profile.get("industry") or signals.get("industry")),
        # "Nothing left to do" covers three cases: an actual gratitude/closing word (is_gratitude_
        # closing) or a pure reaction word (is_vague_reaction) — both bypass the question-gate below,
        # since neither ever genuinely answers a real open question, they're just a closing/reaction
        # regardless of what was asked — OR the LLM's own is_confirmation signal firing on a turn
        # where OUR last message didn't ask anything either — i.e. the exact same "bare ack with
        # nothing open" situation the wants_field_pitch override above targets, so bare_ack and that
        # override stay consistent with each other. Deliberately NOT is_confirmation alone (regardless
        # of whether a question was pending) — a bare "yes"/"ok" answering a real open question (the
        # call offer, a field-confirm question) is a genuine continuation and must keep working as
        # before; that's exactly what last_assistant_asked_question rules out for THAT case. Also
        # requires no OTHER, stronger intent this same turn (e.g. "thanks, and how much does it cost"
        # is pricing, not this).
        "is_bare_ack": (
            (is_gratitude_closing(user_message) or is_vague_reaction(user_message)
             or (is_confirmation and not last_assistant_asked_question))
            and not wants_contact and not wants_field_pitch
            and not asks_services_overview and not wants_more_detail and not pricing
            # ALSO excluded when the message answers the pending onsite/online mode/location choice
            # (2026-07-24): found live — "okay we can schedule an onsite meet" got is_confirmation=
            # true from the LLM (a defensible-but-incomplete read, same failure class as the "i am in
            # dubai" wants_contact false-positive), and since our own last reply (the bare address, per
            # CALL_FLOW_DIRECTIVE's location EXCEPTION) asked no question, is_bare_ack came out true —
            # checked BEFORE call_flow, it won and produced a generic "Great." with no day/time ask at
            # all, even though CALL_FLOW_DIRECTIVE has the exact guarantee for this (address already
            # given, MUST ask day/time). A genuine mode/location answer always has real content (it's
            # exactly what answers_call_mode_or_location is for), so it must never be swallowed as
            # "nothing left to do here."
            and not bool(signals.get("answers_call_mode_or_location"))
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  LLM VERIFICATION & REGEN PIPELINE
# Every check below asks the model the same shape of question — "look at this draft reply (plus some
# context) and tell me in JSON whether it violates one specific rule" — and previously each one
# repeated its own copy of the request/parse/error-handling boilerplate. Shared here once: callers
# just supply the check's own system prompt and user content, and get back the parsed JSON dict (or
# {} on any failure — bad JSON, no braces, network error — so a check never blocks the main reply).
def _llm_check_json(system_prompt: str, user_content: str) -> dict:
    if not user_content:
        return {}
    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            # gpt-5-mini only supports the default temperature (1) — passing any other value is a
            # 400 (unlike gpt-4o-mini, which accepted 0 here for deterministic checks).
            # reasoning_effort="minimal" (2026-07-24, user's explicit latency request): these checks
            # are simple yes/no judgments on a draft reply, not tasks that benefit from deep
            # reasoning — measured live, this cuts a realistic-size check call from ~3.6s to ~1.5s
            # with no observed change in the verdict on the same input. This pipeline runs several of
            # these sequentially per turn, so the saving compounds across all of them.
            reasoning_effort="minimal",
        )
        raw = resp.choices[0].message.content or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(raw[start:end + 1])
    except Exception:
        return {}


# Deterministic backstop for STAY_ON_TASK_RULE (shared by PRICING_DIRECTIVE and CALL_FLOW_DIRECTIVE):
# those turns have exactly ONE transactional/logistical job, and the standing rule already tells the
# model not to pad them with pitch/service content — but the model still does it (found live: a
# pricing reply adding "we can also do a brand book and guidelines for you", a "let's go online" reply
# adding an explainer about what brand identity includes). A regex can't catch this generically since
# the extra content could be about ANY service, not just one specific phrase — so, same approach as
# the other checks below, we ask the model itself. This is the general backstop: it doesn't matter
# WHICH service/topic the model tacks on, only that it added one when this turn's job was to not.
OFF_TASK_PITCH_CHECK_PROMPT = """You are checking a draft reply from a branding agency's chatbot for
ONE thing. This turn's ONLY job is purely transactional or logistical — either acknowledging a
price/cost question and offering a meeting as a choice, or handling meeting/booking logistics (an
address, a link, confirming a day/time). It is explicitly NOT a turn for pitching, explaining, or
describing services or deliverables.
Does the draft reply add ANY content beyond that one job — e.g. explaining what a service includes,
describing what a deliverable does, pitching an additional or different service, or giving detail
about something like "brand identity"/"brand book"/packaging/logo/social media/guidelines that goes
beyond just naming the meeting choice or handling the logistics? A short scope-depends-on-specifics
line, the meeting-choice offer itself, the booking-logistics text (address, day/time, the link), and
a short question asking for the customer's company name or field/business (a qualifying question,
not a pitch) are NOT violations — only EXTRA pitch/explanation content layered on top of those counts.
Output ONLY JSON: {"off_task": true/false, "extra_content": "<short description of the extra content,
or empty>"}"""


def _check_off_task_pitch(reply: str) -> str:
    """Return a short description of the off-task pitch/service content draft `reply` added on a
    purely transactional turn (pricing or call-flow), or "" when there's none or the check itself
    fails (never blocks the main reply)."""
    if not reply:
        return ""
    data = _llm_check_json(OFF_TASK_PITCH_CHECK_PROMPT, f"Draft reply:\n{reply}")
    return (data.get("extra_content") or "").strip() if data.get("off_task") else ""


def off_task_pitch_fix_directive(extra: str) -> str:
    return (
        f"Your previous reply added off-task content this turn must never include: {extra}. This "
        f"turn's job is purely transactional/logistical. Rewrite the SAME reply, same intent, with "
        f"that extra pitch/explanation removed entirely — keep only the scope-dependency line and the "
        f"meeting-choice offer, or the booking-logistics handling itself, nothing more."
    )


# Deterministic backstop for the "wrong-field word" bug: the model can still slip and mention a
# product/industry word that belongs to a DIFFERENT business than the one the customer actually
# stated (e.g. "akel"/food showing up in a reply to a clothing shop) — usually bleed from a few-shot
# example or a retrieved chunk about a different past client. Same failure mode also shows up as a
# wrong-field CLIENT CITATION: the retriever can surface a chunk that's merely semantically close
# (e.g. "Where To?", an APP that gives advice to restaurants, retrieved for a "mataam"/restaurant
# query) and the model cites it as if it were a matching past client, despite the directive telling it
# to only cite genuine field matches. A regex can't catch either generically (the mismatched word or
# client could be anything, for any field), so — same approach as understand.py — we ask the model
# itself in one small, targeted check grounded in the retrieved context, rather than hardcoding a
# lexicon or client list. Runs on every turn where we know the customer's field, whichever directive
# fired.
INDUSTRY_MISMATCH_CHECK_PROMPT = f"""You are checking a draft reply from a branding agency's chatbot
for TWO things, using the retrieved company knowledge below as ground truth:
1) Does it mention any product, deliverable example, or industry-specific word that belongs to a
business DIFFERENT from the customer's own stated field? (e.g. customer's field is "clothing store"
but the reply talks about food/dishes/menu — that's a mismatch.) A word generic enough to fit any
business (e.g. "photography", "social media", "logo", "color palette") is NOT a mismatch by itself.
2) Does it cite a past client/case study as proof for which it is NOT true that the client's
{CLIENT_MATCH_CRITERION}? Three ways this fails: (a) the client IS in the retrieved context but its
actual business, per the context, is a different field (e.g. citing "Where To?" when the context
shows it's an app FOR restaurants, not itself a restaurant, while the customer's field is
"restaurant" — the client's own business must match, not just a topical connection); (b) the client
name does NOT appear anywhere in the retrieved context at all — a fully invented name is exactly as
much a violation as a real but wrong-field one, since the standing rule is "never invent clients";
(c) {CLIENT_MATCH_SPECIFICITY_NOTE}
Output ONLY JSON: {{"mismatch": true/false, "mismatched_words": ["..."]}} — mismatched_words lists the
specific words or client names that are the problem."""


def _check_industry_mismatch(reply: str, industry: str, context_text: str = "") -> list:
    """Return the mismatched words/client names the model flags in `reply` given the customer's real
    `industry` and the retrieved `context_text` (ground truth for client-citation checks), or [] when
    there's no mismatch or the check itself fails (never blocks the main reply)."""
    if not reply or not industry:
        return []
    data = _llm_check_json(
        INDUSTRY_MISMATCH_CHECK_PROMPT,
        f"Customer's stated field: {industry}\n\nRetrieved company knowledge (ground truth):\n"
        f"{context_text}\n\nDraft reply:\n{reply}",
    )
    return data.get("mismatched_words") or ([] if not data.get("mismatch") else ["mismatched industry content"])


def industry_mismatch_fix_directive(words: list, industry: str) -> str:
    return (
        f"Your previous reply mentioned {', '.join(words)}, which doesn't genuinely belong to or match "
        f"the customer's actual field ({industry}) — either an industry-specific word from a different "
        f"business, or a past-client citation whose own business doesn't match per the retrieved context. "
        f"Rewrite the SAME reply, same intent and length, using ONLY words, products, and examples — "
        f"including any client citation — that genuinely fit a {industry} business. If no client in the "
        f"context truly matches, drop the citation entirely rather than forcing one."
    )

# Deterministic backstop for the anti-repeat rules already in SYSTEM_PROMPT ("Never repeat an idea...",
# "don't re-open every later reply with 'For your X company'") — the model still violates them,
# reopening consecutive replies with the identical industry-labeled preamble ("For your clothing
# store, we can provide...", verbatim, turn after turn) instead of answering directly. A regex can't
# catch this generically (the repeated phrase could be anything, in any language), so — same approach
# as the checks above — we ask the model to compare the draft against what it already said.
# NOTE: repeated/second NAMED client citations are deliberately OUT of scope here — that's the
# deterministic _already_cited_client check below, which is the more reliable of the two (this LLM
# check missed a repeated citation live on 3 consecutive turns before that backstop was added).
# Keeping both checks responsible for the same thing was redundant, so this one now only owns the
# fuzzy, non-named-client repeats. User's explicit stance (2026-07-21): repetition of ANY kind is
# the general problem to guard against — "it'll be obvious it's a chatbot" — not just the handful of
# specific patterns (client names, the vague-proof filler line, the meeting choice) that already have
# their own dedicated deterministic checks. This is the general backstop underneath all of those
# narrower ones; it should catch whatever they don't. Found live: a VAGUE, unnamed proof line ("we've
# worked with companies in your domain before") got excluded here as if it were a "client citation,"
# even though it names no one — that carve-out is now explicitly narrowed to NAMED citations only, so
# a reworded vague-proof repeat this check's own dedicated regex doesn't happen to match still gets
# caught here as a backup.
REPETITION_CHECK_PROMPT = """You are checking ONE thing in a draft reply from a branding agency's
chatbot: does it substantially repeat — even if reworded, translated, or in a different language —
an opening label (e.g. "for your clothing store, we can...", reused turn after turn), an idea, a
recommendation, or a vague/unnamed proof claim (e.g. "we've worked with businesses like yours
before") the assistant ALREADY gave earlier in this same conversation? A short natural continuation
of the same topic is NOT a repeat; reusing the same opening phrase, the same specific recommendation,
or the same vague proof claim again IS a repeat. Judge by MEANING, not exact spelling: arabizi
(Arabic written in Latin letters) has no fixed spelling, so the same word can appear as "n2adem"/
"ne2adem"/"mne2adem"/"n3adem", or "aktar"/"aktr"/"ektar" — never let differing arabizi spelling of
the same idea hide a real repeat. (A repeated or second citation of a SPECIFIC NAMED client is handled
by a separate, dedicated check — do not flag that here; but a vague, unnamed proof claim with no
client name attached is NOT that exclusion, and should be flagged like any other repeated idea.
Re-mentioning the SAME booking link/consultation slot is also NOT a repeat to flag here — it's the
same standing action item the customer still hasn't used yet, not a repeated pitch; only flag it if
the surrounding PITCH content around the link is what's repeated.) Output ONLY JSON:
{"repeats": true/false, "repeated_content": "<short description of what's repeated, or empty>"}"""


def _check_repetition(reply: str, chat_history: list) -> str:
    """Return a short description of what draft `reply` repeats from earlier ASSISTANT turns in
    `chat_history`, or "" when there's nothing to repeat (no prior assistant turn), no repeat found,
    or the check itself fails (never blocks the main reply)."""
    prior_assistant = [m.get("content", "") for m in chat_history if m.get("role") == "assistant"]
    if not reply or not prior_assistant:
        return ""
    convo = "\n".join(f"- {a}" for a in prior_assistant[-6:])
    data = _llm_check_json(
        REPETITION_CHECK_PROMPT,
        f"What the assistant already said earlier in this conversation:\n{convo}\n\nDraft reply:\n{reply}",
    )
    return (data.get("repeated_content") or "").strip() if data.get("repeats") else ""


def repetition_fix_directive(repeated: str) -> str:
    return (
        f"Your previous reply repeated something you already said earlier in this conversation: "
        f"{repeated}. Rewrite the SAME reply, same intent and length, WITHOUT that repeated opening or "
        f"idea — answer the question directly with genuinely new substance instead (a different "
        f"concrete detail, angle, or specific), per the anti-repeat rule."
    )


# Deterministic backstop for CALL_FLOW_DIRECTIVE's own explicit rule on the "customer gave a day AND
# time for the onsite meeting" branch ("tell them you'll check the schedule and get back to them
# shortly to confirm — do NOT self-book"). Found live in Arabic: the customer gave "10 صباحا" (10am)
# and the reply answered "رح نلتقي بكرا الساعة 10 صباح. رح كون موجود بالمكتب" ("We'll meet tomorrow at
# 10am, I'll be at the office") — flatly confirming a booking with zero schedule-checking language, a
# real business risk (promising availability that was never actually verified). This is checked only
# on the turn where it matters: we already offered a meeting AND the customer's latest message
# contains an actual TIME (mentions_specific_time — a day alone doesn't count, see its definition).
DAYTIME_HANDLING_CHECK_PROMPT = """You are checking a draft reply from a branding agency's chatbot.
The customer just gave a day and/or time for an onsite meeting. The assistant's instructions require
the reply to do exactly ONE job, in ANY language or phrasing: a brief acknowledgment (e.g. "got it" /
"tamam" / "mni7") followed by saying it will check the schedule and get back to the customer shortly
to confirm — nothing else.
Mark it mishandled (compliant=false) for ANY of these reasons:
1) PREMATURE BOOKING: the reply confirms or commits to the meeting outright instead of deferring to a
schedule check (e.g. "we'll meet at 10am", "I'll be at the office then", "confirmed", "see you then"
— self-booking without actually checking anything).
2) MISREAD AS GREETING: words like "sobh"/"masa"/"صباح"/"مساء" are ambiguous — they can be an AM/PM
time marker (e.g. "3 masa" = 3pm) OR a fragment of a greeting ("sabeh el kheir"/"مسا الخير" = good
morning/evening, with "مسا النور"/"sabeh el nour" as the traditional reply to it). Given the customer
had JUST been asked what time works for them, does the draft reply instead respond as if to a
greeting (e.g. replying "مسا النور"/"sabeh el nour" or similar) INSTEAD of engaging with the time they
gave? This is a real failure mode this exact model makes, not a hypothetical edge case.
3) RESTATES/ECHOES THE VALUE BACK: the reply repeats or spells the day/time value the customer just
gave back to them as a lead-in clause before the acknowledgment (e.g. "Bokra n2dar n3mil meeting
3andna bel office" repeating "tomorrow", or "3pm b2a 3andna" repeating "3pm", or English "okay,
tomorrow at 3pm works" / "got it, 3pm it is"). This reads as a robotic parrot-back, not a natural
human acknowledgment, and is explicitly NOT required — a brief "got it"/"tamam" with no value repeated
is the correct, compliant form.
4) NEVER MENTIONS THE SCHEDULE CHECK: the reply is truncated, degenerate (e.g. just "Hii." or a bare
greeting), or a vague filler like "okay, thanks"/"mnih, shukran" that never mentions checking the
schedule and getting back to confirm.
Be strict: only mark it compliant if the reply is a short, natural acknowledgment (no restated day/time
value) that clearly promises a schedule check is coming.
Output ONLY JSON: {"mishandled": true/false, "reason": "premature_booking" or "misread_as_greeting" or
"restates_value" or "no_schedule_check" or "none"}"""


def _check_daytime_handling(reply: str) -> bool:
    """True if `reply` fails to handle the customer's day/time correctly — self-booking, a
    greeting-misread, restating/echoing the value back (explicitly NOT wanted — a plain "got it" is
    the correct form), or never mentioning the schedule check. False if compliant or the check
    itself fails (never blocks the main reply)."""
    if not reply:
        return False
    data = _llm_check_json(DAYTIME_HANDLING_CHECK_PROMPT, f"Draft reply:\n{reply}")
    return bool(data.get("mishandled"))


DAYTIME_HANDLING_FIX_DIRECTIVE = (
    "Your previous reply mishandled the day/time the customer just gave — either it confirmed or "
    "committed to the meeting outright (e.g. 'we'll meet at...', 'I'll be there') instead of deferring "
    "to a schedule check, OR it misread 'sobh'/'masa'/'صباح'/'مساء' as a greeting and replied with a "
    "greeting-exchange response (e.g. 'مسا النور'/'sabeh el nour') instead of recognizing it as the "
    "AM/PM time they were giving you, OR it restated/echoed the day/time value back as a lead-in "
    "(e.g. 'Bokra n2dar n3mil meeting 3andna bel office', '3pm b2a 3andna', 'okay, tomorrow at 3pm "
    "works') — that reads as a robotic parrot-back and is NOT required, OR it never actually mentioned "
    "checking the schedule at all. Rewrite the SAME reply, same intent and length: a brief, natural "
    "acknowledgment with NO restated day/time value (e.g. 'got it' / 'tamam' / 'mni7') followed by "
    "saying you'll check the schedule and get back to them shortly to confirm — do NOT confirm, commit "
    "to, or self-book the meeting, do NOT respond as if to a greeting, and do NOT repeat their day/time "
    "back to them."
)

# Deterministic backstop for the NEW CALL_FLOW_DIRECTIVE branch above ("already fully given earlier,
# now just a plain acknowledgment or other new non-scheduling info"). Found live TWICE: (1) after the
# bot already said "I'll check the schedule and get back to you" for a Monday-3pm slot, the customer
# replied with a bare "okay" — no branch fit that moment cleanly, and the model improvised a
# fabricated confirmation ("I'm setting the meeting for Monday at 3:00, I'll give you the details
# soon"). (2) the customer instead answered the earlier company-name/field question ("we're a boutique
# now") while the schedule-check was pending — despite an explicit CALL_FLOW_DIRECTIVE branch telling
# it not to, the model pivoted into pitching a brand-identity service AND offering to schedule ANOTHER
# meeting, even though one was already pending confirmation. Both are the same underlying risk — this
# reply's only real job while a meeting is pending is a brief acknowledgment, nothing more — so this
# check now covers all three failure shapes, not just the fabricated-confirmation one. Checked only
# when we've already made the schedule-check promise (_schedule_check_already_promised) and this
# turn's message doesn't introduce a new day/time (gave_specific_time is False), so it never overlaps
# with the day/time-handling check above.
PREMATURE_REBOOKING_CHECK_PROMPT = """You are checking a draft reply from a branding agency's
chatbot. In an EARLIER turn, the assistant already told the customer it would check the schedule and
get back to them to confirm an onsite meeting day/time — that confirmation has NOT actually happened
yet; no human has verified the real schedule, and the customer's latest message does NOT give a new
day/time (it might be a plain acknowledgment, or it might give some other new info like their company
name/field — either way, no new day/time). Since a meeting is already pending confirmation, this
reply's only real job is a brief, warm acknowledgment. Mark it MISHANDLED (mishandled=true) if the
draft reply does ANY of these:
1) Wrongly asserts or implies the meeting is now confirmed, set, booked, or scheduled (e.g. "the
meeting is set for Monday at 3", "I've scheduled it", "see you then") — a promise nobody has actually
verified yet.
2) Pitches, describes, or sells a service/deliverable (e.g. "we can create a brand identity for your
boutique", explaining what a service includes) — there's nothing to sell right now.
3) Suggests, offers, or asks about scheduling ANOTHER meeting or call (e.g. "if you'd like, we can set
up a meeting to discuss this further") — one is ALREADY pending, so offering another is redundant and
confusing.
4) Repeats/restates the day/time value again, or repeats the earlier "I'll check the schedule and get
back to you" promise in different words (it was already said once — saying it again is stale repetition,
not new information), or volunteers the contact email/phone when the customer didn't ask how to reach
you.
A short, warm, generic closing with NONE of the above — e.g. "Great, talk soon!", or briefly
acknowledging new info they gave (like their company name) without pitching or re-booking — is CORRECT
and NOT a violation.
Output ONLY JSON: {"mishandled": true/false}"""


def _check_premature_rebooking(reply: str) -> bool:
    if not reply:
        return False
    data = _llm_check_json(PREMATURE_REBOOKING_CHECK_PROMPT, f"Draft reply:\n{reply}")
    return bool(data.get("mishandled"))


PREMATURE_REBOOKING_FIX_DIRECTIVE = (
    "Your previous reply did something it shouldn't while a meeting is already pending confirmation: "
    "it either wrongly asserted/implied the meeting is confirmed/booked/set, pitched or described a "
    "service/deliverable, suggested/offered ANOTHER meeting, repeated the day/time or the earlier "
    "'I'll check the schedule' promise again, or volunteered the contact email/phone unprompted — none "
    "of that belongs here; nobody has actually verified the real schedule yet, and there's nothing to "
    "sell, re-book, restate, or hand out contact info for right now. "
    "Rewrite the SAME reply, same intent and length, as a short, warm, generic acknowledgment only — "
    "no new meeting/booking claim, no service pitch, no new meeting offer, no repeated day/time or "
    "schedule-check promise, no unprompted contact info. If the customer gave other "
    "new info (like their company name), you may briefly acknowledge it, but nothing more."
)

# Deterministic last resort, same rationale as the other final-safety-net fallbacks in this file.
PREMATURE_REBOOKING_FALLBACK = {
    "en": "Great, talk soon!",
    "ar": "تمام، منحكي قريب.",
}

# Deterministic backstop for FIELD_PITCH_DIRECTIVE's own mandatory rule ("name ONE past client as
# proof ... in this SAME first reply") — the model still skips it on the actual pitch turn, going
# generic instead, even when the retrieved context has a client that's a genuine, obvious match (a
# real estate customer pitched "digital campaign assets and email design" while the context held
# "Elite Home", a real estate website/UI-UX case study, ranked #4). Only checked on the specific turn
# where citing is mandatory (wants_field_pitch), not every turn — elsewhere a missing citation is fine
# per CLIENT_PROOF_DIRECTIVE ("if unsure, don't mention clients at all").
MISSING_CLIENT_CHECK_PROMPT = f"""You are checking ONE thing in a draft reply from a branding agency's
chatbot, for a turn where the customer just revealed their field and the assistant's instructions
require it to cite a genuine past client as proof in THIS reply, if one exists in the retrieved
context. Using the retrieved context below: is there at least one past client/case study in it whose
{CLIENT_MATCH_CRITERION}, which the draft reply FAILED to cite? {CLIENT_MATCH_SPECIFICITY_NOTE}
(If the reply already cites a genuinely matching client — even if that citation is itself wrong-field,
that's a DIFFERENT problem checked elsewhere, not a missing citation — or if no client in the context
truly matches the field, answer false.) Output ONLY JSON: {{"missing_client": true/false,
"client_name": "<the name it should have cited, or empty>"}}"""


def _check_missing_client_citation(reply: str, industry: str, context_text: str) -> str:
    """Return the client name the draft `reply` should have cited but didn't, or "" when it already
    cited one, no genuine match exists in `context_text`, or the check itself fails."""
    if not reply or not industry:
        return ""
    data = _llm_check_json(
        MISSING_CLIENT_CHECK_PROMPT,
        f"Customer's stated field: {industry}\n\nRetrieved company knowledge:\n{context_text}\n\n"
        f"Draft reply:\n{reply}",
    )
    return (data.get("client_name") or "").strip() if data.get("missing_client") else ""


def missing_client_fix_directive(client_name: str) -> str:
    return (
        f"You were required to cite a real matching past client as proof in this reply, and skipped "
        f"it — {client_name} in the retrieved context genuinely matches the customer's field. Rewrite "
        f"the SAME reply, same intent and length, leading with the deliverable that {client_name}'s own "
        f"case study actually showcases, and naming {client_name} as proof."
    )


# ──────────────────────────────────────────────────────────────────────────────
#  DETERMINISTIC POST-PROCESSING
def _strip_exclamations(text: str) -> str:
    """Guarantee no exclamation marks reach the user, even if the model slips. Turn any
    run of '!' into a single period and tidy the artifacts that creates."""
    if not text:
        return text
    text = re.sub(r"\s*!+", ".", text)          # "Great!!" -> "Great."
    text = re.sub(r"([?؟…])\s*\.", r"\1", text)  # "?." -> "?"  (was "?!")
    text = re.sub(r"\.\s*\.", ".", text)          # ".. " artifact -> "."
    return text.strip()


# User's explicit stance (2026-07-21): never use a semicolon. A standing style rule in SYSTEM_PROMPT
# asks for this, but same as every other style rule in this file, a soft instruction alone isn't
# reliable enough on its own — this is the deterministic guarantee. Splits into two sentences rather
# than just swapping in a comma, since a semicolon usually joins two clauses that read fine as their
# own sentences.
def _strip_semicolons(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\s*;\s*", ". ", text)
    text = re.sub(r"\.\s*\.", ".", text)  # tidy a ". ." artifact from a semicolon right before a period
    return text.strip()


def _strip_continuation_words(text: str) -> str:
    """Final, last-resort guarantee that 'also'/'كمان' etc. never reach the customer, even if the
    regen attempt (see _find_continuation_word above) didn't fix it. These are optional connector
    words, so removing them and tidying the resulting spacing/punctuation is safe surgery — unlike
    most other quality issues in this file, there's no risk of mangling the sentence's actual
    meaning."""
    if not text:
        return text
    text = _CONTINUATION_WORD_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)              # double space left by the removed word
    text = re.sub(r"\s+([.,،؛:؟?])", r"\1", text)     # stray space before punctuation
    return text.strip()


# Found live (2026-07-24, gpt-5-mini): a reply started with the literal text "SMALL TALK, hi, I'm
# Lana..." — an internal directive-style label (this file injects system messages like
# SMALL_TALK_ALONGSIDE_DIRECTIVE, UNADDRESSED_INTENT_DIRECTIVE etc.) leaking straight into the
# customer-facing text, apparently because this one turn stacked multiple directive-shaped system
# messages at once (a pleasantry riding alongside a confusion complaint) and the model echoed a
# label instead of just following the instruction. No code anywhere hardcodes this string — it's a
# rare model-side artifact, not a fallback text bug — so this is a cheap, generic safety net rather
# than a fix for one specific label: any reply that opens with 2+ consecutive ALL-CAPS words followed
# by a comma/colon (a shape ordinary conversational text essentially never starts with) has that
# opening stripped.
_LEAKED_LABEL_RE = re.compile(r"^[A-Z]{2,}(?:[ _][A-Z]{2,})+[,:]\s*")


def _strip_leaked_label(text: str) -> str:
    if not text:
        return text
    return _LEAKED_LABEL_RE.sub("", text)


_DOUBLE_QUESTION_JOIN_RE = re.compile(r"([?؟])\s*[,،]\s*(\w)")


def _fix_double_question_join(text: str) -> str:
    """Clean up a run-on like '...to meet?, could you please share your company name?' — TWO
    separate questions glued together with a comma instead of a sentence break. Found live: this
    happens specifically when a regen (e.g. the company-info fallback) tacks a second question onto
    a reply that already ended in one. Split into two proper sentences: '...to meet? Could you...'"""
    if not text:
        return text
    return _DOUBLE_QUESTION_JOIN_RE.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}", text)

# The reflexive "offer to keep talking" tail the model keeps appending despite the prompt
# ("if you want, I can tell you...", "want me to explain?", "بدك اشرحلك؟"). Matches from the
# offer phrase (with any leading connective) to the end, so we can cut it whether it's its own
# sentence or a clause dangling off the last one. The verbs are all "offer to inform" verbs, so
# a value statement like "we can help you grow" won't match.
_OFFER_TAIL_RE = re.compile(
    r"[\s,;:—–-]*(?:and\s+|plus\s+|also\s+)?"
    r"(?:if\s+you'?d?\s*(?:want|like)|whenever\s+you'?re\s+ready|would\s+you\s+like|"
    r"want\s+me\s+to|do\s+you\s+want\s+me\s+to|just\s+let\s+me\s+know|let\s+me\s+know\s+if|"
    r"i\s+can\s+(?:also\s+|go\s+ahead\s+and\s+)?(?:tell|walk|break|explain|share|go\s+over|run)|"
    r"i'?d\s+be\s+happy\s+to|happy\s+to\s+(?:explain|share|walk)|shall\s+i|"
    # the reverse of an offer-to-inform: dumping the ball back on the customer to spell out
    # exactly what they want, instead of the bot committing to a concrete answer/recommendation
    r"tell\s+us\s+(?:exactly\s+)?what\s+you\s+(?:want|need)|"
    r"let\s+us\s+know\s+(?:exactly\s+)?what\s+you\s+(?:want|need)|"
    r"إذا\s*بدك|اذا\s*بدك|إن\s*حبيت|لو\s*حاب\w*|بتحب\s*(?:ياني)?|بدك\s*(?:ياني\s*)?(?:اشرح|قلك|فرجيك|خبرك)|"
    r"بشرحلك|بخبرك\s*اكتر|iza\s*badak|iza\s*b?theb|"
    r"قلولنا\s*شو\s*بدكن|خبرونا\s*شو\s*بدكن|قلنا\s*شو\s*بدك|احكيلنا\s*شو\s*بدك|"
    r"[3ع]t[e]?llna\s*shu\s*bad[a]?k|khabbirna\s*shu\s*bad[a]?k|2elle\s*shu\s*bad[a]?k)"
    r".*$",
    re.IGNORECASE,
)

# The ONE exploratory question we still allow the reply to end on (when we don't yet know the
# customer's field): asking their field/industry, so we can tailor. Any other trailing question
# ("which service do you want to explore?", "do you have a topic?") is a bot-tell and gets cut.
_FIELD_Q_RE = re.compile(
    r"what\s+(?:field|industry|sector|kind of business|type of business|line of (?:work|business))|"
    r"which\s+(?:field|industry|sector)|what'?s\s+your\s+business|what\s+do\s+you\s+do\b|"
    r"شو\s*مجال|أي\s*مجال|اي\s*مجال|بأي\s*مجال|شو\s*شغل|أي\s*قطاع|اي\s*قطاع|شو\s*نوع\s*شغل|"
    r"shu\s*(?:l\s*)?(?:mjel|mjal|majel|shghlak|she?8lak|shughlak)|ay\s*(?:mjel|mjal|8ita3|sector)|"
    r"bi\s*ay\s*mjel|no3\s*shghlak",
    re.IGNORECASE,
)


def _is_field_question(text: str) -> bool:
    return bool(_FIELD_Q_RE.search(text or ""))

def _split_sentences(text: str) -> list:
    """Split into whole sentences on sentence-ending punctuation, EXCEPT the period in "BLNK." —
    the company's own stylized name always carries a trailing period (used throughout the data as
    "BLNK. service - ...", "BLNK. specializes in...", etc.), which a naive splitter misreads as a
    complete one-word sentence. That silently truncated real replies down to just "Hii. BLNK." when
    later capped to 2 sentences (found live) — a fixed-width lookbehind excludes it specifically."""
    return [p for p in re.split(r"(?<=[.?؟…])(?<!BLNK\.)\s+", (text or "").strip()) if p.strip()]


def _cut_after_link(text: str, link: str) -> str:
    """The booking-link reply is meant to be one line pointing the customer to the link and nothing
    else — but the model sometimes tacks on an unrelated trailing sentence after it anyway (e.g. bled
    in from an unrelated few-shot example). A sentence-count cap alone doesn't catch this when the
    whole reply already fits the cap (link sentence + one stray sentence = 2, same as the normal cap).
    So instead we cut deterministically at the link itself: keep through the sentence containing the
    link, drop everything after — whatever that trailing content says."""
    if link not in text:
        return text
    sents = _split_sentences(text)
    kept = []
    for s in sents:
        kept.append(s)
        if link in s:
            break
    return " ".join(kept).strip()


def _cap_sentences(text: str, max_sentences: int) -> str:
    """Hard-cap the reply to its first `max_sentences` WHOLE sentences. The field-pitch turn is the
    one place the model reliably overshoots the 'keep it short' directive, so we guarantee brevity
    in code instead of trusting the prompt. Whole sentences only — we never cut mid-thought."""
    if not text:
        return text
    sents = _split_sentences(text)
    if len(sents) <= max_sentences:
        return text
    return " ".join(sents[:max_sentences]).strip()


def _trim_bot_tail(text: str, allow_question: bool, keep_field_question: bool = False) -> str:
    """Deterministically remove the bot-tell tail the model tacks on: ALWAYS strip a trailing
    'if you want, I can...' offer, and strip a trailing bare question unless it's sanctioned this
    turn (allow_question) or it's the one legitimate field-ask (keep_field_question). Never empties
    the reply — if trimming would leave nothing, the original stands."""
    if not text:
        return text
    for _ in range(2):  # a reply can end with an offer AND a question
        sents = _split_sentences(text)
        if not sents:
            break
        last = sents[-1]

        # 1) trailing offer — as a whole sentence or a clause hanging off the last sentence.
        # Exempted (2026-07-24) when the sentence is actually presenting the real onsite-vs-online
        # choice (both terms present) — _OFFER_TAIL_RE's "would you like" / "let me know if" patterns
        # are meant to catch a generic bot-tell ("let me know if you want me to explain more"), but
        # they ALSO match the call-flow mode question phrased the same way ("Please let me know if
        # you'd prefer onsite or online"). Found live: "we can go for the meet" (mode not chosen yet)
        # got "Great. Please let me know if you'd prefer onsite or online." as its draft — a correct,
        # necessary question — and this strip gutted it down to "Great. Please.", a meaningless
        # fragment with the actual question gone.
        m = _OFFER_TAIL_RE.search(last)
        if m and (_ONSITE_OFFER_TERMS_RE.search(last) and _ONLINE_OFFER_TERMS_RE.search(last)):
            m = None
        if m:
            before = last[: m.start()].rstrip(" ,;:—–-")
            head = " ".join(sents[:-1]).strip()
            combined = (head + " " + before).strip() if before else head
            if combined:  # keep everything up to the offer
                if combined[-1] not in ".?؟…":
                    combined += "."
                text = combined
                continue
            break  # the whole reply was the offer — leave it rather than blank

        # 2) trailing bare question — keep only if sanctioned or it's the field-ask we allow
        if last.rstrip().endswith(("?", "؟")):
            if allow_question or (keep_field_question and _is_field_question(last)):
                break
            head = " ".join(sents[:-1]).strip()
            if head:
                text = head
                continue
            break  # reply is only the question — keep it

        break
    return text

def _format_context(context_chunks: list) -> str:
    """Render retrieved chunks as labeled document blocks instead of a flat bullet list.

    Every chunk in the knowledge base is tagged with a category (and sometimes a
    subcategory) — e.g. 'service', 'portfolio', 'company_overview', 'contact_information'.
    Surfacing that label alongside the text gives the model a handle to ground on: it can
    tell a past-work example apart from a service we actually offer, rather than blurring
    anonymous bullets together. A chunk with no metadata just shows its text, so nothing
    breaks."""
    blocks = []
    for i, c in enumerate(context_chunks, 1):
        meta = c.get("metadata") or {}
        label = " / ".join(
            str(meta[k]) for k in ("category", "subcategory") if meta.get(k)
        )
        header = f"--- Document {i} ---"
        if label:
            header += f"\nCategory: {label}"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────────
#  CORE COMPLETION
def _complete(messages: list) -> str:
    resp = client.chat.completions.create(
        model="gpt-5-mini",   # real OpenAI model; swap for your preferred one
        messages=messages,
        # gpt-5-mini only supports the default temperature (1) — passing 0.3 (the old gpt-4o-mini
        # value) is a 400 Bad Request with this model.
        # reasoning_effort="minimal" (2026-07-24, user's explicit latency request): measured live,
        # this cuts a realistic-size generation call (full SYSTEM_PROMPT + directives + context)
        # from ~8.35s to ~2.8s. If reply quality noticeably suffers, "low" is the next step up
        # (measured ~4.7s) before falling back to the default.
        reasoning_effort="minimal",
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATION
def generate_answer(user_message: str, context_chunks: list, chat_history: list,
                    profile: dict = None, signals: dict = None):
    profile = profile or {}
    signals = signals or {}
    # Reply language: the model's per-turn understanding handles this well, but it occasionally
    # mislabels things. When THIS message is decisively one language on its own (a real English
    # function word, or Arabic script), that reliable local read wins over the LLM guess.
    established_lang = profile.get("preferred_language")
    if established_lang == "arabizi":
        # Migration for sessions persisted before the 2026-07-17 decision to stop generating
        # arabizi output — an old profile row can still have this stored. Treat it as "en" (the
        # language those customers now get), so it never reaches LANG_DIRECTIVE, which no longer
        # has an "arabizi" key and would KeyError on direct indexing (`LANG_DIRECTIVE[lang]`).
        # Also self-heal it on the profile dict (same object main.py saves back after this call),
        # so it's a one-time fix per session instead of re-normalizing forever.
        established_lang = "en"
        profile["preferred_language"] = "en"

    # A bare confirmation/acknowledgment ("okay", "tamam", "deal", "sí", "d'accord"...) carries NO
    # signal about which language to reply in — it's just agreeing to something already on the
    # table, not introducing anything new. Found live TWICE, in both directions: trusting the LLM's
    # per-turn reply_language guess on a bare "yes" flipped an all-English chat to Arabic script; a
    # regex word-list gap ("monday at 3" missing a recognized function word) separately misfired the
    # same way. The general fix isn't a better word list (that just moves the gap to the next unseen
    # phrasing) — it's recognizing that a confirmation shouldn't be asked to determine language AT
    # ALL, in any language or spelling. is_confirmation comes from understand_turn() (general, any
    # language), OR'd with the existing regex is_affirmation() as a fast, no-extra-latency path for
    # the common cases it already covers. Whenever either says "this is just a confirmation," always
    # continue the established language — never let one flip it.
    is_confirmation_turn = bool(signals.get("is_confirmation")) or is_affirmation(user_message)
    # Found live: "eh ana b lebnen" ("yeah I'm in Lebanon") got is_confirmation=True from the LLM
    # (semantically it IS just answering/confirming) and established_lang stuck it to a
    # previously-established "ar" — but this message carries REAL content (their location, in Latin
    # script) and must not be swept into the same "carries no signal" bucket as a bare "yes"/"tamam".
    # Any Latin-script message that names a mode/location/day-time answer is decisive arabizi (the
    # standing rule: arabizi always gets "en"), regardless of what is_confirmation says and
    # regardless of whatever established_lang currently holds (which, as this exact case shows, can
    # itself be wrong from an earlier turn) — the sticky path below is for content-FREE
    # confirmations, and this genuinely isn't one.
    names_mode_or_time = bool(
        _ONSITE_OFFER_TERMS_RE.search(user_message or "")
        or _ONLINE_OFFER_TERMS_RE.search(user_message or "")
        or mentions_day_or_time(user_message)
    )
    # An explicit request to switch language ("can you explain in Arabic") outranks everything else
    # below — those all infer the reply language from what SCRIPT the message happens to be written
    # in, which is the wrong signal here: the customer is directly telling us which language to use,
    # regardless of the script their own request used. Found live: "can you explain in Arabic" is
    # itself plain English (decisive_language reads "en"), so without this override the sticky/
    # decisive logic below just kept replying in English — with the model, forced into English by
    # LANG_DIRECTIVE, sometimes even inventing a false "I can't write in Arabic here" justification.
    requested_lang = signals.get("requested_reply_language")
    if requested_lang in ("en", "ar"):
        lang = requested_lang
    elif names_mode_or_time and detect_language(user_message) == "en":
        lang = "en"
    elif is_confirmation_turn and established_lang:
        lang = established_lang
    else:
        lang = decisive_language(user_message)
        if not lang:
            # The message itself is NOT decisive (a lone loanword, no letters at all — bare
            # confirmations are handled above) — regex genuinely can't tell, by design. An
            # established conversation language is a far more reliable signal on an ambiguous turn
            # than a single fresh LLM guess, so prefer it; only fall back to the LLM's guess when
            # there's no established language yet (e.g. this is the very first turn and the
            # message alone can't decide it either).
            lang = established_lang or signals.get("reply_language")
        if lang not in LANG_DIRECTIVE:
            lang = _resolve_language(user_message, chat_history, established_lang)
    # `pricing` also covers project-timing/duration questions (user's explicit product decision —
    # "how long does it take" gets the exact same treatment as "how much does it cost": no exact
    # number, explain what it depends on, offer the meeting). understand_turn()'s own is_pricing
    # signal is asked to recognize both (see UNDERSTAND_PROMPT); is_timing_question is the regex
    # reinforcement, same role is_pricing_question already plays for pricing itself.
    # Found live: the old "only fall back to the regex when the LLM signal is MISSING (None)" logic
    # left is_pricing_question() powerless whenever understand_turn() confidently (but wrongly)
    # returned is_pricing=false — "can i know the approximate range before" did exactly that, so the
    # regex reinforcement never even got consulted despite matching. OR the regex in unconditionally,
    # the same way is_timing_question already works just below — a deterministic match should never
    # be overridden by the LLM's OWN signal saying otherwise.
    pricing = bool(signals.get("is_pricing")) or is_pricing_question(user_message)
    if is_timing_question(user_message):
        pricing = True
    # Deterministic override for the negation gap _NEGATED_PRICE_OR_TIMING_RE's own comment explains —
    # checked before the LLM signal even gets a say, since a clear negation is a stronger, narrower
    # signal than a generic "is this about pricing" guess either way.
    if pricing and _NEGATED_PRICE_OR_TIMING_RE.search(user_message or ""):
        pricing = False
    # If OUR last reply just gave the scope-dependency acknowledgment ("it depends on the scope of
    # what you need, the services involved, and the size of your business"), the customer's very next
    # message answering exactly that — naming the service, their business size, etc. — is still part
    # of THIS pricing/timing exchange, even when it asks no fresh price/timing question of its own and
    # both understand_turn() and the regexes above miss it entirely. Found live twice: "i want a logo
    # and we are a small business" and "i wanna do a logo and i am a small business" (i.e. the general
    # shape "here are the specifics you just said it depends on", not one fixed wording) both got
    # classified as a fresh field-pitch reveal, silently dropping the pricing thread instead of tying
    # the answer back to it. Scoped to the single immediately-following turn (same lifetime as
    # offered_call below), and self-limiting: it only ever ADDS "pricing" as a genuinely-asked-earlier
    # topic for select_directives/compound_intent_note to route around a higher-priority directive if
    # one applies — the overrides just below (owner/phone, pending-call-reply) still get the final say
    # whenever this message is clearly about something else entirely.
    #
    # Exempted (2026-07-24) on a closing/gratitude/vague-reaction message (is_gratitude_closing or
    # is_vague_reaction — the same narrow, high-confidence word list is_bare_ack itself trusts). Found
    # live, and it was bad: a company-profile conversation ended in "anyway thanks, don't want to do
    # business with you" — is_gratitude_closing correctly recognized it as a closing, but is_bare_ack
    # ALSO requires "not pricing", and this override had just force-set pricing=True from the PRIOR
    # reply's genuine "will depend on the specific details" explanation — so the customer's explicit
    # goodbye got answered with "Pricing and timing specifics are always discussed during the meeting"
    # instead of any kind of closing acknowledgment. A closing is never a continuation of the pricing
    # thread, however the turn right before it was framed.
    if (not pricing and last_assistant_gave_scope_dependency(chat_history)
            and not is_gratitude_closing(user_message) and not is_vague_reaction(user_message)):
        pricing = True
    # Deterministic override, same failure class as the wants_contact-vs-more_detail one above:
    # found live — "who is the owner" got is_pricing=true from the LLM again, hijacking the turn
    # into a redundant "cost depends on scope..." preamble before the (still correct, thanks to the
    # owner-name guarantee) actual answer. A message clearly asking about the owner or phone number
    # (reliable, narrow regexes) is virtually never ALSO a genuine pricing question in the same
    # breath — only distrust pricing here when it's purely the LLM's guess, not the deterministic
    # regex, so an actual "how much does X cost, by the way who owns this place" compound would
    # still work.
    if pricing and not is_pricing_question(user_message) and not is_timing_question(user_message) and (
        _asks_about_owner(user_message) or _asks_about_phone(user_message)
    ):
        pricing = False
    # Same failure class again: found live — "what services do you offer overall" (a plain
    # asks_services_overview question, no price/timing content at all) got is_pricing=true from the
    # LLM anyway, several turns into a pricing-heavy conversation — hijacking a fresh services
    # question into a stale, unrelated "the timeline depends on the scope..." reply. Same guard shape
    # as the owner/phone override above: only distrust pricing when the deterministic price/timing
    # regexes don't independently confirm it, so a genuine compound ("what services do you offer and
    # how much do they cost") still works.
    if pricing and not is_pricing_question(user_message) and not is_timing_question(user_message) and (
        bool(signals.get("asks_services_overview"))
    ):
        pricing = False
    assistant_turns = sum(1 for m in chat_history if m.get("role") == "assistant")
    is_early = assistant_turns < EARLY_TURN_THRESHOLD

    # Did WE offer the call in our last message? This matches our OWN controlled phrasing, so it's
    # reliable — unlike trying to regex-classify the customer's free-text reply. When true, we hand
    # the model one directive and let IT read the reply (any arabizi spelling of "yes", a day/time,
    # or a subject change) and pick the right branch.
    offered_call = last_assistant_offered_call(chat_history)

    # A genuine, standalone location question ("where are you located") deserves the office address
    # regardless of whether OUR own last message happens to still match _CALL_OFFER_RE. Found live:
    # right after a day/time gets confirmed, our reply is "Got it, I'll check the schedule and get
    # back to you shortly to confirm" — which mentions no office/onsite/online wording at all — so
    # offered_call flipped false on the very next turn and "where are you located" fell through every
    # higher-priority directive, landed on bare_ack, and got "Great, talk soon!" instead of an answer.
    # Forcing call_flow to pick this up reuses CALL_FLOW_DIRECTIVE's own location EXCEPTION bullet
    # (give the address, skip the day/time push) — the general, already-existing handling for this
    # category of question, not a new one-off patch for this exact wording.
    if is_pure_location_question(user_message):
        offered_call = True

    # Same failure class as the location-question override just above, found live: a pure location
    # question ("where are you located") correctly gets the address via that override — but per
    # CALL_FLOW_DIRECTIVE's own EXCEPTION bullet, that address-only reply deliberately does NOT
    # mention onsite/online/office wording (so it "STOPS there" instead of also pushing day/time) —
    # which means it does NOT match _CALL_OFFER_RE either, so offered_call flips back to False on the
    # very NEXT turn too. When the customer's OWN next message then explicitly commits ("okay we can
    # schedule an onsite meet"), it fell through every higher-priority directive with offered_call
    # false and call_flow never even considered, so none of CALL_FLOW_DIRECTIVE's guarantees (address
    # already given + MUST ask day/time) applied — the model, with no active directive, just
    # improvised "I'll confirm the meeting time and send details shortly" (the phrasing reserved for
    # AFTER a day/time is actually given) without ever asking for one. A genuine onsite/online
    # commitment in the customer's OWN current message, when the meeting was mentioned at any earlier
    # point in the conversation, is exactly as reliable a local signal as the location-question case
    # — reuses the existing _ONSITE_COMMITMENT_RE/_ONLINE_OFFER_TERMS_RE regexes, no new detection.
    elif (_ONSITE_COMMITMENT_RE.search(user_message or "") or _ONLINE_OFFER_TERMS_RE.search(user_message or ""))\
            and _meeting_ever_mentioned(chat_history):
        offered_call = True

    # Deterministic override, same failure class as the wants_field_pitch/wants_contact overrides in
    # _turn_flags: found live that understand_turn()'s is_pricing signal can ALSO persist as "true"
    # on a turn that's really just continuing an already-active call flow — "eh ana b lebnen"
    # ("yeah I'm in Lebanon", confirming the onsite path) got is_pricing=true again, which — since
    # pricing is checked before call_flow in EXCLUSIVE_DIRECTIVES — hijacked the turn into a generic
    # "pricing already discussed" reply instead of CALL_FLOW_DIRECTIVE's onsite branch (real address
    # + day/time question). Only distrust it when the DETERMINISTIC regex doesn't independently
    # confirm an actual price/timing question, and the message reads like a call-flow continuation
    # (_looks_like_pending_call_reply) — a genuine new pricing question mid-flow, regex-detectable,
    # must still be able to interrupt it exactly as before.
    #
    # Also exempted (2026-07-24) on a fresh field reveal (signals["wants_field_pitch"]), the SAME
    # exemption _turn_flags applies to its own wants_field_pitch/wants_contact resets just below, for
    # the identical reason: a first-time field reveal is substantive new content regardless of how
    # short it reads. Found live: "بدي لوغو وأنا صاحب مشروع صغير" ("I want a logo, I'm a small
    # business") is short and question-free — exactly _looks_like_pending_call_reply's shape — so this
    # reset wiped out last_assistant_gave_scope_dependency's whole point (recognizing this message
    # answers what pricing depends on) a few lines before flags computation even saw it, silently
    # dropping the pricing thread from field_pitch's compound-intent coverage entirely.
    if (pricing and offered_call and not is_pricing_question(user_message)
            and not is_timing_question(user_message) and _looks_like_pending_call_reply(user_message)
            and not bool(signals.get("wants_field_pitch"))):
        pricing = False

    # Same OR-with-the-LLM-signal fallback as onsite_meeting_confirmed() in main.py, applied here to
    # the quality checks (anti-restating, time-ask-not-dropped) instead of the booking trigger — a
    # regex-blind phrasing ("let's meet tomorrow at four") should get the SAME guarantees as one the
    # regex recognizes, not silently skip them. understand_turn()'s confirms_meeting_daytime is the
    # backstop; the regex stays the fast/primary path.
    # BUT (2026-07-20 fix): confirms_meeting_daytime turned out to have its own reliable false-positive
    # — it consistently (5/5 across three prompt-wording attempts) returned true for a plain day name
    # plus an approval word ("monday would be good", no time at all), treating the AFFIRMATIVE TONE as
    # if it were a time. Prose fixes to the understand.py prompt didn't help (the model wasn't unsure,
    # it was confidently wrong — same class of stubborn miss as the "3otoor" industry case earlier this
    # session), so this is a deterministic override instead: when the regex POSITIVELY finds a day
    # mentioned but NO time at all (not even the spelled-out-hour pattern just added to _TIME_RE), that
    # code-level evidence is trusted over the LLM backstop, and confirms_meeting_daytime is ignored for
    # that turn. The backstop still applies normally whenever no day is detected at all (so it can
    # still catch a pure time-only phrasing the regex misses).
    day_only_per_regex = mentions_day(user_message) and not mentions_specific_time(user_message)
    # ALSO distrust confirms_meeting_daytime on a pure location question (found live): "what's the
    # address again", asked right after an actual day/time WAS just confirmed in the prior turn,
    # still got confirms_meeting_daytime=true — momentum from that prior turn, not this message
    # actually containing a time. Since gave_specific_time=true feeds DAYTIME_ACK_FALLBACK, a hard
    # FINAL safety net that overwrites the entire reply, this misfire discarded a correct address
    # answer and replaced it with "Got it, I'll check the schedule..." instead. A message asking
    # where the office is can never simultaneously be giving a day/time, so this exclusion is safe.
    gave_specific_time = mentions_specific_time(user_message) or (
        bool(signals.get("confirms_meeting_daytime")) and not day_only_per_regex
        and not is_pure_location_question(user_message)
    )

    # True on a FOLLOW-UP turn where we already promised to check the schedule for a day/time that
    # was fully settled earlier, and this message doesn't bring a new one — the moment a bare "okay"
    # risks getting fabricated into a false booking confirmation (see PREMATURE_REBOOKING_* above).
    schedule_check_pending = (
        offered_call and not gave_specific_time and _schedule_check_already_promised(chat_history)
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Gold style examples for this language (see STYLE_EXAMPLES at top of file) — they teach the
    # model the exact Lebanese dialect/spelling to imitate. No-op if the slots are empty.
    messages.extend(_few_shot_messages(lang))
    messages.extend(chat_history)  # last N turns

    # Tell the model what we already know about this customer, so it tailors and stops
    # re-asking / re-inferring the field, service, etc. every turn.
    profile_line = profile_directive(profile)
    if profile_line:
        messages.append({"role": "system", "content": profile_line})

    # One rep (Sami or Lana) is assigned per conversation in main.py and carried in the profile —
    # this just tells the model who, so it can answer if asked. Self-gated in the text itself (only
    # surfaces if the customer actually asks), so it's cheap to always include rather than trying to
    # detect the "who am I talking to" intent as its own signal.
    if profile.get("assigned_rep"):
        messages.append({"role": "system", "content": rep_directive(profile["assigned_rep"])})

    # Assemble this turn's directive(s) from the declarative table: the first matching
    # exclusive directive (priority order is policy) plus any additive ones. See
    # select_directives for why this replaced an if/elif ladder.
    flags = _turn_flags(offered_call, pricing, is_early, profile, signals, user_message,
                         _last_assistant_asked_question(chat_history))
    selected_directives = select_directives(flags)
    # Exposed in this function's return value (see the tuple at the very end) so main.py can tell
    # whether this turn was actually handled by a specific, non-retrieval-dependent directive
    # (pricing, contact_request, call_flow, bare_ack, services_overview, field_pitch) rather than a
    # plain "answer from whatever was retrieved" turn — needed to fix the knowledge-gap email
    # firing on turns that were answered correctly regardless of retrieval score (see its own note
    # at the call site).
    primary_directive = _primary_exclusive_ask(flags)
    # Computed once, here, so both compound_intent_note (the secondary-ask path) and the
    # pricing_repeat/contact_repeat gates just below (the primary-directive path) use the exact same
    # "was the meeting mentioned anywhere earlier" read — see compound_intent_note's 2026-07-21 note
    # for why the secondary-ask path needs this too, not just the primary one.
    meeting_ever_mentioned = _meeting_ever_mentioned(chat_history)
    # Compound-intent note (see compound_intent_note above): needs per-turn signals/profile values
    # (the actual service/industry name), so it's computed here rather than as a static
    # ADDITIVE_DIRECTIVES entry — appended AFTER the primary directive(s) so it reads as "also cover
    # this" rather than competing with them.
    extra_note = compound_intent_note(flags, signals, profile, meeting_ever_mentioned)
    if extra_note:
        selected_directives.append(("compound_intent", extra_note))
    selected_directive_names = {name for name, _ in selected_directives}
    for _, directive_text in selected_directives:
        messages.append({"role": "system", "content": directive_text})

    # The actual per-turn paraphrase UNADDRESSED_INTENT_DIRECTIVE refers to — kept out of the static
    # directive text above (see that constant's own comment) since it varies every turn.
    if "unaddressed_intent" in selected_directive_names:
        intent_text = (signals.get("customer_intent") or "").strip()
        if intent_text:
            messages.append({"role": "system", "content": (
                f"What the customer's latest message actually says or means: \"{intent_text}\". "
                "Address this specifically in your reply."
            )})

    # Found live: "do you have a phone number for whatsapp" — a dedicated, self-contained question
    # with its own direct answer already sitting in the retrieved context — still got CONTACT_
    # REQUEST_DIRECTIVE's full BOOKING_INSTRUCTION pitch ("you can choose an onsite meeting or a free
    # 15-minute online consultation...") tacked on BEFORE the number, even though nothing about
    # scheduling was asked. Same category of fix as CALL_FLOW_DIRECTIVE's pure-location-question
    # EXCEPTION: a specific, directly-answerable fact-ask shouldn't be prefaced with an unrelated
    # meeting pitch just because it happens to route through contact_request. Generic contact intent
    # ("how do I reach you", "what's the next step") still gets the normal meeting-choice answer —
    # this only overrides it when the message is unambiguously about the phone/WhatsApp number itself.
    phone_only_contact_request = (
        "contact_request" in selected_directive_names and _asks_about_phone(user_message)
    )
    if phone_only_contact_request:
        messages.append({"role": "system", "content": (
            "The customer specifically asked for the phone number or WhatsApp contact — that's a "
            "direct fact already in the retrieved context. Just give the number plainly and stop "
            "there. Do NOT also offer or mention the onsite meeting, the online consultation, or any "
            "scheduling choice in this reply — they didn't ask about scheduling, only for the number."
        )})

    # Unconditional (see _asks_for_manager_or_human above) — not scoped to any particular directive
    # winning, since this ask needs to override whatever generic meeting-callback text would
    # otherwise fire.
    wants_human_escalation = _asks_for_manager_or_human(user_message)
    if wants_human_escalation:
        messages.append({"role": "system", "content": (
            f"The customer is asking to speak to a manager, boss, or a real person directly — give "
            f"them the WhatsApp number to reach a person: {CONTACT_PHONE} . Say it plainly, do NOT "
            f"deflect with the generic meeting-choice offer or a vague 'go ahead with the meeting "
            f"option' callback — that doesn't answer 'I want to talk to a person'."
        )})

    # OR'd with message_category=="small_talk" (2026-07-26): found live — "Keef akhabrak" ("how are
    # you"/"what's your news", a real small-talk phrasing the fixed _SMALL_TALK_RE lexicon doesn't
    # cover) got no directive at all, so the model improvised a small-talk-shaped reply on its own AND
    # then tacked on an unprompted "email us at hello@blnkcs.com" — SMALL_TALK_DIRECTIVE's own "don't
    # deflect into a sales-mode non-answer" guidance was never in the prompt to prevent it. The LLM's
    # own message_category already correctly classified this as small_talk; the regex is kept as the
    # fast/free first check, same OR-with-the-LLM-signal pattern used throughout this file.
    if is_small_talk(user_message) or signals.get("message_category") == "small_talk":
        messages.append({"role": "system", "content": SMALL_TALK_DIRECTIVE})
    elif _contains_small_talk(user_message):
        messages.append({"role": "system", "content": SMALL_TALK_ALONGSIDE_DIRECTIVE})

    # Flag computed here; the actual directive is appended further below, AFTER pricing_repeat/
    # PRICING_FIRST_EXPLAIN_NO_MEETING_NOTE/contact_repeat — found live (2026-07-24): appending it HERE
    # put it BEFORE those notes in the messages list, which per this file's own "later = more specific,
    # current instruction, takes precedence" convention meant PRICING_REPEAT_NO_MEETING_NOTE's rigid
    # "word for word every time" framing was effectively overriding CORRECTION_DIRECTIVE instead of the
    # other way around — "i did not ask for cost" still got the canned pricing redirect. Moved to
    # actually be last, not just claim to be in its own comment.
    correction_turn = is_correction(user_message)

    # Found live: "and the approximate costs" — unambiguously a cost question (matches _PRICE_RE,
    # not _TIMING_RE) — still got PRICING_DIRECTIVE's TIMING framing ("How long it takes depends on
    # the scope..."). PRICING_DIRECTIVE's own text already tells the model to pick "whichever was
    # actually asked," but leaving that judgment call to the model alone clearly isn't reliable enough
    # on its own (this is the same non-compliance pattern as everywhere else in this file) — likely
    # nudged here by recent conversation history genuinely being about timing. When the deterministic
    # regexes can tell cost and timing apart with no ambiguity (one matches, not the other), reinforce
    # that reading explicitly instead of trusting the directive's general framing line alone.
    if "pricing" in selected_directive_names:
        _is_cost = is_pricing_question(user_message)
        _is_timing = is_timing_question(user_message)
        if _is_cost and not _is_timing:
            messages.append({"role": "system", "content": (
                "This turn's question is specifically about COST/price, not timing/duration — say "
                "'the cost depends on...', not 'how long it takes depends on...', regardless of what "
                "was discussed earlier in this conversation."
            )})
        elif _is_timing and not _is_cost:
            messages.append({"role": "system", "content": (
                "This turn's question is specifically about TIMING/duration, not cost/price — say "
                "'how long it takes depends on...', not 'the cost depends on...', regardless of what "
                "was discussed earlier in this conversation."
            )})
        # When neither regex matches THIS message on its own (e.g. a bare "can i know approximately"),
        # no per-topic reinforcement is injected here — that's the general "read every message as
        # part of the conversation" rule in SYSTEM_PROMPT's job, and it's deliberately written to
        # cover ambiguous follow-ups on ANY topic, not just cost-vs-timing specifically.

    # The OPPOSITE direction of the cost-vs-timing reinforcement just above: found live — right after
    # a pricing/timing exchange, "can you explain the concept" then "no i mean the services" both
    # correctly selected the right directive (more_detail, then services_overview — verified directly,
    # this is NOT a directive-selection bug) but the model's GENERATED TEXT ignored that directive both
    # times and just repeated the pricing/timing framing anyway ("the cost depends on the scope...").
    # Same root cause as the cost-vs-timing case: SYSTEM_PROMPT's "read every message as part of the
    # conversation, it almost always continues the most recent topic" rule is a good general default,
    # but the model over-applies it here even when THIS turn's specific directive explicitly says to
    # do something different — the general rule should never outrank the specific one, but leaving that
    # priority to the model alone isn't reliable enough, same non-compliance pattern as everywhere else
    # in this file. Reinforce explicitly whenever services_overview/more_detail actually won.
    if "services_overview" in selected_directive_names or "more_detail" in selected_directive_names:
        messages.append({"role": "system", "content": (
            "This turn's question is NOT about pricing, cost, or timing — do not mention cost, "
            "price, or how long something takes anywhere in this reply, even if pricing/timing was "
            "discussed earlier in this conversation. Answer what they're actually asking about now, "
            "per the directive above."
        )})

    # True on the exact compound case compound_intent_note's override_note addresses: field_pitch
    # (or field_pitch_unclear) primary, pricing dropped as a secondary ask, meeting never mentioned
    # yet. Found live: the model resolved the field_pitch-vs-pricing contradiction by dropping BOTH
    # the pricing acknowledgment AND the meeting offer rather than picking one — the override_note
    # fix is a soft instruction, so (same as every other soft instruction in this file) it needs a
    # deterministic backstop verifying it actually held. Used below to force a regen if the draft
    # never mentions the meeting at all.
    field_pitch_pricing_compound_owes_meeting = (
        "compound_intent" in selected_directive_names
        and _primary_exclusive_ask(flags) in ("field_pitch", "field_pitch_unclear")
        and "pricing" in _dropped_exclusive_asks(flags)
        and not meeting_ever_mentioned
    )

    # The meeting-already-mentioned counterpart to the guarantee above. When the meeting was already
    # on the table, SECONDARY_ASK_BUILDERS["pricing"] deliberately switches to "note briefly that it
    # depends on the scope... but do NOT mention the meeting again" instead of offering it — so the
    # guarantee above (which checks for the booking CHOICE) doesn't apply here, and nothing else
    # verified the brief scope-dependency acknowledgment itself actually survived. Found live: asked
    # "what are your prices" (got the scope-dependency answer + meeting offer), then answered with
    # concrete scope specifics ("i wanna do a logo and my company is a startup small business") while
    # ALSO explicitly re-asking "what is the approximate price" — field_pitch won the primary slot on
    # the specifics, pricing was correctly dropped into compound_intent's secondary-ask text, but the
    # draft dropped the pricing acknowledgment entirely and gave an unrelated service pitch instead —
    # same non-compliance pattern as every other soft instruction in this file, hence the backstop.
    field_pitch_pricing_compound_repeat_owes_ack = (
        "compound_intent" in selected_directive_names
        and _primary_exclusive_ask(flags) in ("field_pitch", "field_pitch_unclear")
        and "pricing" in _dropped_exclusive_asks(flags)
        and meeting_ever_mentioned
    )

    # True when field_pitch actually won this turn (see the 2026-07-21 note on EXCLUSIVE_DIRECTIVES
    # above for why field_pitch can now win even while `offered_call` is true). Several guarantees
    # below use the raw `offered_call`/`pricing`/`wants_contact` FLAGS — not "which directive won" —
    # to decide "is this a purely transactional turn that must suppress pitch content/client
    # citations." Those flags can all be true on the SAME turn field_pitch wins (a meeting was
    # offered earlier in the conversation), which would otherwise strip exactly the fuller, cited
    # pitch FIELD_PITCH_DIRECTIVE just asked for. Threaded into those checks below to exempt them.
    field_pitch_won = "field_pitch" in selected_directive_names

    # True when this turn's PRIMARY job is a field pitch (clear or unclear-field) with no compound
    # pricing/contact_request ask alongside it — both FIELD_PITCH_DIRECTIVE and FIELD_PITCH_UNCLEAR_
    # DIRECTIVE explicitly forbid mentioning the meeting, but found live (2026-07-21, "c.o company")
    # the model pushed it anyway despite that explicit instruction — same non-compliance pattern as
    # everywhere else in this file. Used below for a deterministic strip. Compound pricing/
    # contact_request asks are the one legitimate exception (SECONDARY_ASK_BUILDERS may genuinely
    # need to mention the meeting for those), so this is only true when NEITHER was also dropped.
    field_pitch_turn_no_meeting = (
        bool(selected_directive_names & {"field_pitch", "field_pitch_unclear"})
        and not (set(_dropped_exclusive_asks(flags)) & {"pricing", "contact_request"})
    )

    # Same "shouldn't mention the meeting, strip it if it slips through anyway" treatment as
    # field_pitch_turn_no_meeting, extended to the other purely-informational turn types: a broad
    # "what do you offer" / "what industries have you worked with" / "tell me more about X" question is
    # not itself a pricing, contact, or scheduling ask (the user's explicit rule: only offer the meeting
    # when actually asked about contact details, prices, or timing), so none of these should end with a
    # meeting offer either — mirrors the standing SYSTEM_PROMPT "Call:" bullet's narrowed scope.
    informational_turn_no_meeting = (
        bool(selected_directive_names & {"services_overview", "client_industries", "more_detail"})
        and not (set(_dropped_exclusive_asks(flags)) & {"pricing", "contact_request"})
    )

    # These exclusive directives end with BOOKING_INSTRUCTION (offer the meeting phrased as an
    # onsite-vs-online CHOICE) — used below so the "did it actually present a real choice" guarantee
    # applies wherever BOOKING_INSTRUCTION was actually invoked, not just on pricing turns.
    # ELABORATION_DIRECTIVE ("more_detail") is DELIBERATELY excluded (2026-07-20 decision): a general
    # "tell me more about this service" question is not itself a signal to offer a meeting — only
    # contact intent, pricing, or field-specific context are. It used to be included here (when
    # ELABORATION_DIRECTIVE still pushed the call), which is exactly why this comment used to warn
    # about a vague meeting-mention slipping through on that turn; now that ELABORATION_DIRECTIVE no
    # longer offers the meeting at all, this guarantee including it would fight that instruction and
    # force a booking-choice sentence back onto a reply that was told not to have one.
    booking_choice_turn = bool(selected_directive_names & {"pricing", "contact_request"})

    # CALL_FLOW_DIRECTIVE's own "plain acceptance, no mode named yet" branch (e.g. customer replies
    # "we can go for the meet" after the meeting was offered, without saying onsite or online) has no
    # deterministic backstop of its own — found live: the model ignored that branch entirely and
    # free-styled an unrelated service pitch instead of asking which mode they'd prefer. That failure
    # is also self-compounding: last_assistant_offered_call() only looks at the LAST assistant
    # message, so once one reply drops every call-related word, offered_call goes false on the NEXT
    # turn too and call_flow stops firing altogether (this is what let the following "online" reply
    # also wander off-script). Detected narrowly so it doesn't fire on the OTHER call_flow branches,
    # which correctly don't mention onsite/online (declining, a genuine subject change, a day/time
    # reply, or a closing after the slot's already pending): mode not chosen yet in this conversation,
    # THIS message doesn't itself name a mode or give a day/time or ask a real question of its own,
    # and reads like a short acceptance rather than a substantive new topic.
    #
    # is_confirmation is also required (found live): "i am in qatar"/"i am in dubai" are SHORT, have
    # no onsite/online keyword, no day/time, no "?" — so they matched the heuristic above just like
    # "we can go for the meet" does, even though they already answer the choice via CALL_FLOW_
    # DIRECTIVE's "not in Lebanon = online" branch (added separately). That made this check force an
    # "ask directly which mode" regen onto a reply that had ALREADY correctly resolved to online,
    # producing a reply that both re-asked the question AND gave the link in the same breath. The
    # naming-a-location messages have real new content (a place), so understand_turn()'s is_confirmation
    # is reliably false for them (verified live) while it's reliably true for a genuine content-free
    # acceptance like "we can go for the meet" — so gating on it here keeps the two cases apart.
    call_flow_mode_pending = (
        "call_flow" in selected_directive_names
        and bool(signals.get("is_confirmation"))
        and not _address_already_given(chat_history)
        and not _link_already_given(chat_history)
        and not _ONSITE_OFFER_TERMS_RE.search(user_message or "")
        and not _ONLINE_OFFER_TERMS_RE.search(user_message or "")
        and not mentions_day_or_time(user_message)
        and "?" not in (user_message or "") and "؟" not in (user_message or "")
        and (is_affirmation(user_message) or len(re.findall(r"\w+", user_message or "")) <= 8)
    )

    # If the meeting was mentioned at ANY point earlier in this conversation, override the directive's
    # "offer the choice" line — added AFTER the directive text above so it takes precedence as the
    # more specific, current instruction. pricing and contact_request get DIFFERENT treatment (2026-
    # 07-21 narrowing, see PRICING_REPEAT_NO_MEETING_NOTE/BOOKING_CHOICE_REPEAT_NOTE above for why):
    # pricing is the bot re-raising the topic on its own on an ordinary content question, so it's
    # suppressed entirely; contact_request is the customer explicitly asking how to move forward, so
    # it still gets a brief reference rather than silence.
    pricing_repeat = "pricing" in selected_directive_names and meeting_ever_mentioned
    # Which dimension (cost or timing) THIS turn is actually about — needed because
    # _topic_scope_already_explained tracks the two separately (see its own 2026-07-24 comment).
    # Prefer the deterministic regexes when either one alone matches; when the message itself is
    # ambiguous (a topic-free follow-up like "but approximately how much" WOULD match cost here, but
    # something like a bare "and that?" wouldn't match either), fall back to whichever framing the
    # LAST assistant reply actually used — the same "read the conversation" principle as SYSTEM_
    # PROMPT's standing rule, just applied to this one deterministic lookup.
    _is_cost_ask = is_pricing_question(user_message)
    _is_timing_ask = is_timing_question(user_message)
    if _is_timing_ask and not _is_cost_ask:
        _repeat_topic_is_timing = True
    elif _is_cost_ask and not _is_timing_ask:
        _repeat_topic_is_timing = False
    else:
        _last_assistant_text = next(
            (m.get("content", "") for m in reversed(chat_history) if m.get("role") == "assistant"), ""
        )
        _repeat_topic_is_timing = (
            bool(_TIMING_SCOPE_EXPLAINED_RE.search(_last_assistant_text))
            and not bool(_COST_SCOPE_EXPLAINED_RE.search(_last_assistant_text))
        )
    pricing_topic_already_explained = pricing_repeat and _topic_scope_already_explained(
        chat_history, _repeat_topic_is_timing
    )
    # The other of the two pricing_repeat sub-cases: meeting already mentioned, but THIS dimension
    # hasn't been explained yet — PRICING_FIRST_EXPLAIN_NO_MEETING_NOTE governs, and its own "do NOT
    # re-offer the meeting" clause needs the same deterministic backstop every other soft "don't
    # mention the meeting" instruction in this file gets (see the checks below).
    pricing_first_explain = pricing_repeat and not pricing_topic_already_explained
    contact_repeat = "contact_request" in selected_directive_names and meeting_ever_mentioned
    booking_choice_repeat = pricing_repeat or contact_repeat
    if pricing_topic_already_explained:
        messages.append({"role": "system", "content": PRICING_REPEAT_NO_MEETING_NOTE})
    elif pricing_first_explain:
        messages.append({"role": "system", "content": PRICING_FIRST_EXPLAIN_NO_MEETING_NOTE})
    elif contact_repeat:
        messages.append({"role": "system", "content": BOOKING_CHOICE_REPEAT_NOTE})

    # Appended LAST among the directive notes (see correction_turn's own comment above for why this
    # exact position matters) so it takes precedence as the more specific, current instruction — the
    # customer correcting a misunderstanding overrides whatever topic/repeat framing the notes above
    # were written for.
    if correction_turn:
        messages.append({"role": "system", "content": CORRECTION_DIRECTIVE})

    # See MODE_ALREADY_SPECIFIED_NOTE's own comment above: a cold-open message that already names a
    # mode (onsite/online) before we've ever offered the call. Excluded on booking_choice_repeat
    # (that's a genuinely different, already-handled case) and requires NOT meeting_ever_mentioned —
    # if the meeting was already mentioned earlier, offered_call/call_flow would be the one handling
    # this, not booking_choice_turn at all.
    # "bare_ack" is included alongside booking_choice_turn (found live): "khalas onsite beje la
    # aandkon" ("okay, I'll come onsite") on a cold open has no pricing/contact/field/services signal
    # of its own to attach to — nothing else in EXCLUSIVE_DIRECTIVES fires, so it fell through to
    # bare_ack and got a bare "You're welcome." despite unambiguously naming a real meeting mode.
    # MODE_ALREADY_SPECIFIED_NOTE's own text is fully general regardless of which exclusive directive
    # nominally "won" — it just needs the mode-naming regex match, which is still required below — so
    # this isn't loosening the guard, just recognizing that a mode statement is real content even when
    # nothing else fired.
    mode_already_specified = (
        bool(selected_directive_names & {"pricing", "contact_request", "bare_ack"})
        and not booking_choice_repeat and not meeting_ever_mentioned
        and bool(_ONSITE_OFFER_TERMS_RE.search(user_message or "")
                 or _ONLINE_OFFER_TERMS_RE.search(user_message or ""))
    )
    if mode_already_specified:
        messages.append({"role": "system", "content": MODE_ALREADY_SPECIFIED_NOTE})

    # User's explicit product decision (2026-07-23): when the customer asks us to SEND/SHARE the
    # location rather than just asking where it is, give the WhatsApp number and tell them to
    # contact it for the location, instead of the raw address text. Scoped to whenever we'd
    # otherwise be about to give the address (call_flow winning, or the onsite half of
    # mode_already_specified) — a no-op otherwise, same pattern as every other conditional note here.
    send_location_instead_of_address = (
        _asks_to_send_location(user_message)
        and ("call_flow" in selected_directive_names or mode_already_specified)
    )
    if send_location_instead_of_address:
        messages.append({"role": "system", "content": (
            "The customer asked us to SEND/SHARE the location, not just where it is — they want a "
            "pin/map link to navigate by, which isn't something you can attach here. IMPORTANT: this "
            "overrides any other instruction above telling you to include the office address text — "
            f"do NOT write out the street address in this reply. Instead, ONLY give BLNK's WhatsApp "
            f"number ({CONTACT_PHONE}) and tell them to contact that number to get the exact "
            "location/pin. Do NOT also push a day/time question in this same reply."
        )})

    # See CALL_FLOW_NEW_TOPIC_REMINDER above: a sharply-scoped, end-of-prompt reinforcement of two
    # things CALL_FLOW_DIRECTIVE's "customer changed the subject" branch already says but which kept
    # getting lost — answer the new question for real, and don't re-explain the onsite/online choice.
    # Harmless no-op on the other 8 branches, so unconditional whenever call_flow is this turn's
    # directive.
    if "call_flow" in selected_directive_names:
        messages.append({"role": "system", "content": CALL_FLOW_NEW_TOPIC_REMINDER})

    messages.append({"role": "system", "content": LANG_DIRECTIVE[lang]})

    context_text = _format_context(context_chunks)
    messages.append({
        "role": "user",
        "content": f"Retrieved company knowledge:\n\n{context_text}\n\nUser question: {user_message}"
    })

    raw = _complete(messages)

    # Tracked from the very first draft, before any regen has a chance to touch it (2026-07-20 fix):
    # found live, non-deterministically, that a LATER regen in this pipeline (repetition, off-task-
    # pitch, client-citation checks all rewrite "the SAME reply, keeping everything else") can
    # accidentally drop or mangle the Calendly link while rewriting the surrounding sentence — not
    # every run, just some, since it depends on which regens fire and how the model handles the
    # rewrite each time. If the link was genuinely present in the model's own first attempt, it's
    # trusted to belong in the final reply too — the final safety net below restores it if any later
    # step silently lost it.
    link_in_first_draft = CALENDLY_LINK in (raw or "")

    # Same idea for Arabic script: the model keeps slipping into fus7a despite the directive.
    # If high-precision MSA markers show up, regenerate once in Lebanese dialect.
    if lang == "ar" and looks_like_fusha(raw or ""):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": FUSHA_FIX_DIRECTIVE},
        ])

    # Deterministic backstop for a same-sentence word stutter (see _find_stutter above), e.g.
    # "khidma mna7a w mna7a". Distinct from, and runs before, the cross-turn _check_repetition
    # below. Regenerate once, clean.
    stutter = _find_stutter(raw)
    if stutter:
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": stutter_fix_directive(stutter)},
        ])

    # Deterministic backstop for the "never use also/كمان" rule (see _CONTINUATION_WORD_RE above).
    # Regenerate once, clean.
    continuation_word = _find_continuation_word(raw)
    if continuation_word:
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": continuation_word_fix_directive(continuation_word)},
        ])

    # Deterministic backstop for the "never say 'which one', always end with a real question" rule
    # (see _services_overview_ending_bad above). Regenerate once, then deterministically replace the
    # ending if it's still bad — the regen itself can produce a different but equally broken fragment
    # ("What."), so the final check re-verifies the ACTUAL ending quality, not just the banned phrase.
    if flags["asks_services_overview"] and _services_overview_ending_bad(raw):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": which_one_fix_directive()},
        ])
        if _services_overview_ending_bad(raw):
            raw = _replace_which_one_ending(raw, lang)

    # On a purely transactional turn (pricing, call-flow, or the customer initiating contact),
    # verify the reply didn't get padded with off-task pitch/service content (see
    # _check_off_task_pitch above). Regenerate once, clean. Skipped when compound_intent_note fired
    # (2026-07-20): that note DELIBERATELY asks for service-detail content this turn because the
    # customer explicitly asked for it in the SAME message as the pricing/contact/call-flow ask — the
    # two guarantees were found live fighting each other, with this one winning and silently
    # stripping the very content the customer asked for and the other guarantee just added.
    if (not field_pitch_won and (pricing or offered_call or flags["wants_contact"])
            and "compound_intent" not in selected_directive_names):
        off_task_extra = _check_off_task_pitch(raw)
        if off_task_extra:
            raw = _complete(messages + [
                {"role": "assistant", "content": raw},
                {"role": "system", "content": off_task_pitch_fix_directive(off_task_extra)},
            ])
            # Re-check once more: found live that the FIRST regen attempt can itself still leave
            # off-task content in (or reintroduce different off-task content), silently reaching the
            # customer since nothing re-verified the regenerated text. Cheap — only runs in the
            # already-rare case this branch fired at all.
            off_task_again = _check_off_task_pitch(raw)
            if off_task_again:
                raw = _complete(messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "system", "content": off_task_pitch_fix_directive(off_task_again)},
                ])

    # Verify the office address wasn't dropped on the turn that's clearly supposed to give it (see
    # _implies_onsite_without_address above) — checked BEFORE the time-ask and company-info checks
    # below, since both of those are gated on the address actually being present; if the address
    # itself got dropped, fixing it here lets those two checks work correctly afterward instead of
    # silently never firing. Best-effort regen only — the guarantee is the final safety net at the
    # end of this function.
    #
    # Scoped to "call_flow" actually winning the turn (found live, not just offered_call being
    # true): a customer message like "ana bde aarf aktr aan l brand identity w kef momken tseedone b
    # mataame" ("I want to know more about Brand Identity and how you can help my restaurant") got
    # wants_contact misread True by the LLM, so CONTACT_REQUEST_DIRECTIVE won instead of more_detail
    # — and this check, gated only on offered_call (which was still true from an earlier turn), then
    # force-injected the FULL office address and a day/time question into what should have been a
    # content answer about Brand Identity, with zero relation to what was actually asked. offered_call
    # alone just means "our last message offered a meeting" — it says nothing about whether THIS
    # reply is actually call_flow's to control; only call_flow winning means the address/day-time
    # guarantee is actually this turn's job. mode_already_specified is ALSO included (found live): its
    # own note explicitly demands "BLNK's real office address from the retrieved context" on a cold
    # open where NOTHING else (pricing/contact/call_flow) is otherwise gating this check — without it,
    # "khalas onsite beje la aandkon" got a fabricated "123 Main Street, Beirut, Lebanon" instead of
    # the real address, since nothing was verifying the fact actually came from the retrieved context.
    if ((offered_call and "call_flow" in selected_directive_names or mode_already_specified)
            and not _address_already_given(chat_history) and not send_location_instead_of_address
            and _implies_onsite_without_address(raw)):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": office_address_fix_directive()},
        ])

    # CALL_FLOW_DIRECTIVE's onsite branch requires the address AND a time-ask in the SAME reply —
    # verify the time-ask wasn't dropped (see TIME_QUESTION_CHECK_PROMPT above). Only relevant once
    # we've actually given the address and the customer hasn't already given an actual TIME
    # themselves (that's the OTHER onsite branch, handled by the daytime-handling check below
    # instead) — a day alone ("tomorrow") does NOT count as already answered, since the time is
    # still unknown and this reply still owes them that question. This is a best-effort regen
    # attempt only — the guarantee is the final safety net at the end of this function (after
    # capping/trimming), same reasoning as the pricing-choice guarantee: a fallback appended THIS
    # early could still get deleted by _cap_sentences before the customer sees it.
    # Scoped to call_flow winning, same reasoning as the office-address check just above. Also
    # excluded on is_pure_location_question — that's CALL_FLOW_DIRECTIVE's own new exception (give
    # the address, don't push day/time), so a missing time-ask there is correct, not a bug to fix.
    if (offered_call and "call_flow" in selected_directive_names and _gave_office_address(raw)
            and not gave_specific_time and not _check_asks_time(raw)
            and not is_pure_location_question(user_message)):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": TIME_QUESTION_FIX_DIRECTIVE},
        ])

    # If we know the customer's field, verify the reply didn't pull in a product/industry word
    # belonging to a different business (see _check_industry_mismatch above). Regenerate once, then
    # RE-CHECK — found live: "MediDate" (an F&B packaging case study) kept reaching the customer as
    # proof for a flower/perfume shop despite the check correctly flagging the mismatch; the single
    # regen attempt just wasn't reliable enough to actually fix it, and nothing re-verified before.
    # If it's STILL mismatched after the regen, don't gamble on a third attempt — deterministically
    # strip the wrong-field client's sentence out instead (see _strip_client_sentence above).
    industry = (profile.get("industry") or signals.get("industry") or "").strip()
    if industry:
        mismatched = _check_industry_mismatch(raw, industry, context_text)
        if mismatched:
            raw = _complete(messages + [
                {"role": "assistant", "content": raw},
                {"role": "system", "content": industry_mismatch_fix_directive(mismatched, industry)},
            ])
            still_mismatched = _check_industry_mismatch(raw, industry, context_text)
            if still_mismatched:
                wrong_client = _first_client_mentioned(raw, ALL_CLIENT_NAMES)
                if wrong_client:
                    raw = _strip_client_sentence(raw, wrong_client)

    # Verify the reply doesn't repeat an opening/idea/citation already given earlier in this
    # conversation (see _check_repetition above). Regenerate once, clean.
    #
    # Excluded on booking_choice_repeat (2026-07-24): that turn's ENTIRE job, per BOOKING_CHOICE_
    # REPEAT_NOTE/PRICING_REPEAT_NO_MEETING_NOTE, is a deliberate brief callback to something already
    # said earlier ("you can check out the meeting options I mentioned before") — there is no "genuinely
    # new substance" to add instead, by design. Found live: a correct, on-brief reply like that got
    # flagged as repetitive by this LLM check, and repetition_fix_directive's "rewrite WITHOUT that
    # repeated idea, answer with genuinely new substance instead" gutted it down to a near-empty
    # fragment ("فيك تخبرني." — "you can let me know") since there was nothing else it was allowed to
    # say. The check's own purpose (catch UNINTENDED repeats) doesn't apply to a turn whose whole point
    # is an intended one.
    repeated = "" if booking_choice_repeat else _check_repetition(raw, chat_history)
    if repeated:
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": repetition_fix_directive(repeated)},
        ])

    # On the field-pitch turn specifically, verify the mandatory client citation wasn't skipped
    # (see _check_missing_client_citation above). Regenerate once, clean. Guarded the same way as
    # the final safety net below: don't trust an LLM "missing" verdict blindly when the name is
    # already sitting right there in the text — that would just prompt the model to add it a
    # second time.
    if flags["wants_field_pitch"] and industry:
        missing_client = _check_missing_client_citation(raw, industry, context_text)
        if missing_client and missing_client not in (raw or ""):
            raw = _complete(messages + [
                {"role": "assistant", "content": raw},
                {"role": "system", "content": missing_client_fix_directive(missing_client)},
            ])
            # This step just ADDED a client name back in — the one regen most likely to reintroduce
            # a repetition the earlier repetition check (above) had already cleared. Re-check once,
            # cheap since it only runs in the already-rare case this branch fired at all.
            repeated_again = _check_repetition(raw, chat_history)
            if repeated_again:
                raw = _complete(messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "system", "content": repetition_fix_directive(repeated_again)},
                ])
            # Final safety net (see FIELD_PITCH_CLIENT_FALLBACK above): re-check the regenerated
            # text, and if the model STILL skipped the citation, append it deterministically rather
            # than gamble on a third regeneration for this conversion-critical moment. Found live:
            # the LLM re-check itself can false-positive here — the regen step just above already
            # added the citation into the body text ("just like we did for Cira Beauty"), but
            # _check_missing_client_citation still reported it missing, so the fallback appended a
            # SECOND, redundant mention of the same client at the end of the same reply. The check
            # is only ever used to decide WHETHER to append — the actual text is right there to
            # verify deterministically, so do that instead of trusting the LLM's verdict blindly.
            still_missing = _check_missing_client_citation(raw, industry, context_text)
            if still_missing and still_missing not in (raw or ""):
                raw = (raw or "").rstrip()
                if raw and raw[-1] not in ".?؟…":
                    raw += "."
                raw += field_pitch_client_fallback(still_missing, lang)

        # This whole block can introduce (or re-introduce) a client citation — via the regen or the
        # deterministic fallback above — that was never checked against industry-mismatch, since
        # that check already ran and finished BEFORE this block started. Found live: "Farm Market"
        # (F&B) got added back in exactly this way for a flower shop, right after an earlier
        # industry-mismatch pass had correctly dropped a different wrong-field client — the two
        # checks run sequentially with no cross-verification, so whichever runs last wins blindly.
        # Re-verify one last time now that citation handling for this turn is done.
        final_mismatch = _check_industry_mismatch(raw, industry, context_text)
        if final_mismatch:
            wrong_client = _first_client_mentioned(raw, ALL_CLIENT_NAMES)
            if wrong_client:
                raw = _strip_client_sentence(raw, wrong_client)

    # Deterministic (no LLM judgment needed): a client already cited earlier in this conversation
    # must not be cited again — checked here for every turn EXCEPT pricing/call-flow/contact-request
    # turns, which already have their own stricter "no client at all" deterministic check below.
    if field_pitch_won or (not pricing and not offered_call and not flags["wants_contact"]):
        already_cited = _already_cited_client(chat_history)
        if already_cited:
            new_mention = _first_client_mentioned(raw, ALL_CLIENT_NAMES)
            if new_mention:
                raw = _complete(messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "system", "content": repeated_client_fix_directive(
                        already_cited, new_mention, _vague_client_filler_already_used(chat_history)
                    )},
                ])

    # Deterministic (no LLM judgment needed): the vague no-name fallback line ("worked with companies
    # in your domain before") repeating itself across turns — see _VAGUE_CLIENT_FILLER_RE above.
    # Simple strip rather than a regen: this filler is always a bolt-on clause, never load-bearing
    # content, so removing the sentence it lives in is safe and cheap (no extra round trip).
    if _vague_client_filler_already_used(chat_history) and _VAGUE_CLIENT_FILLER_RE.search(raw or ""):
        sentences = _split_sentences(raw)
        raw = " ".join(s for s in sentences if not _VAGUE_CLIENT_FILLER_RE.search(s)).strip()

    # Verify the customer's day/time didn't get self-booked/confirmed outright, or misread as a
    # greeting exchange (see DAYTIME_HANDLING_CHECK_PROMPT above). Only relevant when we've offered a
    # meeting and they just gave an actual TIME — a day alone ("tomorrow") isn't a confirmed slot
    # yet and must NOT trigger the "I'll check the schedule" acknowledgment; it still owes them the
    # time question (CALL_FLOW_DIRECTIVE's second branch). Regenerate once, clean.
    if offered_call and gave_specific_time and _check_daytime_handling(raw):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": DAYTIME_HANDLING_FIX_DIRECTIVE},
        ])

    # Verify a follow-up plain acknowledgment (after we already promised a schedule check) didn't
    # get fabricated into a false booking confirmation (see PREMATURE_REBOOKING_* above). Regenerate
    # once, clean.
    if schedule_check_pending and _check_premature_rebooking(raw):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": PREMATURE_REBOOKING_FIX_DIRECTIVE},
        ])

    # Deterministic (no LLM judgment needed): found live — after the customer self-booked via the
    # Calendly link and said so ("7ajazit maw3ad", then "shokran"), the model still reached for "I'll
    # check the schedule and get back to you," which is the ONSITE-only phrasing (that path needs a
    # human to actually check availability; the online path is self-served and Calendly confirms it
    # the instant they pick a slot). _SCHEDULE_CHECK_PROMISED_RE matches our own controlled phrasing,
    # so this is reliable. Only fires when the online link was given and the onsite address never was
    # — if the onsite path is (also) active, the schedule-check line is legitimate and left alone.
    if (offered_call and _link_already_given(chat_history) and not _address_already_given(chat_history)
            and _SCHEDULE_CHECK_PROMISED_RE.search(raw or "")):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply said you'll check the schedule and get back to the customer to "
                "confirm — that phrasing is ONLY correct for the onsite/office meeting path, and this "
                "conversation is on the ONLINE path (the Calendly link was already given, never the "
                "office address). The online booking is self-service and already confirmed the moment "
                "the customer picked their own slot — there is nothing for you to check or confirm. "
                "Rewrite the SAME reply as a brief, warm closing acknowledgment instead (e.g. 'Great, "
                "see you then!' / 'تمام، منتطلع نلتقي!'), nothing more."
            )},
        ])

    # Deterministic (no LLM judgment needed): PRICING_DIRECTIVE/CONTACT_REQUEST_DIRECTIVE/
    # ELABORATION_DIRECTIVE all end with BOOKING_INSTRUCTION, which explicitly requires actually
    # offering the meeting AS A CHOICE (onsite or online) — not trailing off with a vague "I can
    # give you more accurate info once we discuss it further", and not even a soft "we can discuss
    # this during a meeting" that mentions a meeting without presenting the actual choice. Checking
    # for BOTH an onsite-side term and an online-side term is stricter than _CALL_OFFER_RE (which
    # matches on the word "meeting" alone and let that soft case through in testing) and catches the
    # bare-mention case too. Gated on booking_choice_turn (any of the three directives above), not
    # just `pricing` — narrowing it to pricing-only was itself the bug that let a vague elaboration-
    # turn offer slip through with no guardrail at all. Excluded on booking_choice_repeat: THAT turn
    # is deliberately instructed NOT to spell out both options again (BOOKING_CHOICE_REPEAT_NOTE), so
    # this guarantee would otherwise fight its own instruction and force the repeat back in.
    def _offers_real_choice(text: str) -> bool:
        return bool(_ONSITE_OFFER_TERMS_RE.search(text or "") and _ONLINE_OFFER_TERMS_RE.search(text or ""))

    # PRICING_DIRECTIVE is explicit that a pricing reply has EXACTLY one job (scope-dependency
    # acknowledgment + the booking choice) and must NOT explain any service/deliverable, "even if it
    # seems related" — found live it still opened with 1-2 unrelated pitch sentences ("we can create
    # digital campaign assets...", "we provide social media design services...") before ever getting
    # to pricing. A sentence-level strip (same shape as field_pitch_turn_no_meeting below) is more
    # reliable than trusting the instruction alone: keep only sentences that are actually about
    # pricing/scope or the meeting/booking choice, drop everything else. Deliberately not gated on
    # pricing_repeat — a repeat pricing turn still shouldn't have pitch content either, it just also
    # gets the meeting-mention stripped separately by that check.
    def _is_pricing_or_booking_sentence(s: str) -> bool:
        return bool(
            _PRICE_RE.search(s) or _CALL_OFFER_RE.search(s) or is_timing_question(s)
            or _SCOPE_DEPENDENCY_RE.search(s)
        )

    # Excluded when compound_intent also fired (found live): a compound turn like "pricings... and
    # how you could help my business" legitimately needs an extra sentence beyond pure pricing/
    # booking content (e.g. the service categories compound_intent_note's override just sanctioned) —
    # stripping every non-pricing sentence here would delete exactly that sanctioned addition right
    # back out again.
    if "pricing" in selected_directive_names and "compound_intent" not in selected_directive_names:
        _sentences = _split_sentences(raw)
        _pitch_free = [s for s in _sentences if _is_pricing_or_booking_sentence(s)]
        if _pitch_free and len(_pitch_free) != len(_sentences):
            raw = " ".join(_pitch_free).strip()

    # Deterministic strip for the BARE_ACK_DIRECTIVE guarantee above (see _BARE_ACK_INVITATION_RE's
    # own comment): drop any trailing "feel free to ask"-style invitation clause, keeping the rest of
    # the (short) acknowledgment intact.
    if "bare_ack" in selected_directive_names:
        _stripped = _BARE_ACK_INVITATION_RE.sub("", raw or "").strip()
        if _stripped and _stripped != raw:
            raw = _stripped

    # Found live: "who is the owner", asked mid call-flow (offered_call true from an earlier turn),
    # got CALL_FLOW_DIRECTIVE's own "changed subject — answer it, no meeting content" branch, but the
    # model didn't follow it — the raw draft opened with "I'm here to help with your questions about
    # the meeting," gave the FULL office address, asked what day/time works, AND squeezed the owner's
    # name in as one clause in the middle. Unlike is_pure_location_question (where giving the address
    # IS still correct, just without the day/time push), an owner or phone question has NOTHING to do
    # with meeting logistics at all — the right reply doesn't mention the office/address/day/time in
    # any form. _asks_about_owner/_asks_about_phone are reliable, deterministic detectors for this
    # specific "completely unrelated factual question" case, so use them to catch stray meeting
    # content the model reintroduced despite the directive telling it not to. Best-effort regen only
    # (same as call_flow_off_script above, for the same reason: there's no clean deterministic way to
    # strip "address + day/time" back out of a reply without risking mangling the actual answer).
    if ("call_flow" in selected_directive_names
            and (_asks_about_owner(user_message) or _asks_about_phone(user_message))
            and (_gave_office_address(raw) or _check_asks_time(raw) or CALENDLY_LINK in (raw or ""))):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "The customer's message has NOTHING to do with the meeting/logistics flow — it's a "
                "plain factual question (about the owner or the phone/WhatsApp number). Rewrite the "
                "reply to answer ONLY that question, directly, using the retrieved context. Do NOT "
                "mention the office, the address, the online link, or ask about day/time in this "
                "reply at all — none of that belongs here, no matter how many turns into the meeting "
                "flow this conversation already is."
            )},
        ])

    # Same guarantee, for the owner's name (see OWNER_NAME's own comment for the documented failure
    # mode: retrieval succeeds, generation still declines to answer). Independent of offered_call —
    # this can come up on any turn. Placed AFTER the pricing-pitch-strip above (not before, where it
    # used to be): a customer asking about the owner alongside a pricing question, with no OTHER
    # tracked signal also dropped (so compound_intent never fires), would otherwise get this fix
    # applied first, only to have the pricing strip immediately delete the "Dina Rifai" sentence right
    # back out again since it doesn't look pricing-related. Best-effort regen only — the guarantee is
    # the final safety net.
    if (_asks_about_owner(user_message) and not _mentions_owner_name(raw)
            and not _owner_name_already_given(chat_history)):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                f"The customer asked who owns/founded the company. The retrieved context DOES contain "
                f"the answer — the owner and co-founder is {OWNER_NAME} — this is public business "
                f"information the company itself shares, not private data to withhold. Rewrite the "
                f"SAME reply, same length and tone, but actually state the name plainly instead of "
                f"deflecting or saying you'll confirm and follow up."
            )},
        ])

    # Same guarantee, for the phone/WhatsApp number (see CONTACT_PHONE's own comment) — found live in
    # the same pass: asked directly, on a fresh conversation, and the reply skipped the number and
    # pivoted straight to offering a meeting instead. Best-effort regen only; final safety net below.
    if (_asks_about_phone(user_message) and not _mentions_phone(raw)
            and not _phone_already_given(chat_history)):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                f"The customer asked for a phone number / WhatsApp contact. The retrieved context DOES "
                f"contain it — {CONTACT_PHONE} — this is BLNK's own public contact number. Rewrite the "
                f"SAME reply, same length and tone, but actually state the number plainly instead of "
                f"pivoting straight to offering a meeting without ever giving the number they asked for."
            )},
        ])

    # Excluded on mode_already_specified: that note deliberately wants ONLY one side (the address OR
    # the link), never both — forcing the full choice back in here would directly undo it. Excluded
    # on phone_only_contact_request the same way (found live): that note deliberately wants NO meeting
    # mention at all, so this check seeing "no choice offered" and forcing one back in directly undid
    # it — the phone-number reply got the full onsite/online pitch shoved back in front of the number.
    if (booking_choice_turn and not booking_choice_repeat and not mode_already_specified
            and not phone_only_contact_request and not _offers_real_choice(raw)):
        # Best-effort attempt at a natural, model-written fix. This is NOT the guarantee — a final
        # deterministic safety net at the end of this function catches whatever slips through here
        # (including cases where a later trim/cap step deletes a choice that WAS present at this
        # point, e.g. the model phrasing it as one trailing question that _trim_bot_tail then strips).
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply mentioned a vague 'meeting' or 'more detail' without actually "
                "presenting it as a choice (e.g. 'we can discuss this during a meeting' is NOT "
                "acceptable — that names no real options). Rewrite the SAME reply, same intent and "
                "length, but state BOTH options as a real choice — a plain statement, NOT a trailing "
                "question: (1) an onsite meeting at BLNK's office, for customers based in Lebanon, "
                "and (2) a free 15-minute online consultation, for everyone else. Both must be spelled "
                "out, and the sentence must not end in a question mark."
            )},
        ])

    # Backstop for call_flow_mode_pending (see its own comment above): the customer accepted the
    # meeting without saying which mode, and CALL_FLOW_DIRECTIVE's own branch for that case was
    # ignored — the draft neither asked which mode nor mentioned the meeting at all. Force it back on
    # track rather than letting an unrelated pitch stand.
    if call_flow_mode_pending and not _offers_real_choice(raw):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "The customer just accepted the meeting you offered earlier, without saying which mode "
                "they want. Your previous reply did not ask that — instead it drifted into unrelated "
                "content the customer didn't ask for. Rewrite the SAME reply as ONE short line: a brief "
                "acknowledgment, then ask directly whether they'd prefer the onsite meeting at BLNK's "
                "office or the free 15-minute online consultation. Nothing else — no service pitch, no "
                "example, no re-explaining anything already covered."
            )},
        ])

    # General catch-all for call_flow going completely off-script — found live a THIRD time (after
    # the bare-acceptance case above and the online-choice case): customer confirmed the onsite path
    # by saying "yes i am in lebanon", CALL_FLOW_DIRECTIVE's own onsite branch requires the real
    # office address plus a day/time question in that SAME reply, and the draft instead pitched an
    # unrelated "cohesive visual identity" service — zero mention of the office, a day, a time, the
    # online link, or anything meeting-related at all. Rather than keep adding a narrow, input-shaped
    # check for every individual phrasing that trips this (there will always be another one — this is
    # exactly the generic "detect turns that don't need a dataset pitch, just a logical reply" gap),
    # this checks the OUTPUT instead: whenever call_flow wins the turn, the reply MUST touch the
    # meeting flow somehow (_CALL_OFFER_RE is deliberately broad — office/onsite, online/calendly, a
    # day/time question, "check the schedule", etc.) UNLESS the customer's message was a genuine,
    # different question that CALL_FLOW_DIRECTIVE's own "changed subject" branch correctly answers
    # with no meeting mention at all — the fix instruction below explicitly preserves that legitimate
    # case instead of forcing meeting content onto it. Checked after call_flow_mode_pending (above),
    # so a raw already fixed by that block naturally passes and doesn't get regenerated twice.
    # Excluded on _asks_about_owner/_asks_about_phone (found live): "who is the owner" is EXACTLY
    # the "genuine subject-change, no call-offer words is CORRECT" case the comment below already
    # warns about — but until now nothing actually excluded it, so this check saw the (correct)
    # owner-only reply, decided it looked "off script," and regenerated the address+day-time content
    # right back in, undoing the dedicated owner/phone guard above. These two signals are reliable
    # enough (same regexes that guard, elsewhere) to safely carve out of the general check.
    call_flow_off_script = (
        "call_flow" in selected_directive_names and not _CALL_OFFER_RE.search(raw or "")
        and not (_asks_about_owner(user_message) or _asks_about_phone(user_message))
    )
    # Narrower companion flag, computed here for use as a FINAL safety net later (unlike
    # call_flow_off_script above, which only gets a soft mid-pipeline regen, no hard guarantee — see
    # that decision's own note near the final-safety-net section). Found live a FOURTH time: "tb wen
    # maktabkon" ("so where's your office") unambiguously names the office, the model's own `notes`
    # field even said "User is asking about the office location," and it STILL produced "Great, talk
    # soon!" — the regen above evidently didn't hold this time. The reason a hard guarantee was
    # deliberately left out for the broader call_flow_off_script is the "who is the owner"-style
    # genuine subject-change case, where a reply with NO call-offer words is CORRECT — but that risk
    # only exists when the customer's OWN message is ambiguous. When it explicitly names onsite/
    # online (as this one does), there's no such ambiguity: the reply MUST address it, so forcing a
    # fallback here is safe.
    call_flow_ignored_clear_mode_mention = (
        "call_flow" in selected_directive_names
        and not _address_already_given(chat_history) and not _link_already_given(chat_history)
        and bool(_ONSITE_OFFER_TERMS_RE.search(user_message or "")
                 or _ONLINE_OFFER_TERMS_RE.search(user_message or ""))
    )
    if call_flow_off_script:
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply ignored the meeting-scheduling conversation that's already active "
                "in this chat and instead gave a generic, unfocused service pitch the customer didn't "
                "ask for. Look again at exactly what the customer's LAST message said and respond to "
                "THAT specifically: if it's about the meeting in any way (confirming they're in "
                "Lebanon or want the onsite option, choosing online, giving a day/time, or just a bare "
                "acknowledgment), handle that exact step — for onsite confirmation this means giving "
                "BLNK's real office address from the retrieved context AND asking what day/time works, "
                "in this same reply; for online it means the Calendly link. If instead their message "
                "was a genuinely different, unrelated question, answer THAT directly using the "
                "retrieved context and do NOT mention the meeting at all — it was already offered "
                "earlier and must not be re-raised unprompted. Either way, do not give a generic "
                "sales pitch that fits neither case."
            )},
        ])

    # Deterministic (no LLM judgment needed): both FIELD_PITCH_DIRECTIVE and FIELD_PITCH_UNCLEAR_
    # DIRECTIVE explicitly forbid mentioning the meeting — found live it didn't reliably hold, same
    # non-compliance pattern as everywhere else in this file. field_pitch_turn_no_meeting already
    # excludes the one legitimate exception (a compound pricing/contact_request ask alongside it).
    if field_pitch_turn_no_meeting and _CALL_OFFER_RE.search(raw or ""):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply mentioned the meeting/call/consultation, even though the customer "
                "only revealed or asked about their field/business — not the meeting itself, and not "
                "pricing or how to get in touch either. Rewrite the SAME reply keeping the rest of its "
                "content intact, but remove every word about the meeting, the call, booking, or a "
                "consultation entirely."
            )},
        ])

    # Deterministic (no LLM judgment needed): same treatment for the other purely-informational turn
    # types (services_overview, client_industries, more_detail) — see informational_turn_no_meeting's
    # own comment for why none of these are a pricing/contact/timing ask.
    if informational_turn_no_meeting and _CALL_OFFER_RE.search(raw or ""):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply mentioned the meeting/call/consultation, even though the customer "
                "only asked a general informational question — not about pricing, how to get in touch, "
                "or scheduling/timing. Rewrite the SAME reply keeping the rest of its content intact, "
                "but remove every word about the meeting, the call, booking, or a consultation entirely."
            )},
        ])

    # Deterministic (no LLM judgment needed): the phone_only_contact_request equivalent of the check
    # just above — its own directive text explicitly says "do NOT also offer or mention the onsite "
    # meeting... they didn't ask about scheduling, only for the number," but nothing enforced that; the
    # booking_choice_turn guarantee above only EXCLUDES phone_only_contact_request from being forced to
    # offer a full choice, it doesn't stop a partial/vague mention from slipping through. Found live:
    # "do you have a phone number for whatsapp" got the number correctly, followed by "We can go ahead
    # with that meeting whenever works for you" tacked on anyway.
    if phone_only_contact_request and _CALL_OFFER_RE.search(raw or ""):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply mentioned the meeting/call/consultation, even though the customer "
                "only asked for the phone/WhatsApp number. Rewrite the SAME reply keeping the number "
                "intact, but remove every word about the meeting, the call, booking, or a consultation "
                "entirely — they didn't ask about scheduling."
            )},
        ])

    # Deterministic (no LLM judgment needed): PRICING_FIRST_EXPLAIN_NO_MEETING_NOTE's own "do NOT
    # re-offer the meeting" clause, same non-compliance pattern as everywhere else in this file. Found
    # live: "shu howe l waet yale bethejo" (first-ever timing ask, price already explained/meeting
    # already offered) got the correct "how long it takes depends on..." explanation, followed by "If
    # you're interested, we can set up a meeting to discuss this further" tacked on anyway — which then
    # cascaded into the NEXT turn misreading offered_call as freshly true and hijacking a genuine field
    # reveal into a stale onsite-address reply.
    if pricing_first_explain and _CALL_OFFER_RE.search(raw or ""):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply mentioned the meeting/call/consultation again, even though it was "
                "already offered earlier in this conversation for a different question. Rewrite the "
                "SAME reply keeping the scope-dependency explanation intact, but remove every word "
                "about the meeting, the call, booking, or a consultation entirely."
            )},
        ])

    # Deterministic (no LLM judgment needed): the opposite guarantee from the one just above — see
    # field_pitch_pricing_compound_owes_meeting's comment for the exact contradiction this resolves.
    # Here the draft dropped the meeting mention (and usually the pricing acknowledgment with it)
    # entirely, when the compound-intent override_note said it was owed.
    if field_pitch_pricing_compound_owes_meeting and not _offers_real_choice(raw or ""):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "The customer's message ALSO asked about pricing, alongside the field-pitch content — "
                "your previous reply either dropped the pricing part entirely, or mentioned a meeting/"
                "call too vaguely (e.g. 'let's schedule a call') without actually presenting the real "
                "choice. Rewrite the SAME reply, keeping the field-pitch content intact, and add a "
                "brief scope-dependency acknowledgment (e.g. 'pricing depends on the specifics of what "
                "you need') followed by offering a meeting AS A CHOICE, both sides spelled out: an "
                "onsite meeting at BLNK's office if they're based in Lebanon, and a free 15-minute "
                "online consultation otherwise — as a natural continuation of the same reply, not a "
                "disconnected afterthought."
            )},
        ])

    # Deterministic (no LLM judgment needed): PRICING_REPEAT_NO_MEETING_NOTE's bar on a repeat pricing
    # question is narrower than it used to be (2026-07-24 revision) — it now explicitly WANTS one
    # brief, fixed line ("pricing and timing specifics are always discussed during the meeting"), it
    # only forbids re-explaining or re-offering the full onsite-vs-online CHOICE. So the check here
    # uses _offers_real_choice (both sides spelled out), the same bar contact_repeat/compound_repeat
    # already use just above — not a bare _CALL_OFFER_RE match, which would also catch (and wrongly
    # strip) the very sentence this note now asks for.
    if pricing_repeat and _offers_real_choice(raw):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply spelled out the full onsite-office-vs-online-consultation choice "
                "again, even though it was already given earlier in this conversation. Rewrite the SAME "
                "reply, keeping the brief 'pricing and timing are always discussed during the meeting' "
                "line intact, but remove the re-explained choice — do not spell out the onsite office "
                "or 15-minute online consultation details again."
            )},
        ])

    # Deterministic (no LLM judgment needed): PRICING_REPEAT_NO_MEETING_NOTE's own rule — once THIS
    # dimension (cost, or timing) has already been explained once, it must NEVER be recited again in
    # any form, only the one-line meeting redirect. Gated on pricing_topic_already_explained, NOT the
    # broader pricing_repeat (2026-07-24 correction: pricing_repeat also covers the FIRST-ever ask
    # about a dimension that hasn't been explained yet, e.g. timing asked for the first time after
    # price was already explained — PRICING_FIRST_EXPLAIN_NO_MEETING_NOTE's whole job there is TO
    # give that "depends on..." explanation, so stripping it in that case would be exactly backwards).
    # _SCOPE_DEPENDENCY_RE's "depend" branch is the one reliable fingerprint of the forbidden
    # re-explanation (the redirect line itself never uses that word), so strip any sentence matching
    # it rather than the broader regex, which would also strip the redirect line itself.
    if pricing_topic_already_explained and re.search(r"\bdepend\w*\b", raw or "", re.IGNORECASE):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply recited the 'depends on the scope/services/size' explanation "
                "again, even though it was already given earlier in this conversation. Rewrite the "
                "SAME reply removing that sentence entirely — keep only the one-line 'pricing and "
                "timing specifics are always discussed during the meeting' redirect (plus any other "
                "genuinely new content this reply owes), nothing about scope/services/size again."
            )},
        ])

    # Deterministic (no LLM judgment needed): the contact_request equivalent of the pricing_repeat
    # backstop above. BOOKING_CHOICE_REPEAT_NOTE (soft instruction) asks for a brief, DIFFERENTLY-
    # worded reference instead of restating the full choice or re-pitching earlier content on a
    # repeat contact-request turn — found live it didn't reliably hold (the model gave the full
    # "onsite at BLNK's office... or a free 15-minute online consultation" sentence again, padded with
    # an extra re-pitch sentence on top). _offers_real_choice (both sides spelled out) is the same
    # fingerprint used for the mid-pipeline pricing/contact "missing choice" check above, just
    # inverted: here its PRESENCE on a repeat turn is the violation, not its absence.
    if contact_repeat and _offers_real_choice(raw):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply spelled out the full onsite-office-vs-online-consultation choice "
                "again, even though it was already given earlier in this conversation. Rewrite the SAME "
                "reply as ONE short sentence pointing the customer back to that meeting option in "
                "different, much shorter wording than before (e.g. 'you can check out the meeting "
                "options I mentioned before' or 'we can go ahead with that meeting whenever works for "
                "you') — do not spell out the onsite office or 15-minute online consultation details "
                "again, and do not add any other service/deliverable content."
            )},
        ])

    # Deterministic (no LLM judgment needed): a NARROWER violation of the same BOOKING_CHOICE_REPEAT_
    # NOTE instruction — the check just above only catches spelling out BOTH sides again; this catches
    # the model instead silently PICKING onsite and giving the real office address, without the
    # customer ever having said they're in Lebanon or want to meet in person (not in this message, not
    # anywhere earlier). Found live: "كيف فيني تواصل معكون" ("how can I contact you"), with the
    # customer's location never mentioned anywhere in the conversation, got the real office address
    # handed over directly — assuming onsite instead of pointing back to the still-open onsite-or-
    # online choice from earlier. Scoped to "contact_request" in selected_directive_names generally
    # (not just contact_repeat) — found the SAME assumption on what turned out to be a non-repeat turn
    # too, so both need this guarantee, not just the repeat path.
    if ("contact_request" in selected_directive_names and _gave_office_address(raw)
            and not _ONSITE_OFFER_TERMS_RE.search(user_message or "")
            and not any(m.get("role") == "user" and _ONSITE_OFFER_TERMS_RE.search(m.get("content", ""))
                        for m in chat_history or [])):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply gave the real office address, committing to an ONSITE meeting — "
                "but the customer has never said they're in Lebanon or want to meet in person, here or "
                "earlier in this conversation. Rewrite the SAME reply WITHOUT committing to onsite or "
                "online and WITHOUT giving the specific address: if the meeting was already mentioned "
                "earlier in this conversation, ONE short neutral sentence pointing back to it (e.g. "
                "'you can go ahead with the meeting we mentioned before, onsite or online, whichever "
                "works for you'); if this is the FIRST time it's coming up, present it as a real choice "
                "— onsite at BLNK's office if they're in Lebanon, or a free 15-minute online "
                "consultation otherwise — without picking one for them."
            )},
        ])

    # Deterministic (no LLM judgment needed): the compound_intent equivalent of the two backstops
    # above. Found live (2026-07-21) — the exact bug: field_pitch (checked before pricing/
    # contact_request in EXCLUSIVE_DIRECTIVES) won the primary slot, so the customer's ALSO-asked
    # contact-request only reached the reply via compound_intent_note's secondary-ask text — a path
    # pricing_repeat/contact_repeat (both keyed on being the PRIMARY selected directive) never covers.
    # The prompt-level fix is in SECONDARY_ASK_BUILDERS now, but the same non-compliance risk applies
    # here as everywhere else in this file, hence the backstop.
    compound_repeat = "compound_intent" in selected_directive_names and meeting_ever_mentioned
    if compound_repeat and _offers_real_choice(raw):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply spelled out the full onsite-office-vs-online-consultation choice "
                "again, even though it was already given earlier in this conversation. Rewrite the SAME "
                "reply, keeping the rest of its content intact, but replace that full choice with AT "
                "MOST one short, DIFFERENTLY-worded clause pointing back to it (e.g. 'you can check out "
                "the meeting options I mentioned before') — do not spell out the onsite office or "
                "15-minute online consultation details again."
            )},
        ])

    # Deterministic (no LLM judgment needed): field_pitch_pricing_compound_repeat_owes_ack's own
    # comment above explains the gap this closes — the meeting-already-mentioned counterpart of
    # field_pitch_pricing_compound_owes_meeting, which only guarantees the booking CHOICE is present
    # (or absent) and says nothing about the brief scope-dependency acknowledgment itself. Uses
    # _has_pricing_ack (price/cost/timing word + dependency language together), not a bare
    # _SCOPE_DEPENDENCY_RE match — see that helper's own comment for the false positive a looser
    # check let through live (a field-pitch closer mentioning "specifics"/"meeting" with zero actual
    # price or timing content).
    if field_pitch_pricing_compound_repeat_owes_ack and not _has_pricing_ack(raw or ""):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "The customer's message ALSO asked about pricing again, alongside giving new details "
                "about their field/business — your previous reply dropped the pricing part entirely. "
                "The scope-dependency explanation was already given earlier in this conversation, so "
                "do NOT recite it again. Rewrite the SAME reply, keeping the field-pitch content "
                "intact, and add ONE short, clear sentence that pricing and timing specifics are "
                "always discussed during the meeting — do NOT give a number or range, and do NOT "
                "re-explain or re-offer the onsite-vs-online choice itself (it was already given "
                "earlier in this conversation)."
            )},
        ])

    # Deterministic (no LLM judgment needed): the call_flow equivalent of the booking_choice_repeat
    # guard above. CALL_FLOW_NEW_TOPIC_REMINDER (a soft instruction) asks the model not to re-explain
    # the onsite/online choice on a "customer changed the subject" turn when it was already given
    # earlier — but found live that instruction alone didn't reliably hold (e.g. answering "do you
    # have restaurant clients?" correctly, then still tacking on the full "onsite at BLNK's office...
    # or a free 15-minute online consultation" sentence again). The one LEGITIMATE call_flow branch
    # that's allowed to mention both onsite and online again is "customer gave a plain acceptance with
    # no mode named yet, ask briefly which they prefer" — but that's a short question triggered by the
    # customer's message itself being a bare confirmation, which is_confirmation_turn already detects.
    # So: if the choice was already offered earlier, this isn't that plain-acceptance case, and the
    # draft still spells out both sides again, it's the bug, not the legitimate branch.
    if ("call_flow" in selected_directive_names and _booking_choice_already_offered(chat_history)
            and not is_confirmation_turn and _offers_real_choice(raw)):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply re-explained the full onsite-office-vs-online-consultation "
                "choice again, even though it was already given earlier in this conversation and the "
                "customer didn't ask to hear it again. Rewrite the SAME reply, keeping the actual "
                "answer to the customer's question intact and unchanged, but replace the re-explained "
                "choice with AT MOST one short clause noting the meeting is still an option (e.g. 'we "
                "can go ahead with that meeting whenever works for you') — do not spell out the onsite "
                "office details or the 15-minute online consultation again."
            )},
        ])

    # Deterministic (no LLM judgment needed): pricing/meeting-logistics turns must never cite a
    # client at all — same rule for the customer-initiated contact-request turn (CONTACT_REQUEST_
    # DIRECTIVE also forbids any citation, not just a repeated one). Check the real client names
    # straight from context metadata (see _client_names_in_context above) rather than trusting the
    # model to notice its own mistake.
    if not field_pitch_won and (pricing or offered_call or flags["wants_contact"]):
        mentioned_client = _first_client_mentioned(raw, _client_names_in_context(context_chunks))
        if mentioned_client:
            raw = _complete(messages + [
                {"role": "assistant", "content": raw},
                {"role": "system", "content": no_client_this_turn_fix_directive(mentioned_client)},
            ])

    # FINAL re-verification that no off-task pitch content snuck back in (see _check_off_task_pitch
    # above) — the earlier check+regen only guards against ITS OWN regen reintroducing the problem,
    # not against ANY of the several regens after it (company-info, industry-mismatch, repetition,
    # client checks) doing so. Found live: the company-info regen ("rewrite the SAME reply... adding
    # a brief question") reintroduced a full branding-pitch sentence on a call-flow turn that had
    # already been cleared of it. Same reasoning as the arabic-script-leak check below being placed
    # LAST rather than right after the first draft — checked here, after everything else, for the
    # same reason. Same compound_intent exclusion as the mid-pipeline check above.
    if (not field_pitch_won and (pricing or offered_call or flags["wants_contact"])
            and "compound_intent" not in selected_directive_names):
        off_task_final = _check_off_task_pitch(raw)
        if off_task_final:
            raw = _complete(messages + [
                {"role": "assistant", "content": raw},
                {"role": "system", "content": off_task_pitch_fix_directive(off_task_final)},
            ])

    # "en" now covers both real English and what used to be arabizi output, so an "en" reply must
    # never contain Arabic-script characters at all (model starts a word in Latin and finishes in
    # Arabic script, e.g. 'y3كس' — a leak that used to only matter for arabizi replies, now matters
    # for every "en" reply). Regenerate once, clean. Checked here, LAST — after every other regen
    # step above — rather than right after the first draft: any of those later regen calls (off-
    # task, industry-mismatch, repetition, client checks) can reintroduce the same script leak, and
    # a check placed earlier would miss a leak introduced by a LATER regen (found live).
    if lang == "en" and _ARABIC_RE.search(raw or ""):
        raw = _complete(messages + [
            {"role": "assistant", "content": raw},
            {"role": "system", "content": (
                "Your previous reply contained Arabic-script letters, which is not allowed in an "
                "English reply. Rewrite the SAME message with identical meaning entirely in English, "
                "with zero Arabic-script characters."
            )},
        ])

    answer = _strip_leaked_label(raw)
    answer = _strip_exclamations(answer)
    answer = _strip_semicolons(answer)
    answer = _strip_continuation_words(answer)
    answer = _fix_double_question_join(answer)

    # Deterministic backstop for the habit the prompt can't fully suppress. Strip the reflexive
    # "if you want, I can..." offer always. For trailing questions, three are legitimate to end
    # on: asking the day during the call flow, asking the field when we don't know it yet (handled
    # inside _trim_bot_tail), and the booking choice itself when the model phrases it as a question
    # despite BOOKING_INSTRUCTION asking for a plain statement. That last one matters: found live,
    # stripping it as an "unsanctioned trailing question" cut the reply off mid-choice (e.g. right
    # after "...or." with the actual online-option clause chopped away), and the separate booking-
    # choice guarantee then appended the FULL fallback on top of that dangling fragment — a garbled,
    # visibly duplicated mess. Keeping a question-phrased choice intact is better than destroying it.
    # Everything else — "which service?", "do you have a topic?" — still gets cut, so the reply
    # lands on a statement.
    # services_overview is ALSO a sanctioned trailing question (2026-07-20 fix): SERVICES_OVERVIEW_
    # DIRECTIVE explicitly requires ending with a real invitation to continue ("what are you
    # interested in?") — without this, _trim_bot_tail (built for the UNSANCTIONED-question case)
    # silently deleted that mandated question outright whenever it was well-formed, leaving only a
    # period-ended fragment like "Which one." to survive untouched (it doesn't end in "?", so the
    # strip logic below never even looked at it) — exactly backwards from the intended guarantee.
    allow_question = offered_call or booking_choice_turn or flags["asks_services_overview"]
    keep_field_question = not (profile.get("industry") or signals.get("industry"))
    answer = _trim_bot_tail(answer, allow_question, keep_field_question=keep_field_question)
    # A greeting is only appropriate ONCE, on the conversation's first assistant reply — never
    # again later even if the customer greets a second time.
    if assistant_turns == 0:
        answer = _ensure_greeting(answer, user_message, lang)
    else:
        answer = _strip_leading_greeting(answer)

    # Guarantee brevity in code — the model overshoots the "keep it short" directive no matter what
    # the prompt says. Cap every reply to 2 whole sentences by default (3 for pricing turns, so the
    # "depends on scope + book the call" line is never cut). This ALSO applies to call-flow replies
    # (offered_call=True): the directive promises "ONE short line," but that promise isn't reliably
    # kept either — e.g. the model has appended an entirely unrelated trailing sentence bled in from
    # an unrelated few-shot example (a stray "tell me more about the food you want to photograph" on
    # a booking-link reply with zero food context). The cap trims any such stray sentence for free,
    # regardless of content. The field-pitch turn gets a much higher cap (2026-07-17 decision): it's
    # now deliberately a fuller, multi-category pitch (FIELD_PITCH_DIRECTIVE), so capping it at 2-3
    # sentences the same as everything else would silently undo that directive. flags["wants_field_
    # pitch"] is the same signal FIELD_PITCH_DIRECTIVE itself was selected from — reliable on this
    # exact turn, not something that's only known after the profile is merged.
    if pricing:
        sentence_cap = 3
    elif flags["wants_field_pitch"]:
        # Lowered from 7 (2026-07-17): the broader multi-category pitch was landing noticeably
        # long/paragraph-y — FIELD_PITCH_DIRECTIVE itself now targets ~3 sentences total, so the
        # cap only needs enough headroom for that plus the client-proof sentence, not a full essay.
        sentence_cap = 5
    else:
        sentence_cap = 2
    if "compound_intent" in selected_directive_names:
        # The customer asked multiple separate things at once (see compound_intent_note) — whatever
        # the base cap already allows, add headroom per extra ask, or the cap would silently cut the
        # very content that note just asked for. Generic: works for any combination of dropped asks,
        # not just the specific pairings found live so far. more_detail gets EXTRA headroom (+2, not
        # +1): found live that a real deliverables-based answer (see its own SECONDARY_ASK_BUILDERS
        # entry) genuinely doesn't fit in one sentence the way the other, more compact secondary asks
        # (a one-line category list, a brief field-relevance note) do — capped too tight, it kept
        # landing as vague filler instead of the concrete answer just asked for.
        for _dropped_name in _dropped_exclusive_asks(flags):
            if _dropped_name in SECONDARY_ASK_BUILDERS:
                sentence_cap += 2 if _dropped_name == "more_detail" else 1
    capped = _cap_sentences(answer, sentence_cap)
    # Never let the cap truncate away the booking link — it's the conversion action. If the
    # link was in the reply but fell outside the kept sentences, keep the full reply instead.
    if CALENDLY_LINK in answer and CALENDLY_LINK not in capped:
        capped = answer
    answer = capped

    # The link is the conversion action and the reply should end there — cut anything the model
    # tacked on after it (see _cut_after_link above).
    answer = _cut_after_link(answer, CALENDLY_LINK)


    # Final safety net for the booking-choice guarantee above: that check verified `raw` BEFORE
    # _trim_bot_tail/_cap_sentences ran, but those steps can still delete the choice afterward — e.g.
    # the model phrases the whole choice as one trailing question ("would you prefer onsite at our
    # office, or a free 15-minute online consultation?"), which _trim_bot_tail then strips as an
    # unsanctioned trailing question, since it can't tell that this particular question also carried
    # the only copy of the offer (observed live). So re-check the ACTUAL FINAL text here, after every
    # other transform, and only append the deterministic fallback if it's truly still missing.
    # Excluded on booking_choice_repeat, mode_already_specified, and phone_only_contact_request —
    # same reasoning as the mid-pipeline check above.
    if (booking_choice_turn and not booking_choice_repeat and not mode_already_specified
            and not phone_only_contact_request and not (
        _ONSITE_OFFER_TERMS_RE.search(answer or "") and _ONLINE_OFFER_TERMS_RE.search(answer or "")
    )):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += BOOKING_CHOICE_FALLBACK.get(lang, BOOKING_CHOICE_FALLBACK["en"])

    # Final safety net for the call_flow_mode_pending guarantee above: re-check the ACTUAL FINAL text
    # (the mid-pipeline regen is only best-effort, same reasoning as every other check in this
    # function) and guarantee the mode question reaches the customer regardless of what the model did.
    if call_flow_mode_pending and not _offers_real_choice(answer or ""):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += BOOKING_CHOICE_FALLBACK.get(lang, BOOKING_CHOICE_FALLBACK["en"])

    # NOTE: there used to be a hard final-safety-net here that force-appended BOOKING_CHOICE_FALLBACK
    # whenever call_flow_off_script's mid-pipeline regen still left the reply with no meeting content.
    # Removed (found live): call_flow_off_script's own trigger can't reliably tell "customer answered
    # the call flow but the model went off-script" (should force meeting content back) apart from
    # "customer asked a genuinely different question and CALL_FLOW_DIRECTIVE's own 'changed subject'
    # branch correctly left out any meeting mention" (forcing it back here is flat-out WRONG — e.g.
    # "who is the owner" got a correct, meeting-free answer, and this safety net then force-appended
    # "If you're based in Lebanon, we can schedule..." onto it anyway, directly violating the user's
    # explicit product rule that the meeting must never be re-raised unprompted after the first
    # offer). The mid-pipeline regen above is left in place — it's a soft nudge that explicitly gives
    # the model the "if this is a genuine different question, leave the meeting out entirely" escape
    # hatch, so it can still get this right; it just no longer gets forcibly overridden afterward.

    # Final safety net for call_flow_ignored_clear_mode_mention (see its own comment above): unlike
    # the general call_flow_off_script case just above, this one IS safe to force — the customer's
    # own message unambiguously named onsite or online, so a reply with no call-related content at
    # all is never correct here, there's no "genuine subject change" ambiguity to worry about.
    if call_flow_ignored_clear_mode_mention and not _CALL_OFFER_RE.search(answer or ""):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += BOOKING_CHOICE_FALLBACK.get(lang, BOOKING_CHOICE_FALLBACK["en"])

    # Final safety net for the field_pitch_pricing_compound_owes_meeting guarantee above: the
    # mid-pipeline regen is only a best-effort attempt (found live: it can still land on a vague
    # "let's schedule a quick meeting" instead of actually spelling out the onsite/online choice) —
    # re-check the ACTUAL FINAL text and append the real choice if it's still missing.
    if field_pitch_pricing_compound_owes_meeting and not _offers_real_choice(answer or ""):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += BOOKING_CHOICE_FALLBACK.get(lang, BOOKING_CHOICE_FALLBACK["en"])

    # Final safety net for the field_pitch_pricing_compound_repeat_owes_ack guarantee above: same
    # "mid-pipeline regen is only best-effort" reasoning as the owes_meeting fallback just above —
    # re-check the ACTUAL FINAL text and append PRICING_SCOPE_FALLBACK (the same last-resort text
    # PRICING_REPEAT_NO_MEETING_NOTE's own final-safety-net strip uses) if the acknowledgment is
    # still missing, so the customer's re-asked pricing question is never silently dropped.
    if field_pitch_pricing_compound_repeat_owes_ack and not _has_pricing_ack(answer or ""):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += " " + PRICING_SCOPE_FALLBACK.get(lang, PRICING_SCOPE_FALLBACK["en"])

    # Final safety net for the field_pitch_turn_no_meeting guarantee above: re-check the ACTUAL
    # FINAL text and strip the sentence outright if a meeting mention is still in there.
    if field_pitch_turn_no_meeting and _CALL_OFFER_RE.search(answer or ""):
        sentences = _split_sentences(answer)
        kept = [s for s in sentences if not _CALL_OFFER_RE.search(s)]
        if kept and kept[-1].rstrip().endswith(("?", "؟")):
            kept = kept[:-1]
        answer = " ".join(kept).strip()

    if informational_turn_no_meeting and _CALL_OFFER_RE.search(answer or ""):
        sentences = _split_sentences(answer)
        kept = [s for s in sentences if not _CALL_OFFER_RE.search(s)]
        if kept and kept[-1].rstrip().endswith(("?", "؟")):
            kept = kept[:-1]
        answer = " ".join(kept).strip()

    # Final safety net for the phone_only_contact_request guarantee above: same reasoning and same
    # sentence-level strip as field_pitch_turn_no_meeting's final check just above.
    if phone_only_contact_request and _CALL_OFFER_RE.search(answer or ""):
        sentences = _split_sentences(answer)
        kept = [s for s in sentences if not _CALL_OFFER_RE.search(s)]
        if kept and kept[-1].rstrip().endswith(("?", "؟")):
            kept = kept[:-1]
        answer = " ".join(kept).strip()

    # Final safety net for the pricing_first_explain guarantee above: same reasoning and same
    # sentence-level strip as field_pitch_turn_no_meeting's final check above.
    if pricing_first_explain and _CALL_OFFER_RE.search(answer or ""):
        sentences = _split_sentences(answer)
        kept = [s for s in sentences if not _CALL_OFFER_RE.search(s)]
        if kept and kept[-1].rstrip().endswith(("?", "؟")):
            kept = kept[:-1]
        answer = " ".join(kept).strip()

    # Final safety net for the pricing-pitch-strip guarantee above: a later regen in this pipeline
    # (e.g. the booking-choice guarantee rewriting "the SAME reply") can reintroduce a pitch sentence
    # that the mid-pipeline strip already removed — re-check the ACTUAL FINAL text here, after every
    # other transform, same reasoning as every other final-safety-net check in this function.
    # Same compound_intent exclusion as the mid-pipeline check above.
    if "pricing" in selected_directive_names and "compound_intent" not in selected_directive_names:
        _sentences = _split_sentences(answer)
        _pitch_free = [s for s in _sentences if _is_pricing_or_booking_sentence(s)]
        if _pitch_free and len(_pitch_free) != len(_sentences):
            answer = " ".join(_pitch_free).strip()

    # Final safety net for the bare_ack invitation-strip guarantee above: re-check the ACTUAL FINAL
    # text and strip a leftover "feel free to ask" clause if a later transform reintroduced it.
    if "bare_ack" in selected_directive_names:
        _stripped = _BARE_ACK_INVITATION_RE.sub("", answer or "").strip()
        if _stripped and _stripped != answer:
            answer = _stripped

    # Final safety net for the pricing-repeat guarantee above: re-check the ACTUAL FINAL text and
    # strip only a re-explained full CHOICE (both onsite and online terms in the same sentence) —
    # same reasoning as every other final-safety-net check in this function (a mid-pipeline regen fix
    # can still get reintroduced or missed by a later transform). Narrowed to _offers_real_choice
    # (2026-07-24, same change as the mid-pipeline check above): PRICING_REPEAT_NO_MEETING_NOTE now
    # deliberately wants one brief "pricing and timing are always discussed during the meeting" line,
    # so a bare _CALL_OFFER_RE match would wrongly strip that sentence right back out.
    if pricing_repeat and _offers_real_choice(answer or ""):
        sentences = _split_sentences(answer)
        kept = [
            s for s in sentences
            if not (_ONSITE_OFFER_TERMS_RE.search(s) and _ONLINE_OFFER_TERMS_RE.search(s))
        ]
        # A trailing question ("Which option do you prefer?") almost always belongs to the
        # meeting-choice sentence just removed, not to whatever content sentence is left — left in
        # place it reads as a dangling non-sequitur with no options actually presented. Drop it too.
        if kept and kept[-1].rstrip().endswith(("?", "؟")):
            kept = kept[:-1]
        answer = " ".join(kept).strip()
        if not answer:
            answer = PRICING_SCOPE_FALLBACK.get(lang, PRICING_SCOPE_FALLBACK["en"])

    # Final safety net for the mid-pipeline "depends" re-recitation guarantee above: re-check the
    # ACTUAL FINAL text and strip any sentence still using "depend" — same reasoning as every other
    # final-safety-net check in this function (a mid-pipeline regen fix can still get reintroduced or
    # missed by a later transform). Gated on pricing_topic_already_explained, not pricing_repeat — see
    # the mid-pipeline check's own comment for why the broader flag would wrongly strip a legitimate
    # first-ever explanation of the OTHER dimension.
    if pricing_topic_already_explained and re.search(r"\bdepend\w*\b", answer or "", re.IGNORECASE):
        sentences = _split_sentences(answer)
        kept = [s for s in sentences if not re.search(r"\bdepend\w*\b", s, re.IGNORECASE)]
        answer = " ".join(kept).strip()
        if not answer:
            answer = PRICING_SCOPE_FALLBACK.get(lang, PRICING_SCOPE_FALLBACK["en"])

    # Final safety net for the contact_repeat guarantee above: re-check the ACTUAL FINAL text and
    # replace it outright with the short, differently-worded fallback if the full choice is still in
    # there — same reasoning as every other final-safety-net check in this function.
    if contact_repeat and _offers_real_choice(answer or ""):
        answer = BOOKING_CHOICE_REPEAT_FALLBACK.get(lang, BOOKING_CHOICE_REPEAT_FALLBACK["en"])

    # Final safety net for the "don't assume onsite" guarantee above: re-check the ACTUAL FINAL text
    # and replace it with the same neutral fallback (it already doesn't commit to either side) if the
    # real office address is still in there without the customer ever having stated an onsite/Lebanon
    # preference.
    if ("contact_request" in selected_directive_names and _gave_office_address(answer or "")
            and not _ONSITE_OFFER_TERMS_RE.search(user_message or "")
            and not any(m.get("role") == "user" and _ONSITE_OFFER_TERMS_RE.search(m.get("content", ""))
                        for m in chat_history or [])):
        answer = (BOOKING_CHOICE_REPEAT_FALLBACK.get(lang, BOOKING_CHOICE_REPEAT_FALLBACK["en"])
                  if meeting_ever_mentioned else
                  NEUTRAL_MEETING_CHOICE_FALLBACK.get(lang, NEUTRAL_MEETING_CHOICE_FALLBACK["en"]))

    # Final safety net for the vague-client-filler guarantee above: a later regen (company-info,
    # industry-mismatch, repetition) can reintroduce it after the mid-pipeline check already cleared
    # it — re-check the ACTUAL FINAL text here, after every other transform.
    if _vague_client_filler_already_used(chat_history) and _VAGUE_CLIENT_FILLER_RE.search(answer or ""):
        sentences = _split_sentences(answer)
        answer = " ".join(s for s in sentences if not _VAGUE_CLIENT_FILLER_RE.search(s)).strip()

    # Final safety net for the compound_repeat guarantee above: unlike contact_repeat, a compound-
    # intent reply has OTHER genuine content that must survive (the primary directive's answer plus
    # whichever other secondary asks), so this strips out just the offending sentence(s) rather than
    # replacing the whole answer — same sentence-level approach as the pricing-repeat final check.
    # Deliberately does NOT append a replacement reference: which secondary ask actually triggered
    # this could have been pricing (wants ZERO mention) or contact_request (wants a brief one), and
    # with no way to tell which from here, silence is the safe default either way.
    if compound_repeat and _offers_real_choice(answer or ""):
        sentences = _split_sentences(answer)
        kept = [
            s for s in sentences
            if not (_ONSITE_OFFER_TERMS_RE.search(s) and _ONLINE_OFFER_TERMS_RE.search(s))
        ]
        answer = " ".join(kept).strip()

    # Final safety net for the day/time-handling guarantee: the mid-pipeline regen (above) is only
    # a best-effort attempt — re-check the ACTUAL FINAL text here, after every other transform, and
    # replace it outright with a guaranteed-safe acknowledgment if it's still mishandled. Gated on
    # gave_specific_time (regex OR the LLM's confirms_meeting_daytime signal), not mentions_day_or_time
    # — a day alone must never be forced into an "I'll check the schedule" acknowledgment.
    if offered_call and gave_specific_time and _check_daytime_handling(answer):
        answer = DAYTIME_ACK_FALLBACK.get(lang, DAYTIME_ACK_FALLBACK["en"])

    # Final safety net for the premature-rebooking guarantee above: re-check the ACTUAL FINAL text
    # and replace it outright with a safe, generic closing if it's still mishandled.
    if schedule_check_pending and _check_premature_rebooking(answer):
        answer = PREMATURE_REBOOKING_FALLBACK.get(lang, PREMATURE_REBOOKING_FALLBACK["en"])

    # Final safety net for the office-address guarantee above: the mid-pipeline regen is only a
    # best-effort attempt, and capping/trimming could delete an address that WAS present at that
    # point — re-check the ACTUAL FINAL text here, after every other transform, and only append the
    # deterministic fallback if it's truly still missing. Placed BEFORE the time-ask and company-
    # info final safety nets below, since both check `_gave_office_address(answer)` themselves — if
    # the address is still missing at this point, appending it first lets those two see it correctly.
    # Scoped to call_flow winning (see the mid-pipeline check's own comment for why) — offered_call
    # alone isn't enough, or this fires on turns a completely different directive is actually
    # governing. mode_already_specified included too, same reasoning as the mid-pipeline check above.
    if ((offered_call and "call_flow" in selected_directive_names or mode_already_specified)
            and not _address_already_given(chat_history) and not send_location_instead_of_address
            and _implies_onsite_without_address(answer)):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += OFFICE_ADDRESS_FALLBACK.get(lang, OFFICE_ADDRESS_FALLBACK["en"])

    # Final safety net for the send-location guarantee above: guarantee the WhatsApp number reaches
    # the customer when they asked us to send/share the location, regardless of what the model did.
    if (send_location_instead_of_address and not _mentions_phone(answer)
            and not _phone_already_given(chat_history)):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += f" You can reach us on WhatsApp at {CONTACT_PHONE} to get the exact location."

    # Found live (distinct from the two "name/number missing" guarantees below): the model can also
    # give the fact AND keep the "I'll confirm that for you shortly" hedge in the SAME reply — "who is
    # the owner of blnk" got "I'll confirm that for you shortly. Our owner and co-founder is Dina
    # Rifai." The hedge is only ever appropriate when we genuinely don't have the answer (see
    # SYSTEM_PROMPT's own framing of it); once the real fact is right there in the same breath, it's a
    # flat self-contradiction, not a mid-pipeline forced-fix case (_mentions_owner_name(answer) is
    # already true, so neither guarantee below would otherwise touch it). Checked unconditionally,
    # ahead of both, so it also covers the mid-pipeline regen's own output.
    if _asks_about_owner(user_message) and _mentions_owner_name(answer):
        answer = _strip_confirm_followup_hedge(answer)
    if _asks_about_phone(user_message) and _mentions_phone(answer):
        answer = _strip_confirm_followup_hedge(answer)

    # Final safety net for the owner-name guarantee above: re-check the ACTUAL FINAL text and
    # guarantee the name reaches the customer regardless of what the model did, same reasoning as
    # every other final-safety-net check in this function. Strips a leftover "I'll confirm and follow
    # up" hedge first (found live: appending the name after that hedge without removing it produced a
    # directly self-contradictory reply — "I'll get back to you" immediately followed by the actual
    # answer in the same breath). ALSO strips a trailing "feel free to ask"-style invitation the same
    # way BARE_ACK_DIRECTIVE's own guarantee does (_BARE_ACK_INVITATION_RE) — found live: "who is the
    # owner" got "If you have any other questions in the meantime, feel free to ask." with no name at
    # all, so this safety net appended the name at the very end, reading as a non-sequitur tacked onto
    # unrelated filler ("...feel free to ask. Our owner is Dina Rifai."). Stripping the filler first
    # lets the guaranteed fact read as the actual answer, not an afterthought bolted onto one.
    if (_asks_about_owner(user_message) and not _mentions_owner_name(answer)
            and not _owner_name_already_given(chat_history)):
        answer = _strip_confirm_followup_hedge(answer)
        answer = _BARE_ACK_INVITATION_RE.sub("", answer).rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        # Language-matched append (_mentions_owner_name now recognizes both scripts, but the fallback
        # text itself still needs to pick one) — an "ar" conversation getting an English sentence
        # bolted onto an Arabic reply is the same two-languages-in-one-reply problem this whole check
        # exists to prevent, just introduced by the fallback instead of by the model.
        if lang == "ar":
            answer += f" صاحبة الشركة والمؤسسة المشاركة هي {OWNER_NAME_AR}."
        else:
            answer += f" Our owner and co-founder is {OWNER_NAME}."

    # Final safety net for the phone-number guarantee above: same reasoning and same stripping as the
    # owner-name check just above.
    if (_asks_about_phone(user_message) and not _mentions_phone(answer)
            and not _phone_already_given(chat_history)):
        answer = _strip_confirm_followup_hedge(answer)
        answer = _BARE_ACK_INVITATION_RE.sub("", answer).rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += f" You can reach us on WhatsApp at {CONTACT_PHONE}."

    # Final safety net for the office-address time-ask guarantee above: the mid-pipeline regen is
    # only a best-effort attempt, and _cap_sentences/_trim_bot_tail could delete a time-ask that WAS
    # present at that point — re-check the ACTUAL FINAL text here, after every other transform, and
    # only append the deterministic fallback if it's truly still missing.
    # Scoped to call_flow winning, same reasoning as the office-address final safety net above. Also
    # excluded on is_pure_location_question, same reasoning as the mid-pipeline check above.
    if (offered_call and "call_flow" in selected_directive_names and _gave_office_address(answer)
            and not gave_specific_time and not _check_asks_time(answer)
            and not is_pure_location_question(user_message)):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += TIME_QUESTION_FALLBACK.get(lang, TIME_QUESTION_FALLBACK["en"])

    # Final safety net for the Calendly link (see link_in_first_draft above): a later regen can
    # silently drop it while "keeping everything else" — restore it deterministically rather than
    # ever letting the customer's actual conversion action vanish.
    if link_in_first_draft and CALENDLY_LINK not in answer:
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += f" {BOOKING_INSTRUCTION_LINK_FALLBACK.get(lang, BOOKING_INSTRUCTION_LINK_FALLBACK['en'])}"

    # Final safety net for the human-escalation guarantee above: guarantee the WhatsApp number
    # actually reaches the customer regardless of what the model did with the soft instruction.
    if wants_human_escalation and CONTACT_PHONE not in (answer or ""):
        answer = (answer or "").rstrip()
        if answer and answer[-1] not in ".?؟…":
            answer += "."
        answer += f" You can reach a member of our team directly on WhatsApp at {CONTACT_PHONE}."

    # NOTE: no blanket "force the escalation hedge" backstop here (removed 2026-07-24) — CORRECTION_
    # DIRECTIVE now deliberately has TWO valid outcomes (answer directly when the correction itself
    # supplied the answer, or escalate when it didn't), and there's no reliable deterministic way to
    # tell "the model correctly used their clarification" apart from "the model guessed wrong" from
    # the text alone. Forcing a hard-coded fallback whenever the hedge phrase was absent would
    # overwrite a genuinely correct direct answer with a needless "let me check and get back to you"
    # — exactly the over-escalation the directive was just rewritten to stop doing.

    # Final safety net for the repetition check earlier in the pipeline (see _check_repetition above):
    # that check only regenerates ONCE, well before several LATER regens in this function (client
    # citation, office address, time-ask, company-info, off-task-pitch, arabic-leak) that could each
    # reintroduce a repeated idea or leave the original repeat untouched if the first regen didn't
    # hold — same "any regen after the one that fixed something can silently undo that fix" risk this
    # file's other checks already guard against with a final re-verification, this one just never got
    # it. Found live: a compound field-reveal + pricing turn re-explained Digital Design's Figma UI/UX
    # and front-end/back-end deliverables verbatim, having already given that exact list one turn
    # earlier — the mid-pipeline check missed it. Excluded on booking_choice_repeat, same reason as the
    # mid-pipeline check: that turn's entire job IS an intentional callback to something already said.
    if not booking_choice_repeat:
        repeated_final = _check_repetition(answer, chat_history)
        if repeated_final:
            answer = _complete(messages + [
                {"role": "assistant", "content": answer},
                {"role": "system", "content": repetition_fix_directive(repeated_final)},
            ])

    return answer, primary_directive
