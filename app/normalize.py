#write the user question in a normalized way 
from openai import OpenAI
client = OpenAI()

NORMALIZE_PROMPT = """Rewrite the user's message into clear English, preserving the exact meaning
and intent. This is ONLY for searching a knowledge base that is written entirely in English — it
has nothing to do with what language the reply to the customer will be in, so ALWAYS output English
regardless of what language the input is in.
The input may be Arabic script, or Arabizi (spoken Arabic transliterated into Latin letters and
digits standing in for Arabic letters, e.g. 3=ع, 7=ح, 2=ء — like "mataam" for "مطعم"/restaurant,
"mahal" for "محل"/shop). DECODE Arabic/Arabizi words into their actual English meaning before
rewriting — never leave a word transliterated or untranslated, and never treat it as an
untranslatable proper noun/industry name (e.g. "the mataam industry" is WRONG; it must become
"the restaurant industry").
Do not answer the question. Do not add anything. Output ONLY the rewritten English sentence."""

def normalize_query(text: str) -> str:
    # Normalization is a retrieval enhancement, not a hard requirement: bge-m3 embeds raw
    # multilingual/arabizi text fine on its own. So if the rewrite call fails (network blip,
    # rate limit, etc.) we fall back to the original text instead of 500ing the whole reply.
    try:
        resp = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": NORMALIZE_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text