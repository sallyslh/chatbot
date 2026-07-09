#write the user question in a normalized way 
from openai import OpenAI
client = OpenAI()

NORMALIZE_PROMPT = """Rewrite the user's message into clear, formal Modern Standard Arabic 
or English (keep whichever language dominates), preserving the exact meaning and intent. 
Do not answer the question. Do not add anything. Output ONLY the rewritten sentence."""

def normalize_query(text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": NORMALIZE_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()