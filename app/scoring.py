from openai import OpenAI
import json
client = OpenAI()
#json for parsing the model's structured output

SCORING_PROMPT = """Given this conversation, output ONLY a JSON object with:
- industry (string or null) — the customer's field/business, only if they stated it
- interested_service (string or null) — the BLNK. service they're asking about (e.g.
  "Brand Identity", "Social Media Design", "Content Creation"), only if clear
- customer_name (string or null) — only if they gave their name
- budget_signal ("low", "medium", "high", or null)
- timeline (string or null)
- score (0-100, likelihood this is a qualified lead)
- notes (short string)
Use null for anything not clearly stated — never guess. No preamble, no markdown, just JSON."""
# The fixed instructions telling the model exactly what fields to extract from the conversation,

def score_lead(chat_history: list):
    # function that analyzes a conversation and returns a lead-quality summary

    convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)
    # Combines the whole conversation history into one readable block of text,
    # labeling each line with who said it ("user: ..." / "assistant: ...")

    resp = client.chat.completions.create(
        model="gpt-5.4-nano",
        # Uses the same cheap model, since this is a simple extraction task, not creative writing

        messages=[
            {"role": "system", "content": SCORING_PROMPT},
            {"role": "user", "content": convo_text}
        ],
        # Sends the scoring instructions as the system message, and the actual conversation as the user message

        temperature=0
        # not creative 
    )
    text = resp.choices[0].message.content.strip().strip("```json").strip("```")
    # Extracts the model's raw text reply, then removes any stray markdown code-fence formatting
    # in case the model wraps its JSON in ```json ... ``` despite being told not to

    try:
        return json.loads(text)
        # Attempts to convert the model's text output into an actual Python dictionary

    except json.JSONDecodeError:
        return {"error": "could not parse", "raw": text}
        # If the model's output wasn't valid JSON for some reason, this safety net
        # returns an error dictionary instead of crashing the whole app