# draft prompts


def build_rfp_prompt(query, context_chunks):
    context_text = "\n\n".join(context_chunks)

    return f"""
You are an intelligent assistant helping users understand documents.

Use the provided context to answer the question.

IMPORTANT RULES:
- If the answer is partially available, try to infer logically.
- Do NOT say "NOT FOUND" unless absolutely no relevant info exists.
- If abbreviation is present (like RFP), expand it if possible.
- Be helpful and explain clearly.

Context:
{context_text}

Question:
{query}

Answer:
"""

# compatibility with old code
def build_draft_prompt(question: str, context_chunks: list):
    return build_rfp_prompt(question, context_chunks)