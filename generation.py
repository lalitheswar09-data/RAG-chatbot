import os
import re

from dotenv import load_dotenv
from openai import OpenAI


# Load .env from the project directory
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. "
        "Make sure your .env file exists in the project root."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = """
You are a retrieval-augmented question answering system.

Rules:
1. Use ONLY the supplied context.
2. Do not use outside knowledge.
3. If the context is insufficient, say that the answer cannot be
   determined from the provided context.
4. Every factual claim must cite its supporting chunk.
5. Use citations exactly like [chunk_id].
6. Never invent a chunk ID.
"""


def build_augmented_prompt(query, chunks):
    context = "\n\n".join(
        f"[{chunk['chunk_id']}]\n"
        f"Source: {chunk['source_doc']}\n\n"
        f"{chunk['text']}"
        for chunk in chunks
    )

    return f"""
USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Answer using only the retrieved context.
"""


def generate_answer(query, chunks):
    user_prompt = build_augmented_prompt(query, chunks)

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )

    answer = response.output_text

    cited_ids = re.findall(r"\[([^\]]+)\]", answer)

    valid_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    used_ids = [
        x for x in cited_ids
        if x in valid_ids
    ]

    used_chunks = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] in used_ids
    ]

    return {
        "answer": answer,
        "used_chunks": used_chunks,
    }