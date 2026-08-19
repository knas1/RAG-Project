"""Answer questions about the standards using an LLM, with citations.

    python chatbot.py
    python chatbot.py "minimum password length"

Needs OPENAI_API_KEY. Set OPENAI_BASE_URL to use a local server such as Ollama.
"""
import os
import sys
import json
import textwrap

from openai import OpenAI

import config
from retrieval import retrieve_and_rerank

SYSTEM_PROMPT = """Answer questions about technical standards using only the given excerpts.
- If the excerpts do not contain the answer, reply exactly: INSUFFICIENT_CONTEXT.
- Cite standard, version, section and page for every claim, e.g. [ISO 9001:2015 §4.2.1, p.12].
- Quote exact values verbatim; never paraphrase a numeric requirement.
- Include the short supporting excerpt for each claim.
- If versions conflict, say so."""

GRADER_PROMPT = """For each numbered excerpt, decide if it helps answer the question.
Return only a JSON array of the relevant numbers, e.g. [0,2,3]."""


def _client():
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def _tag(m):
    return f"[{m['standard']}:{m['version']} §{m['section']} p.{m['page_start']}]"


def _format_context(hits):
    return "\n\n---\n\n".join(f"{_tag(h['meta'])}\n{h['text']}" for h in hits)


def _grade(client, question, hits):
    listing = "\n\n".join(f"[{i}] {h['text'][:600]}" for i, h in enumerate(hits))
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": GRADER_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{listing}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        keep = set(json.loads(raw[raw.find("["): raw.rfind("]") + 1]))
        return [h for i, h in enumerate(hits) if i in keep] or hits
    except Exception:
        return hits


def answer(question, where=None):
    hits = retrieve_and_rerank(question, where)
    if not hits:
        return {"answer": "No relevant content found in the standards.", "sources": []}

    client = _client()
    if config.USE_LLM_GRADER:
        hits = _grade(client, question, hits)

    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Excerpts:\n\n{_format_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    text = resp.choices[0].message.content.strip()
    if text == "INSUFFICIENT_CONTEXT":
        text = ("The standards do not contain enough information to answer this. "
                "Retrieved sections are listed below.")
    return {"answer": text, "sources": hits}


def _print(result):
    print("\n" + textwrap.fill(result["answer"], width=100) + "\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            m = s["meta"]
            score = s.get("score")
            extra = f", score {score:.2f}" if score is not None else ""
            print(f"  - {m['standard']}:{m['version']} §{m['section']} (p.{m['page_start']}{extra})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _print(answer(" ".join(sys.argv[1:])))
    else:
        print("Standards chatbot. Ctrl-C to exit.")
        try:
            while True:
                q = input("\n> ").strip()
                if q:
                    _print(answer(q))
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
