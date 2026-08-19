"""Search the standards without an LLM. Prints the most relevant clauses.

    python search.py
    python search.py "minimum password length"
    python search.py --standard "ISO 9001" "password length"
"""
import argparse
import textwrap

from retrieval import retrieve_and_rerank


def _tag(m):
    return f"[{m['standard']}:{m['version']} §{m['section']} p.{m['page_start']}]"


def show(query, where):
    hits = retrieve_and_rerank(query, where)
    if not hits:
        print("\nNo relevant passages found.\n")
        return
    print(f"\nTop {len(hits)} passages for: {query!r}\n" + "=" * 70)
    for rank, h in enumerate(hits, start=1):
        score = h.get("score")
        extra = f"  (score {score:.2f})" if score is not None else ""
        print(f"\n[{rank}] {_tag(h['meta'])}{extra}")
        print("-" * 70)
        print(textwrap.fill(h["text"].strip(), width=100))
    print("\n" + "=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--standard")
    ap.add_argument("--version")
    args = ap.parse_args()

    where = {}
    if args.standard:
        where["standard"] = args.standard
    if args.version:
        where["version"] = args.version
    where = where or None

    if args.query:
        show(" ".join(args.query), where)
    else:
        print("Search mode (no LLM). Ctrl-C to exit.")
        try:
            while True:
                q = input("\nsearch> ").strip()
                if q:
                    show(q, where)
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
