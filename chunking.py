"""Split standards PDFs into sections, keeping clause numbers together."""
import re
import fitz

import config

# Matches clause numbers like "4.2.1 Title" and annex headings like "Annex B".
_CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\s+(\S.{0,120})$")
_ANNEX_RE = re.compile(r"^\s*(Annex\s+[A-Z])\b(.{0,120})$", re.IGNORECASE)


def _looks_like_heading(line):
    line = line.strip()
    if not line or len(line) > 140:
        return None
    m = _CLAUSE_RE.match(line)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    m = _ANNEX_RE.match(line)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}".strip()
    return None


def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    current = {"section": "(preamble)", "text": "", "page_start": 1, "page_end": 1}

    for page_num, page in enumerate(doc, start=1):
        for line in page.get_text("text").splitlines():
            heading = _looks_like_heading(line)
            if heading:
                if current["text"].strip():
                    sections.append(current)
                current = {"section": heading, "text": "",
                           "page_start": page_num, "page_end": page_num}
            else:
                current["text"] += line + "\n"
                current["page_end"] = page_num

    if current["text"].strip():
        sections.append(current)
    doc.close()
    return sections


def _split_long(text):
    text = text.strip()
    if len(text) <= config.MAX_CHUNK_CHARS:
        return [text]

    parts, start = [], 0
    while start < len(text):
        end = start + config.MAX_CHUNK_CHARS
        window = text[start:end]
        cut = max(window.rfind("\n\n"), window.rfind(". "))
        if cut > config.MAX_CHUNK_CHARS * 0.5:
            end = start + cut + 1
        parts.append(text[start:end].strip())
        start = end - config.CHUNK_OVERLAP_CHARS
    return [p for p in parts if p]


def chunk_pdf(pdf_path, meta):
    chunks = []
    for sec in extract_sections(pdf_path):
        for i, piece in enumerate(_split_long(sec["text"])):
            if len(piece) < config.MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "text": piece,
                "metadata": {
                    "standard": meta["standard"],
                    "version": meta["version"],
                    "source_file": meta["source_file"],
                    "section": sec["section"],
                    "page_start": sec["page_start"],
                    "page_end": sec["page_end"],
                    "part": i,
                },
            })
    return chunks
