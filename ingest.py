"""Build the search index from the PDFs in the pdfs/ folder.

    python ingest.py            add or update files
    python ingest.py --reset    rebuild the index from scratch
"""
import argparse
import hashlib

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

import config
from chunking import chunk_pdf


def _meta_for(pdf_path):
    declared = config.MANIFEST.get(pdf_path.name, {})
    return {
        "standard": declared.get("standard", pdf_path.stem),
        "version": declared.get("version", "unspecified"),
        "source_file": pdf_path.name,
    }


def _chunk_id(meta, text):
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{meta['source_file']}::{meta['section'][:40]}::{meta['part']}::{h}"


def main(reset):
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )

    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.PDF_DIR.resolve()}")

    total = 0
    for pdf in tqdm(pdfs, desc="PDFs"):
        meta = _meta_for(pdf)
        chunks = chunk_pdf(str(pdf), meta)
        if not chunks:
            print(f"  ! no text extracted from {pdf.name} (scanned PDF? needs OCR)")
            continue

        seen, ids, docs, metas = set(), [], [], []
        for c in chunks:
            cid = _chunk_id(c["metadata"], c["text"])
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            docs.append(c["text"])
            metas.append(c["metadata"])

        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        total += len(ids)
        print(f"  {pdf.name}: {len(ids)} chunks [{meta['standard']} {meta['version']}]")

    print(f"\nDone. {total} chunks indexed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    main(ap.parse_args().reset)
