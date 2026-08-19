"""Hybrid search (keyword + vector) with cross-encoder reranking."""
import re
import math
import functools

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config

# Keeps clause numbers like "4.2.1" as a single token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _meta_match(meta, where):
    if not where:
        return True
    return all(meta.get(k) == v for k, v in where.items())


def _rrf(ranked_lists, k):
    # Reciprocal Rank Fusion: merge several ranked id lists into one order.
    scores = {}
    for ids in ranked_lists:
        for rank, _id in enumerate(ids):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL
        )
        self.collection = client.get_collection(
            config.COLLECTION_NAME, embedding_function=embed_fn
        )

        # Load the whole corpus once, for keyword search and id lookup.
        dump = self.collection.get(include=["documents", "metadatas"])
        self.ids = dump["ids"]
        self.docs = dump["documents"]
        self.metas = dump["metadatas"]
        self._by_id = {i: (d, m) for i, d, m in zip(self.ids, self.docs, self.metas)}
        self._bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

    def _dense(self, query, where):
        res = self.collection.query(
            query_texts=[query], n_results=config.DENSE_K, where=where or None
        )
        return res["ids"][0]

    def _sparse(self, query, where):
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if _meta_match(self.metas[i], where):
                out.append(self.ids[i])
            if len(out) >= config.SPARSE_K:
                break
        return out

    def retrieve(self, query, where=None):
        dense = self._dense(query, where)
        sparse = self._sparse(query, where)
        fused = _rrf([dense, sparse], config.RRF_K)[: config.FUSED_K]
        results = []
        for _id in fused:
            if _id in self._by_id:
                doc, meta = self._by_id[_id]
                results.append({"id": _id, "text": doc, "meta": meta})
        return results


class Reranker:
    def __init__(self):
        self._model = CrossEncoder(config.RERANK_MODEL)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(self, query, candidates):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        for c, logit in zip(candidates, self._model.predict(pairs)):
            c["score"] = self._sigmoid(float(logit))
        kept = [c for c in candidates if c["score"] >= config.RERANK_MIN_SCORE]
        kept.sort(key=lambda c: c["score"], reverse=True)
        return kept[: config.RERANK_TOP_N]


@functools.lru_cache(maxsize=1)
def _pipeline():
    # Load models and the BM25 index once, then reuse them.
    return HybridRetriever(), Reranker()


def retrieve_and_rerank(query, where=None):
    retriever, reranker = _pipeline()
    return reranker.rerank(query, retriever.retrieve(query, where))
