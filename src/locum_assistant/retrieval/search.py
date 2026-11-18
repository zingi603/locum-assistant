"""Hybrid retrieval: keyword and vector search run separately, then reciprocal rank fusion (RRF)
merges the two rankings.

Why RRF: the two scores are on different scales (ts_rank_cd vs cosine similarity), so adding them
is meaningless. RRF uses only each hit's rank: score = sum of 1 / (k + rank). A chunk ranked high
by both retrievers wins; k = 60 is the constant from the original RRF paper (Cormack et al. 2009).
"""

from collections import defaultdict
from typing import Literal, Protocol

from locum_assistant.retrieval.embed import Embedder
from locum_assistant.retrieval.store import Hit

Mode = Literal["hybrid", "keyword", "vector"]


class SearchBackend(Protocol):
    def keyword_search(self, query: str, limit: int) -> list[Hit]: ...

    def vector_search(self, embedding: list[float], limit: int) -> list[Hit]: ...


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[int, float] = defaultdict(float)
    by_id: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] += 1.0 / (k + rank)
            by_id.setdefault(hit.chunk_id, hit)
    fused = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    out = []
    for cid in fused:
        hit = by_id[cid]
        hit.score = round(scores[cid], 5)
        out.append(hit)
    return out


def limit_per_document(hits: list[Hit], k: int, max_per_doc: int) -> list[Hit]:
    """Keep the best `k` hits with at most `max_per_doc` from any one document, so a long
    document can't push every other source out of the answer."""
    counts: dict[str, int] = defaultdict(int)
    out = []
    for hit in hits:
        if counts[hit.doc_id] < max_per_doc:
            out.append(hit)
            counts[hit.doc_id] += 1
        if len(out) == k:
            break
    return out


class Retriever:
    def __init__(
        self, backend: SearchBackend, embedder: Embedder, candidates: int = 20,
        max_per_doc: int = 2,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.candidates = candidates
        self.max_per_doc = max_per_doc

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Hit]:
        rankings = []
        if mode in ("hybrid", "keyword"):
            rankings.append(self.backend.keyword_search(query, self.candidates))
        if mode in ("hybrid", "vector"):
            vector = self.embedder.embed_query(query)
            rankings.append(self.backend.vector_search(vector, self.candidates))
        return limit_per_document(reciprocal_rank_fusion(rankings), k, self.max_per_doc)
