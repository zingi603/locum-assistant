import re

from locum_assistant.retrieval.embed import HashingEmbedder
from locum_assistant.retrieval.search import Retriever, limit_per_document, reciprocal_rank_fusion
from locum_assistant.retrieval.store import Hit


def hit(cid: int, doc: str = "d", text: str = "") -> Hit:
    return Hit(cid, doc, "Title", "http://x", "", None, text)


def test_rrf_rewards_agreement_between_rankings() -> None:
    keyword = [hit(1), hit(2), hit(3)]
    vector = [hit(3), hit(4), hit(1)]
    fused = [h.chunk_id for h in reciprocal_rank_fusion([keyword, vector])]
    # 1 and 3 appear in both lists, so they beat 2 and 4 which appear once
    assert set(fused[:2]) == {1, 3}
    assert set(fused[2:]) == {2, 4}


def test_rrf_single_ranking_keeps_order() -> None:
    assert [h.chunk_id for h in reciprocal_rank_fusion([[hit(5), hit(2), hit(9)]])] == [5, 2, 9]


def test_limit_per_document() -> None:
    hits = [hit(1, "a"), hit(2, "a"), hit(3, "a"), hit(4, "b"), hit(5, "c")]
    assert [h.chunk_id for h in limit_per_document(hits, k=4, max_per_doc=2)] == [1, 2, 4, 5]


def test_citation_prefers_page_then_heading() -> None:
    assert Hit(1, "d", "Guide", "u", "", 4, "t").citation == "Guide, p. 4"
    assert Hit(1, "d", "Regs", "u", "Absence", None, "t").citation == "Regs, Absence"
    assert Hit(1, "d", "Regs", "u", "", None, "t").citation == "Regs"


class MemoryBackend:
    """In-memory stand-in for the Postgres store: same interface, naive scoring."""

    def __init__(self, texts: dict[int, tuple[str, str]], embedder: HashingEmbedder) -> None:
        self.texts = texts
        self.vectors = {cid: embedder.embed_query(t) for cid, (_, t) in texts.items()}

    def _hit(self, cid: int) -> Hit:
        doc, text = self.texts[cid]
        return hit(cid, doc, text)

    def keyword_search(self, query: str, limit: int) -> list[Hit]:
        words = set(re.findall(r"\w+", query.lower()))
        scored = [(len(words & set(re.findall(r"\w+", t.lower()))), cid)
                  for cid, (_, t) in self.texts.items()]
        return [self._hit(cid) for s, cid in sorted(scored, reverse=True) if s][:limit]

    def vector_search(self, embedding: list[float], limit: int) -> list[Hit]:
        sims = sorted(((sum(a * b for a, b in zip(embedding, v, strict=True)), cid)
                       for cid, v in self.vectors.items()), reverse=True)
        return [self._hit(cid) for _, cid in sims[:limit]]


def test_retriever_finds_the_relevant_chunk() -> None:
    embedder = HashingEmbedder()
    backend = MemoryBackend({
        1: ("rp", "The responsible pharmacist may be absent for two hours."),
        2: ("consent", "Consent is valid when the person has capacity."),
        3: ("online", "Use good judgement on social media."),
    }, embedder)
    retriever = Retriever(backend, embedder, candidates=3)
    for mode in ("hybrid", "keyword", "vector"):
        top = retriever.search("how long can the pharmacist be absent", k=1, mode=mode)[0]
        assert top.doc_id == "rp"
