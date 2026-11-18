"""Retrieval eval: does the right passage come back for a realistic question?

Each case names the documents that hold the answer and a phrase (regex) from the answering
passage. Two scores per search mode:
- doc recall@k: an expected document is in the top k
- passage recall@k: a top-k chunk from an expected document contains the phrase
plus MRR (mean reciprocal rank of the first correct passage), which rewards ranking it first.
Running all three modes side by side shows what hybrid search actually buys.

Two datasets: retrieval.jsonl is the dev set, used while tuning search. retrieval_test.jsonl
is held out: written and committed before any tuning, and never tuned against, so its score
is the honest one."""

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from locum_assistant.retrieval.search import Mode, Retriever

DATASET = Path("evals/retrieval.jsonl")
RESULTS = Path("evals/results")


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score(retriever: Retriever, cases: list[dict], k: int, mode: Mode) -> dict:
    doc_hits = passage_hits = 0
    rr_total = 0.0
    misses = []
    for case in cases:
        hits = retriever.search(case["q"], k=k, mode=mode)
        doc_ok = any(h.doc_id in case["docs"] for h in hits)
        rank = next((i for i, h in enumerate(hits, start=1) if h.doc_id in case["docs"]
                     and re.search(case["phrase"], h.text, re.I)), None)
        doc_hits += doc_ok
        passage_hits += rank is not None
        rr_total += 1 / rank if rank else 0.0
        if rank is None:
            misses.append({"q": case["q"], "got": [h.citation for h in hits[:3]]})
    n = len(cases)
    return {"mode": mode, "k": k, "n": n, "doc_recall": round(doc_hits / n, 3),
            "passage_recall": round(passage_hits / n, 3), "mrr": round(rr_total / n, 3),
            "misses": misses}


def main(retriever: Retriever, k: int = 5, min_recall: float = 0.0,
         dataset: Path = DATASET) -> int:
    cases = load_cases(dataset)
    print(f"{dataset} ({len(cases)} questions)")
    report = [score(retriever, cases, k, mode) for mode in ("keyword", "vector", "hybrid")]
    print(f"{'mode':8s} {'doc@' + str(k):>7s} {'passage@' + str(k):>10s} {'MRR':>6s}")
    for r in report:
        print(f"{r['mode']:8s} {r['doc_recall']:7.3f} {r['passage_recall']:10.3f} {r['mrr']:6.3f}")
    hybrid = report[-1]
    for miss in hybrid["misses"]:
        print(f"  MISS (hybrid): {miss['q']}\n        got: {miss['got']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{dataset.stem}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if hybrid["passage_recall"] < min_recall:
        print(f"FAIL: hybrid passage recall {hybrid['passage_recall']} < {min_recall}",
              file=sys.stderr)
        return 1
    return 0
