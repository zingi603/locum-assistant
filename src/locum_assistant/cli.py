"""Command line: `locum ingest`, `locum search "question"`, `locum eval-retrieval`."""

import argparse
import logging
import sys

from locum_assistant.config import get_settings
from locum_assistant.ingest.sources import BY_ID, SOURCES


def _retriever():  # type: ignore[no-untyped-def]
    from locum_assistant.retrieval.embed import FastEmbedder
    from locum_assistant.retrieval.search import Retriever
    from locum_assistant.retrieval.store import Store

    s = get_settings()
    store = Store(s.database_url, s.embedding_dim)
    embedder = FastEmbedder(s.embedding_model, s.embedding_dim)
    return store, embedder, Retriever(store, embedder, s.candidates, s.max_chunks_per_doc)


def cmd_ingest(args: argparse.Namespace) -> int:
    from locum_assistant.ingest.pipeline import ingest_source

    s = get_settings()
    store, embedder, _ = _retriever()
    store.init_schema()
    sources = [BY_ID[i] for i in args.only] if args.only else SOURCES
    failed = []
    for source in sources:
        try:
            n = ingest_source(source, store, embedder, s.data_dir / "raw", args.refresh)
            print(f"  ok  {source.id:32s} {n:4d} chunks")
        except Exception as exc:  # one broken source shouldn't stop the rest
            failed.append(source.id)
            print(f"  ERR {source.id:32s} {exc}", file=sys.stderr)
    store.refresh_stats()
    return 1 if failed else 0


def cmd_search(args: argparse.Namespace) -> int:
    _, _, retriever = _retriever()
    for i, hit in enumerate(retriever.search(args.query, args.k, args.mode), start=1):
        print(f"{i}. [{hit.score:.4f}] {hit.citation}\n   {hit.text[:240]}...\n")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from locum_assistant.api import build_agent, estimate_cost

    agent, _ = build_agent(get_settings())
    result = agent.run(args.question)
    print(result.answer, "\n")
    for src in result.sources:
        print(f"  [{src['n']}] {src['citation']}")
    tools = ", ".join(f"{t.name}({'ok' if t.ok else 'error'})" for t in result.tool_calls)
    u = result.usage
    print(f"\nstop={result.stop} grounded={result.grounded} steps={result.steps} tools: {tools}")
    print(f"model={result.model} tokens in={u.input_tokens} out={u.output_tokens} "
          f"cache_read={u.cache_read_tokens} cost=${estimate_cost(result.model, result):.4f}")
    return 0


JUDGE_PRICES = {"openai/gpt-5.6-sol": (4.0, 20.0), "anthropic/claude-sonnet-5.5": (2.0, 10.0)}


def cmd_eval_answers(args: argparse.Namespace) -> int:
    from locum_assistant.api import build_agent, estimate_cost
    from locum_assistant.evals.answers import Judge
    from locum_assistant.evals.answers import main as run

    s = get_settings()
    agent, _ = build_agent(s)
    judge = None if args.no_judge else Judge(
        s.openrouter_api_key, args.judge_model, s.openrouter_base_url,
        JUDGE_PRICES.get(args.judge_model, (0.0, 0.0)))
    return run(agent, judge, lambda r: estimate_cost(r.model, r), args.workers, args.only)


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    from pathlib import Path

    from locum_assistant.evals.retrieval import main as run

    _, _, retriever = _retriever()
    return run(retriever, args.k, args.min_recall, Path(args.dataset))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="locum")
    sub = p.add_subparsers(required=True)

    ing = sub.add_parser("ingest", help="download, chunk, embed and store the sources")
    ing.add_argument("--only", nargs="*", choices=list(BY_ID), help="source ids to (re)ingest")
    ing.add_argument("--refresh", action="store_true", help="download again even if cached")
    ing.set_defaults(fn=cmd_ingest)

    se = sub.add_parser("search", help="run a retrieval query")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--mode", choices=["hybrid", "keyword", "vector"], default="hybrid")
    se.set_defaults(fn=cmd_search)

    ak = sub.add_parser("ask", help="ask the agent one question")
    ak.add_argument("question")
    ak.set_defaults(fn=cmd_ask)

    ea = sub.add_parser("eval-answers", help="run evals/answers.jsonl through the agent and judge")
    ea.add_argument("--judge-model", default="openai/gpt-5.6-sol")
    ea.add_argument("--no-judge", action="store_true", help="deterministic checks only (cheaper)")
    ea.add_argument("--workers", type=int, default=4)
    ea.add_argument("--only", nargs="*", help="case ids to run, e.g. g01 r03")
    ea.set_defaults(fn=cmd_eval_answers)

    ev = sub.add_parser("eval-retrieval", help="score retrieval against evals/retrieval.jsonl")
    ev.add_argument("-k", type=int, default=5)
    ev.add_argument("--dataset", default="evals/retrieval.jsonl",
                    help="evals/retrieval.jsonl (dev) or evals/retrieval_test.jsonl (held out)")
    ev.add_argument("--min-recall", type=float, default=0.0,
                    help="exit non-zero if hybrid recall@k is below this (used in CI)")
    ev.set_defaults(fn=cmd_eval_retrieval)

    args = p.parse_args()
    sys.exit(args.fn(args))
