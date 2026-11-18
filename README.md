# Locum Assistant

An AI assistant for locum pharmacists in Great Britain. It answers questions about professional
standards and pharmacy law from the official sources, cites the passage behind every claim, and
uses tools for the things a language model gets wrong: pay sums, bank holiday dates and pharmacy
lookups.

I built it because I run [Locumhire](https://locumhire.co.uk), a marketplace for locum
pharmacists, and these are the questions locums ask: can the responsible pharmacist leave the
premises, what goes in the RP record, what changed in 2025 about supervision, what is a shift
worth after the unpaid break.

## What it does

```
question ──> agent loop (Claude, hand-written) ──> answer with [1] [2] citations
                 │
                 ├── search_guidance ──> hybrid search: BM25 in Postgres + pgvector, fused with RRF
                 ├── uk_bank_holidays ─> gov.uk bank holidays API
                 ├── find_pharmacy ────> NHS Organisation Data Service API
                 └── calculate_shift_pay (plain Python: no arithmetic by the model)
```

- **Sources**: 14 GPhC standards and guidance documents, the Responsible Pharmacist Regulations
  2008 and the Human Medicines (Authorisation by Pharmacists and Supervision by Pharmacy
  Technicians) Order 2025. They are downloaded at ingest time, never committed.
- **Retrieval**: sentence-aligned chunks of about 900 characters with overlap; keyword search
  (BM25 computed in Postgres over a `tsvector` GIN index) and semantic search (`BAAI/bge-small-en-v1.5` embeddings run
  locally, pgvector HNSW index), merged with reciprocal rank fusion, at most two passages per
  document.
- **Agent**: a tool-use loop written by hand, with strict tool schemas, a step limit and prompt
  caching on the tools and system prompt. The model sits behind a small interface with two
  providers: the Anthropic Messages API (with server-side refusal fallback) and any
  OpenAI-compatible API such as OpenRouter, through an adapter that translates tool calls and
  results between the two formats.
- **Guardrails**: input length limits; scope and refusal rules in the prompt (no clinical advice
  about a specific patient, Northern Ireland has a different regulator); every `[n]` citation is
  checked against the passages actually retrieved, and invented ones are removed and flagged.
- **API**: FastAPI with `/ask`, `/ask/stream` (server-sent events showing each tool call),
  `/health` and `/metrics` (Prometheus: latency, tokens, estimated cost, tool calls). API-key
  auth, per-client rate limiting, JSON logs with request IDs.
- **No-key mode**: without an API key it runs extractively: the same search, then the top
  passages quoted with citations, clearly labelled.

## Run it

```bash
docker compose up -d db          # Postgres 17 + pgvector
uv sync
uv run locum ingest              # download, chunk, embed, store
uv run locum search "can the RP leave the pharmacy for lunch"
uv run locum eval-retrieval      # keyword vs vector vs hybrid
uv run locum-api                 # http://localhost:8080/docs
```

To use a model, set `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`, or
`LLM_PROVIDER=openrouter` with `OPENROUTER_API_KEY`, in `.env` (see `.env.example`).

## Evaluation

Two question sets, phrased the way locums ask, each question labelled with the document and
passage that answers it: `evals/retrieval.jsonl` (26 questions, used while tuning) and
`evals/retrieval_test.jsonl` (20 questions, written and committed before any tuning and never
tuned against). The score is passage recall@5: is the answering passage in the top five?

| Search | Dev, first version | Dev, with BM25 | Held out, first version | Held out, with BM25 |
|---|---|---|---|---|
| Keyword | 0.46 | 0.85 | 0.55 | 0.80 |
| Semantic | 0.65 | 0.65 | 0.75 | 0.75 |
| Hybrid (RRF) | 0.58 | 0.69 | 0.70 | **0.85** |

What happened: the first keyword search used Postgres's `ts_rank_cd`, which has no inverse
document frequency, so words like "pharmacy" that appear in almost every passage outweighed
the words that matter. Hybrid search scored *below* semantic search alone. Replacing it with BM25
computed in SQL (document frequencies in a materialised view refreshed after each ingest) raised
keyword recall from 0.46 to 0.85 on the dev set and hybrid from 0.70 to 0.85 on the held-out set.

On the dev set keyword-only search now beats hybrid; on the held-out set hybrid wins. With 20 to
26 questions one question is worth four to five points, so I kept equal-weight fusion, the
default chosen before measuring, rather than pick a weighting that only the dev set supports.
CI fails if hybrid dev recall drops below 0.65.

## Tests and CI

`uv run pytest` runs the unit tests: chunking, rank fusion, the agent loop against a scripted
model (tool errors, refusals, the step limit, invented citations), the tools against mocked
HTTP, and the API. GitHub Actions runs ruff, mypy and the tests, then ingests the corpus into a
real pgvector service container and runs the retrieval eval, and builds the Docker image.

## Decisions

- **A hand-written loop, not a framework.** The loop is about 60 lines and every step can be
  tested with a scripted model. A framework would hide the part worth understanding.
- **Hybrid search.** Locums ask with abbreviations ("RP", "SOP") that keyword search matches
  exactly, and with paraphrases ("nip out for lunch") that only semantic search catches.
- **Local embeddings.** No second API key, no per-query cost, and CI can run the retrieval eval
  for free.
- **Honest data limits.** legislation.gov.uk only has the 2008 Regulations as originally made,
  so the 2025 Order is ingested too and the prompt says which one wins. The GPhC's supervision
  FAQ page blocks bots, so it is left out rather than scraped with a fake browser user agent.
  The NHS pharmacy directory covers England and Wales only, and the tool says so.

This is information, not legal advice.
