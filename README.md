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
                 ├── search_guidance ──> hybrid search: Postgres full-text + pgvector, fused with RRF
                 ├── uk_bank_holidays ─> gov.uk bank holidays API
                 ├── find_pharmacy ────> NHS Organisation Data Service API
                 └── calculate_shift_pay (plain Python: no arithmetic by the model)
```

- **Sources**: 14 GPhC standards and guidance documents, the Responsible Pharmacist Regulations
  2008 and the Human Medicines (Authorisation by Pharmacists and Supervision by Pharmacy
  Technicians) Order 2025. They are downloaded at ingest time, never committed.
- **Retrieval**: sentence-aligned chunks of about 900 characters with overlap; keyword search
  (Postgres `tsvector`, GIN index) and semantic search (`BAAI/bge-small-en-v1.5` embeddings run
  locally, pgvector HNSW index), merged with reciprocal rank fusion, at most two passages per
  document.
- **Agent**: a tool-use loop written by hand on the Anthropic Messages API, with strict tool
  schemas, a step limit, prompt caching on the tools and system prompt, and server-side refusal
  fallback.
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

Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env` to use Claude (see
`.env.example`).

## Evaluation

`evals/retrieval.jsonl` holds questions phrased the way locums ask them, each with the document
and the passage that answers it. `locum eval-retrieval` scores keyword-only, vector-only and
hybrid search side by side (passage recall@5 and MRR), and CI fails if hybrid recall drops.

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
