"""HTTP API.

POST /ask         question in, answer + numbered sources + tool trace out (JSON)
POST /ask/stream  the same, as server-sent events: tool_call, tool_result, answer, done
GET  /health      liveness plus a database check
GET  /metrics     Prometheus metrics: requests, latency, tokens, cost, tool calls"""

import asyncio
import json
import logging
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from locum_assistant.agent.loop import MAX_QUESTION_CHARS, Agent, AgentResult
from locum_assistant.config import Settings, get_settings

log = logging.getLogger("locum_assistant.api")

REQUESTS = Counter("ask_requests_total", "Questions answered", ["stop"])
LATENCY = Histogram("ask_latency_seconds", "Time to answer a question",
                    buckets=(0.5, 1, 2, 4, 8, 15, 30, 60))
TOKENS = Counter("llm_tokens_total", "Tokens used", ["kind"])
COST = Counter("llm_cost_usd_total", "Estimated model cost in US dollars")
TOOL_CALLS = Counter("tool_calls_total", "Tool calls", ["tool", "ok"])

# US dollars per million tokens for claude-opus-5-5: input, output, cache read, cache write.
PRICES = {"claude-opus-5-5": (4.00, 20.00, 0.20, 5.00)}


def estimate_cost(model: str, r: AgentResult) -> float:
    price = PRICES.get(model)
    if not price:
        return 0.0
    u = r.usage
    return (u.input_tokens * price[0] + u.output_tokens * price[1]
            + u.cache_read_tokens * price[2] + u.cache_write_tokens * price[3]) / 1e6


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class RateLimiter:
    """Sliding one-minute window per client. In memory, so per instance: fine for one small
    container, and the honest limit of this design (a shared limit would need Redis)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def build_agent(settings: Settings) -> tuple[Agent, Callable[[], bool]]:
    """Wire the real agent: Postgres store, local embeddings, the chosen model."""
    from locum_assistant.agent.llm import LLM, AnthropicLLM, ExtractiveLLM
    from locum_assistant.agent.tools import default_tools
    from locum_assistant.retrieval.embed import FastEmbedder
    from locum_assistant.retrieval.search import Retriever
    from locum_assistant.retrieval.store import Store

    store = Store(settings.database_url, settings.embedding_dim)
    retriever = Retriever(store, FastEmbedder(settings.embedding_model, settings.embedding_dim),
                          settings.candidates, settings.max_chunks_per_doc)
    llm: LLM
    if settings.llm_provider == "anthropic":
        llm = AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model,
                           settings.anthropic_effort)
    else:
        llm = ExtractiveLLM()

    def db_ok() -> bool:
        try:
            with store.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    agent = Agent(llm, lambda reg: default_tools(retriever, reg), settings.max_agent_steps)
    return agent, db_ok


def create_app(settings: Settings | None = None, agent: Agent | None = None,
               db_ok: Callable[[], bool] | None = None) -> FastAPI:
    settings = settings or get_settings()
    keys = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if agent is None:
            app.state.agent, app.state.db_ok = build_agent(settings)
        else:
            app.state.agent, app.state.db_ok = agent, db_ok or (lambda: True)
        yield

    app = FastAPI(title="Locum Assistant", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_log(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info("request", extra={"request_id": request_id, "method": request.method,
                                   "path": request.url.path, "status": response.status_code,
                                   "ms": int((time.perf_counter() - start) * 1000)})
        return response

    def guard(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        if keys and not any(secrets.compare_digest(x_api_key or "", k) for k in keys):
            raise HTTPException(401, "missing or wrong X-API-Key")
        client = x_api_key or (request.client.host if request.client else "unknown")
        if not limiter.allow(client):
            raise HTTPException(429, "rate limit: try again in a minute")

    def record(result: AgentResult, seconds: float) -> dict[str, Any]:
        REQUESTS.labels(result.stop).inc()
        LATENCY.observe(seconds)
        for kind, n in [("input", result.usage.input_tokens),
                        ("output", result.usage.output_tokens),
                        ("cache_read", result.usage.cache_read_tokens),
                        ("cache_write", result.usage.cache_write_tokens)]:
            TOKENS.labels(kind).inc(n)
        cost = estimate_cost(result.model, result)
        COST.inc(cost)
        for t in result.tool_calls:
            TOOL_CALLS.labels(t.name, str(t.ok).lower()).inc()
        body = asdict(result)
        body["mode"] = settings.llm_provider
        body["latency_ms"] = int(seconds * 1000)
        body["cost_usd"] = round(cost, 5)
        return body

    @app.post("/ask", dependencies=[Depends(guard)])
    def ask(req: AskRequest, request: Request) -> dict[str, Any]:
        start = time.perf_counter()
        result = request.app.state.agent.run(req.question)
        return record(result, time.perf_counter() - start)

    @app.post("/ask/stream", dependencies=[Depends(guard)])
    async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

        def on_event(kind: str, data: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

        def work() -> None:
            start = time.perf_counter()
            try:
                result = request.app.state.agent.run(req.question, on_event)
                on_event("done", record(result, time.perf_counter() - start))
            except Exception:
                log.exception("stream failed")
                on_event("error", {"message": "something went wrong"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        async def events() -> AsyncIterator[str]:
            task = loop.run_in_executor(None, work)
            while (item := await queue.get()) is not None:
                kind, data = item
                yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
            await task

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "database": request.app.state.db_ok(),
                "llm": settings.llm_provider}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    return app


class JsonFormatter(logging.Formatter):
    """One JSON object per line: what Cloud Logging and CloudWatch parse into fields."""

    EXTRA = ("request_id", "method", "path", "status", "ms")

    def format(self, record: logging.LogRecord) -> str:
        entry = {"severity": record.levelname, "logger": record.name,
                 "message": record.getMessage()}
        entry.update({k: getattr(record, k) for k in self.EXTRA if hasattr(record, k)})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(as_json: bool) -> None:
    handler = logging.StreamHandler()
    if as_json:
        handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def serve() -> None:
    import os

    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_json)
    uvicorn.run(create_app(settings), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                log_config=None)
