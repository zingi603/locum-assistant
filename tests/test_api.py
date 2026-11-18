from fastapi.testclient import TestClient

from locum_assistant.agent.llm import ScriptedLLM, text_turn, tool_turn
from locum_assistant.agent.loop import Agent
from locum_assistant.agent.tools import search_guidance_tool
from locum_assistant.api import create_app
from locum_assistant.config import Settings
from tests.test_agent import FakeRetriever


def app_with(turns: list, **settings: object) -> TestClient:
    llm = ScriptedLLM(turns)
    agent = Agent(llm, lambda reg: [search_guidance_tool(FakeRetriever(), reg)])  # type: ignore[arg-type]
    # _env_file=None: tests must not pick up a developer's local .env
    config = Settings(_env_file=None, llm_provider="extractive", **settings)  # type: ignore[call-arg, arg-type]
    return TestClient(create_app(config, agent=agent))


def test_ask_returns_answer_sources_and_trace() -> None:
    with app_with([tool_turn("search_guidance", {"query": "absence"}),
                   text_turn("Two hours [1].")]) as client:
        r = client.post("/ask", json={"question": "How long can the RP be away?"})
    body = r.json()
    assert r.status_code == 200 and body["answer"] == "Two hours [1]."
    assert body["sources"][0]["doc_id"] == "rp-regulations" and body["grounded"]
    assert body["tool_calls"][0]["name"] == "search_guidance"
    assert "x-request-id" in r.headers


def test_ask_validates_input() -> None:
    with app_with([]) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={}).status_code == 422


def test_api_key_required_when_configured() -> None:
    with app_with([text_turn("hi")], api_keys="secret1,secret2") as client:
        assert client.post("/ask", json={"question": "q"}).status_code == 401
        assert client.post("/ask", json={"question": "q"},
                           headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/ask", json={"question": "q"},
                           headers={"X-API-Key": "secret2"}).status_code == 200


def test_rate_limit() -> None:
    with app_with([text_turn("a"), text_turn("b")], rate_limit_per_minute=2) as client:
        codes = [client.post("/ask", json={"question": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_stream_sends_tool_events_then_done() -> None:
    with app_with([tool_turn("search_guidance", {"query": "x"}), text_turn("A [1].")]) as client:
        r = client.post("/ask/stream", json={"question": "q"})
    events = [line.split(": ", 1)[1] for line in r.text.splitlines() if line.startswith("event:")]
    assert events == ["tool_call", "tool_result", "answer", "done"]


def test_health_and_metrics() -> None:
    with app_with([text_turn("a")]) as client:
        assert client.get("/health").json() == {"status": "ok", "database": True,
                                                "llm": "extractive"}
        client.post("/ask", json={"question": "q"})
        assert "ask_requests_total" in client.get("/metrics").text
