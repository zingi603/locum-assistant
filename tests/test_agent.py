from locum_assistant.agent.llm import ExtractiveLLM, LLMTurn, ScriptedLLM, text_turn, tool_turn
from locum_assistant.agent.loop import Agent, check_citations
from locum_assistant.agent.tools import SourceRegistry, Tool, ToolError, search_guidance_tool
from locum_assistant.retrieval.store import Hit

HITS = [Hit(11, "rp-regulations", "RP Regulations", "http://leg", "Absence", None,
            "The maximum period of absence is two hours."),
        Hit(12, "gphc-consent", "Consent guidance", "http://gphc", "", 7,
            "For consent to be valid the person must have capacity.")]


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        self.queries.append(query)
        return HITS[:k]


def make_agent(llm, extra: list[Tool] | None = None, max_steps: int = 6) -> tuple[Agent, FakeRetriever]:  # noqa: E501
    retriever = FakeRetriever()

    def factory(registry: SourceRegistry) -> list[Tool]:
        return [search_guidance_tool(retriever, registry), *(extra or [])]  # type: ignore[arg-type]

    return Agent(llm, factory, max_steps), retriever


def test_search_then_answer_with_valid_citations() -> None:
    llm = ScriptedLLM([tool_turn("search_guidance", {"query": "RP absence"}),
                       text_turn("Up to two hours [1].")])
    agent, retriever = make_agent(llm)
    result = agent.run("How long can the RP be absent?")
    assert result.stop == "answered" and result.grounded
    assert result.answer == "Up to two hours [1]."
    assert result.sources == [{"n": 1, "citation": "RP Regulations, Absence",
                               "url": "http://leg", "doc_id": "rp-regulations",
                               "text": "The maximum period of absence is two hours."}]
    assert retriever.queries == ["RP absence"]
    # the tool result went back to the model, tied to the right call id
    tool_result = llm.calls[1][-1]["content"][0]
    assert tool_result["tool_use_id"] == "call_1" and "[1] RP Regulations" in tool_result["content"]


def test_invented_citations_are_removed_and_flagged() -> None:
    llm = ScriptedLLM([tool_turn("search_guidance", {"query": "x"}),
                       text_turn("Two hours [1], see also [7].")])
    result = make_agent(llm)[0].run("q")
    assert result.answer == "Two hours [1], see also ."
    assert result.removed_citations == [7] and not result.grounded


def test_answer_without_search_or_citations_is_grounded_only_if_no_search() -> None:
    result = make_agent(ScriptedLLM([text_turn("Hello")]))[0].run("hi")
    assert result.grounded and result.sources == []
    llm = ScriptedLLM([tool_turn("search_guidance", {"query": "x"}), text_turn("No idea.")])
    assert not make_agent(llm)[0].run("q").grounded


def test_tool_errors_go_back_to_the_model() -> None:
    def broken(**_: object) -> str:
        raise ToolError("hours must be between 0 and 24")

    tool = Tool("calculate_shift_pay", "d", {"type": "object", "properties": {}}, broken)
    llm = ScriptedLLM([tool_turn("calculate_shift_pay", {"hours": 99}),
                       text_turn("That shift length isn't possible.")])
    result = make_agent(llm, [tool])[0].run("pay for 99 hours?")
    sent = llm.calls[1][-1]["content"][0]
    assert sent["is_error"] and "between 0 and 24" in sent["content"]
    assert result.tool_calls[0].ok is False and result.stop == "answered"


def test_unknown_tool_and_crashing_tool_do_not_crash_the_run() -> None:
    def crash(**_: object) -> str:
        raise ConnectionError("network down")

    tool = Tool("uk_bank_holidays", "d", {"type": "object", "properties": {}}, crash)
    llm = ScriptedLLM([tool_turn("no_such_tool", {}), tool_turn("uk_bank_holidays", {}),
                       text_turn("Sorry, I couldn't check.")])
    result = make_agent(llm, [tool])[0].run("bank holidays?")
    assert [t.ok for t in result.tool_calls] == [False, False]
    assert "ConnectionError" in llm.calls[2][-1]["content"][0]["content"]


def test_step_limit_stops_a_looping_model() -> None:
    llm = ScriptedLLM([tool_turn("search_guidance", {"query": str(i)}, f"c{i}") for i in range(3)])
    result = make_agent(llm, max_steps=3)[0].run("q")
    assert result.stop == "step_limit" and result.steps == 3


def test_refusal_is_handled() -> None:
    result = make_agent(ScriptedLLM([LLMTurn(stop_reason="refusal")]))[0].run("q")
    assert result.stop == "refused" and "can't help" in result.answer


def test_input_guard_rejects_empty_and_huge_questions() -> None:
    llm = ScriptedLLM([])
    agent = make_agent(llm)[0]
    assert agent.run("   ").stop == "rejected_input"
    assert agent.run("x" * 5000).stop == "rejected_input"
    assert llm.calls == []  # no model call was made


def test_events_are_emitted_in_order() -> None:
    llm = ScriptedLLM([tool_turn("search_guidance", {"query": "x"}), text_turn("A [2].")])
    events: list[str] = []
    make_agent(llm)[0].run("q", lambda kind, data: events.append(kind))
    assert events == ["tool_call", "tool_result", "answer"]


def test_extractive_mode_quotes_passages_with_citations() -> None:
    agent, retriever = make_agent(ExtractiveLLM())
    result = agent.run("How long can the RP be absent?")
    assert retriever.queries == ["How long can the RP be absent?"]
    assert "two hours" in result.answer and "[1]" in result.answer and result.grounded


def test_check_citations_keeps_valid_numbers() -> None:
    reg = SourceRegistry()
    reg.number(HITS[0])
    assert check_citations("A [1] B [2]", reg) == ("A [1] B ", [1], [2])
