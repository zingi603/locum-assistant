"""The agent loop: ask the model, run the tools it asks for, send the results back, repeat until
it answers. Written by hand (not the SDK's tool runner) so every step is visible and testable.

Guardrails, in order:
1. Input: empty or over-long questions are rejected before any model call.
2. Prompt: scope (GB pharmacy practice and law), cite every claim, refuse clinical advice for a
   specific patient, say so when the documents don't cover something.
3. Loop: a step limit, so a confused model can't call tools forever.
4. Output: every [n] citation must point to a passage that search_guidance actually returned.
   Invented numbers are removed and the answer is marked ungrounded."""

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from locum_assistant.agent.llm import LLM, Usage
from locum_assistant.agent.tools import SourceRegistry, Tool, ToolError

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000

SYSTEM_PROMPT = """You help locum pharmacists and pharmacy teams in Great Britain with questions \
about professional standards, pharmacy law and day-to-day locum work.

How to answer:
- For any question about standards, guidance or law, call search_guidance first, and answer \
only from the passages it returns. After each claim, cite the number of the passage that \
supports it, in square brackets. Never cite a number that search_guidance did not give you.
- If the passages don't answer the question, say so plainly and suggest where to check (the \
GPhC, the employer's SOPs, a defence organisation such as the PDA). Don't fill gaps from memory.
- The Responsible Pharmacist Regulations 2008 in the documents are the original 2008 text. The \
Human Medicines (Authorisation by Pharmacists and Supervision by Pharmacy Technicians) Order \
2025 changed the law on supervision; where the two differ, say the 2025 Order applies and cite it.
- The GPhC regulates England, Scotland and Wales. Northern Ireland has its own regulator (PSNI); \
say so if a question is about Northern Ireland.
- Use calculate_shift_pay for any pay arithmetic, uk_bank_holidays for bank holiday dates and \
find_pharmacy to look up a pharmacy in England or Wales.
- Do not give clinical advice about a specific patient (doses, interactions, whether to supply \
a particular medicine to a particular person). Say that needs the pharmacist's own clinical \
judgement and a clinical reference such as the BNF.
- This is information, not legal advice. Keep answers short and plain: a direct answer first, \
then the detail. Use British English."""


@dataclass
class ToolRecord:
    name: str
    input: dict[str, Any]
    ok: bool
    ms: int


@dataclass
class AgentResult:
    answer: str
    sources: list[dict[str, Any]]
    tool_calls: list[ToolRecord]
    usage: Usage
    steps: int
    stop: str  # answered | refused | step_limit | max_tokens | rejected_input
    grounded: bool
    model: str = ""
    removed_citations: list[int] = field(default_factory=list)


CITATION = re.compile(r"\[(\d+)\]")


def check_citations(answer: str, registry: SourceRegistry) -> tuple[str, list[int], list[int]]:
    """Return the answer with invented citation numbers removed, the numbers that were valid,
    and the numbers that were removed."""
    cited = [int(n) for n in CITATION.findall(answer)]
    valid = sorted({n for n in cited if n in registry.hits})
    bad = sorted({n for n in cited if n not in registry.hits})
    if bad:
        answer = CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in registry.hits else "",
                              answer)
    return answer, valid, bad


class Agent:
    def __init__(self, llm: LLM, tool_factory: Callable[[SourceRegistry], list[Tool]],
                 max_steps: int = 6) -> None:
        self.llm = llm
        self.tool_factory = tool_factory
        self.max_steps = max_steps

    def run(self, question: str,
            on_event: Callable[[str, dict[str, Any]], None] | None = None) -> AgentResult:
        emit = on_event or (lambda kind, data: None)
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            return AgentResult(
                answer=f"Please ask a question of 1 to {MAX_QUESTION_CHARS} characters.",
                sources=[], tool_calls=[], usage=Usage(), steps=0, stop="rejected_input",
                grounded=False)

        registry = SourceRegistry()
        tools = {t.name: t for t in self.tool_factory(registry)}
        specs = [t.spec() for t in tools.values()]
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        usage, records = Usage(), cast(list[ToolRecord], [])
        answer, stop, model, steps = "", "step_limit", "", 0

        for _ in range(self.max_steps):
            steps += 1
            turn = self.llm.complete(SYSTEM_PROMPT, messages, specs)
            usage.add(turn.usage)
            model = turn.model or model
            if turn.stop_reason == "refusal":
                answer, stop = "I can't help with that request.", "refused"
                break
            messages.append({"role": "assistant", "content": turn.content})
            if not turn.tool_calls:
                answer = turn.text
                stop = "max_tokens" if turn.stop_reason == "max_tokens" else "answered"
                break

            results = []
            for call in turn.tool_calls:
                emit("tool_call", {"name": call.name, "input": call.input})
                content, ok = self._run_tool(tools, call.name, call.input, records)
                emit("tool_result", {"name": call.name, "ok": ok})
                results.append({"type": "tool_result", "tool_use_id": call.id,
                                "content": content, "is_error": not ok})
            # all results for one turn go back in a single user message
            messages.append({"role": "user", "content": results})
        else:
            answer = ("I couldn't finish answering within the step limit. Please try a more "
                      "specific question.")

        answer, valid, bad = check_citations(answer, registry)
        if bad:
            log.warning("removed invented citations %s", bad)
        searched = any(r.name == "search_guidance" and r.ok for r in records)
        sources = [{"n": n, "citation": registry.hits[n].citation, "url": registry.hits[n].url,
                    "doc_id": registry.hits[n].doc_id, "text": registry.hits[n].text}
                   for n in valid]
        result = AgentResult(
            answer=answer.strip(), sources=sources, tool_calls=records, usage=usage, steps=steps,
            stop=stop, grounded=stop == "answered" and not bad and (bool(valid) or not searched),
            model=model, removed_citations=bad)
        emit("answer", {"answer": result.answer, "sources": sources})
        return result

    @staticmethod
    def _run_tool(tools: dict[str, Tool], name: str, args: dict[str, Any],
                  records: list[ToolRecord]) -> tuple[str, bool]:
        start = time.perf_counter()
        tool = tools.get(name)
        try:
            if tool is None:
                raise ToolError(f"unknown tool {name}")
            content, ok = tool.run(**args), True
        except ToolError as exc:
            content, ok = f"Error: {exc}", False
        except Exception as exc:  # network failures etc.: tell the model, don't crash the request
            log.exception("tool %s failed", name)
            content = f"Error: the {name} tool failed ({type(exc).__name__}). Try another way."
            ok = False
        records.append(ToolRecord(name, args, ok, int((time.perf_counter() - start) * 1000)))
        log.info("tool %s ok=%s %s", name, ok, json.dumps(args)[:200])
        return content, ok
