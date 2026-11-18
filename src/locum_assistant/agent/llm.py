"""The model behind the agent, behind one small interface.

AnthropicLLM calls Claude. ScriptedLLM replays canned turns, so the agent loop can be tested
without a network or an API key. The loop only ever sees an LLMTurn."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass
class LLMTurn:
    stop_reason: str  # end_turn | tool_use | max_tokens | refusal | ...
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: list[Any] = field(default_factory=list)  # raw blocks, appended back verbatim
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn: ...


class AnthropicLLM:
    """Claude over the Messages API.

    - The system prompt carries a cache breakpoint. Tools render before the system prompt, so
      tools + system are one stable prefix, cached across every step of every question.
    - Opus 5.5 always thinks; `effort` is the depth control, set explicitly ("medium").
    - fallbacks="default": if a safety classifier declines, the API re-runs the request on
      Anthropic's recommended fallback model inside the same call instead of failing."""

    def __init__(self, api_key: str | None, model: str, effort: str = "medium",
                 max_tokens: int = 16000) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        from anthropic.types.beta import BetaMessageParam, BetaTextBlockParam, BetaToolParam

        system_blocks: list[BetaTextBlockParam] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            tools=cast(list[BetaToolParam], tools),
            messages=cast(list[BetaMessageParam], messages),
            output_config=cast(Any, {"effort": self.effort}),
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        u = response.usage
        return LLMTurn(
            stop_reason=response.stop_reason or "",
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[ToolCall(b.id, b.name, dict(b.input)) for b in response.content
                        if b.type == "tool_use"],
            content=list(response.content),
            usage=Usage(u.input_tokens, u.output_tokens, u.cache_read_input_tokens or 0,
                        u.cache_creation_input_tokens or 0),
            model=response.model,
        )


class ScriptedLLM:
    """Returns pre-written turns in order. A turn can also be a function of the conversation,
    for tests that need to see what the agent sent back."""

    def __init__(self, turns: list[LLMTurn | Callable[[list[dict]], LLMTurn]]) -> None:
        self.turns = list(turns)
        self.calls: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        self.calls.append([dict(m) for m in messages])
        turn = self.turns.pop(0)
        return turn(messages) if callable(turn) else turn


def text_turn(text: str) -> LLMTurn:
    return LLMTurn(stop_reason="end_turn", text=text, content=[{"type": "text", "text": text}])


def tool_turn(name: str, args: dict, call_id: str = "call_1") -> LLMTurn:
    block = {"type": "tool_use", "id": call_id, "name": name, "input": args}
    return LLMTurn(stop_reason="tool_use", tool_calls=[ToolCall(call_id, name, args)],
                   content=[block])


class ExtractiveLLM:
    """No-key mode. Searches with the user's own words, then answers by quoting the top
    passages with their citations. No generation, so nothing can be invented; it is also
    worse at answering. The API labels these answers as extractive."""

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        last = messages[-1]["content"]
        if isinstance(last, str):  # first step: search for the question as asked
            return tool_turn("search_guidance", {"query": last}, call_id="extractive_1")
        result = next(b["content"] for b in last if b.get("type") == "tool_result")
        passages = [p for p in result.split("\n\n") if p.startswith("[")][:3]
        if not passages:
            return text_turn("I couldn't find anything on that in the guidance I have.")
        lines = ["No language model is configured, so here are the most relevant passages:"]
        for p in passages:
            head, _, body = p.partition("\n")
            n = head[: head.index("]") + 1]
            lines.append(f"- {body[:400].rstrip()}... {n}")
        return text_turn("\n".join(lines))


# --- OpenAI-compatible APIs (OpenRouter, OpenAI, local Ollama) --------------------------------
#
# The agent loop keeps its conversation in Anthropic's shape (content blocks: text, tool_use,
# tool_result). This adapter translates that history to the OpenAI chat format on every call and
# translates the reply back, so the loop never knows which provider it is talking to.

FINISH_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
                  "content_filter": "refusal"}


def _block(b: Any, key: str) -> Any:
    return b.get(key) if isinstance(b, dict) else getattr(b, key, None)


def to_openai_tools(specs: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": s["name"], "description": s["description"], "parameters": s["input_schema"],
        "strict": s.get("strict", False)}} for s in specs]


def to_openai_messages(system: str, messages: list[dict], cache: bool = False) -> list[dict]:
    """Anthropic-shaped history -> OpenAI chat messages. With cache=True the system prompt
    carries an Anthropic cache breakpoint, which OpenRouter passes through to Claude."""
    sys_content: Any = system
    if cache:
        sys_content = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    out: list[dict] = [{"role": "system", "content": sys_content}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "user":  # tool results: one "tool" message per result
            for b in content:
                if _block(b, "type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": _block(b, "tool_use_id"),
                                "content": _block(b, "content")})
        else:  # assistant: text plus tool calls
            text = "".join(_block(b, "text") or "" for b in content if _block(b, "type") == "text")
            calls = [{"id": _block(b, "id"), "type": "function", "function": {
                "name": _block(b, "name"), "arguments": json.dumps(_block(b, "input"))}}
                for b in content if _block(b, "type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
    return out


def from_openai_response(response: Any) -> LLMTurn:
    choice = response.choices[0]
    message = choice.message
    calls = []
    for c in message.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_invalid_arguments": c.function.arguments}  # the tool call will fail cleanly
        calls.append(ToolCall(c.id, c.function.name, args))
    text = message.content or ""
    content: list[Any] = [{"type": "text", "text": text}] if text else []
    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
    stop = "tool_use" if calls else FINISH_REASONS.get(choice.finish_reason or "", "end_turn")
    u = response.usage
    details = getattr(u, "prompt_tokens_details", None) if u else None
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    usage = Usage(input_tokens=(u.prompt_tokens - cached) if u else 0,
                  output_tokens=u.completion_tokens if u else 0, cache_read_tokens=cached)
    return LLMTurn(stop_reason=stop, text=text, tool_calls=calls, content=content, usage=usage,
                   model=response.model or "")


class OpenAICompatLLM:
    """Any OpenAI-compatible chat API. Used with OpenRouter, which can route to Claude
    ("anthropic/claude-opus-5.5") or to other providers' models for comparison."""

    def __init__(self, api_key: str | None, model: str, base_url: str,
                 effort: str = "medium", max_tokens: int = 16000) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=cast(Any, to_openai_messages(system, messages,
                                                  cache=self.model.startswith("anthropic/"))),
            tools=cast(Any, to_openai_tools(tools)),
            extra_body={"reasoning": {"effort": self.effort}},  # OpenRouter's unified setting
        )
        return from_openai_response(response)
