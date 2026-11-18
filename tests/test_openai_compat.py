import json
from types import SimpleNamespace as NS

from locum_assistant.agent.llm import from_openai_response, to_openai_messages, to_openai_tools
from locum_assistant.agent.tools import calculate_shift_pay_tool


def test_tools_translate_to_function_specs() -> None:
    spec = to_openai_tools([calculate_shift_pay_tool().spec()])[0]
    assert spec["type"] == "function" and spec["function"]["name"] == "calculate_shift_pay"
    assert spec["function"]["strict"] is True
    assert spec["function"]["parameters"]["additionalProperties"] is False


def test_history_translates_tool_calls_and_results() -> None:
    history = [
        {"role": "user", "content": "RP absence?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "t1", "name": "search_guidance", "input": {"query": "RP"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "[1] two hours",
             "is_error": False}]},
    ]
    out = to_openai_messages("SYS", history, cache=True)
    assert out[0]["role"] == "system"
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out[1] == {"role": "user", "content": "RP absence?"}
    assert out[2]["content"] == "Let me check."
    call = out[2]["tool_calls"][0]
    assert call["id"] == "t1" and json.loads(call["function"]["arguments"]) == {"query": "RP"}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "[1] two hours"}


def test_no_cache_breakpoint_for_non_anthropic_models() -> None:
    assert to_openai_messages("SYS", [], cache=False)[0] == {"role": "system", "content": "SYS"}


def _response(content=None, calls=None, finish="stop", cached=0):  # type: ignore[no-untyped-def]
    usage = NS(prompt_tokens=100, completion_tokens=20,
               prompt_tokens_details=NS(cached_tokens=cached))
    return NS(model="anthropic/claude-opus-5.5", usage=usage, choices=[NS(
        finish_reason=finish, message=NS(content=content, tool_calls=calls))])


def test_reply_with_tool_call_parses_back() -> None:
    call = NS(id="c9", function=NS(name="uk_bank_holidays",
                                   arguments='{"region": "scotland", "year": 2026}'))
    turn = from_openai_response(_response(calls=[call], finish="tool_calls", cached=60))
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].input == {"region": "scotland", "year": 2026}
    assert turn.content[-1]["type"] == "tool_use"
    assert (turn.usage.input_tokens, turn.usage.cache_read_tokens) == (40, 60)


def test_reply_text_and_finish_reasons() -> None:
    assert from_openai_response(_response("Two hours [1].")).text == "Two hours [1]."
    assert from_openai_response(_response("", finish="length")).stop_reason == "max_tokens"
    assert from_openai_response(_response(None, finish="content_filter")).stop_reason == "refusal"


def test_malformed_arguments_do_not_crash() -> None:
    call = NS(id="c1", function=NS(name="calculate_shift_pay", arguments="{not json"))
    turn = from_openai_response(_response(calls=[call], finish="tool_calls"))
    assert turn.tool_calls[0].input == {"_invalid_arguments": "{not json"}
