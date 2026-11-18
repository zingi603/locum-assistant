import httpx
import pytest

from locum_assistant.agent.tools import (
    ToolError,
    bank_holidays_tool,
    calculate_shift_pay_tool,
    find_pharmacy_tool,
)

HOLIDAYS = {"england-and-wales": {"events": [{"date": "2026-12-25", "title": "Christmas Day"},
                                             {"date": "2027-01-01", "title": "New Year's Day"}]},
            "scotland": {"events": [{"date": "2026-11-30", "title": "St Andrew's Day"}]},
            "northern-ireland": {"events": []}}


def client_for(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_shift_pay_sums_in_code() -> None:
    run = calculate_shift_pay_tool().run
    out = run(hours=9, hourly_rate=45, unpaid_break_minutes=30, mileage_miles=20,
              pence_per_mile=45)
    assert "Paid hours: 8.50" in out and "Shift pay: £382.50" in out
    assert "Mileage: £9.00" in out and "Total before tax: £391.50" in out


@pytest.mark.parametrize("args", [
    {"hours": 0, "hourly_rate": 40}, {"hours": 25, "hourly_rate": 40},
    {"hours": 8, "hourly_rate": -1}, {"hours": 8, "hourly_rate": 40, "unpaid_break_minutes": 480},
])
def test_shift_pay_rejects_impossible_values(args: dict) -> None:
    full = {"unpaid_break_minutes": 0, "mileage_miles": 0, "pence_per_mile": 0, **args}
    with pytest.raises(ToolError):
        calculate_shift_pay_tool().run(**full)


def test_bank_holidays_filters_by_region_and_year_and_caches() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200, json=HOLIDAYS)

    tool = bank_holidays_tool(client_for(handler))
    assert tool.run(region="england-and-wales", year=2026) == "2026-12-25: Christmas Day"
    assert "St Andrew" in tool.run(region="scotland", year=2026)
    assert len(calls) == 1  # second call served from cache
    with pytest.raises(ToolError):
        tool.run(region="wales", year=2026)
    with pytest.raises(ToolError):
        tool.run(region="northern-ireland", year=2026)


def test_find_pharmacy_by_postcode_and_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/FA536"):
            return httpx.Response(200, json={"Organisation": {
                "Name": "EVEREST PHARMACY", "Status": "Active", "GeoLoc": {"Location": {
                    "AddrLn1": "117B WITHINGTON ROAD", "Town": "MANCHESTER",
                    "PostCode": "M16 8EE"}}}})
        if request.url.path.endswith("/ZZZ99"):
            return httpx.Response(404)
        assert request.url.params["PrimaryRoleId"] == "RO182"
        return httpx.Response(200, json={"Organisations": [
            {"Name": "EVEREST PHARMACY", "OrgId": "FA536", "PostCode": "M16 8EE"}]})

    tool = find_pharmacy_tool(client_for(handler))
    assert tool.run(postcode="m16", ods_code="") == "EVEREST PHARMACY (ODS FA536), M16 8EE"
    assert "WITHINGTON ROAD" in tool.run(postcode="", ods_code="fa536")
    with pytest.raises(ToolError):
        tool.run(postcode="", ods_code="ZZZ99")
    with pytest.raises(ToolError):
        tool.run(postcode="not a postcode", ods_code="")


def test_find_pharmacy_explains_scotland_gap() -> None:
    tool = find_pharmacy_tool(client_for(lambda r: httpx.Response(200, json={})))
    assert "England and Wales" in tool.run(postcode="G1", ods_code="")


def test_tool_specs_are_strict() -> None:
    spec = calculate_shift_pay_tool().spec()
    assert spec["strict"] is True and spec["input_schema"]["additionalProperties"] is False
    assert set(spec["input_schema"]["required"]) == set(spec["input_schema"]["properties"])
