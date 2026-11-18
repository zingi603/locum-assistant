"""The agent's tools. Each one has a JSON schema the model sees, with `strict: true` so the API
guarantees arguments match it, plus our own range checks, because a schema can't say
"hours must be under 24".

- search_guidance: hybrid search over GPhC guidance and the legislation (the RAG tool)
- uk_bank_holidays: the gov.uk bank holidays API, for shift planning and pay
- find_pharmacy: the NHS Organisation Data Service API (England and Wales pharmacies)
- calculate_shift_pay: arithmetic, done in code, because language models get sums wrong
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from locum_assistant.retrieval.search import Retriever
from locum_assistant.retrieval.store import Hit

USER_AGENT = "locum-assistant/0.1 (portfolio project; github.com/zingi603)"


class ToolError(Exception):
    """A problem the model can act on (bad input, nothing found). Sent back with is_error."""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    run: Callable[..., str]

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "strict": True,
                "input_schema": {**self.schema, "additionalProperties": False}}


@dataclass
class SourceRegistry:
    """Numbers each retrieved passage once per question ([1], [2], ...), so the answer's
    citations can be checked against what was actually retrieved."""

    by_chunk: dict[int, int] = field(default_factory=dict)
    hits: dict[int, Hit] = field(default_factory=dict)

    def number(self, hit: Hit) -> int:
        if hit.chunk_id not in self.by_chunk:
            n = len(self.by_chunk) + 1
            self.by_chunk[hit.chunk_id] = n
            self.hits[n] = hit
        return self.by_chunk[hit.chunk_id]


def search_guidance_tool(retriever: Retriever, registry: SourceRegistry, k: int = 5) -> Tool:
    def run(query: str) -> str:
        query = query.strip()
        if not query:
            raise ToolError("query is empty")
        hits = retriever.search(query, k=k)
        if not hits:
            return "No matching passages found."
        return "\n\n".join(
            f"[{registry.number(h)}] {h.citation} ({h.url})\n{h.text}" for h in hits)

    return Tool(
        name="search_guidance",
        description=(
            "Search GPhC standards and guidance and UK pharmacy legislation (the Responsible "
            "Pharmacist Regulations 2008 and the 2025 supervision Order). Returns numbered "
            "passages. Call this before answering ANY question about professional standards, "
            "law, the responsible pharmacist, supervision, consent, confidentiality, raising "
            "concerns or other pharmacy practice rules. Call it again with different words if "
            "the first results don't answer the question."),
        schema={"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."}}},
        run=run,
    )


REGIONS = {"england-and-wales", "scotland", "northern-ireland"}


def bank_holidays_tool(client: httpx.Client, cache_seconds: int = 86400) -> Tool:
    cache: dict[str, Any] = {}

    def fetch() -> dict[str, Any]:
        if "data" not in cache or time.time() - cache["at"] > cache_seconds:
            r = client.get("https://www.gov.uk/bank-holidays.json", timeout=15)
            r.raise_for_status()
            cache.update(data=r.json(), at=time.time())
        return cache["data"]

    def run(region: str, year: int) -> str:
        if region not in REGIONS:
            raise ToolError(f"region must be one of {sorted(REGIONS)}")
        events = [e for e in fetch()[region]["events"] if e["date"].startswith(f"{year}-")]
        if not events:
            raise ToolError(f"gov.uk lists no {year} bank holidays for {region} (yet)")
        return "\n".join(f"{e['date']}: {e['title']}" for e in events)

    return Tool(
        name="uk_bank_holidays",
        description=(
            "List official UK bank holidays for a region and year from gov.uk. Call this when "
            "the user asks whether a date is a bank holiday, or about bank holiday shifts, "
            "opening or rates. Regions differ: Scotland and Northern Ireland have extra days."),
        schema={"type": "object", "required": ["region", "year"], "properties": {
            "region": {"type": "string", "enum": sorted(REGIONS)},
            "year": {"type": "integer", "description": "Four-digit year, e.g. 2026"}}},
        run=run,
    )


POSTCODE_PREFIX = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?( ?[0-9][A-Z]{0,2})?$")
ODS_CODE = re.compile(r"^[A-Z0-9]{3,6}$")
ODS = "https://directory.spineservices.nhs.uk/ORD/2-0-0/organisations"


def find_pharmacy_tool(client: httpx.Client) -> Tool:
    def run(postcode: str, ods_code: str) -> str:
        postcode, ods_code = postcode.strip().upper(), ods_code.strip().upper()
        if ods_code:
            if not ODS_CODE.match(ods_code):
                raise ToolError("ods_code looks wrong: pharmacy codes are like FA536")
            r = client.get(f"{ODS}/{ods_code}", timeout=15)
            if r.status_code == 404:
                raise ToolError(f"no organisation with ODS code {ods_code}")
            r.raise_for_status()
            org = r.json()["Organisation"]
            addr = org.get("GeoLoc", {}).get("Location", {})
            return (f"{org['Name']} (ODS {ods_code}), status {org['Status']}, "
                    f"{addr.get('AddrLn1', '')}, {addr.get('Town', '')} {addr.get('PostCode', '')}")
        if not POSTCODE_PREFIX.match(postcode):
            raise ToolError("give a UK postcode or the start of one, e.g. 'M16' or 'M16 8EE'")
        r = client.get(ODS, params={"PostCode": postcode, "PrimaryRoleId": "RO182",
                                    "Status": "Active", "Limit": 10}, timeout=15)
        r.raise_for_status()
        orgs = r.json().get("Organisations", [])
        if not orgs:
            return (f"No active pharmacies found for {postcode}. The NHS ODS directory covers "
                    "England and Wales; Scotland and Northern Ireland pharmacies aren't in it.")
        return "\n".join(f"{o['Name']} (ODS {o['OrgId']}), {o['PostCode']}" for o in orgs)

    return Tool(
        name="find_pharmacy",
        description=(
            "Look up community pharmacies in the NHS Organisation Data Service (England and "
            "Wales only). Give either a postcode or postcode district to list active "
            "pharmacies there, or an ODS code (e.g. FA536) for one pharmacy's details. Use an "
            "empty string for the field you don't use. Call this when the user names a "
            "pharmacy or area and wants its details or nearby pharmacies."),
        schema={"type": "object", "required": ["postcode", "ods_code"], "properties": {
            "postcode": {"type": "string", "description": "Postcode or district, or ''"},
            "ods_code": {"type": "string", "description": "ODS code, or ''"}}},
        run=run,
    )


def calculate_shift_pay_tool() -> Tool:
    def run(hours: float, hourly_rate: float, unpaid_break_minutes: float,
            mileage_miles: float, pence_per_mile: float) -> str:
        if not 0 < hours <= 24:
            raise ToolError("hours must be between 0 and 24")
        if not 0 < hourly_rate <= 500:
            raise ToolError("hourly_rate must be between 0 and 500 pounds")
        if unpaid_break_minutes < 0 or unpaid_break_minutes >= hours * 60:
            raise ToolError("unpaid_break_minutes must be at least 0 and shorter than the shift")
        if mileage_miles < 0 or pence_per_mile < 0:
            raise ToolError("mileage values can't be negative")
        paid_hours = hours - unpaid_break_minutes / 60
        pay = round(paid_hours * hourly_rate, 2)
        travel = round(mileage_miles * pence_per_mile / 100, 2)
        return (f"Paid hours: {paid_hours:.2f}\nShift pay: £{pay:.2f}\n"
                f"Mileage: £{travel:.2f}\nTotal before tax: £{pay + travel:.2f}")

    return Tool(
        name="calculate_shift_pay",
        description=(
            "Work out a locum shift's pay: paid hours after an unpaid break, pay at an hourly "
            "rate, plus mileage. Always use this for pay sums instead of doing arithmetic "
            "yourself. Use 0 for values the user didn't give."),
        schema={"type": "object",
                "required": ["hours", "hourly_rate", "unpaid_break_minutes", "mileage_miles",
                             "pence_per_mile"],
                "properties": {
                    "hours": {"type": "number", "description": "Shift length in hours"},
                    "hourly_rate": {"type": "number", "description": "Pounds per hour"},
                    "unpaid_break_minutes": {"type": "number"},
                    "mileage_miles": {"type": "number"},
                    "pence_per_mile": {"type": "number"}}},
        run=run,
    )


def default_tools(retriever: Retriever, registry: SourceRegistry,
                  http: httpx.Client | None = None) -> list[Tool]:
    http = http or httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    return [search_guidance_tool(retriever, registry), bank_holidays_tool(http),
            find_pharmacy_tool(http), calculate_shift_pay_tool()]
