"""Turn downloaded files into Documents made of Sections. A section keeps the locator a citation
needs: the heading it sits under, or the PDF page it came from."""

import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import cast

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

from locum_assistant.ingest.sources import Source

DOT_LEADERS = re.compile(r"(?:\.\s?){6,}")


@dataclass
class Section:
    heading: str
    text: str
    page: int | None = None


@dataclass
class Document:
    source: Source
    sections: list[Section] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Join PDF line breaks and collapse whitespace. A hyphen at a line end followed by a
    lower-case letter is a broken word ("pro-\\nfessional"), so the hyphen goes."""
    text = text.replace(" ", " ")
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_pdf(source: Source, data: bytes) -> Document:
    reader = PdfReader(BytesIO(data))
    doc = Document(source)
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if len(DOT_LEADERS.findall(text)) >= 4:  # a contents page: headings and page numbers only
            continue
        if len(text) > 40:  # skip blank and cover-only pages
            doc.sections.append(Section(heading="", text=text, page=number))
    return doc


def _walk_html(root: Tag, source: Source) -> Document:
    """Collect text under each h2-h4 heading. Paragraphs and list items before the first
    heading go under the page title."""
    doc = Document(source)
    heading, buffer = source.title, cast(list[str], [])

    def flush() -> None:
        text = clean_text(" ".join(buffer))
        if text:
            doc.sections.append(Section(heading=heading, text=text))
        buffer.clear()

    for el in root.find_all(["h2", "h3", "h4", "p", "li", "td"]):
        if el.name in ("h2", "h3", "h4"):
            flush()
            heading = clean_text(el.get_text(" "))
        elif not el.find_parent("li") or el.name == "li":
            buffer.append(el.get_text(" "))
    flush()
    return doc


def parse_html(source: Source, data: bytes) -> Document:
    soup = BeautifulSoup(data, "html.parser")
    for junk in soup(["script", "style", "nav", "header", "footer", "form"]):
        junk.decompose()
    root = soup.find("main") or soup.body
    assert isinstance(root, Tag), f"no page body in {source.url}"
    return _walk_html(root, source)


def parse_legislation(source: Source, data: bytes) -> Document:
    """legislation.gov.uk puts the provisions inside #viewLegSnippet."""
    soup = BeautifulSoup(data, "html.parser")
    root = soup.find(id="viewLegSnippet")
    assert isinstance(root, Tag), f"no legislation text in {source.url}"
    return _walk_html(root, source)


PARSERS = {"pdf": parse_pdf, "html": parse_html, "legislation": parse_legislation}


def parse(source: Source, data: bytes) -> Document:
    return PARSERS[source.kind](source, data)
