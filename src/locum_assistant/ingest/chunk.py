"""Split sections into overlapping, sentence-aligned chunks.

Why these numbers: about 900 characters is roughly 200 tokens, small enough that a retrieved
chunk is about one thing, big enough to hold a full rule with its condition. The 150-character
overlap means a rule that straddles a boundary still appears whole in one of the two chunks."""

import re
from dataclasses import dataclass

from locum_assistant.ingest.parse import Section

SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\"'“•-])")


@dataclass
class Chunk:
    ord: int
    heading: str
    page: int | None
    text: str


def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text) if s.strip()]


def _hard_split(sentence: str, size: int) -> list[str]:
    """A 'sentence' longer than a whole chunk (tables, lists without full stops) is cut on
    word boundaries."""
    words, parts, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_section(
    section: Section, size: int = 900, overlap: int = 150, min_chars: int = 80
) -> list[tuple[str, str, int | None]]:
    sentences: list[str] = []
    for s in split_sentences(section.text):
        sentences.extend(_hard_split(s, size) if len(s) > size else [s])

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current)) + len(sentence) + 1 > size:
            chunks.append(" ".join(current))
            # carry the last sentences (up to `overlap` characters) into the next chunk
            tail: list[str] = []
            for prev in reversed(current):
                if len(" ".join([prev, *tail])) > overlap or len(tail) + 1 == len(current):
                    break
                tail.insert(0, prev)
            current = tail
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))

    # a tiny last chunk is folded into the one before it
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        last = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {last}"
    return [(section.heading, text, section.page) for text in chunks if len(text) >= min_chars
            or len(chunks) == 1]


def chunk_document(sections: list[Section], size: int = 900, overlap: int = 150) -> list[Chunk]:
    out: list[Chunk] = []
    for section in sections:
        for heading, text, page in chunk_section(section, size, overlap):
            out.append(Chunk(ord=len(out), heading=heading, page=page, text=text))
    return out
