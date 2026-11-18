from locum_assistant.ingest.chunk import chunk_document, chunk_section, split_sentences
from locum_assistant.ingest.parse import Section, clean_text, parse_html
from locum_assistant.ingest.sources import SOURCES, Source


def test_clean_text_joins_broken_words_and_whitespace() -> None:
    assert clean_text("pro-\nfessional   duty\n of care") == "professional duty of care"
    assert clean_text("well-\nKnown") == "well- Known"  # capital after hyphen: not a broken word


def test_split_sentences_keeps_abbreviation_free_text_whole() -> None:
    assert split_sentences("First rule. Second rule! (3) Third.") == [
        "First rule.", "Second rule!", "(3) Third."]


def test_chunks_respect_size_and_overlap() -> None:
    text = " ".join(f"Sentence number {i} says something useful." for i in range(60))
    chunks = chunk_section(Section("H", text), size=300, overlap=80)
    assert len(chunks) > 5
    assert all(len(t) <= 300 + 60 for _, t, _ in chunks)  # one sentence may tip it over
    # consecutive chunks share text (the overlap)
    first, second = chunks[0][1], chunks[1][1]
    assert first.split(". ")[-1].rstrip(".") in second


def test_tiny_last_chunk_is_merged() -> None:
    text = "A" * 280 + ". " + "Short end."
    chunks = chunk_section(Section("", text), size=300, overlap=0, min_chars=80)
    assert len(chunks) == 1
    assert chunks[0][1].endswith("Short end.")


def test_overlong_sentence_is_split_on_words() -> None:
    chunks = chunk_section(Section("", "word " * 400), size=200, overlap=0)
    assert all(len(t) <= 200 for _, t, _ in chunks)


def test_chunk_document_numbers_chunks_and_keeps_locators() -> None:
    sections = [Section("Absence", "Rule one applies. " * 40, page=None),
                Section("", "Page text here. " * 40, page=7)]
    chunks = chunk_document(sections, size=300)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert chunks[0].heading == "Absence" and chunks[-1].page == 7


def test_parse_html_groups_text_under_headings() -> None:
    html = b"""<html><body><nav>Menu</nav><main><h1>Title</h1><p>Intro text.</p>
    <h2>Absence</h2><p>The RP may be absent for two hours.</p><ul><li>Item one</li></ul>
    <h3>Record</h3><p>Keep a record.</p></main><footer>Footer</footer></body></html>"""
    doc = parse_html(Source("x", "Page", "http://x", "html"), html)
    assert [(s.heading, s.text) for s in doc.sections] == [
        ("Page", "Intro text."),
        ("Absence", "The RP may be absent for two hours. Item one"),
        ("Record", "Keep a record."),
    ]


def test_source_ids_are_unique() -> None:
    ids = [s.id for s in SOURCES]
    assert len(ids) == len(set(ids))
