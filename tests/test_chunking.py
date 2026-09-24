from ragbench.chunking import WordChunker
from ragbench.models import DocumentInput


def test_chunker_preserves_overlap_and_stable_ids() -> None:
    document = DocumentInput(document_id="doc", title="Doc", text="one two three four five six")
    chunker = WordChunker(chunk_size=4, overlap=2)

    first = chunker.split(document)
    second = chunker.split(document)

    assert [chunk.text for chunk in first] == ["one two three four", "three four five six"]
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == "doc-faa569a5f3d5"


def test_chunker_preserves_page_number_and_avoids_cross_page_ids() -> None:
    chunker = WordChunker(chunk_size=4, overlap=0)
    page_one = DocumentInput(
        document_id="guide", title="Guide", text="same text on pages", page_number=1
    )
    page_two = page_one.model_copy(update={"page_number": 2})

    first = chunker.split(page_one)[0]
    second = chunker.split(page_two)[0]

    assert first.page_number == 1
    assert second.page_number == 2
    assert first.chunk_id != second.chunk_id
