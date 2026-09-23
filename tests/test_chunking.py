from ragbench.chunking import WordChunker
from ragbench.models import DocumentInput


def test_chunker_preserves_overlap_and_stable_ids() -> None:
    document = DocumentInput(document_id="doc", title="Doc", text="one two three four five six")
    chunker = WordChunker(chunk_size=4, overlap=2)

    first = chunker.split(document)
    second = chunker.split(document)

    assert [chunk.text for chunk in first] == ["one two three four", "three four five six"]
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]

