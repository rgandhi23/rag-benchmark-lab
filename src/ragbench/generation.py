from typing import Protocol

from openai import OpenAI

from ragbench.models import SearchResult


class Generator(Protocol):
    provider_name: str

    def answer(self, question: str, results: list[SearchResult]) -> str: ...


class DemoGenerator:
    """Deterministic extractive answerer for development without API calls."""

    provider_name = "demo"

    def answer(self, question: str, results: list[SearchResult]) -> str:
        del question
        if not results:
            return "I do not have enough indexed evidence to answer that question."
        evidence = " ".join(result.chunk.text for result in results[:2])
        return f"Based on the indexed evidence: {evidence}"


class OpenAIGenerator:
    provider_name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I do not have enough indexed evidence to answer that question."
        context = "\n\n".join(
            f"[{index}] {result.chunk.title}\n{result.chunk.text}"
            for index, result in enumerate(results, start=1)
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Answer only from the supplied evidence. Cite supporting passages with [n]. "
                "If the evidence is insufficient, say so clearly. Do not invent facts."
            ),
            input=f"Question: {question}\n\nEvidence:\n{context}",
        )
        return response.output_text

