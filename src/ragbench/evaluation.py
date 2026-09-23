from time import perf_counter

from ragbench.models import EvaluationCaseResult, EvaluationExample, EvaluationResponse
from ragbench.store import VectorStore


def evaluate_recall(
    store: VectorStore,
    examples: list[EvaluationExample],
    k: int,
    name: str = "ad-hoc",
) -> EvaluationResponse:
    cases: list[EvaluationCaseResult] = []
    for example in examples:
        started = perf_counter()
        results = store.search(example.question, k=k)
        latency_ms = (perf_counter() - started) * 1_000
        retrieved = list(dict.fromkeys(result.chunk.document_id for result in results))
        relevant = set(example.relevant_document_ids)
        matching = relevant.intersection(retrieved)
        first_relevant_rank = next(
            (
                index
                for index, document_id in enumerate(retrieved, start=1)
                if document_id in relevant
            ),
            None,
        )
        cases.append(
            EvaluationCaseResult(
                question=example.question,
                relevant_document_ids=example.relevant_document_ids,
                retrieved_document_ids=retrieved,
                precision_at_k=round(len(matching) / len(retrieved), 4) if retrieved else 0.0,
                recall_at_k=round(len(matching) / len(relevant), 4),
                reciprocal_rank=round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
                latency_ms=round(latency_ms, 2),
            )
        )
    total = len(cases)
    return EvaluationResponse(
        name=name,
        hit_rate_at_k=round(sum(case.reciprocal_rank > 0 for case in cases) / total, 4),
        precision_at_k=round(sum(case.precision_at_k for case in cases) / total, 4),
        recall_at_k=round(sum(case.recall_at_k for case in cases) / total, 4),
        mean_reciprocal_rank=round(sum(case.reciprocal_rank for case in cases) / total, 4),
        average_latency_ms=round(sum(case.latency_ms for case in cases) / total, 2),
        cases=cases,
        k=k,
    )
