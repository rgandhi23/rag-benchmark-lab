import argparse

from ragbench.api import build_service, load_sample_documents
from ragbench.config import get_settings
from ragbench.evaluation import evaluate_recall
from ragbench.models import EvaluationExample


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG Benchmark Lab baseline")
    parser.add_argument("question", nargs="?", default="What improved Atlas search latency?")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    service = build_service(get_settings())
    service.ingest(load_sample_documents())
    if args.evaluate:
        examples = [
            EvaluationExample(
                question="What improved Atlas search latency?", relevant_document_ids=["atlas"],
            ),
            EvaluationExample(
                question="How does Beacon monitor models?", relevant_document_ids=["beacon"],
            ),
            EvaluationExample(
                question="What security controls does Harbor use?",
                relevant_document_ids=["harbor"],
            ),
        ]
        print(
            evaluate_recall(
                service.store, examples, k=3, name="sample-baseline"
            ).model_dump_json(indent=2)
        )
        return
    print(service.query(args.question).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
