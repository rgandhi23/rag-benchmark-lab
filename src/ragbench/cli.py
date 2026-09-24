import argparse
import json
from collections.abc import Sequence

from ragbench.api import build_service, load_sample_documents, load_sample_evaluation_set
from ragbench.config import Settings, get_settings
from ragbench.evaluation import evaluate_recall
from ragbench.experiments import BenchmarkRunRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbench",
        description="Inspect and benchmark a retrieval-augmented generation pipeline.",
    )
    commands = parser.add_subparsers(dest="command")

    query = commands.add_parser("query", help="Ask a question and print ranked citations")
    query.add_argument(
        "question",
        nargs="?",
        default="What improved Atlas search latency?",
    )
    query.add_argument("--top-k", type=int, default=None)

    evaluate = commands.add_parser(
        "evaluate", help="Run the sample retrieval benchmark and persist its metrics"
    )
    evaluate.add_argument("--name", default="cli-baseline")
    evaluate.add_argument("--k", type=int, default=3)
    evaluate.add_argument("--min-mrr", type=float, default=None)
    evaluate.add_argument("--min-recall", type=float, default=None)
    evaluate.add_argument("--min-hit-rate", type=float, default=None)
    evaluate.add_argument("--max-latency-ms", type=float, default=None)

    commands.add_parser(
        "doctor", help="Show the active components without revealing credentials"
    )

    serve = commands.add_parser("serve", help="Start the API and visual dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    return parser


def doctor_report(settings: Settings) -> dict[str, object]:
    return {
        "provider": settings.provider,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "api_key_configured": bool(settings.openai_api_key),
        "store_backend": settings.store_backend,
        "retrieval_mode": settings.retrieval_mode,
        "persistent_vectors": settings.store_backend == "qdrant",
        "qdrant_path": settings.qdrant_path,
        "experiment_db_path": settings.experiment_db_path,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
    }


def evaluation_gate_failures(result: object, args: argparse.Namespace) -> list[str]:
    checks = (
        ("MRR", result.mean_reciprocal_rank, args.min_mrr, "minimum"),
        ("recall", result.recall_at_k, args.min_recall, "minimum"),
        ("hit rate", result.hit_rate_at_k, args.min_hit_rate, "minimum"),
        ("latency", result.average_latency_ms, args.max_latency_ms, "maximum"),
    )
    failures = []
    for label, actual, threshold, direction in checks:
        if threshold is None:
            continue
        failed = actual < threshold if direction == "minimum" else actual > threshold
        if failed:
            failures.append(f"{label} {actual} violates {direction} {threshold}")
    return failures


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    settings = get_settings()
    if args.command == "doctor":
        print(json.dumps(doctor_report(settings), indent=2))
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run("ragbench.api:app", host=args.host, port=args.port, reload=args.reload)
        return

    service = build_service(settings)
    try:
        service.ingest(load_sample_documents())
        if args.command == "query":
            print(service.query(args.question, args.top_k).model_dump_json(indent=2))
            return

        result = evaluate_recall(
            service.store,
            load_sample_evaluation_set(),
            k=args.k,
            name=args.name,
        )
        repository = BenchmarkRunRepository(settings.experiment_db_path)
        try:
            result.run_id = repository.save(
                result,
                provider=service.generator.provider_name,
                store_backend=service.store.backend_name,
            )
        finally:
            repository.close()
        print(result.model_dump_json(indent=2))
        failures = evaluation_gate_failures(result, args)
        if failures:
            parser.exit(1, "Evaluation gate failed: " + "; ".join(failures) + "\n")
    finally:
        service.store.close()


if __name__ == "__main__":
    main()
