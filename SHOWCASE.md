# RAG Benchmark Lab Showcase Kit

Use this file when describing the project on a resume, in an interview, on GitHub, or in a social
post. Keep the measured numbers with the surrounding context: nine synthetic labeled questions,
local Qdrant storage, deterministic demo embeddings, and `K=3`.

## 30-second explanation

> I built an inspectable RAG evaluation platform rather than a black-box chatbot. It supports
> persistent Qdrant vectors, an in-memory cosine-search baseline, OpenAI or offline providers, a
> CLI and FastAPI dashboard, and SQLite experiment history. I created a nine-question retrieval
> benchmark, found a ranking miss, added BM25 plus reciprocal rank fusion, and improved MRR from
> 0.9444 to 1.0000 for a 0.49 millisecond average latency increase.

## Two-minute explanation

1. **Problem:** A plausible answer does not prove that retrieval found the right evidence.
2. **Baseline:** The system chunks documents, embeds passages, stores vectors in Qdrant, retrieves
   Top K evidence, and creates cited answers.
3. **Measurement:** Nine labeled questions produce Hit Rate, Precision, Recall, MRR, and latency.
   SQLite preserves every run.
4. **Finding:** The vector baseline recovered all relevant documents at `K=3`, but one Atlas
   question placed its correct document second. Baseline MRR was `0.9444`.
5. **Experiment:** A transparent BM25 ranker and weighted reciprocal rank fusion combined lexical
   and vector evidence.
6. **Result:** Hybrid retrieval moved the Atlas document to rank one and raised MRR to `1.0000`.
   Average retrieval latency increased from `0.80 ms` to `1.29 ms`.
7. **Engineering:** The same interfaces support memory or Qdrant storage, offline or OpenAI
   providers, API or CLI workflows, and repeatable tests in CI.

## Resume bullets

- Built a modular RAG benchmarking platform with FastAPI, Qdrant, SQLite, Docker, and configurable
  OpenAI adapters, exposing ranked evidence and cited answers through CLI and web interfaces.
- Designed a nine-case retrieval evaluation harness measuring Hit Rate@K, Precision@K, Recall@K,
  MRR, and latency, with persistent experiment history and case-level failure analysis.
- Implemented BM25 plus weighted reciprocal rank fusion, improving MRR from `0.9444` to `1.0000`
  on the labeled sample while quantifying a `0.49 ms` average retrieval-latency tradeoff.
- Added deterministic offline execution, persistent-vector restart tests, Docker packaging, and
  GitHub Actions-ready lint and test automation with 90% measured test coverage.

## GitHub repository description

> Inspectable RAG benchmark lab with Qdrant persistence, hybrid BM25-vector retrieval, cited
> answers, evaluation metrics, experiment history, CLI, FastAPI, Docker, and an interactive
> dashboard.

Suggested topics:

```text
rag llm evaluation information-retrieval qdrant fastapi bm25 mlops openai python
```

## Social post draft

I built RAG Benchmark Lab to answer a question that many AI demos skip: did retrieval actually
find the right evidence?

The project includes persistent Qdrant vectors, transparent in-memory cosine search, a nine-case
evaluation set, SQLite run history, cited answers, a CLI, and a FastAPI dashboard. The first
vector baseline exposed one ranking miss. I added BM25 plus reciprocal rank fusion and improved
MRR from 0.9444 to 1.0000, with a measured average-latency increase of 0.49 ms.

The repository runs without paid API calls in deterministic demo mode and can switch to OpenAI
embeddings and generation through configuration.

## Interview questions and concise answers

**Why is this more than a chatbot wrapper?**

The primary artifact is an evaluation system. Retrieval results, labels, metrics, latency, and
experiment history are first-class parts of the application.

**Why use reciprocal rank fusion?**

Vector and lexical scores have different scales. Reciprocal rank fusion combines their ordering
without pretending the raw scores are directly comparable.

**Why weight BM25 slightly higher?**

The observed failure contained strong project and dashboard terms that the deterministic vector
baseline underweighted. The lexical weight is explicit and should be tuned on a larger held-out
set rather than assumed universal.

**Is MRR 1.0000 enough to claim the system is solved?**

No. The dataset is intentionally small and synthetic. The result proves the experiment harness
and fixes one observed ranking error. A larger held-out corpus is required for a general claim.

**What would you build next?**

A cross-encoder reranker and a larger public corpus, followed by answer faithfulness and citation
correctness metrics. Each addition would be compared with the saved vector and hybrid runs.

