# Five-Minute Demo Script

Use this when showing the project to a recruiter, engineer, classmate, or interviewer.

## 0:00 - The problem

Say:

> Many RAG demos produce an answer but do not show whether retrieval actually worked. This
> project exposes the ranked evidence, measures retrieval quality, and saves each experiment.

Show the dashboard title and evaluation panel.

## 0:40 - One query

Ask: `What improved Atlas search latency?`

Point out:

- The answer is generated from retrieved evidence.
- Every citation includes a document title, excerpt, and similarity score.
- The ranking can be inspected instead of trusted blindly.

## 1:30 - The architecture

Explain the path:

```text
documents -> chunks -> embeddings -> vector search -> generator -> cited answer
```

Then explain that evaluation compares the retrieved IDs with known relevant document IDs.

## 2:15 - Persistence

Say:

> The default vector store is Qdrant local mode. It keeps vectors and payloads on disk. Chunk IDs
> become deterministic UUIDs, so restarting or ingesting again does not create duplicate points.

Show `.ragbench/qdrant` in the configuration, but do not open or commit its runtime files.

## 3:00 - Evaluation

Run:

```bash
ragbench evaluate --name live-demo --k 3
```

Explain:

- Hit Rate and Recall of `1.00` mean the relevant documents were found.
- Precision of roughly `0.33` is expected because one of three returned documents is relevant.
- MRR shows how early the first relevant document appeared.
- SQLite keeps the run so the next retriever can be compared against it.

## 4:00 - Engineering decisions

Show these files:

- `store.py`: in-memory cosine search beside persistent Qdrant search.
- `service.py`: small orchestration layer using interfaces.
- `evaluation.py`: ranking metrics calculated from explicit cases.
- `tests/`: behavior checks for API, retrieval, persistence, and CLI.

Say:

> I kept an offline deterministic mode so the repository is reproducible without API costs, but
> the same application can switch to OpenAI embeddings and generation through configuration.

## 4:40 - The measured improvement

Close with:

> I added BM25 and reciprocal rank fusion, then reran the same nine questions. MRR improved from
> 0.9444 to 1.0000 because the missed Atlas case moved from rank two to rank one. Average retrieval
> latency increased by 0.49 milliseconds. The next experiment is a cross-encoder reranker.

## Questions you should be ready to answer

**Why use overlapping chunks?**

Important context can cross a chunk boundary. Overlap reduces the chance that a useful sentence
is split away from related text.

**Why keep the in-memory store?**

It is a transparent reference implementation. It makes cosine similarity easy to inspect and
helps validate database-backed behavior.

**Why evaluate retrieval separately from answers?**

An answer generator cannot cite evidence it never received. Retrieval quality is a distinct
failure mode and should be measured directly.

**Why is precision lower than recall?**

The benchmark asks for three results but each sample question has one relevant document. Finding
that document gives full recall, while the other two results reduce precision.

**What would make this production-ready?**

A larger labeled dataset, remote concurrent storage, authentication, request tracing, cost and
tail-latency metrics, hybrid search, reranking, and answer-quality evaluation.
