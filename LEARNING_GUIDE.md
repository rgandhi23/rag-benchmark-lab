# RAG Benchmark Lab Learning Guide

This guide is the fastest way to understand the project well enough to change it and explain it
to someone else. Work through it in order. Every section points to the code that implements the
idea.

## 1. The project in one sentence

RAG Benchmark Lab finds evidence related to a question, generates an answer from that evidence,
and measures whether the retrieval step found the documents it was supposed to find.

RAG stands for **retrieval-augmented generation**. The important idea is that the model does not
answer from the question alone. A retrieval system first supplies relevant passages.

## 2. The five-stage mental model

```text
document -> chunks -> vectors -> ranked evidence -> cited answer
                         |
                         +-> evaluation metrics -> saved benchmark run
```

1. **Chunk:** Long documents are divided into smaller overlapping passages.
2. **Embed:** Each passage becomes a numeric vector.
3. **Search:** The question vector is compared with stored passage vectors.
4. **Generate:** The highest-ranked passages become evidence for the answer.
5. **Evaluate:** Expected document IDs are compared with retrieved document IDs.

## 3. Follow one query through the code

Start with `RAGService.query()` in `src/ragbench/service.py`.

```text
question
  -> store.search(question, k)
  -> generator.answer(question, results)
  -> Citation objects
  -> QueryResponse
```

The service is deliberately small. It coordinates other components but does not know how an
embedding is produced or where vectors are stored. That separation makes experiments possible.

The HTTP version begins at `POST /v1/query` in `src/ragbench/api.py`. The command-line version
begins in `src/ragbench/cli.py`.

## 4. Understand the in-memory cosine search

Open `InMemoryVectorStore.search()` in `src/ragbench/store.py`.

For each stored chunk, the implementation calculates:

```text
cosine_similarity = dot(query_vector, chunk_vector)
                    ---------------------------------
                    norm(query_vector) * norm(chunk_vector)
```

The results are sorted from highest score to lowest score, then only the first `k` are returned.
This implementation is intentionally visible and small. It is useful as the baseline against
which a database-backed implementation can be checked.

## 5. Understand persistent Qdrant storage

`QdrantVectorStore` implements the same `add`, `search`, `clear`, `size`, and `close` contract as
the in-memory store.

The differences are operational:

- Vectors and chunk payloads live on disk under `.ragbench/qdrant`.
- Every chunk ID is converted into a deterministic UUID.
- Ingesting the same sample corpus again updates the same points instead of creating duplicates.
- Qdrant performs cosine search and returns the stored chunk payload with each result.

Because both stores satisfy the `VectorStore` protocol, configuration can swap them without
changing `RAGService`.

## 6. Understand the evaluation metrics

The evaluation loop is in `src/ragbench/evaluation.py`.

| Metric | Question it answers |
| --- | --- |
| Hit Rate@K | Did the top K contain at least one relevant document? |
| Precision@K | What fraction of the top K results were relevant? |
| Recall@K | What fraction of all relevant documents appeared in the top K? |
| MRR | How early did the first relevant document appear? |
| Average latency | How long did retrieval take per question? |

The current sample benchmark has one known relevant document per question and asks for three
results. That is why recall can be `1.00` while precision is `0.33`: the correct document was
found, but two additional documents were also returned.

`BenchmarkRunRepository` in `src/ragbench/experiments.py` saves the aggregate metrics to SQLite.
That makes each run comparable with a later retrieval configuration.

## 7. Run each surface

```bash
# Explain the active configuration without printing secrets
ragbench doctor

# Inspect a single answer and its ranked citations
ragbench query "What improved Atlas search latency?" --top-k 3

# Run and persist the sample evaluation
ragbench evaluate --name my-first-run --k 3

# Start the visual dashboard and API
ragbench serve --reload
```

Open `http://127.0.0.1:8000` for the dashboard and `http://127.0.0.1:8000/docs` for FastAPI's
interactive API reference.

## 8. Know where to make changes

| Goal | File to start with |
| --- | --- |
| Change chunk sizes or overlap | `src/ragbench/config.py`, `chunking.py` |
| Add an embedding provider | `src/ragbench/embeddings.py` |
| Add a vector database | `src/ragbench/store.py` |
| Add a retrieval metric | `src/ragbench/evaluation.py`, `models.py` |
| Change answer generation | `src/ragbench/generation.py` |
| Add an API route | `src/ragbench/api.py` |
| Change the dashboard | `src/ragbench/static/index.html` |
| Change CLI commands | `src/ragbench/cli.py` |

## 9. Three exercises that build real understanding

### Exercise A: Change K

Run the benchmark with `--k 1` and then `--k 3`. Explain why precision changes and whether MRR
changes. This teaches how result-set size affects metrics.

### Exercise B: Break and repair retrieval

Change one sample question so it no longer shares useful terms with its document. Run the
benchmark, observe the affected case, then improve the question or embedding method.

### Exercise C: Inspect hybrid search

Run the vector and hybrid configurations at the same value of K. Follow `HybridVectorStore` as it
calculates BM25 scores and combines lexical and vector ranks through reciprocal rank fusion.
Confirm that the Atlas dashboard question moves from rank two to rank one, then explain why the
latency increases.

## 10. The explanation to remember

> The project separates retrieval, generation, and evaluation. I added hybrid BM25 plus vector
> retrieval, reran the same nine labeled questions, and improved MRR from 0.9444 to 1.0000 for a
> measured latency increase of 0.49 milliseconds.
