# RAG Evaluation Protocol

## Current automated check

Run:

```bash
python evaluation/evaluate.py evaluation/predictions.example.jsonl
```

The script reports mean, minimum and maximum expected-term coverage across recorded answers. An optional smoke gate can be set with `--min-term-coverage 0.70`.

**Interpretation:** a term appearing in an answer does not prove that the answer is correct, that the term came from a retrieved source, or that a citation supports the claim. The example JSONL file is a format demonstration, not a benchmark result.

## Recommended evaluation layers

1. **Retrieval:** recall@k, precision@k, MRR and nDCG against a manually labelled question-to-source set.
2. **Groundedness:** verify each factual claim against retrieved passages; report citation precision and citation coverage.
3. **Answer quality:** answer relevance, completeness and correctness against reviewed reference answers.
4. **Safety:** prompt-injection resistance, refusal when evidence is missing, cross-document leakage and access-control tests.
5. **Operations:** p50/p95 latency, token usage, cache hit rate, model/API errors and cost per successful query.

## Dataset format

Each JSONL record contains an `answer` and a non-empty `expected_terms` list. Keep test questions separate from prompt and retrieval tuning. Version the evaluation dataset and record the model, embedding model, chunk settings, top-k and prompt version with every measured run.

## Release gate proposal

Do not publish a project quality claim from a single scalar. Before a release, review retrieval metrics, groundedness/citations, safety tests, and operational measurements separately. Establish thresholds only after creating and manually reviewing a representative evaluation set.
