# LAB DAY 19 Report

## Overview

This project implements a compact GraphRAG pipeline over a tech-company corpus using:

- OpenAI-first entity extraction with a rule-based fallback
- a directed knowledge graph built with NetworkX
- graph-based multi-hop question answering
- a retrieval baseline based on lexical TF-IDF-style scoring over paragraphs
- a 20-question benchmark comparing Flat RAG vs GraphRAG

## Research Questions

### 1. How does an LLM distinguish entities from attributes?

An LLM can identify entities by recognizing stable names, organizations, people, and locations as discrete nodes, while treating years, headquarters, and parent companies as attributes or relations attached to those nodes.

In this project, the extractor converts text into triples such as:

- `(OpenAI, FOUNDED_BY, Sam Altman)`
- `(OpenAI, FOUNDED_IN, 2015)`
- `(OpenAI, HEADQUARTERED_IN, San Francisco)`

This is useful because the graph stores facts in a relation-centric form rather than raw prose.

### 2. Why is deduplication critical when building a knowledge graph?

Deduplication prevents the graph from becoming noisy and inflated with repeated nodes or repeated facts. Without it, the same company or person could appear many times, which would:

- increase graph size unnecessarily
- create duplicate edges
- make traversal less reliable
- hurt answer consistency during multi-hop querying

This project deduplicates triples before graph construction, so repeated facts do not create redundant structure.

### 3. How does BFS graph traversal differ from standard vector search?

BFS explores explicit graph connections hop by hop from a starting node. It follows actual edges, so it is good for structured multi-hop reasoning such as:

- `Satya Nadella -> Microsoft -> Bill Gates`

Vector search, by contrast, ranks passages by semantic similarity and does not guarantee that the retrieved text reflects a valid chain of relations.

In short:

- BFS is best for structured reasoning on known entities
- vector search is best for broad semantic retrieval

## Implementation

### GraphRAG Pipeline

1. Read the corpus from `data/raw/tech_companies.txt`
2. Extract triples from each paragraph
3. Build a directed graph in NetworkX
4. Answer queries by locating the entity, collecting local context, and synthesizing the final answer
5. Export graph visualizations and benchmark results

### Flat RAG Baseline

The baseline now uses paragraph retrieval with lexical TF-IDF-style scoring instead of simple keyword picking. This makes the comparison against GraphRAG more realistic and closer to a true retrieval pipeline.

## Results

- Corpus size: 16 shorter paragraphs
- Extracted triples: 47
- Graph size: 52 nodes, 46 edges
- Extraction time: about 0.0000s in offline fallback mode
- Graph build time: sub-millisecond on this corpus
- Flat RAG accuracy: 75.0%
- GraphRAG accuracy: 100.0%

The benchmark contains:

- 18 answerable questions
- 2 out-of-corpus questions

The out-of-corpus questions are important because they test whether the system can abstain instead of hallucinating.

## Cost Analysis

### Token / API Cost

- Offline fallback mode: 0 API tokens used
- OpenAI extraction mode: cost depends on the selected model and token usage

Because the current run fell back to the rule-based extractor, the pipeline completed with zero API cost.

### Runtime Cost

- Graph construction with NetworkX is effectively instantaneous on this small corpus
- Retrieval and BFS traversal are also lightweight
- Visualization is the slowest step relative to graph build, but still small for this dataset

## Outputs

- Graph image: `outputs/graphs/knowledge_graph.png`
- Interactive graph: `outputs/graphs/knowledge_graph_interactive.html`
- Benchmark results: `outputs/results/benchmark_results.json`
- Comparison table: `outputs/results/comparison_table.md`
- Extracted triples: `data/processed/triples.json`

## Strengths

- Clear separation between retrieval, graph construction, and answering
- Strong GraphRAG performance on direct and multi-hop factual questions
- Useful visual artifacts for presentation

## Limitations

- The corpus is small, so the benchmark is only a local evaluation
- The flat baseline is lexical rather than embedding-based
- OpenAI extraction is optional and may fall back to rules if API access fails

## Conclusion

This lab shows that structured graph retrieval is especially effective for relation-heavy questions and multi-hop reasoning. On this harder benchmark, GraphRAG stays perfect while the flat baseline drops once the supporting facts are split across multiple paragraphs. That gap is exactly what we want to demonstrate for a GraphRAG-style lab.
