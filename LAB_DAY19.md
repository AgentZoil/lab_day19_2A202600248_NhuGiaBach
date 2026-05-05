# LAB DAY 19: GraphRAG System with Tech Company Corpus

## METADATA
- **Topic:** GraphRAG (Graph Retrieval-Augmented Generation)
- **Domain:** NLP / Knowledge Graph / RAG
- **Tools:** NetworkX, Neo4j, NodeRAG, OpenAI API

---

## OBJECTIVES
1. Extract entities (nodes) and relations (edges) from raw text
2. Build and query a knowledge graph using NetworkX / Neo4j / NodeRAG
3. Implement a full GraphRAG pipeline: Indexing → Multi-hop Querying
4. Benchmark GraphRAG vs Flat RAG on accuracy

---

## PART 1: RESEARCH — KEY CONCEPTS

### Questions to Answer Before Coding
| # | Question |
|---|----------|
| 1 | **Entity Extraction:** How does an LLM distinguish entities (nodes) from attributes? |
| 2 | **Graph Construction:** Why is deduplication critical when building a knowledge graph? |
| 3 | **Query Answering:** How does BFS graph traversal differ from standard vector search? |

### Tool Overview
| Tool | Type | Best For |
|------|------|----------|
| **NetworkX** | Python library | Offline prototyping, algorithm research |
| **Neo4j** | Graph database | Visual exploration (Bloom/Browser UI) |
| **NodeRAG** | Framework (built on NetworkX) | All-in-one GraphRAG, no DB config needed |

---

## PART 2: ENVIRONMENT SETUP

```bash
# Core libraries
pip install networkx matplotlib neo4j openai pandas

# GraphRAG framework
pip install noderag

# Optional: LangChain pipeline support
pip install langchain langchain-openai
```

> **Note:** For Neo4j, use Neo4j Desktop or Docker to access the visual interface (Bloom/Browser).

---

## PART 3: STEP-BY-STEP IMPLEMENTATION

### Step 1 — Entity & Relation Extraction (Indexing)
Use an LLM to convert raw text into **triples**.

**Input:**
```
"OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015."
```

**Output Triples:**
```
(OpenAI, FOUNDED_BY, Sam Altman)
(OpenAI, FOUNDED_BY, Elon Musk)
(OpenAI, FOUNDED_IN,  2015)
```

---

### Step 2 — Graph Construction
Push triples into one of three backends:

| Option | Tool | When to Use |
|--------|------|-------------|
| **A** | NetworkX | Offline, Jupyter Notebook |
| **B** | Neo4j | Need visual link exploration |
| **C** | NodeRAG | Want an optimized all-in-one solution |

---

### Step 3 — Query Execution (Multi-hop)
Query logic flow:

```
User Question
    → Extract key entity (e.g., "Google")
    → Find matching node in graph
    → Traverse neighbors within 2-hop radius
    → Textualize gathered info into a passage
    → Send passage to LLM for final answer
```

---

### Step 4 — Evaluation: Flat RAG vs GraphRAG
Run **5 complex questions** on both systems:

| System | Backend |
|--------|---------|
| Flat RAG | ChromaDB or FAISS |
| GraphRAG | Knowledge graph (Steps 1–3) |

**Key task:** Document cases where Flat RAG hallucinates but GraphRAG answers correctly.

---

## PART 4: TOOL RECOMMENDATION SUMMARY

| Goal | Recommended Tool | Reason |
|------|-----------------|--------|
| Easiest start | **NodeRAG** | Built-in GraphRAG logic, zero DB config |
| Best visualization | **Neo4j** | GUI shows how knowledge nodes connect |
| Algorithm research | **NetworkX** | Full control over graph math/algorithms |

---

## DELIVERABLES CHECKLIST

- [ ] `code.py` or `notebook.ipynb` — full pipeline implementation
- [ ] Screenshot of the knowledge graph (Neo4j or Matplotlib)
- [ ] Comparison table — **20 benchmark questions**: Flat RAG vs GraphRAG results
- [ ] Short analysis of **cost**: token usage & time to build the graph

---

## IMPLEMENTATION NOTES FOR THIS REPO

- Main entry point: `run.py`
- Notebook demo: `notebooks/graphrag_demo.ipynb`
- Graph backend: NetworkX
- Graph reasoning: rule-based multi-hop query engine over graph context
- Flat baseline: lexical TF-IDF-style retrieval over paragraphs
- Extraction mode: OpenAI-first with rule-based fallback for offline runs
- Current corpus: 16 shorter paragraphs about 8 tech companies
- Current benchmark: 20 questions
- Current results: GraphRAG `100%`, Flat RAG `75%`
