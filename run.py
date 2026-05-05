"""Main script - LAB DAY 19: GraphRAG with NetworkX"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.append("src")

from entity_extraction import EntityExtractor
from flat_rag import FlatRAGRetriever
from graph_builder import GraphBuilder
from query_engine import QueryEngine


PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "tech_companies.txt"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "triples.json"
GRAPH_PATH = PROJECT_ROOT / "outputs" / "graphs" / "knowledge_graph.png"
GRAPH_HTML_PATH = PROJECT_ROOT / "outputs" / "graphs" / "knowledge_graph_interactive.html"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "benchmark_results.json"
COMPARISON_PATH = PROJECT_ROOT / "outputs" / "results" / "comparison_table.md"


benchmark_questions = [
    {
        "query": "Who founded OpenAI?",
        "type": "Founder",
        "expected": "Sam Altman, Elon Musk, Greg Brockman",
    },
    {
        "query": "What year was OpenAI founded?",
        "type": "Year",
        "expected": "2015",
    },
    {
        "query": "Which city is OpenAI headquartered in?",
        "type": "Location",
        "expected": "San Francisco",
    },
    {
        "query": "Name the founders of Google.",
        "type": "Founder",
        "expected": "Larry Page, Sergey Brin",
    },
    {
        "query": "What is the parent company of the company founded by Larry Page?",
        "type": "Parent",
        "expected": "Alphabet Inc.",
    },
    {
        "query": "Which company does Satya Nadella lead?",
        "type": "Multi-hop CEO",
        "expected": "Microsoft",
    },
    {
        "query": "Who founded the company led by Satya Nadella?",
        "type": "Multi-hop Founder",
        "expected": "Bill Gates, Paul Allen",
    },
    {
        "query": "Who founded the company led by Andy Jassy?",
        "type": "Multi-hop Founder",
        "expected": "Jeff Bezos",
    },
    {
        "query": "What is the founding year of the company founded by Jeff Bezos?",
        "type": "Multi-hop Year",
        "expected": "1994",
    },
    {
        "query": "Which company was founded by Mark Zuckerberg?",
        "type": "Founder",
        "expected": "Meta",
    },
    {
        "query": "Who is the CEO of Microsoft?",
        "type": "CEO",
        "expected": "Satya Nadella",
    },
    {
        "query": "Which city is Microsoft headquartered in?",
        "type": "Location",
        "expected": "Redmond",
    },
    {
        "query": "Who founded Tesla?",
        "type": "Founder",
        "expected": "Martin Eberhard, Marc Tarpenning",
    },
    {
        "query": "When was Tesla founded?",
        "type": "Year",
        "expected": "2003",
    },
    {
        "query": "Which city is Tesla headquartered in?",
        "type": "Location",
        "expected": "Austin",
    },
    {
        "query": "Who is the CEO of Amazon?",
        "type": "CEO",
        "expected": "Andy Jassy",
    },
    {
        "query": "Who founded Apple?",
        "type": "Founder",
        "expected": "Steve Jobs, Steve Wozniak, Ronald Wayne",
    },
    {
        "query": "Who founded the company whose CEO is Satya Nadella?",
        "type": "Multi-hop Founder",
        "expected": "Bill Gates, Paul Allen",
    },
    {
        "query": "Who is the CFO of Anthropic?",
        "type": "Unknown",
        "expected": "Không tìm thấy",
    },
    {
        "query": "Where is SpaceX headquartered?",
        "type": "Unknown",
        "expected": "Không tìm thấy",
    },
]


STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "is",
    "was",
    "what",
    "who",
    "where",
    "when",
    "to",
    "by",
    "in",
    "and",
    "company",
    "current",
    "whose",
    "is",
}


COMPANY_NAMES = [
    "OpenAI",
    "Google",
    "Microsoft",
    "Meta",
    "Amazon",
    "Apple",
    "Tesla",
    "NVIDIA",
]


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if token not in STOPWORDS
    ]


def load_paragraphs() -> List[str]:
    text = CORPUS_PATH.read_text(encoding="utf-8")
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def select_best_paragraph(query: str, paragraphs: List[str]) -> str:
    query_tokens = tokenize(query)
    query_lower = query.lower()
    best_score = -1
    best_paragraph = paragraphs[0]

    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        score = sum(1 for token in query_tokens if token in paragraph_lower)
        score += sum(2 for company in COMPANY_NAMES if company.lower() in query_lower and company.lower() in paragraph_lower)
        if score > best_score:
            best_score = score
            best_paragraph = paragraph

    return best_paragraph


def extract_company_from_query(query: str) -> str:
    for company in COMPANY_NAMES:
        if company.lower() in query.lower():
            return company
    return ""


def flat_rag_answer(query: str, paragraphs: List[str]) -> str:
    paragraph = select_best_paragraph(query, paragraphs)
    query_lower = query.lower()
    company = extract_company_from_query(query)

    if "who founded" in query_lower or "founder" in query_lower:
        match = re.search(r"by ([^.]+?)(?:\s+in\s+|\.)", paragraph, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    if "who is the ceo" in query_lower or "ceo of" in query_lower:
        if company:
            match = re.search(
                rf"([A-Z][A-Za-z.\- ]+?) is the current CEO(?: of {re.escape(company)})?",
                paragraph,
            )
            if match:
                return match.group(1).strip()
            match = re.search(r"([A-Z][A-Za-z.\- ]+?) is the current CEO", paragraph)
            if match:
                return match.group(1).strip()

    if "headquarter" in query_lower or "based in" in query_lower or "located" in query_lower:
        match = re.search(r"headquartered in ([^.]+)", paragraph, re.IGNORECASE)
        if not match:
            match = re.search(r"based in ([^.]+)", paragraph, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    if "when was" in query_lower or "founded in" in query_lower or "year" in query_lower:
        match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", paragraph)
        if match:
            return match.group(1)

    if "parent company" in query_lower:
        match = re.search(r"parent company is ([^.]+)", paragraph, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return paragraph


def is_expected_in_answer(expected: str, answer: str) -> bool:
    expected_norm = normalize_text(expected)
    answer_norm = normalize_text(answer)
    return expected_norm in answer_norm


def save_comparison_table(rows: List[Dict[str, str]], output_path: Path) -> None:
    lines = [
        "| # | Question | Expected | Flat RAG | GraphRAG |",
        "|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | {row['query']} | {row['expected']} | {row['flat_answer']} | {row['graph_answer']} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 60)
    print("LAB DAY 19 - GRAPHRAG WITH NETWORKX")
    print("Student: Nhu Gia Bach (2A202600248)")
    print("=" * 60)

    print("\n[Step 1] Building Knowledge Graph with NetworkX...")
    extractor = EntityExtractor()
    os.makedirs(PROCESSED_PATH.parent, exist_ok=True)
    extraction_start = time.perf_counter()
    triples = extractor.extract_from_corpus(str(CORPUS_PATH), str(PROCESSED_PATH))
    extraction_time = time.perf_counter() - extraction_start

    build_start = time.perf_counter()
    builder = GraphBuilder()
    builder.add_triples(triples)
    build_time = time.perf_counter() - build_start

    stats = builder.get_stats()
    print("✅ Graph built successfully!")
    print(f"   - Nodes: {stats['num_nodes']}")
    print(f"   - Edges: {stats['num_edges']}")
    print(f"   - Build time: {build_time:.3f}s")

    print("\n[Step 2] Visualizing Graph...")
    os.makedirs(GRAPH_PATH.parent, exist_ok=True)
    builder.visualize(str(GRAPH_PATH))
    builder.export_interactive_html(str(GRAPH_HTML_PATH))

    print("\n[Step 3] Testing BFS Traversal...")
    for entity in ["OpenAI", "Microsoft", "Tesla", "Apple"]:
        if entity in builder.graph.nodes:
            bfs_result = builder.bfs_traversal(entity, max_depth=2)
            print(f"\n🔍 BFS from '{entity}':")
            print(f"   {bfs_result}")

    print("\n[Step 4] Testing Query Engine...")
    engine = QueryEngine(builder.graph)
    corpus_paragraphs = load_paragraphs()
    flat_retriever = FlatRAGRetriever(corpus_paragraphs)

    query_results = []
    flat_results = []
    graph_correct = 0
    flat_correct = 0

    for index, item in enumerate(benchmark_questions, start=1):
        query = item["query"]
        expected = item["expected"]

        flat_start = time.perf_counter()
        flat_answer = flat_retriever.answer(query)
        flat_time = time.perf_counter() - flat_start

        graph_start = time.perf_counter()
        graph_answer = engine.answer(query)
        graph_time = time.perf_counter() - graph_start

        flat_ok = is_expected_in_answer(expected, flat_answer)
        graph_ok = is_expected_in_answer(expected, graph_answer)

        flat_correct += int(flat_ok)
        graph_correct += int(graph_ok)

        print(f"\nQ{index}: {query}")
        print(f"  Flat RAG  : {flat_answer}")
        print(f"  GraphRAG  : {graph_answer}")

        query_results.append(
            {
                "index": index,
                "query": query,
                "type": item["type"],
                "expected": expected,
                "graph_answer": graph_answer,
                "graph_correct": graph_ok,
                "graph_time_sec": round(graph_time, 6),
            }
        )
        flat_results.append(
            {
                "index": index,
                "query": query,
                "type": item["type"],
                "expected": expected,
                "flat_answer": flat_answer,
                "flat_correct": flat_ok,
                "flat_time_sec": round(flat_time, 6),
            }
        )

    graph_accuracy = round(graph_correct / len(benchmark_questions) * 100, 2)
    flat_accuracy = round(flat_correct / len(benchmark_questions) * 100, 2)

    os.makedirs(RESULTS_PATH.parent, exist_ok=True)
    results_payload = {
        "graph_stats": {
            **stats,
            "extraction_time_sec": round(extraction_time, 6),
            "build_time_sec": round(build_time, 6),
        },
        "graph_rag_results": query_results,
        "flat_rag_results": flat_results,
        "summary": {
            "num_questions": len(benchmark_questions),
            "graph_accuracy_percent": graph_accuracy,
            "flat_accuracy_percent": flat_accuracy,
        },
        "total_triples": len(triples),
    }

    RESULTS_PATH.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    save_comparison_table(
        [
            {
                "index": row["index"],
                "query": row["query"],
                "expected": row["expected"],
                "flat_answer": flat_row["flat_answer"],
                "graph_answer": row["graph_answer"],
            }
            for row, flat_row in zip(query_results, flat_results)
        ],
        COMPARISON_PATH,
    )

    print("\n[Step 5] Summary")
    print(f"Flat RAG accuracy : {flat_accuracy:.2f}%")
    print(f"GraphRAG accuracy : {graph_accuracy:.2f}%")
    print(f"Graph saved to    : {GRAPH_PATH}")
    print(f"Interactive HTML  : {GRAPH_HTML_PATH}")
    print(f"Results saved to  : {RESULTS_PATH}")
    print(f"Comparison saved  : {COMPARISON_PATH}")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("📁 Results saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
