import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple


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
    "for",
    "on",
    "with",
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

KNOWN_PEOPLE = [
    "Sam Altman",
    "Elon Musk",
    "Greg Brockman",
    "Ilya Sutskever",
    "John Schulman",
    "Wojciech Zaremba",
    "Larry Page",
    "Sergey Brin",
    "Bill Gates",
    "Paul Allen",
    "Satya Nadella",
    "Mark Zuckerberg",
    "Eduardo Saverin",
    "Andrew McCollum",
    "Dustin Moskovitz",
    "Chris Hughes",
    "Jeff Bezos",
    "Andy Jassy",
    "Steve Jobs",
    "Steve Wozniak",
    "Ronald Wayne",
    "Tim Cook",
    "Martin Eberhard",
    "Marc Tarpenning",
    "Jensen Huang",
    "Chris Malachowsky",
    "Curtis Priem",
]


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if token not in STOPWORDS
    ]


class FlatRAGRetriever:
    def __init__(self, paragraphs: Sequence[str]):
        self.paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph and paragraph.strip()]
        self._doc_tokens = [tokenize(paragraph) for paragraph in self.paragraphs]
        self._idf = self._build_idf(self._doc_tokens)

    def _build_idf(self, docs: Sequence[Sequence[str]]) -> Dict[str, float]:
        doc_count = len(docs)
        df = Counter()
        for tokens in docs:
            for token in set(tokens):
                df[token] += 1
        return {
            token: math.log((1 + doc_count) / (1 + freq)) + 1.0
            for token, freq in df.items()
        }

    def _score_paragraph(self, query_tokens: Sequence[str], paragraph_index: int, query_text: str) -> float:
        paragraph_tokens = self._doc_tokens[paragraph_index]
        if not paragraph_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        para_counts = Counter(paragraph_tokens)
        score = 0.0

        for token, q_count in query_counts.items():
            if token in para_counts:
                tf = 1.0 + math.log(para_counts[token])
                idf = self._idf.get(token, 0.0)
                score += q_count * tf * idf

        query_lower = query_text.lower()
        paragraph_lower = self.paragraphs[paragraph_index].lower()
        for company in COMPANY_NAMES:
            if company.lower() in query_lower and company.lower() in paragraph_lower:
                score += 2.5

        # Give a small boost for exact token order overlaps and numeric matches.
        if any(token.isdigit() for token in query_tokens):
            score += sum(0.5 for token in query_tokens if token.isdigit() and token in paragraph_tokens)

        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_tokens = tokenize(query)
        scored = [
            (self.paragraphs[i], self._score_paragraph(query_tokens, i, query))
            for i in range(len(self.paragraphs))
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def answer(self, query: str) -> str:
        top_hits = self.retrieve(query, top_k=1)
        if not top_hits or top_hits[0][1] < 2.0:
            return f"❌ Không tìm thấy thông tin liên quan đến câu hỏi: '{query}'"

        paragraph = top_hits[0][0]
        query_lower = query.lower()
        company = self._extract_company_from_query(query)

        if "who founded" in query_lower or "founder" in query_lower:
            founders = self._extract_people_from_paragraph(paragraph)
            if founders:
                return ", ".join(founders)
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

    def _extract_company_from_query(self, query: str) -> str:
        query_lower = query.lower()
        for company in COMPANY_NAMES:
            if company.lower() in query_lower:
                return company
        return ""

    def _extract_people_from_paragraph(self, paragraph: str) -> List[str]:
        found = []
        paragraph_lower = paragraph.lower()
        for person in KNOWN_PEOPLE:
            if person.lower() in paragraph_lower:
                found.append(person)
        return found
