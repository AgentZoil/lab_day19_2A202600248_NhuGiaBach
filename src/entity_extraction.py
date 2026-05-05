import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

class EntityExtractor:
    KNOWN_COMPANIES = [
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

    HQ_LOCATIONS = [
        "San Francisco",
        "Redmond",
        "Menlo Park",
        "Cupertino",
        "Austin",
        "Bellevue",
        "Mountain View",
        "Seattle",
        "Santa Clara",
        "California",
        "Washington",
        "Texas",
    ]

    def __init__(self, use_openai: Optional[bool] = None):
        self.project_root = Path(__file__).resolve().parents[1]
        self._load_environment()

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        force_disable = os.getenv("USE_OPENAI_EXTRACTOR", "").strip().lower() in {"0", "false", "no"}

        if use_openai is None:
            self.use_openai = bool(api_key) and not force_disable
        else:
            self.use_openai = bool(use_openai) and bool(api_key) and not force_disable

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        self.client = None
        self._openai_fallback_warned = False

        if self.use_openai:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=api_key)
            except Exception as exc:
                self.use_openai = False
                self._warn_openai_fallback(f"OpenAI init failed, falling back to rules: {exc}")

    def _warn_openai_fallback(self, message: str) -> None:
        if not self._openai_fallback_warned:
            print(f"[EntityExtractor] {message}")
            self._openai_fallback_warned = True

    def _load_environment(self) -> None:
        try:
            from dotenv import load_dotenv

            env_path = self.project_root / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
        except Exception:
            pass
    
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        if self.use_openai:
            try:
                triples = self._extract_with_openai(text)
                if triples:
                    return triples
            except Exception as exc:
                self._warn_openai_fallback(f"OpenAI extraction failed, falling back to rules: {exc}")
        else:
            return self._extract_with_rules(text)

        return self._extract_with_rules(text)
    
    def _extract_with_openai(self, text: str) -> List[Tuple[str, str, str]]:
        prompt = f"""Extract all (entity1, relation, entity2) triples from the text below.
Return ONLY a JSON array of triples, nothing else.

Text: {text}

Format: [["entity1", "relation", "entity2"], ...]
Example: [["OpenAI", "FOUNDED_BY", "Sam Altman"], ["OpenAI", "FOUNDED_IN", "2015"]]
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        parsed = json.loads(content)
        triples = []
        for item in parsed:
            if isinstance(item, list) and len(item) == 3:
                triples.append(tuple(str(part).strip() for part in item))
        return triples
    
    def _extract_with_rules(self, text: str) -> List[Tuple[str, str, str]]:
        triples = []
        company = self._detect_company(text)
        if not company:
            return []

        lower_text = text.lower()

        founder_clause = self._extract_founder_clause(text)
        if founder_clause:
            for person in self.KNOWN_PEOPLE:
                if person.lower() in founder_clause.lower():
                    triples.append((company, "FOUNDED_BY", person))

        year = self._extract_founding_year(text)
        if year:
            triples.append((company, "FOUNDED_IN", year))

        ceo = self._extract_ceo(text, company)
        if ceo:
            triples.append((company, "CEO", ceo))

        location = self._extract_headquarters(text)
        if location:
            triples.append((company, "HEADQUARTERED_IN", location))

        parent_company = self._extract_parent_company(text)
        if parent_company:
            triples.append((company, "PARENT_COMPANY", parent_company))

        renamed_to = self._extract_renamed_to(text)
        if renamed_to:
            triples.append((company, "RENAMED_TO", renamed_to))

        return list(dict.fromkeys(triples))

    def _detect_company(self, text: str) -> str:
        lower_text = text.lower()
        if "formerly facebook" in lower_text and "meta" in lower_text:
            return "Meta"

        for company in self.KNOWN_COMPANIES:
            if company.lower() in lower_text:
                return company
        return ""

    def _extract_founder_clause(self, text: str) -> str:
        patterns = [
            r"founded by (?P<clause>.+?)(?: in | on | while |\.|, the |$)",
            r"was founded by (?P<clause>.+?)(?: in | on | while |\.|, the |$)",
            r"founded (?P<clause>.+?)(?: in | on | while |\.|, the |$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group("clause")
        return ""

    def _extract_founding_year(self, text: str) -> str:
        match = re.search(
            r"(?:founded|founding|founded in|was founded in).{0,60}?\b(19[0-9]{2}|20[0-9]{2})\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)

        years = re.findall(r"\b(19[0-9]{2}|20[0-9]{2})\b", text)
        return years[0] if years else ""

    def _extract_ceo(self, text: str, company: str) -> str:
        if "ceo" not in text.lower():
            return ""

        for ceo in [
            "Satya Nadella",
            "Andy Jassy",
            "Tim Cook",
            "Elon Musk",
            "Jensen Huang",
        ]:
            if ceo.lower() in text.lower():
                return ceo

        patterns = [
            rf"([A-Z][A-Za-z.\- ]+?) is the current CEO of {re.escape(company)}",
            rf"([A-Z][A-Za-z.\- ]+?) is the current CEO(?:\.|$)",
            rf"([A-Z][A-Za-z.\- ]+?) has been the CEO(?:\.|$)",
            rf"([A-Z][A-Za-z.\- ]+?) later became CEO",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_headquarters(self, text: str) -> str:
        patterns = [
            r"headquartered in ([A-Z][A-Za-z ]+)",
            r"based in ([A-Z][A-Za-z ]+)",
            r"located in ([A-Z][A-Za-z ]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                for known in self.HQ_LOCATIONS:
                    if known.lower() in location.lower():
                        return known
                return location
        return ""

    def _extract_parent_company(self, text: str) -> str:
        match = re.search(r"parent company is ([A-Z][A-Za-z.& ]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if "alphabet" in text.lower():
            return "Alphabet Inc."
        return ""

    def _extract_renamed_to(self, text: str) -> str:
        match = re.search(r"formerly [^(]*\(([^)]+)\)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if "formerly facebook" in text.lower():
            return "Meta"
        return ""
    
    def extract_from_corpus(self, corpus_path: str, output_path: str) -> List[Tuple[str, str, str]]:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        all_triples = []
        for para in paragraphs:
            triples = self.extract_triples(para)
            all_triples.extend(triples)
            print(f"Extracted {len(triples)} triples")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([list(t) for t in all_triples], f, indent=2)
        return all_triples
