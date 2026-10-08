"""Tiny dependency-free BM25 retriever over the Flexsin knowledge base.

The corpus is small (tens of documents), so an in-memory index rebuilt at
startup is plenty -- no vector DB or embedding call needed. Swap this module
for a vector store if the corpus grows past a few hundred documents.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

KB_PATH = Path(__file__).parent / "data" / "knowledge.json"

_TOKEN_RE = re.compile(r"[a-z0-9\+#\.]+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "into", "to", "from", "in", "on", "is", "are", "was",
    "were", "be", "been", "do", "does", "did", "you", "your", "we", "our",
    "i", "me", "my", "it", "its", "this", "that", "these", "those", "can",
    "could", "would", "should", "will", "what", "how", "who", "which",
    "there", "have", "has", "had", "any", "some", "tell", "know", "like",
}

# Cheap morphological folding so "services" matches "service".
def _stem(token):
    for suffix in ("ing", "ies", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            if suffix == "ies":
                return token[: -len(suffix)] + "y"
            return token[: -len(suffix)]
    return token


def tokenize(text):
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Retriever:
    K1 = 1.5
    B = 0.75

    def __init__(self, documents):
        self.docs = documents
        self.tokens = []
        for doc in documents:
            blob = " ".join([doc["title"], " ".join(doc.get("tags", [])), doc["text"]])
            self.tokens.append(Counter(tokenize(blob)))

        self.lengths = [sum(c.values()) for c in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        self.df = Counter()
        for counts in self.tokens:
            self.df.update(counts.keys())
        self.n = len(documents)

    def _idf(self, term):
        df = self.df.get(term, 0)
        if not df:
            return 0.0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query, top_k=4, min_score=1.0):
        """Return [(doc, score)] above min_score, best first."""
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scored = []
        for idx, counts in enumerate(self.tokens):
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                norm = 1 - self.B + self.B * (self.lengths[idx] / (self.avg_len or 1))
                score += self._idf(term) * (tf * (self.K1 + 1)) / (tf + self.K1 * norm)
            if score > 0:
                scored.append((self.docs[idx], score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scored[:top_k] if pair[1] >= min_score]


def load_retriever(path=KB_PATH):
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return Retriever(payload["documents"])


def format_context(hits):
    """Render retrieved docs into the CONTEXT block the system prompt expects."""
    if not hits:
        return "(no relevant context found)"
    blocks = []
    for doc, _score in hits:
        blocks.append(
            "[{id} | {type}] {title}\n{text}".format(
                id=doc["id"], type=doc["type"], title=doc["title"], text=doc["text"]
            )
        )
    return "\n\n---\n\n".join(blocks)
