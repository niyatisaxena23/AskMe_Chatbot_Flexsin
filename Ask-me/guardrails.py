"""Output guardrails: think-tag stripping and client anonymization.

The system prompt already forbids naming clients. This module enforces the
same rule mechanically, so a prompt injection or a leaky context chunk still
cannot surface a client name to the visitor.
"""

import re
from pathlib import Path

NAMES_PATH = Path(__file__).parent / "data" / "client_names.txt"

REDACTION = "[client name withheld]"

# Qwen3 emits chain-of-thought in <think>...</think>. Groq's reasoning_format
# should suppress it, but strip it here too -- an unterminated block on a
# truncated response would otherwise leak reasoning to the visitor.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)

_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)


def strip_reasoning(text):
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.strip()


def load_client_names(path=NAMES_PATH):
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    # Longest first so "Acme Health Systems" is matched before "Acme".
    return sorted(names, key=len, reverse=True)


class Anonymizer:
    def __init__(self, names=None, allowed_urls=()):
        self.names = names if names is not None else load_client_names()
        self.allowed_urls = tuple(allowed_urls)
        self._patterns = [
            re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            for name in self.names
        ]

    def scrub(self, text):
        """Redact known client names. Safe to run on context and on answers."""
        if not text:
            return text
        for pattern in self._patterns:
            text = pattern.sub(REDACTION, text)
        return text

    def scrub_answer(self, text):
        """Answer-side scrub: names plus any URL that isn't ours."""
        text = self.scrub(text)

        def _replace(match):
            url = match.group(0)
            if any(url.startswith(allowed) for allowed in self.allowed_urls):
                return url
            if "flexsin.com" in url.lower():
                return url
            return "[link removed]"

        return _URL.sub(_replace, text)
