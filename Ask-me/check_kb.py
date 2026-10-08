#!/usr/bin/env python3
"""Validate the knowledge base and test what a question retrieves.

    python check_kb.py                        # validate every entry
    python check_kb.py "do you do salesforce" # see what a question matches
"""

import json
import sys

from guardrails import load_client_names
from retrieval import Retriever

VALID_TYPES = {"service", "process", "company", "case_study"}
REQUIRED = ("id", "type", "title", "tags", "text")


def validate(docs):
    problems = []
    seen = set()
    names = load_client_names()

    for i, doc in enumerate(docs):
        where = doc.get("id") or "entry #{}".format(i)

        for field in REQUIRED:
            if not doc.get(field):
                problems.append("{}: missing '{}'".format(where, field))

        if doc.get("id") in seen:
            problems.append("{}: duplicate id".format(where))
        seen.add(doc.get("id"))

        if doc.get("type") not in VALID_TYPES:
            problems.append(
                "{}: type '{}' not one of {}".format(
                    where, doc.get("type"), sorted(VALID_TYPES)
                )
            )

        text = doc.get("text", "")
        if len(text) < 80:
            problems.append("{}: text is very short, retrieval will struggle".format(where))
        if "http" in text:
            problems.append("{}: text contains a URL — remove it".format(where))
        for name in names:
            if name.lower() in text.lower():
                problems.append("{}: contains listed client name '{}'".format(where, name))

    return problems


def main():
    with open("data/knowledge.json", encoding="utf-8") as fh:
        docs = json.load(fh)["documents"]

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        hits = Retriever(docs).search(question, top_k=5)
        print('"{}"\n'.format(question))
        if not hits:
            print("  NO MATCH — the bot will answer 'I don't have specific")
            print("  information on that'. Add tags or an entry to fix it.")
            return 1
        for doc, score in hits:
            print("  {:6.2f}  {:<28} {}".format(score, doc["id"], doc["title"]))
        return 0

    problems = validate(docs)
    print("{} entries".format(len(docs)))
    for problem in problems:
        print("  ! " + problem)
    print("OK" if not problems else "{} problem(s)".format(len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
