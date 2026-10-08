#!/usr/bin/env python3
"""Load data/knowledge.json into the vector store.

    python seed.py           # add seed docs if they aren't there yet
    python seed.py --force   # transactionally replace all seed docs

Ingested (pasted/uploaded) documents are never touched.
"""

import json
import sys
from pathlib import Path

from vectorstore import VectorStore

KB_PATH = Path(__file__).parent / "data" / "knowledge.json"


def seed(store, force=False):
    docs = json.loads(
        KB_PATH.read_text(encoding="utf-8")
    )["documents"]

    if force:
        count = store.bulk_replace(
            docs,
            origin="seed",
        )
        return count

    existing = {
        s["title"]
        for s in store.sources
        if s["origin"] == "seed"
    }

    added = 0

    for doc in docs:
        if doc["title"] in existing:
            continue

        # Seed entries are already one tight paragraph -- keep each as one
        # chunk so the retrieved unit matches what a human curated.
        body = "{}\n\n{}".format(
            doc["title"],
            doc["text"],
        )

        store.add(
            [body],
            source_title=doc["title"],
            source_type=doc["type"],
            origin="seed",
            tags=doc.get("tags", []),
        )

        added += 1

    return added


if __name__ == "__main__":
    store = VectorStore.load()

    count = seed(
        store,
        force="--force" in sys.argv,
    )

    print(
        "processed {} seed document(s); store now holds {} "
        "chunks across {} sources".format(
            count,
            len(store.chunks),
            len(store.sources),
        )
    )
