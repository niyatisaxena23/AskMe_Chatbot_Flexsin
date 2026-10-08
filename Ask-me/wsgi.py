"""WSGI entrypoint for gunicorn.

    gunicorn -c gunicorn.conf.py wsgi:app

Warm-up and seeding live in app.py's __main__ block, which gunicorn never runs,
so they happen here instead.
"""

import logging

import admin
import embeddings
from app import app, llm, store
from seed import seed

log = logging.getLogger("askme")

store.embeddings_ok = embeddings.warm_up()

if not store.chunks:
    log.info("store is empty — seeding from data/knowledge.json")
    seed(store)

log.info(
    "ready: %d chunks / %d sources, chat=%s, embeddings=%s",
    len(store.chunks), len(store.sources), llm.model, store.embeddings_ok,
)
if app.config.get("ADMIN_ENABLED", True) and admin.TOKEN_IS_GENERATED:
    log.warning("ADMIN_TOKEN not set — this run's token is: %s", admin.ADMIN_TOKEN)

if __name__ == "__main__":
    app.run()
