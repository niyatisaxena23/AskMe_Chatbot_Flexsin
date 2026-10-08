"""Text embedding backend.

Uses fastembed (ONNX runtime) so embedding happens locally: no API key, no
per-token cost, and visitor questions never leave the box. The model is
downloaded once (~130MB) to the fastembed cache on first use.

Groq serves chat models but not embedding models, which is why this is local
rather than another call to the same provider.
"""

import logging
import os
import threading

import numpy as np

log = logging.getLogger("askme.embeddings")

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DIM = int(os.getenv("EMBED_DIM", "384"))

# bge models want an instruction prefix on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None
_lock = threading.Lock()


def _get_model():
    """Load lazily -- importing this module must stay cheap for tests."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding

                log.info("loading embedding model %s", MODEL_NAME)
                _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_documents(texts):
    """Embed passages. Returns a float32 (n, DIM) array of unit vectors."""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    vectors = list(_get_model().embed(list(texts)))
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text):
    """Embed one question. Returns a float32 (DIM,) unit vector."""
    vectors = list(_get_model().query_embed([text]))
    return np.asarray(vectors[0], dtype=np.float32)


def warm_up():
    """Pay the model-load cost at startup instead of on the first visitor."""
    try:
        embed_query("warm up")
        return True
    except Exception as exc:  # noqa: BLE001 - startup must not hard-fail
        log.warning("embedding model unavailable (%s); falling back to keyword search", exc)
        return False
