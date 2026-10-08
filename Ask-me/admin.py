"""RAG admin panel: ingest content, inspect what retrieval returns."""

import functools
import logging
import os
import secrets

from flask import Blueprint, current_app, jsonify, render_template, request

from ingest import SUPPORTED_EXTENSIONS, ExtractionError, chunk, clean, extract_text

log = logging.getLogger("askme.admin")

bp = Blueprint("admin", __name__)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PASTE_CHARS = 200_000

# Set ADMIN_TOKEN in .env for anything shared. Unset, we mint a random one per
# process and log it -- convenient locally, and never accidentally wide open.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN") or secrets.token_urlsafe(12)
TOKEN_IS_GENERATED = not os.getenv("ADMIN_TOKEN")


def _store():
    return current_app.extensions["askme_store"]


def require_token(view):
    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        supplied = request.headers.get("X-Admin-Token") or request.args.get("token", "")
        if not secrets.compare_digest(supplied, ADMIN_TOKEN):
            return jsonify(error="Invalid admin token."), 401
        return view(*args, **kwargs)

    return wrapper


@bp.get("/admin")
def panel():
    return render_template("admin.html", extensions=", ".join(SUPPORTED_EXTENSIONS))


@bp.post("/api/admin/auth")
def auth():
    supplied = (request.get_json(silent=True) or {}).get("token", "")
    if not secrets.compare_digest(supplied, ADMIN_TOKEN):
        return jsonify(error="Invalid admin token."), 401
    return jsonify(ok=True)


@bp.get("/api/admin/sources")
@require_token
def list_sources():
    store = _store()
    return jsonify(
        sources=store.sources,
        chunks=len(store.chunks),
        embeddings_ok=store.embeddings_ok,
    )


@bp.post("/api/admin/ingest")
@require_token
def ingest_document():
    store = _store()

    upload = request.files.get("file")
    if upload and upload.filename:
        blob = upload.read(MAX_UPLOAD_BYTES + 1)
        if len(blob) > MAX_UPLOAD_BYTES:
            return jsonify(error="File is larger than 8MB."), 413
        try:
            text = extract_text(upload.filename, blob)
        except ExtractionError as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:  # noqa: BLE001 - malformed files are user error
            log.warning("extraction failed for %s: %s", upload.filename, exc)
            return jsonify(error="Couldn't read that file — it may be corrupt."), 400

        title = (request.form.get("title") or "").strip() or upload.filename
        source_type = (request.form.get("type") or "document").strip()
        origin = "upload"
    else:
        payload = request.get_json(silent=True) or request.form
        text = (payload.get("text") or "")[:MAX_PASTE_CHARS]
        title = (payload.get("title") or "").strip()
        source_type = (payload.get("type") or "document").strip()
        origin = "paste"
        if not title:
            return jsonify(error="Give the content a title."), 400

    text = clean(text)
    if len(text) < 40:
        return jsonify(error="Not enough text to index (need ~40+ characters)."), 400

    chunks = chunk(text)
    if not chunks:
        return jsonify(error="Nothing indexable found in that content."), 400

    try:
        source_id = store.add(
            chunks, source_title=title, source_type=source_type, origin=origin
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest failed")
        return jsonify(error="Indexing failed: {}".format(exc)), 500

    log.info("ingested '%s' as %s (%d chunks)", title, source_id, len(chunks))
    return jsonify(
        ok=True,
        source_id=source_id,
        title=title,
        chunks=len(chunks),
        chars=len(text),
        preview=chunks[0][:400],
    )


@bp.delete("/api/admin/sources/<source_id>")
@require_token
def delete_source(source_id):
    removed = _store().delete_source(source_id)
    if not removed:
        return jsonify(error="No such source."), 404
    return jsonify(ok=True, removed=removed)


@bp.post("/api/admin/search")
@require_token
def search():
    """Retrieval inspector -- run a question, see exactly what it would feed the model."""
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify(error="Enter a question to test."), 400

    hits = _store().search(query, top_k=int(body.get("top_k") or 5))
    return jsonify(
        query=query,
        results=[
            {
                "chunk_id": hit["chunk"]["chunk_id"],
                "source_id": hit["chunk"]["source_id"],
                "title": hit["chunk"].get("source_title", ""),
                "origin": hit["chunk"].get("origin", ""),
                "text": hit["chunk"]["text"],
                "score": round(hit["score"], 5),
                "similarity": hit["similarity"],
                "rank_dense": hit["rank_dense"],
                "rank_keyword": hit["rank_keyword"],
            }
            for hit in hits
        ],
    )
