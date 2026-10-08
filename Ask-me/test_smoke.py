"""Smoke tests -- no network, the Groq call is stubbed."""

import json

import app as askme
from guardrails import Anonymizer, strip_reasoning


def client():
    askme.app.config["TESTING"] = True
    askme.SESSIONS.clear()
    askme._hits.clear()
    return askme.app.test_client()


def post(c, message, session_id=None):
    return c.post("/api/chat", json={"message": message, "session_id": session_id})


def test_health():
    res = client().get("/healthz")
    assert res.status_code == 200 and res.get_json()["chunks"] > 0


def test_index_renders():
    assert client().get("/").status_code == 200


def test_empty_message_rejected():
    assert post(client(), "  ").status_code == 400


def test_out_of_scope_skips_the_model(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("model should not be called with no context")

    monkeypatch.setattr(askme.llm, "complete", boom)
    body = post(client(), "what is the capital of France?").get_json()
    assert body["grounded"] is False
    assert "reaching out to the team" in body["reply"]


def test_grounded_answer_and_history(monkeypatch):
    seen = {}

    def fake(messages, **_k):
        seen["messages"] = messages
        return "<think>internal</think>We build native and cross-platform apps."

    monkeypatch.setattr(askme.llm, "complete", fake)
    c = client()
    body = post(c, "do you build mobile apps?").get_json()

    assert body["grounded"] is True
    assert "<think>" not in body["reply"]
    assert any("Mobile" in s["title"] for s in body["sources"])
    assert "CONTEXT:" in seen["messages"][-1]["content"]

    post(c, "what about iOS specifically?", session_id=body["session_id"])
    assert len(seen["messages"]) == 4  # system + prior user/assistant + new user


def test_client_names_are_scrubbed_from_answers(monkeypatch):
    askme.anonymizer = Anonymizer(names=["Acme Health"], allowed_urls=(askme.CONTACT_URL,))
    monkeypatch.setattr(
        askme.llm, "complete", lambda *_a, **_k: "That was Acme Health, see https://acmehealth.com"
    )
    reply = post(client(), "who was the healthcare client?").get_json()["reply"]
    assert "Acme" not in reply and "acmehealth.com" not in reply


def test_llm_failure_returns_502(monkeypatch):
    def fail(*_a, **_k):
        raise askme.LLMError("down")

    monkeypatch.setattr(askme.llm, "complete", fail)
    res = post(client(), "do you build mobile apps?")
    assert res.status_code == 502 and "error" in res.get_json()


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(askme, "RATE_LIMIT_PER_MIN", 3)
    monkeypatch.setattr(askme.llm, "complete", lambda *_a, **_k: "ok")
    c = client()
    codes = [post(c, "do you build mobile apps?").status_code for _ in range(5)]
    assert 429 in codes


def test_knowledge_base_has_no_bare_client_names():
    docs = json.load(open("data/knowledge.json"))["documents"]
    for doc in docs:
        if doc["type"] == "case_study":
            assert "client name" not in doc["text"].lower()


def test_strip_unterminated_think():
    assert strip_reasoning("<think>cut off") == ""


# --------------------------------------------------------------- RAG panel

import io

import numpy as np
import pytest

import admin as admin_mod
from ingest import chunk, extract_text, ExtractionError
from vectorstore import VectorStore


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Deterministic bag-of-words vectors -- no model download in tests."""

    def encode(text):
        vec = np.zeros(askme.embeddings.DIM, dtype=np.float32)
        for word in text.lower().split():
            vec[hash(word) % askme.embeddings.DIM] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    monkeypatch.setattr(
        askme.embeddings, "embed_documents",
        lambda texts: np.vstack([encode(t) for t in texts]).astype(np.float32),
    )
    monkeypatch.setattr(askme.embeddings, "embed_query", encode)


@pytest.fixture
def admin_client(monkeypatch, tmp_path, fake_embeddings):
    monkeypatch.setattr("vectorstore.STORE_DIR", tmp_path)
    monkeypatch.setattr("vectorstore.CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr("vectorstore.VECTORS_PATH", tmp_path / "vectors.npy")
    store = VectorStore()
    askme.store = store
    askme.app.extensions["askme_store"] = store
    askme.app.config["TESTING"] = True
    askme.SESSIONS.clear()
    askme._hits.clear()
    c = askme.app.test_client()
    c.environ_base["HTTP_X_ADMIN_TOKEN"] = admin_mod.ADMIN_TOKEN
    return c, store


def test_admin_requires_token(admin_client):
    c, _ = admin_client
    c.environ_base.pop("HTTP_X_ADMIN_TOKEN")
    assert c.get("/api/admin/sources").status_code == 401


def test_admin_panel_renders(admin_client):
    c, _ = admin_client
    assert c.get("/admin").status_code == 200


def test_paste_ingest_then_retrieve(admin_client):
    c, store = admin_client
    res = c.post(
        "/api/admin/ingest",
        json={"title": "Salesforce", "type": "service",
              "text": "Flexsin implements Salesforce Sales Cloud with custom Apex "
                      "development and integrates it with client ERP systems." * 2},
    )
    assert res.status_code == 200 and res.get_json()["chunks"] >= 1

    found = c.post("/api/admin/search", json={"query": "Salesforce Apex"}).get_json()
    assert found["results"] and found["results"][0]["title"] == "Salesforce"
    assert len(store.chunks) >= 1


def test_docx_upload(admin_client):
    docx = pytest.importorskip("docx")
    c, _ = admin_client

    document = docx.Document()
    document.add_paragraph("Flexsin delivered a warehouse management system.")
    document.add_paragraph("The stack was Django, Postgres and Kubernetes on GCP.")
    buffer = io.BytesIO()
    document.save(buffer)

    res = c.post(
        "/api/admin/ingest",
        data={"file": (io.BytesIO(buffer.getvalue()), "brief.docx"), "type": "case_study"},
        content_type="multipart/form-data",
    )
    body = res.get_json()
    assert res.status_code == 200, body
    assert body["title"] == "brief.docx" and body["chunks"] >= 1
    assert "warehouse" in body["preview"].lower()


def test_unsupported_file_rejected(admin_client):
    c, _ = admin_client
    res = c.post(
        "/api/admin/ingest",
        data={"file": (io.BytesIO(b"binary"), "logo.png")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400 and "Unsupported" in res.get_json()["error"]


def test_legacy_doc_gets_a_useful_message():
    with pytest.raises(ExtractionError, match="re-save it as .docx"):
        extract_text("brief.doc", b"\xd0\xcf")


def test_too_short_paste_rejected(admin_client):
    c, _ = admin_client
    res = c.post("/api/admin/ingest", json={"title": "T", "text": "hi"})
    assert res.status_code == 400


def test_paste_without_title_rejected(admin_client):
    c, _ = admin_client
    res = c.post("/api/admin/ingest", json={"text": "x" * 200})
    assert res.status_code == 400


def test_delete_source_removes_chunks_and_vectors(admin_client):
    c, store = admin_client
    source_id = c.post(
        "/api/admin/ingest",
        json={"title": "Temp", "text": "Flexsin builds logistics platforms. " * 20},
    ).get_json()["source_id"]

    assert c.delete("/api/admin/sources/" + source_id).status_code == 200
    assert store.chunks == [] and len(store.vectors) == 0
    assert c.delete("/api/admin/sources/" + source_id).status_code == 404


def test_store_survives_reload(admin_client):
    c, store = admin_client
    c.post("/api/admin/ingest", json={"title": "Persist", "text": "Flexsin builds portals. " * 20})
    reloaded = VectorStore.load()
    assert len(reloaded.chunks) == len(store.chunks)
    assert reloaded.vectors.shape == store.vectors.shape


def test_ingested_content_reaches_the_chat_context(admin_client, monkeypatch):
    c, _ = admin_client
    c.post("/api/admin/ingest",
           json={"title": "Drone survey work",
                 "text": "Flexsin built a drone survey pipeline for mining sites "
                         "using photogrammetry and Python." * 3})

    seen = {}
    monkeypatch.setattr(askme.llm, "complete",
                        lambda messages, **_k: (seen.update(m=messages), "Yes, we have.")[1])

    body = post(c, "drone survey photogrammetry").get_json()
    assert body["grounded"] is True
    assert "drone survey" in seen["m"][-1]["content"].lower()


def test_chunking_overlaps_and_respects_size():
    chunks = chunk("Flexsin builds software. " * 400)
    assert len(chunks) > 1
    assert all(len(c) <= 1400 for c in chunks)
    assert "…" in chunks[1]
