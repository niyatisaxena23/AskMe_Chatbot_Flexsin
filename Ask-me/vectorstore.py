"""Persistent chunk store with hybrid (vector + PostgreSQL FTS) retrieval.

Dense vector search is performed by PostgreSQL with pgvector.
Keyword search is performed by PostgreSQL full-text search.
Ranking is fused with Reciprocal Rank Fusion in Python.
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import psycopg

import embeddings

log = logging.getLogger("askme.store")

STORE_DIR = Path(__file__).parent / "data" / "store"
CHUNKS_PATH = STORE_DIR / "chunks.json"
VECTORS_PATH = STORE_DIR / "vectors.npy"

DB_NAME = "askme_db"
DB_USER = "askme_app"
DB_HOST = "localhost"
DB_PORT = 5432
RRF_K = 60


class VectorStore:
    def __init__(self, chunks=None, vectors=None):
        self.chunks = chunks or []
        self.vectors = (
            vectors
            if vectors is not None
            else np.zeros((0, embeddings.DIM), dtype=np.float32)
        )
        self.embeddings_ok = True
        self._lock = threading.RLock()

    @property
    def sources(self):
        """Ingested documents, newest first, with their chunk counts."""
        grouped = {}

        for chunk in self.chunks:
            sid = chunk["source_id"]

            entry = grouped.setdefault(
                sid,
                {
                    "source_id": sid,
                    "title": chunk.get("source_title", "Untitled"),
                    "type": chunk.get("source_type", "document"),
                    "origin": chunk.get("origin", "paste"),
                    "url": chunk.get("url"),
                    "added_at": chunk.get("added_at", 0),
                    "chunks": 0,
                    "chars": 0,
                },
            )

            entry["chunks"] += 1
            entry["chars"] += len(chunk["text"])

        return sorted(
            grouped.values(),
            key=lambda e: e["added_at"],
            reverse=True,
        )

    def add(
        self,
        texts,
        source_title,
        source_type="document",
        origin="paste",
        tags=(),
    ):
        """Embed and persist chunks in PostgreSQL.

        PostgreSQL is the source of truth. The local JSON/NumPy mirror
        is updated only after the database transaction commits successfully.

        Returns the PostgreSQL source id as a string.
        """
        texts = [t for t in texts if t and t.strip()]

        if not texts:
            raise ValueError("nothing to add")

        source_title = (source_title or "").strip()
        source_type = (source_type or "document").strip()
        origin = (origin or "paste").strip()
        tags = list(tags)

        if not source_title:
            raise ValueError("source title is required")

        vectors = embeddings.embed_documents(texts)

        if len(vectors) != len(texts):
            raise ValueError(
                "embedding count does not match chunk count"
            )

        now = time.time()

        # Keep a unique local prefix for chunk_id values.
        chunk_prefix = uuid.uuid4().hex[:12]

        db_source_id = None

        # PostgreSQL is updated first. If anything fails, the transaction
        # rolls back and the local mirror is left untouched.
        with self._db_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO sources
                            (title, type, origin, url, feed_key, synced_at)
                        VALUES
                            (%s, %s, %s, NULL, NULL, CURRENT_TIMESTAMP)
                        RETURNING id
                        """,
                        (
                            source_title,
                            source_type,
                            origin,
                        ),
                    )

                    db_source_id = cur.fetchone()[0]

                    with cur.copy(
                        """
                        COPY chunks
                            (source_id, ordinal, text, tags, embedding,
                             title, chunk_id)
                        FROM STDIN
                        """
                    ) as copy:
                        for index, (text, vector) in enumerate(
                            zip(texts, vectors)
                        ):
                            vector = np.asarray(
                                vector,
                                dtype=np.float32,
                            ).tolist()

                            vector_literal = (
                                "["
                                + ",".join(map(str, vector))
                                + "]"
                            )

                            copy.write_row(
                                (
                                    db_source_id,
                                    index,
                                    text,
                                    json.dumps(tags),
                                    vector_literal,
                                    source_title,
                                    "{}-{:03d}".format(
                                        chunk_prefix,
                                        index,
                                    ),
                                )
                            )

                conn.commit()

            except Exception:
                conn.rollback()
                raise

        # DB transaction succeeded. Only now update the local mirror.
        source_id = str(db_source_id)

        new_chunks = []

        for index, text in enumerate(texts):
            new_chunks.append(
                {
                    "chunk_id": "{}-{:03d}".format(
                        chunk_prefix,
                        index,
                    ),
                    "source_id": source_id,
                    "source_title": source_title,
                    "source_type": source_type,
                    "origin": origin,
                    "tags": tags,
                    "text": text,
                    "added_at": now,
                }
            )

        with self._lock:
            self.chunks.extend(new_chunks)

            self.vectors = (
                np.vstack(
                    [
                        self.vectors,
                        np.asarray(
                            vectors,
                            dtype=np.float32,
                        ),
                    ]
                )
                if len(self.vectors)
                else np.asarray(
                    vectors,
                    dtype=np.float32,
                )
            )

            self.save()

        return source_id

    def delete_source(self, source_id):
        """Delete a source from PostgreSQL and the local mirror."""
        source_id = str(source_id).strip()

        if not source_id:
            return 0

        try:
            db_source_id = int(source_id)
        except ValueError:
            # Older local-only source IDs cannot exist in the current
            # PostgreSQL source of truth.
            return 0

        with self._db_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM chunks
                        WHERE source_id = %s
                        """,
                        (db_source_id,),
                    )

                    removed = cur.fetchone()[0]

                    if not removed:
                        conn.rollback()
                        return 0

                    cur.execute(
                        """
                        DELETE FROM sources
                        WHERE id = %s
                        """,
                        (db_source_id,),
                    )

                conn.commit()

            except Exception:
                conn.rollback()
                raise

        # PostgreSQL deletion succeeded. Update local mirror.
        with self._lock:
            keep = [
                i
                for i, chunk in enumerate(self.chunks)
                if str(chunk["source_id"]) != source_id
            ]

            self.chunks = [
                self.chunks[i]
                for i in keep
            ]

            self.vectors = (
                self.vectors[keep]
                if keep
                else np.zeros(
                    (0, embeddings.DIM),
                    dtype=np.float32,
                )
            )

            self.save()

        return removed

    def _db_connection(self):
        return psycopg.connect(
            f"host={DB_HOST} "
            f"port={DB_PORT} "
            f"dbname={DB_NAME} "
            f"user={DB_USER} "
            f"password={os.getenv('ASKME_DB_PASSWORD', '').strip()}"
        )

    def _refresh_from_db(self):
        """Refresh the local chunk mirror from PostgreSQL.

        PostgreSQL is the shared source of truth. Each Gunicorn worker has
        its own Python process and therefore its own in-memory state, so
        retrieval refreshes this mirror before searching.
        """
        try:
            with self._db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            c.chunk_id,
                            c.source_id,
                            s.title,
                            s.type,
                            s.origin,
                            s.url,
                            c.ordinal,
                            c.text,
                            c.tags,
                            c.embedding,
                            COALESCE(
                                EXTRACT(EPOCH FROM s.synced_at),
                                c.id
                            ) AS added_at
                        FROM chunks AS c
                        JOIN sources AS s
                          ON s.id = c.source_id
                        ORDER BY c.source_id, c.ordinal, c.id
                        """
                    )

                    rows = cur.fetchall()

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "PostgreSQL refresh unavailable; keeping local mirror: %s",
                exc,
            )
            return False

        chunks = []
        vectors = []

        for (
            chunk_id,
            source_id,
            title,
            source_type,
            origin,
            url,
            ordinal,
            text,
            tags,
            embedding,
            added_at,
        ) in rows:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": str(source_id),
                    "source_title": title,
                    "source_type": source_type,
                    "origin": origin,
                    "url": url,
                    "tags": tags or [],
                    "text": text,
                    "added_at": float(added_at or 0),
                }
            )

            if embedding is None:
                vectors.append(
                    np.zeros(
                        embeddings.DIM,
                        dtype=np.float32,
                    )
                )
            else:
                if isinstance(embedding, str):
                    embedding = embedding.strip("[]")
                    values = (
                        []
                        if not embedding
                        else embedding.split(",")
                    )
                    embedding = [
                        float(value)
                        for value in values
                    ]

                vectors.append(
                    np.asarray(
                        embedding,
                        dtype=np.float32,
                    )
                )

        if vectors:
            refreshed_vectors = np.vstack(vectors).astype(
                np.float32
            )
        else:
            refreshed_vectors = np.zeros(
                (0, embeddings.DIM),
                dtype=np.float32,
            )

        with self._lock:
            self.chunks = chunks
            self.vectors = refreshed_vectors

        return True

    def bulk_replace(self, documents, origin="seed"):
        """Replace every document for an origin in one DB transaction.

        Embeddings are generated before the transaction starts. PostgreSQL
        then atomically deletes the old corpus and inserts the replacement
        corpus. The local JSON/NumPy mirror is updated only after the DB
        transaction commits successfully.

        Each document must contain:
            title
            type
            tags
            text

        Returns the number of replacement chunks.
        """
        documents = list(documents)

        prepared = []

        for document in documents:
            title = document["title"]
            source_type = document.get("type", "document")
            tags = list(document.get("tags", []))
            text = document["text"]

            if not text or not text.strip():
                continue

            source_id = uuid.uuid4().hex[:12]
            now = time.time()

            vectors = embeddings.embed_documents([text])

            prepared.append(
                {
                    "source_id": source_id,
                    "source_title": title,
                    "source_type": source_type,
                    "origin": origin,
                    "tags": tags,
                    "text": text,
                    "added_at": now,
                    "vectors": vectors,
                }
            )

        new_chunks = []
        new_vectors = []

        for document in prepared:
            source_id = document["source_id"]
            vector = document["vectors"][0]

            new_chunks.append(
                {
                    "chunk_id": "{}-000".format(source_id),
                    "source_id": source_id,
                    "source_title": document["source_title"],
                    "source_type": document["source_type"],
                    "origin": origin,
                    "tags": document["tags"],
                    "text": document["text"],
                    "added_at": document["added_at"],
                }
            )
            new_vectors.append(vector)

        if new_vectors:
            replacement_vectors = np.asarray(
                new_vectors,
                dtype=np.float32,
            )
        else:
            replacement_vectors = np.zeros(
                (0, embeddings.DIM),
                dtype=np.float32,
            )

        with self._db_connection() as conn:
            try:
                with conn.cursor() as cur:
                    # Delete the complete old corpus for this origin.
                    # The chunks are removed automatically through
                    # ON DELETE CASCADE.
                    cur.execute(
                        """
                        DELETE FROM sources
                        WHERE origin = %s
                        """,
                        (origin,),
                    )

                    db_source_ids = {}

                    # Insert replacement sources inside the same transaction.
                    for document in prepared:
                        cur.execute(
                            """
                            INSERT INTO sources
                                (title, type, origin, url, feed_key, synced_at)
                            VALUES
                                (%s, %s, %s, NULL, NULL, CURRENT_TIMESTAMP)
                            RETURNING id
                            """,
                            (
                                document["source_title"],
                                document["source_type"],
                                origin,
                            ),
                        )

                        db_source_ids[document["source_id"]] = (
                            cur.fetchone()[0]
                        )

                    # COPY all replacement chunks as one bulk operation.
                    with cur.copy(
                        """
                        COPY chunks
                            (source_id, ordinal, text, tags, embedding,
                             title, chunk_id)
                        FROM STDIN
                        """
                    ) as copy:
                        for document in prepared:
                            db_source_id = db_source_ids[
                                document["source_id"]
                            ]

                            vector = document["vectors"][0].astype(
                                np.float32
                            ).tolist()

                            vector_literal = (
                                "["
                                + ",".join(map(str, vector))
                                + "]"
                            )

                            copy.write_row(
                                (
                                    db_source_id,
                                    0,
                                    document["text"],
                                    json.dumps(document["tags"]),
                                    vector_literal,
                                    document["source_title"],
                                    "{}-000".format(
                                        document["source_id"]
                                    ),
                                )
                            )

                conn.commit()

            except Exception:
                conn.rollback()
                raise

        # PostgreSQL is now committed. Only now update the local mirror.
        with self._lock:
            self.chunks = new_chunks
            self.vectors = replacement_vectors
            self.save()

        return len(new_chunks)

    def _dense(self, query, top_k):
        """Dense vector search using PostgreSQL + pgvector.

        Keeps the original interface:
            [(chunk_index, cosine_similarity), ...]

        PostgreSQL returns cosine distance.
        Since the embeddings are unit-normalized:

            cosine similarity = 1 - cosine distance
        """
        if not self.embeddings_ok or not len(self.chunks):
            return []

        try:
            q = embeddings.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "dense search unavailable (%s); keyword only",
                exc,
            )
            self.embeddings_ok = False
            return []

        if len(q) != embeddings.DIM:
            log.warning(
                "dense search unavailable: expected query dimension %d, got %d",
                embeddings.DIM,
                len(q),
            )
            self.embeddings_ok = False
            return []

        vector_literal = (
            "[" + ",".join(map(str, q.astype(np.float32))) + "]"
        )

        index_of = {
            chunk["chunk_id"]: index
            for index, chunk in enumerate(self.chunks)
        }

        try:
            with self._db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            c.chunk_id,
                            c.embedding <=> %s::vector AS distance
                        FROM chunks AS c
                        WHERE c.embedding IS NOT NULL
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector_literal,
                            vector_literal,
                            top_k,
                        ),
                    )

                    rows = cur.fetchall()

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "PostgreSQL dense search unavailable (%s); keyword only",
                exc,
            )
            self.embeddings_ok = False
            return []

        results = []

        for chunk_id, distance in rows:
            if chunk_id not in index_of:
                continue

            index = index_of[chunk_id]
            similarity = 1.0 - float(distance)

            results.append((index, similarity))

        return results

    def _keyword(self, query, top_k):
        """Keyword search using PostgreSQL full-text search."""
        if not self.chunks or not query.strip():
            return []

        try:
            with self._db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            c.chunk_id,
                            ts_rank_cd(
                                c.search_vector,
                                websearch_to_tsquery('simple', %s)
                            ) AS rank
                        FROM chunks AS c
                        WHERE c.search_vector @@
                              websearch_to_tsquery('simple', %s)
                        ORDER BY rank DESC, c.id
                        LIMIT %s
                        """,
                        (
                            query,
                            query,
                            top_k,
                        ),
                    )

                    rows = cur.fetchall()

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "PostgreSQL keyword search unavailable: %s",
                exc,
            )
            return []

        index_of = {
            chunk["chunk_id"]: index
            for index, chunk in enumerate(self.chunks)
        }

        results = []

        for chunk_id, score in rows:
            if chunk_id not in index_of:
                continue

            results.append(
                (
                    index_of[chunk_id],
                    float(score),
                )
            )

        return results

    def search(
        self,
        query,
        top_k=5,
        candidates=20,
        min_similarity=0.62,
    ):
        """Hybrid search.

        Returns:
            [
                {
                    "chunk": chunk,
                    "score": fused_rrf_score,
                    "similarity": dense_similarity,
                    "rank_dense": dense_rank,
                    "rank_keyword": keyword_rank,
                }
            ]
        """
        if not query.strip():
            return []

        self._refresh_from_db()

        if not self.chunks:
            return []

        dense = self._dense(query, candidates)
        keyword = self._keyword(query, candidates)

        fused = {}

        for rank, (idx, sim) in enumerate(dense):
            entry = fused.setdefault(
                idx,
                {
                    "score": 0.0,
                    "similarity": sim,
                },
            )

            entry["score"] += 1.0 / (
                RRF_K + rank + 1
            )
            entry["rank_dense"] = rank + 1

        for rank, (_idx, _score) in enumerate(keyword):
            idx = _idx

            entry = fused.setdefault(
                idx,
                {
                    "score": 0.0,
                    "similarity": 0.0,
                },
            )

            entry["score"] += 1.0 / (
                RRF_K + rank + 1
            )
            entry["rank_keyword"] = rank + 1

        results = []

        for idx, entry in fused.items():
            if (
                entry["similarity"] >= min_similarity
                or "rank_keyword" in entry
            ):
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "score": entry["score"],
                        "similarity": round(
                            entry["similarity"],
                            4,
                        ),
                        "rank_dense": entry.get(
                            "rank_dense"
                        ),
                        "rank_keyword": entry.get(
                            "rank_keyword"
                        ),
                    }
                )

        results.sort(
            key=lambda r: r["score"],
            reverse=True,
        )

        return results[:top_k]

    def save(self):
        STORE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp_chunks = CHUNKS_PATH.with_suffix(
            ".json.tmp"
        )
        tmp_vectors = STORE_DIR / "vectors.tmp.npy"

        tmp_chunks.write_text(
            json.dumps(
                {"chunks": self.chunks},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        np.save(
            tmp_vectors,
            self.vectors,
        )

        tmp_chunks.replace(CHUNKS_PATH)
        tmp_vectors.replace(VECTORS_PATH)

    @classmethod
    def load(cls):
        if (
            not CHUNKS_PATH.exists()
            or not VECTORS_PATH.exists()
        ):
            return cls()

        chunks = json.loads(
            CHUNKS_PATH.read_text(
                encoding="utf-8"
            )
        )["chunks"]

        vectors = np.load(VECTORS_PATH)

        if len(chunks) != len(vectors):
            log.error(
                "store is inconsistent (%d chunks, %d vectors) "
                "— starting empty",
                len(chunks),
                len(vectors),
            )
            return cls()

        return cls(
            chunks=chunks,
            vectors=vectors,
        )