import json
from pathlib import Path

import numpy as np
import psycopg


BASE_DIR = Path(__file__).parent
CHUNKS_PATH = BASE_DIR / "data" / "store" / "chunks.json"
VECTORS_PATH = BASE_DIR / "data" / "store" / "vectors.npy"

DB_NAME = "askme_db"
DB_USER = "askme_app"
DB_HOST = "localhost"
DB_PORT = 5432


def load_store():
    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    vectors = np.load(VECTORS_PATH)

    if len(chunks) != len(vectors):
        raise RuntimeError(
            f"Store mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
        )

    if vectors.ndim != 2 or vectors.shape[1] != 384:
        raise RuntimeError(
            f"Expected vectors with shape (N, 384), got {vectors.shape}"
        )

    if vectors.dtype != np.float32:
        raise RuntimeError(
            f"Expected float32 vectors, got {vectors.dtype}"
        )

    return chunks, vectors


def connection_string():
    return (
        f"host={DB_HOST} "
        f"port={DB_PORT} "
        f"dbname={DB_NAME} "
        f"user={DB_USER}"
    )


def source_key(chunk):
    return (
        chunk["source_id"],
        chunk.get("source_title", "Untitled"),
        chunk.get("source_type", "document"),
        chunk.get("origin", "paste"),
    )


def migrate():
    chunks, vectors = load_store()

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loaded vectors: shape={vectors.shape}, dtype={vectors.dtype}")

    # Preserve the first-seen source order from the JSON store.
    source_keys = []
    seen = set()

    for chunk in chunks:
        key = source_key(chunk)
        if key not in seen:
            seen.add(key)
            source_keys.append(key)

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            # Remove previous copies of the sources belonging to this migration.
            # ON DELETE CASCADE removes their chunks automatically.
            for _, title, source_type, origin in source_keys:
                cur.execute(
                    """
                    DELETE FROM sources
                    WHERE title = %s
                      AND type = %s
                      AND origin = %s
                    """,
                    (title, source_type, origin),
                )

            source_map = {}

            # Insert sources first.
            for original_source_id, title, source_type, origin in source_keys:
                cur.execute(
                    """
                    INSERT INTO sources
                        (title, type, origin, url, feed_key, synced_at)
                    VALUES
                        (%s, %s, %s, NULL, NULL, NULL)
                    RETURNING id
                    """,
                    (title, source_type, origin),
                )

                db_source_id = cur.fetchone()[0]
                source_map[
                    (original_source_id, title, source_type, origin)
                ] = db_source_id

            # Insert chunks while preserving the original JSON order
            # through source + ordinal.
            for index, chunk in enumerate(chunks):
                key = source_key(chunk)
                db_source_id = source_map[key]

                vector = vectors[index].astype(np.float32).tolist()
                vector_literal = "[" + ",".join(map(str, vector)) + "]"

                cur.execute(
                    """
                    INSERT INTO chunks
                        (source_id, ordinal, text, tags, embedding)
                    VALUES
                        (%s, %s, %s, %s::jsonb, %s::vector)
                    """,
                    (
                        db_source_id,
                        index,
                        chunk["text"],
                        json.dumps(chunk.get("tags", [])),
                        vector_literal,
                    ),
                )

        conn.commit()

    print("Migration committed successfully.")


def verify():
    chunks, original_vectors = load_store()

    # Build the expected order using the same source + ordinal information
    # used during migration.
    expected = {
        (
            chunk["source_id"],
            chunk.get("source_title", "Untitled"),
            chunk.get("source_type", "document"),
            chunk.get("origin", "paste"),
            index,
        ): vectors
        for index, (chunk, vectors) in enumerate(zip(chunks, original_vectors))
    }

    source_keys = []
    seen = set()

    for chunk in chunks:
        key = source_key(chunk)
        if key not in seen:
            seen.add(key)
            source_keys.append(key)

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM sources")
            source_count = cur.fetchone()[0]

            migrated = []

            for original_source_id, title, source_type, origin in source_keys:
                cur.execute(
                    """
                    SELECT c.ordinal, c.embedding::text
                    FROM chunks AS c
                    JOIN sources AS s
                      ON s.id = c.source_id
                    WHERE s.title = %s
                      AND s.type = %s
                      AND s.origin = %s
                    ORDER BY c.ordinal
                    """,
                    (title, source_type, origin),
                )

                rows = cur.fetchall()

                for ordinal, embedding_text in rows:
                    migrated.append((ordinal, embedding_text))

    print(f"PostgreSQL sources: {source_count}")
    print(f"PostgreSQL chunks: {chunk_count}")
    print(f"Expected chunks: {len(chunks)}")

    if chunk_count != len(chunks):
        raise RuntimeError(
            f"Chunk count mismatch: expected={len(chunks)}, "
            f"PostgreSQL={chunk_count}"
        )

    if len(migrated) != len(original_vectors):
        raise RuntimeError(
            f"Vector count mismatch: expected={len(original_vectors)}, "
            f"PostgreSQL={len(migrated)}"
        )

    # Fetch vectors in the exact ordinal order used by the migration.
    migrated_vectors = []

    for ordinal, embedding_text in migrated:
        values = [
            float(value)
            for value in embedding_text.strip("[]").split(",")
        ]
        migrated_vectors.append(values)

    migrated_vectors = np.asarray(migrated_vectors, dtype=np.float32)

    max_error = float(
        np.max(
            np.abs(
                original_vectors.astype(np.float32)
                - migrated_vectors
            )
        )
    )

    print(f"Round-trip vectors: shape={migrated_vectors.shape}")
    print(f"Maximum float32 round-trip error: {max_error}")

    if max_error > 1e-6:
        raise RuntimeError(
            f"Vector round-trip verification failed: max error={max_error}"
        )

    print("Vector round-trip verification PASSED.")


if __name__ == "__main__":
    migrate()
    verify()
