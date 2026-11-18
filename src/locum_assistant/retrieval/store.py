"""PostgreSQL storage for documents and chunks, with two indexes over the same rows:
a GIN full-text index (keyword search) and an HNSW pgvector index (semantic search)."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from locum_assistant.ingest.chunk import Chunk
from locum_assistant.ingest.sources import Source


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    title: str
    url: str
    heading: str
    page: int | None
    text: str
    score: float = 0.0

    @property
    def citation(self) -> str:
        where = f"p. {self.page}" if self.page else self.heading
        return f"{self.title}, {where}" if where and where != self.title else self.title


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,
    title       text NOT NULL,
    url         text NOT NULL,
    publisher   text NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id        bigserial PRIMARY KEY,
    doc_id    text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord       int  NOT NULL,
    heading   text NOT NULL DEFAULT '',
    page      int,
    text      text NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || text)) STORED,
    embedding vector({dim}) NOT NULL,
    UNIQUE (doc_id, ord)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
-- BM25 needs each chunk's length in words and each word's document frequency
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS len int NOT NULL DEFAULT 0;
-- schema-qualified: Postgres 17 refreshes materialized views with a locked-down search_path
CREATE MATERIALIZED VIEW IF NOT EXISTS lexeme_df AS
    SELECT word AS lexeme, ndoc AS df FROM ts_stat('SELECT tsv FROM public.chunks');
CREATE UNIQUE INDEX IF NOT EXISTS lexeme_df_idx ON lexeme_df (lexeme);
"""

HIT_COLUMNS = "c.id, c.doc_id, d.title, d.url, c.heading, c.page, c.text"

# Keyword search scored with BM25, computed in SQL.
#
# Postgres's own ts_rank has no IDF: "pharmacy" (in nearly every chunk) counts as much as
# "absent" (in a handful), so common words swamped the ranking (passage recall@5 0.46 on the dev
# set). BM25 weights each query word by how rare it is (IDF), saturates repeated words (k1) and
# normalises for chunk length (b):
#     score = sum over query words of  idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))
#     idf   = ln(1 + (N - df + 0.5) / (df + 0.5))
# The question's words are OR-ed, so the GIN index still finds every chunk with any of them.
BM25_K1, BM25_B = 1.2, 0.75
KEYWORD_SQL = f"""
WITH q AS (
    SELECT DISTINCT l AS lexeme FROM unnest(tsvector_to_array(to_tsvector('english', %(q)s))) l
),
query AS (
    SELECT array_to_string(ARRAY(SELECT quote_literal(lexeme) FROM q), ' | ')::tsquery AS tsq
),
corpus AS (SELECT count(*)::float AS n, avg(len)::float AS avglen FROM chunks),
terms AS (
    SELECT q.lexeme, ln(1 + (corpus.n - d.df + 0.5) / (d.df + 0.5)) AS idf
    FROM q JOIN lexeme_df d ON d.lexeme = q.lexeme, corpus
),
scored AS (
    SELECT c.id, sum(t.idf * u.tf * ({BM25_K1} + 1)
               / (u.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * c.len / corpus.avglen))) AS score
    FROM chunks c, query, corpus,
         LATERAL (SELECT lexeme, array_length(positions, 1) AS tf FROM unnest(c.tsv)) u
         JOIN terms t ON t.lexeme = u.lexeme
    WHERE c.tsv @@ query.tsq
    GROUP BY c.id
)
SELECT {HIT_COLUMNS}, s.score
FROM scored s JOIN chunks c ON c.id = s.id JOIN documents d ON d.id = c.doc_id
ORDER BY s.score DESC
LIMIT %(limit)s
"""

VECTOR_SQL = f"""
SELECT {HIT_COLUMNS}, 1 - (c.embedding <=> %s) AS score
FROM chunks c JOIN documents d ON d.id = c.doc_id
ORDER BY c.embedding <=> %s
LIMIT %s
"""


class Store:
    def __init__(self, database_url: str, dim: int) -> None:
        self.database_url = database_url
        self.dim = dim

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.database_url, autocommit=False) as conn:
            register_vector(conn)
            yield conn

    def init_schema(self) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(SCHEMA.format(dim=self.dim))

    def replace_document(
        self, source: Source, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Swap a document's chunks in one transaction, so a search never sees half a document."""
        import numpy as np

        with self.connect() as conn:
            conn.execute(
                """INSERT INTO documents (id, title, url, publisher) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, url = EXCLUDED.url,
                   publisher = EXCLUDED.publisher, ingested_at = now()""",
                (source.id, source.title, source.url, source.publisher),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (source.id,))
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (doc_id, ord, heading, page, text, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    [(source.id, c.ord, c.heading, c.page, c.text, np.array(e, dtype=np.float32))
                     for c, e in zip(chunks, embeddings, strict=True)],
                )
            conn.execute(
                """UPDATE chunks SET len = (SELECT coalesce(sum(array_length(positions, 1)), 0)
                                           FROM unnest(tsv)) WHERE doc_id = %s""",
                (source.id,))

    def refresh_stats(self) -> None:
        """Recompute word document frequencies for BM25. Run after ingesting."""
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute("REFRESH MATERIALIZED VIEW lexeme_df")

    def keyword_search(self, query: str, limit: int) -> list[Hit]:
        with self.connect() as conn:
            rows = conn.execute(KEYWORD_SQL, {"q": query, "limit": limit}).fetchall()
        return [Hit(*row) for row in rows]

    def vector_search(self, embedding: list[float], limit: int) -> list[Hit]:
        import numpy as np

        v = np.array(embedding, dtype=np.float32)
        with self.connect() as conn:
            rows = conn.execute(VECTOR_SQL, (v, v, limit)).fetchall()
        return [Hit(*row) for row in rows]

    def stats(self) -> list[tuple[str, int]]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT d.id, count(c.id) FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
                "GROUP BY d.id ORDER BY d.id"
            ).fetchall()
