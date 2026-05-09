# Key Architectural Decisions — MLMS

Each entry records what was decided, why, and which alternatives were rejected.

Dates are the days the decisions were made. Commit history before 2026-08 was reconstructed afterwards, so early dates are approximate. Entry numbers are stable identifiers, not chronological order; gaps are intentional.

---

## Decision 1: EMBEDDING_DIM = 1536 (Matryoshka truncation)

**Decided:** 2026-05-09

**Decision:** `EMBEDDING_DIM = 1536` instead of the native 4096 of Qwen3-Embedding-8B.

**Rationale:** pgvector 0.8.2 limits HNSW and IVFFlat indexes on `vector` to 2000 dimensions, and `halfvec` to 4000 ([pgvector README](https://github.com/pgvector/pgvector)). Neither fits 4096. Qwen3-Embedding-8B supports Matryoshka Representation Learning with output dimensions from 32 to 4096 ([model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B); technique: [Kusupati et al., 2022](https://arxiv.org/abs/2205.13147)), so its vectors can be truncated to a prefix with a limited loss of quality. Truncating to 1536 brings the column under the 2000-dimension limit and makes an HNSW index possible (see Decision 2). The recall loss from truncation has not been measured in this project.

**Alternatives:**
- Sequential scan with no index on `vector(4096)`: exact (recall 100%), but query cost grows linearly with table size.
- `halfvec(4096)`: the 4000-dimension limit is below 4096.
- Expression index on a 1536-dimension prefix (`subvector(embedding, 1, 1536)`, documented in the pgvector README for models that support it): a valid option, not chosen. It keeps the full 4096 dimensions in storage (16392 bytes per row versus 6152 at 1536, from the README formula `4 * dimensions + 8`), and every query has to use the same expression to hit the index.

**Affected files:** `alembic/versions/006_change_embedding_dim.py`

**See also:** Decision 2.

---

## Decision 2: HNSW index on embedding columns

**Decided:** 2026-05-09

**Decision:** HNSW index on `facts.embedding` and `session_log.embedding`.

**Rationale:** without an ANN index, similarity search is a sequential scan, so its cost grows linearly with table size. The schema initially used `VECTOR(4096)` with no index (the first versions of `002_create_facts.py` and `004_create_session_log.py`). IVFFlat was tried first and hit the same 2000-dimension limit as HNSW, so no ANN index was possible at 4096 dimensions. Truncating to 1536 (Decision 1) removed the limit and allowed the HNSW index.

**Alternatives:**
- Sequential scan: exact, but linear cost.
- IVFFlat: same 2000-dimension limit at 4096.

**Trade-offs:** HNSW search is approximately logarithmic in table size, at the price of higher memory use and a slower index build. At the expected volume (up to about 100K rows, an estimate rather than a measurement) this is acceptable.

**Affected files:** `alembic/versions/002_create_facts.py`, `alembic/versions/004_create_session_log.py`, `alembic/versions/006_change_embedding_dim.py`

**See also:** Decision 1.

---
