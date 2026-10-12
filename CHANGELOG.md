# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

First public version (0.1.0).

### Added

- MCP server with ten tools over four memory layers: semantic facts, episodic timeline events,
  procedural skills and session-scoped working memory.
- Fact versioning with conflict detection by cosine similarity, run in a `SERIALIZABLE`
  transaction, and `revert_fact`.
- Storage in Postgres (pgvector + TimescaleDB) with HNSW indexes, and Redis for working memory.
- Transports: `stdio` (default), `sse` and `streamable-http`, with a Host/Origin allowlist and an
  optional shared-secret gate (`MCP_SHARED_SECRET`, required on a non-loopback bind).
- Docker Compose setup: Postgres and Redis by default, the server itself behind the `app` profile.
- One Alembic migration, `001_initial_schema.py`, that creates the whole schema.
- CI: ruff, mypy, unit and integration tests, and a documentation consistency check
  (`scripts/check_docs.py`).
