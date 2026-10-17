"""Shared pgvector helpers."""

from __future__ import annotations


def vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"


# pgvector HNSW is approximate: it pulls its ef_search candidate window (default 40)
# before any WHERE filter runs, so a selective filter can leave fewer than LIMIT matches
# even when plenty exist in the table. The iterative scan keeps expanding the search until
# LIMIT is filled (or hnsw.max_scan_tuples is hit). SET LOCAL needs an explicit transaction.
ITERATIVE_SCAN = "SET LOCAL hnsw.iterative_scan = 'strict_order'"
