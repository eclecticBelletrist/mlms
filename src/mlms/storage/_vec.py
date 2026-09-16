"""Shared pgvector literal serialization."""

from __future__ import annotations


def vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"
