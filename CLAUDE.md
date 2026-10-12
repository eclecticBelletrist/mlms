# CLAUDE.md — MLMS

MLMS is a self-hosted MCP server giving LLM agents persistent cross-session memory across 4 CoALA layers.

---

## Behavioral Rules

1. **Don't assume** — ambiguous intent gets a question, not a guess. Never silently pick an interpretation.
2. **Minimum code** — solve only what was asked. No speculative features, no single-use abstractions. If 50 lines works, don't write 200.
3. **Surgical changes** — touch only what the task requires. Don't refactor adjacent code, don't "improve" comments. Unrelated dead code gets flagged, not deleted.
4. **Goal-driven loops** — when given success criteria, verify them before stopping. Don't report done until tests pass.
5. **Verify, don't trust** — re-check any claim (agent report, "already fixed", assumed state) against live state before acting on it. "Fixed earlier this session" is never sufficient — verify fresh, every time.
6. **No irreversible action without a fresh yes** — commit, push, force-push, history-rewrite: explicit per-instance approval required, even for a verified fix, even if a similar action was approved minutes ago. Approval doesn't carry forward to the next instance of the same action.

---

## Key Constants — never change without explicit instruction

```python
EMBEDDING_DIM            = 1536       # Matryoshka truncation of Qwen3-Embedding-8B (4096→1536), pgvector HNSW limit is 2000
EMBEDDING_MODEL          = "qwen3-embedding-8b"
EMBEDDING_FALLBACK       = "qwen3-embedding-0.6b"

COSINE_CONFLICT_THRESHOLD = 0.96      # calibration needed

REDIS_SESSION_TTL        = 172800
REDIS_KEY_PATTERN        = "session:{chat_id}"

LATENCY_P99_TARGET_MS    = 200        # cached path only
LATENCY_SESSION_CTX_MS   = 5         # get_session_context() hard limit
PRECISION_AT_5_TARGET    = 0.80      # Recall@5 metric — renamed from precision@5, sparse ground truth (1-2 relevant per scenario)

MEMORY_CONTENT_MAX_LENGTH = 10_000            # memorize(content) hard reject limit — sized with margin for the fallback model served locally, not the primary (Decision 15)
```

---

## memorize() Routing

```mermaid
flowchart TD
  IN[memorize called] --> T{type?}

  T -->|fact| EMB[embed content via Qwen3]
  EMB --> CC{cosine >= 0.96\nwith any is_current=TRUE\nfact in same project?}
  CC -->|yes — conflict| UPD[UPDATE old: is_current=FALSE\nINSERT new: is_current=TRUE, version+1\nwrap in transaction]
  CC -->|no| INS[INSERT new: is_current=TRUE, version=1]
  UPD --> CH{chat_id in\nmetadata?}
  INS --> CH
  CH -->|yes| RC[update Redis session context]
  CH -->|no| DONE0[return, no Redis write]

  T -->|event| TLE[INSERT timeline_events\nno embedding needed]
  TLE --> ECH{chat_id in\nmetadata?}
  ECH -->|yes| SLE[INSERT session_log entry]
  SLE --> RE[update Redis]
  ECH -->|no| DONE1[return, no session_log/Redis write]

  T -->|skill| SKI[INSERT skills\nincrement version if name exists]
  SKI --> SCH{chat_id in\nmetadata?}
  SCH -->|yes| SLS[INSERT session_log entry]
  SLS --> RES[update Redis]
  SCH -->|no| DONE2[return, no session_log/Redis write]

  T -->|session_log| EMBL[embed label]
  EMBL --> SLL[INSERT session_log]
  SLL --> RL[update Redis last_activity]
```

---

## Key Decisions Log

All non-obvious architectural decisions (index type changes, schema trade-offs, library limitations)
must be recorded in `KEY_DECISIONS.md` at the project root.
Format: `Decided` date (plus `Last updated` when the entry text changes later), decision, reason,
alternatives rejected, affected files, trade-offs.

**Write the final state, not the process.** A decision entry records what was decided and why —
not a transcript of every attempt or false start that led there ("first thought it was X, then
it turned out to be Y, then testing surfaced Z"). If it took three tries to land the fix, the
entry says what the bug was, why it happened, and how it was fixed — not the sequence of who
found what. `KEY_DECISIONS.md` shows how the project thinks and what got fixed, not a log of how
the work went.

**The private working notes file is local-only — never committed, never public.** Any file that
IS committed/public (`KEY_DECISIONS.md`, `docs/known-issues.md`, `README.md`, code comments, commit
messages) must never reference it by name or point a reader at it. If a point from it
needs to survive into a public doc, restate the substance inline instead of citing the file — a
committed doc pointing at a file that doesn't exist in the clone is broken for anyone reading it.
