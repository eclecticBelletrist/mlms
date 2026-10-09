"""Consistency checks for the documentation. Exit code 1 if any check fails.

    python scripts/check_docs.py

Checks:
  - KEY_DECISIONS.md: unique decision numbers, a valid `Decided` date, `Last updated` not before it
  - repo paths written in backticks in the docs exist
  - "N tools" claims in README.md and CLAUDE.md match the tools registered in src/mlms/server.py
"""

import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [
    ROOT / "README.md",
    ROOT / "CLAUDE.md",
    ROOT / "KEY_DECISIONS.md",
    *sorted((ROOT / "docs").glob("*.md")),
]
PATH_PREFIXES = ("src/", "tests/", "scripts/", "docs/", "alembic/", ".github/")
errors: list[str] = []


def fail(where: str, msg: str) -> None:
    errors.append(f"{where}: {msg}")


def parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def check_key_decisions() -> None:
    text = (ROOT / "KEY_DECISIONS.md").read_text(encoding="utf-8")
    parts = re.split(r"^## Decision ", text, flags=re.M)[1:]
    seen: set[str] = set()
    for part in parts:
        number = part.split(":", 1)[0]
        where = f"KEY_DECISIONS.md Decision {number}"
        if number in seen:
            fail(where, "duplicate number")
        seen.add(number)
        decided = re.search(r"^\*\*Decided:\*\* (\S+)$", part, re.M)
        decided_on = parse_date(decided.group(1)) if decided else None
        if not decided_on:
            fail(where, "missing or invalid `Decided` date")
            continue
        updated = re.search(r"^\*\*Last updated:\*\* (\d{4}-\d\d-\d\d)", part, re.M)
        if updated:
            updated_on = parse_date(updated.group(1))
            if not updated_on or updated_on < decided_on:
                fail(where, "`Last updated` is invalid or earlier than `Decided`")


def check_paths() -> None:
    for doc in DOCS:
        for line_no, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for ref in re.findall(r"`([^`\s]+)`", line):
                path = ref.split(":", 1)[0].rstrip(".,;)")
                if not path.startswith(PATH_PREFIXES) or any(c in path for c in "*<>{}$"):
                    continue
                if not (ROOT / path).exists():
                    fail(f"{doc.relative_to(ROOT)}:{line_no}", f"`{path}` does not exist")


def check_tool_count() -> None:
    server = (ROOT / "src" / "mlms" / "server.py").read_text(encoding="utf-8")
    actual = len(re.findall(r"^@mcp\.tool\(", server, flags=re.M))
    for doc in (ROOT / "README.md", ROOT / "CLAUDE.md"):
        for line_no, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for claimed in re.findall(r"\b(\d+) tools\b", line):
                if int(claimed) != actual:
                    fail(
                        f"{doc.relative_to(ROOT)}:{line_no}",
                        f"claims {claimed} tools, server.py registers {actual}",
                    )


if __name__ == "__main__":
    check_key_decisions()
    check_paths()
    check_tool_count()
    for e in errors:
        print(e)
    print(f"{len(errors)} problem(s)" if errors else "docs OK")
    sys.exit(1 if errors else 0)
