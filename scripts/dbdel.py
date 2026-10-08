import os
import sys
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlms.config import pg_dsn


def main() -> None:
    url = pg_dsn(os.environ["DATABASE_URL"])
    target = conninfo_to_dict(url)
    dbname = target.get("dbname", "")
    print(
        "This deletes all rows from facts, timeline_events, session_log and skills "
        f"in {dbname!r} on {target.get('host', 'localhost')}."
    )
    if "--yes" not in sys.argv and input("Type the database name to continue: ") != dbname:
        sys.exit("aborted")

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM timeline_events")
        conn.execute("DELETE FROM session_log")
        conn.execute("DELETE FROM skills")
    print("done")


if __name__ == "__main__":
    main()
