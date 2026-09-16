import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlms.config import pg_dsn

url = pg_dsn(os.environ["DATABASE_URL"])

with psycopg.connect(url, autocommit=True) as conn:
    conn.execute("DELETE FROM facts")
    conn.execute("DELETE FROM timeline_events")
    conn.execute("DELETE FROM session_log")
    conn.execute("DELETE FROM skills")
    print("done")
