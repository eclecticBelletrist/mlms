import os
import psycopg

url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)

with psycopg.connect(url, autocommit=True) as conn:
    conn.execute("DELETE FROM facts")
    conn.execute("DELETE FROM timeline_events")
    conn.execute("DELETE FROM session_log")
    conn.execute("DELETE FROM skills")
    print("done")
