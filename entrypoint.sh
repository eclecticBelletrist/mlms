#!/bin/sh
set -e

# depends_on: condition: service_healthy doesn't close the race with Postgres: the
# first `alembic upgrade head` can fail with connection refused although postgres is
# already marked healthy (cause not established). A bounded retry here means we
# don't lean on the container's restart policy (noisy crash-loop traceback, no cap)
# to paper over that gap.
attempt=0
max_attempts=10
until alembic upgrade head; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "entrypoint: alembic upgrade head failed after $max_attempts attempts, giving up" >&2
        exit 1
    fi
    echo "entrypoint: alembic upgrade head failed (attempt $attempt/$max_attempts), retrying in 2s..." >&2
    sleep 2
done

exec python -m mlms.server
