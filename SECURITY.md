# Security policy

MLMS is early-stage (pre-1.0) software for self-hosted, single-user use (one person with a few
trusted clients). Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting: on the
repository page, open the **Security** tab and choose **Report a vulnerability**. Include what you
found, how to reproduce it, and which version or commit you tested.

This is a personal project, so there is no service-level agreement: I will respond as soon as I can.

## Scope

The server's trust model is described in the README ("What this is" and "Homelab/VPS hardening")
and in `KEY_DECISIONS.md` (Decisions 4, 22 and 23). In short:

- The `stdio` transport has no network port and no authentication.
- `sse` and `streamable-http` are protected by one shared secret (`MCP_SHARED_SECRET`). The
  secret is required on a non-loopback bind and optional on a loopback one. Every holder has full
  read and write access; there is no per-user isolation.
- Redis in the shipped `docker-compose.yml` has no authentication, and the default Postgres
  credentials are `mlms`/`mlms`. The defaults are meant for a local setup.

Reports about behavior that the documentation lists as a known limitation are still welcome, but
they are treated as improvements rather than vulnerabilities.
