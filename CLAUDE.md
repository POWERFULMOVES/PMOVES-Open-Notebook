@AGENTS.md

---

# PMOVES.AI Fork — Integration Notes

This is the PMOVES.AI-hardened fork of Open Notebook. The sections below are
PMOVES-specific and layer on top of the upstream agent rules imported above.

## Security Posture (PMOVES hardening)

- **Auth middleware fail-closed:** `PasswordAuthMiddleware` raises HTTP 500 when
  `OPEN_NOTEBOOK_PASSWORD` is unset (upstream skips auth when unset). See
  `api/auth.py`.
- **Remote-User SSO trust is proof-gated:** `RemoteUserMiddleware` honors the
  reverse-proxy `Remote-User` header only when `TRUST_REMOTE_USER_HEADER` is
  enabled AND a matching `X-Forward-Auth-Secret` proves the request transited the
  Traefik forward-auth edge (constant-time check against `SSO_FORWARD_AUTH_SECRET`).
  Any failure falls back to the password check — never served unauthenticated.
  Covered by `tests/test_remote_user_auth.py`.
- **SurrealDB credentials** use env-var substitution (`${SURREAL_PASSWORD:-...}`).
- **`/healthz` alias + `/metrics` Prometheus endpoint** are registered and
  excluded from auth.
- Credential encryption via Fernet (upstream); content processing sandboxed
  within the API container.

## CHIT & Geometry Bus Integration

**Status: No CHIT integration (by design).** Open Notebook is a SurrealDB-backed
knowledge base. It operates as a content storage/retrieval layer, not a geometry
producer/consumer, and has no NATS connection (communication is HTTP API from
DeepResearch and Notebook Sync). CGP data may flow in indirectly via DeepResearch
results, but Open Notebook does not parse or produce CGP.

**Related:** DeepResearch (`pmoves/services/deep-research/`) publishes research
results here and does interact with geometry subjects.

## TensorZero Provider Mode

When `NOTEBOOK_PROVIDER_MODE=tensorzero` (default in PMOVES), all LLM/embedding
calls route through the TensorZero gateway via a managed `openai_compatible`
credential. The `pmoves_provider/` module bootstraps this on API startup (see the
`bootstrap_tensorzero()` call in `api/main.py` lifespan).

## PMOVES.AI Skill Hints

**Primary Skills:** `/search:deepresearch`, `/db:query`, `/deploy:up`, `/health:quick`
**Context Files:** `services-catalog.md`, `nats-subjects.md`
**Domain Tags:** `knowledge`, `documents`
**Context Tier:** 2 (On-Demand — Major Subsystem)
