import os
try:
    from jose import jwt
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False
    jwt = None

import hmac
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from open_notebook.utils.encryption import get_secret_from_env


class PasswordAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to check password authentication for all API requests.
    Fail-closed: if OPEN_NOTEBOOK_PASSWORD is not set, every request is denied
    (500) rather than served unauthenticated (PMOVES hardening — upstream skips
    auth when unset).
    Supports Docker secrets via OPEN_NOTEBOOK_PASSWORD_FILE.
    """

    def __init__(
        self, app: ASGIApp, excluded_paths: Optional[list[str]] = None
    ) -> None:
        super().__init__(app)
        self.password = get_secret_from_env("OPEN_NOTEBOOK_PASSWORD")
        self.excluded_paths: list[str] = excluded_paths or [
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Fail-closed: if no password is configured, deny all requests
        if not self.password:
            return JSONResponse(
                status_code=500,
                content={"detail": "OPEN_NOTEBOOK_PASSWORD not configured — authentication unavailable"},
            )

        # Skip authentication for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Skip authentication for CORS preflight requests (OPTIONS)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Check authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Expected format: "Bearer {password}"
        try:
            scheme, credentials = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                raise ValueError("Invalid authentication scheme")
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authorization header format"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check password (constant-time to avoid a timing side-channel)
        if not secrets.compare_digest(
            credentials.encode("latin-1"), self.password.encode("utf-8")
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid password"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Password is correct, proceed with the request
        response = await call_next(request)
        return response


def _valid_remote_user(value: str) -> bool:
    """Reject empty / over-long / non-printable Remote-User values before trusting
    them (a forged header could carry control chars or absurd length)."""
    return bool(value) and len(value) <= 320 and value.isprintable()


class RemoteUserMiddleware(PasswordAuthMiddleware):
    """Trust the reverse-proxy `Remote-User` header ONLY with proof the request
    actually transited the PMOVES Traefik forward-auth edge. Trusting a bare
    header is an auth bypass — any peer that reaches this app off-proxy could
    forge `Remote-User`. So the header is honored only when ALL hold:

      1. TRUST_REMOTE_USER_HEADER is explicitly enabled (fail-closed default —
         without it this behaves exactly like PasswordAuthMiddleware), AND
      2. a proof-of-proxy secret is configured (SSO_FORWARD_AUTH_SECRET), AND
      3. the request carries X-Forward-Auth-Secret matching it (constant-time) —
         the SSO edge injects this and Traefik overwrites any client value, so a
         peer bypassing the proxy cannot produce it, AND
      4. the Remote-User value is well-formed.

    Any failure falls back to the inherited password check — nothing is ever
    served unauthenticated."""

    def __init__(
        self, app: ASGIApp, excluded_paths: Optional[list[str]] = None
    ) -> None:
        super().__init__(app, excluded_paths)
        raw = (get_secret_from_env("TRUST_REMOTE_USER_HEADER") or "").strip().lower()
        self.trust_header = raw in ("1", "true", "yes", "on")
        self.forward_auth_secret = get_secret_from_env("SSO_FORWARD_AUTH_SECRET")

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if self.trust_header and self.forward_auth_secret:
            remote_user = request.headers.get("Remote-User", "")
            proxy_secret = request.headers.get("X-Forward-Auth-Secret", "")
            if (
                _valid_remote_user(remote_user)
                and hmac.compare_digest(proxy_secret, self.forward_auth_secret)
            ):
                request.state.user = remote_user
                return await call_next(request)
        # Not proven to come from the proxy — fall back to the password check.
        return await super().dispatch(request, call_next)


# Optional: HTTPBearer security scheme for OpenAPI documentation
security = HTTPBearer(auto_error=False)


def check_api_password(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """
    Utility function to check API password.
    Can be used as a dependency in individual routes if needed.
    Supports Docker secrets via OPEN_NOTEBOOK_PASSWORD_FILE.
    Fail-closed: raises 500 if OPEN_NOTEBOOK_PASSWORD is not configured, 401 if
    credentials are missing or don't match the configured password.
    """
    password = get_secret_from_env("OPEN_NOTEBOOK_PASSWORD")

    # Fail-closed: if no password is configured, deny access
    if not password:
        raise HTTPException(
            status_code=500,
            detail="OPEN_NOTEBOOK_PASSWORD not configured — authentication unavailable",
        )

    # No credentials provided
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check password
    if credentials.credentials != password:
        raise HTTPException(
            status_code=401,
            detail="Invalid password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


class SupabaseJWTMiddleware(BaseHTTPMiddleware):
    """
    Validates Supabase-issued JWT Bearer tokens as an alternative to
    password auth. Enabled when SUPABASE_JWT_SECRET is set.

    Pattern sourced from PMOVES-BoTZ features/mcp_bridge/auth.py (PR #181,
    commit 1ba7a05). Validates HS256 JWTs against the same JWT_SECRET
    the GoTrue auth server uses. Rejects anon keys; accepts authenticated
    and service_role tokens.

    Falls through to PasswordAuthMiddleware when:
    - SUPABASE_JWT_SECRET is not set
    - No Bearer token in the Authorization header
    - Token validation fails
    """

    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.secret = os.getenv("SUPABASE_JWT_SECRET", "")
        self.algorithm = "HS256"
        self.excluded_paths = excluded_paths or [
            "/", "/health", "/docs", "/openapi.json", "/redoc",
        ]

    async def dispatch(self, request: Request, call_next):
        if not self.secret or not HAS_JOSE:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return await call_next(request)

        token = auth_header.replace("Bearer ", "", 1)
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                options={"verify_signature": True, "verify_aud": False, "verify_exp": True},
            )
        except Exception:
            return await call_next(request)  # fall through to password auth

        role = payload.get("role", "")
        if role == "anon":
            return await call_next(request)  # anon tokens don't grant access

        # Valid authenticated or service_role token — mark request as authenticated
        request.state.authenticated = True
        request.state.user_id = payload.get("sub", "")
        request.state.user_role = role
        return await call_next(request)


try:
    from jose import jwt
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False
    jwt = None
