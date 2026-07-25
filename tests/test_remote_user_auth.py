# tests/test_remote_user_auth.py
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from api.auth import RemoteUserMiddleware

SECRET = "proxy-shared-secret-xyz"


def _client(monkeypatch, password="secret", trust="1", fwd_secret=SECRET):
    monkeypatch.setenv("OPEN_NOTEBOOK_PASSWORD", password)
    # Header-trust is fail-closed: OFF unless explicitly opted in AND a proof-of-proxy
    # secret is configured. Tests set these to exercise the trust path.
    if trust is None:
        monkeypatch.delenv("TRUST_REMOTE_USER_HEADER", raising=False)
    else:
        monkeypatch.setenv("TRUST_REMOTE_USER_HEADER", trust)
    if fwd_secret is None:
        monkeypatch.delenv("SSO_FORWARD_AUTH_SECRET", raising=False)
    else:
        monkeypatch.setenv("SSO_FORWARD_AUTH_SECRET", fwd_secret)

    async def home(request):
        return PlainTextResponse(getattr(request.state, "user", "nouser"))

    app = Starlette(routes=[Route("/x", home)])
    app.add_middleware(RemoteUserMiddleware)
    return TestClient(app)


def test_remote_user_with_proxy_secret_authenticates(monkeypatch):
    # Opt-in ON + correct proof-of-proxy secret + valid Remote-User -> trusted.
    c = _client(monkeypatch)
    r = c.get("/x", headers={"Remote-User": "a@b.co", "X-Forward-Auth-Secret": SECRET})
    assert r.status_code == 200 and r.text == "a@b.co"


def test_remote_user_without_secret_is_rejected(monkeypatch):
    # THE CRITICAL FIX: a forged Remote-User with NO proxy secret must NOT be
    # trusted — it falls back to the password check (401 without a valid password).
    c = _client(monkeypatch)
    assert c.get("/x", headers={"Remote-User": "attacker@evil"}).status_code == 401


def test_remote_user_with_wrong_secret_is_rejected(monkeypatch):
    # A guessed/wrong proxy secret must not unlock header-trust.
    c = _client(monkeypatch)
    r = c.get("/x", headers={"Remote-User": "attacker@evil", "X-Forward-Auth-Secret": "wrong"})
    assert r.status_code == 401


def test_header_trust_off_by_default_ignores_header(monkeypatch):
    # Fail-closed: without TRUST_REMOTE_USER_HEADER, even a correct secret + header
    # is ignored (behaves like plain PasswordAuthMiddleware).
    c = _client(monkeypatch, trust=None)
    r = c.get("/x", headers={"Remote-User": "a@b.co", "X-Forward-Auth-Secret": SECRET})
    assert r.status_code == 401


def test_no_secret_configured_ignores_header(monkeypatch):
    # Fail-closed: opt-in on but no SSO_FORWARD_AUTH_SECRET configured -> header ignored.
    c = _client(monkeypatch, fwd_secret=None)
    r = c.get("/x", headers={"Remote-User": "a@b.co", "X-Forward-Auth-Secret": SECRET})
    assert r.status_code == 401


def test_password_fallback_still_works(monkeypatch):
    # No header path -> valid password still authenticates (direct/non-proxied access).
    c = _client(monkeypatch)
    assert c.get("/x", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert c.get("/x").status_code == 401
