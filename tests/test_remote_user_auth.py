# tests/test_remote_user_auth.py
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from api.auth import RemoteUserMiddleware


def _client(monkeypatch, password="secret"):
    monkeypatch.setenv("OPEN_NOTEBOOK_PASSWORD", password)

    async def home(request):
        return PlainTextResponse(getattr(request.state, "user", "nouser"))

    app = Starlette(routes=[Route("/x", home)])
    app.add_middleware(RemoteUserMiddleware)
    return TestClient(app)


def test_remote_user_header_authenticates(monkeypatch):
    # Header present -> authenticated by the SSO gateway; request.state.user set.
    c = _client(monkeypatch)
    r = c.get("/x", headers={"Remote-User": "a@b.co"})
    assert r.status_code == 200 and r.text == "a@b.co"


def test_no_header_no_password_denies(monkeypatch):
    # No header AND no/invalid password token -> 401 (fall back to password path).
    c = _client(monkeypatch)
    assert c.get("/x").status_code == 401
    assert c.get("/x", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_no_header_valid_password_allows(monkeypatch):
    # No header but valid password -> allowed (direct/non-proxied fallback works).
    c = _client(monkeypatch)
    r = c.get("/x", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
