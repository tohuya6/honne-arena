"""Static mounts: /datasets serves data/public, and / still serves the web UI."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_datasets_served_from_public_dir(cfg, client):
    public = Path(cfg.public_dir)
    assert public.is_dir()                            # created when missing
    (public / "lie").mkdir()
    (public / "lie" / "items.csv").write_text("id\nt1\n")
    r = client.get("/datasets/lie/items.csv")
    assert r.status_code == 200 and r.text == "id\nt1\n"
    assert client.get("/datasets/nope.csv").status_code == 404


def test_datasets_cannot_escape_the_public_dir(cfg, client, tmp_path):
    (tmp_path / "secret.txt").write_text("hidden")
    assert client.get("/datasets/../secret.txt").status_code == 404
    assert client.get("/datasets/%2e%2e/secret.txt").status_code == 404


def test_web_root_still_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_api_routes_win_over_static(cfg, db):
    c = TestClient(create_app(cfg, db=db))
    assert c.get("/api/season").json()["name"] == "Season 1"
    assert c.get("/health").json()["ok"] is True
