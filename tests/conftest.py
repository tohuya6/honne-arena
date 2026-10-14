import pytest
from fastapi.testclient import TestClient

from app.challenges import steering
from app.config import Settings
from app.db import Database
from app.main import create_app


@pytest.fixture()
def cfg(tmp_path):
    # Never a real NDIF key in tests. ndif_enabled is flipped per test when needed,
    # and steering tests always replace the scorer factory with a fake.
    return Settings(_env_file=None, db_path=":memory:", admin_token="admin-secret",
                    ip_hash_salt="salt", ndif_enabled=False, ndif_api_key="",
                    cache_dir=str(tmp_path / "cache"))


@pytest.fixture()
def db():
    d = Database(":memory:")
    d.migrate()
    d.create_season("Season 1")
    return d


@pytest.fixture(autouse=True)
def _reset_scorers():
    steering.reset_scorers()
    yield
    steering.reset_scorers()


@pytest.fixture()
def client(cfg, db):
    return TestClient(create_app(cfg, db=db))


ADMIN = {"X-Admin-Token": "admin-secret"}
