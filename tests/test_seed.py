"""scripts/seed.py: loads fixture seeds, hashes flag files, never prints a flag,
and is safe to run twice."""

import importlib.util
import json
from pathlib import Path

from app import challenges
from app.db import Database

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "seeds"
FLAG = "test{fixture-only}"


def _seed_module():
    spec = importlib.util.spec_from_file_location("seed", ROOT / "scripts" / "seed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _private(tmp_path):
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "fixture-flag.txt").write_text(f"  {FLAG}\n")
    (tmp_path / "private" / "fixture-labels.json").write_text(json.dumps({"a": 1, "b": 0}))


def _run(tmp_path, *args):
    lines = []
    code = _seed_module().main(["--db", str(tmp_path / "arena.db"), "--root", str(tmp_path),
                                *args], out=lines.append)
    return code, "\n".join(lines)


def test_seed_loads_hashes_and_is_idempotent(tmp_path):
    _private(tmp_path)
    files = [str(FIXTURES / "ctf.json"), str(FIXTURES / "lie.json")]
    code, out = _run(tmp_path, "--season", "Fall 2026", *files)
    assert code == 1                                  # the broken entry is an error
    assert "created fixture-flag" in out and "created fixture-liar" in out
    assert "error   fixture-broken" in out
    assert FLAG not in out and "fixture-only" not in out

    db = Database(str(tmp_path / "arena.db"))
    season = db.get_active_season()
    assert season["name"] == "Fall 2026"
    flag = db.get_challenge(season["id"], "fixture-flag")
    assert FLAG not in flag["config"]
    ch = challenges.load(flag)
    assert ch.evaluate(FLAG).correct is True          # stripped before hashing
    assert flag["tier"] == "learn" and flag["points_initial"] == 200
    assert [(h["position"], h["body"], h["cost"]) for h in db.list_hints(flag["id"])] == [
        (1, "First hint.", 10), (2, "Second hint.", 20)]
    liar = db.get_challenge(season["id"], "fixture-liar")
    # The path is kept as written, never replaced by the file's contents.
    assert json.loads(liar["config"])["labels_file"] == "private/fixture-labels.json"

    code, out = _run(tmp_path, "--season", "Fall 2026", *files)
    assert "skipped fixture-flag" in out and "skipped fixture-liar" in out
    assert "created" not in out.replace("0 created", "")
    db2 = Database(str(tmp_path / "arena.db"))
    assert len(db2.list_challenges(season["id"], include_hidden=True)) == 2
    assert len(db2.list_hints(flag["id"])) == 2
    assert [s["name"] for s in db2.list_seasons()] == ["Fall 2026"]


def test_missing_flag_file_is_an_error(tmp_path):
    code, out = _run(tmp_path, "--season", "S", str(FIXTURES / "ctf.json"))
    assert code == 1 and "flag_file not found" in out


def test_needs_an_active_season(tmp_path):
    _private(tmp_path)
    code, out = _run(tmp_path, str(FIXTURES / "ctf.json"))
    assert code == 1 and "--season" in out


def test_missing_config_file_only_warns(tmp_path):
    (tmp_path / "private").mkdir()
    code, out = _run(tmp_path, "--season", "S", str(FIXTURES / "lie.json"))
    assert "created fixture-liar" in out
    assert "warning fixture-liar: config.labels_file not found" in out


def test_season_switch_reuses_existing(tmp_path):
    _private(tmp_path)
    _run(tmp_path, "--season", "A", str(FIXTURES / "ctf.json"))
    _run(tmp_path, "--season", "B", str(FIXTURES / "ctf.json"))
    _run(tmp_path, "--season", "A", str(FIXTURES / "ctf.json"))
    db = Database(str(tmp_path / "arena.db"))
    assert [(s["name"], s["active"]) for s in db.list_seasons()] == [("A", 1), ("B", 0)]
