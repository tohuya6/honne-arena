"""Load starter challenges from seeds/*.json into the active season.

    python scripts/seed.py                         # every seeds/*.json, active season
    python scripts/seed.py --season "Fall 2026"    # create (or reuse) and activate it
    python scripts/seed.py seeds/ctf.json --db data/arena.db

Run it from the repository root, since seed paths such as data/private/... are
relative to it (or pass --root). Each file is a JSON list of challenge objects with the
admin API's fields (see seeds/README.md). They go through the same validation as
POST /api/admin/challenges.

Safe to run again: a slug that already exists in the season is skipped and reported.
A `flag_file` is read, stripped, and hashed. Its contents are never printed.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError as PydanticError  # noqa: E402

from app.admin import ChallengeIn, create_challenge  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.errors import SubmitError  # noqa: E402


def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _prepare(obj: dict, root: Path) -> tuple[dict, list[str]]:
    """Turn a seed object into admin API fields. Returns (fields, warnings)."""
    obj = dict(obj)
    warnings = []
    flag_file = obj.pop("flag_file", None)
    if flag_file:
        p = _resolve(flag_file, root)
        if not p.is_file():
            raise ValueError(f"flag_file not found: {flag_file}")
        flag = p.read_text(encoding="utf-8").strip()
        if not flag:
            raise ValueError(f"flag_file is empty: {flag_file}")
        obj["flag"] = flag
    for key, value in (obj.get("config") or {}).items():
        # Paths stay as written. Only check they exist, never read them here.
        if key.endswith("_file") and isinstance(value, str) and \
                not _resolve(value, root).exists():
            warnings.append(f"config.{key} not found: {value}")
    return obj, warnings


def load_files(db, season_id: int, paths: list[str], root: Path, out=print) -> dict:
    report = {"created": [], "skipped": [], "errors": []}
    for path in paths:
        try:
            items = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report["errors"].append(path)
            out(f"error   {path}: cannot read ({exc.__class__.__name__})")
            continue
        if not isinstance(items, list):
            report["errors"].append(path)
            out(f"error   {path}: expected a JSON list")
            continue
        for obj in items:
            slug = obj.get("slug", "?") if isinstance(obj, dict) else "?"
            if not isinstance(obj, dict):
                report["errors"].append(slug)
                out(f"error   {path}: every item must be an object")
                continue
            if db.get_challenge(season_id, slug):
                report["skipped"].append(slug)
                out(f"skipped {slug} (already in this season)")
                continue
            try:
                fields, warnings = _prepare(obj, root)
                row = create_challenge(db, season_id, ChallengeIn(**fields))
            except PydanticError as exc:
                report["errors"].append(slug)
                bad = ", ".join(".".join(str(x) for x in e["loc"]) for e in exc.errors())
                out(f"error   {slug}: invalid {bad}")
                continue
            except (SubmitError, ValueError) as exc:
                report["errors"].append(slug)
                out(f"error   {slug}: {getattr(exc, 'message', exc)}")
                continue
            report["created"].append(slug)
            hints = len(fields.get("hints") or [])
            out(f"created {slug} (id {row['id']}, {hints} hints)")
            for w in warnings:
                out(f"warning {slug}: {w}")
    return report


def main(argv: list[str] | None = None, out=print) -> int:
    ap = argparse.ArgumentParser(description="Load seeds/*.json into a season.")
    ap.add_argument("files", nargs="*", help="seed files (default: seeds/*.json)")
    ap.add_argument("--season", help="create this season if needed and make it active")
    ap.add_argument("--db", default=settings.db_path, help="SQLite path")
    ap.add_argument("--root", default=".", help="base for relative paths in seeds")
    args = ap.parse_args(argv)

    db = Database(args.db)
    db.migrate()
    if args.season:
        season = db.get_season_by_name(args.season)
        if season:
            db.activate_season(season["id"])
            out(f"season  {args.season} (existing, now active)")
        else:
            db.create_season(args.season, active=True)
            out(f"season  {args.season} (created, active)")
    season = db.get_active_season()
    if not season:
        out("No active season. Pass --season \"Name\" to create one.")
        return 1

    files = args.files or sorted(glob.glob(str(ROOT / "seeds" / "*.json")))
    report = load_files(db, season["id"], files, Path(args.root), out=out)
    out(f"done: {len(report['created'])} created, {len(report['skipped'])} skipped, "
        f"{len(report['errors'])} errors")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
