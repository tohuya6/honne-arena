"""Shared helpers for the lie-detection dataset and baseline scripts.

Imported by the scripts in this folder (run them from the repo root). Kept free of
heavy imports so the dataset builders run without torch.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "data" / "public" / "lie"
PRIVATE = ROOT / "data" / "private" / "lie"

GPT2 = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def secret_seed(name: str) -> int:
    """A random seed kept in data/private/lie/<name>_seed.txt and made on first use.

    The scripts are public, so a seed written in them would let anyone rerun a
    builder and get the hidden answer key. Keeping the seed private means only an
    organizer with data/private/ can rebuild the exact same files."""
    path = PRIVATE / f"{name}_seed.txt"
    if not path.exists():
        PRIVATE.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{secrets.randbits(31)}\n", encoding="utf-8")
    return int(path.read_text(encoding="utf-8").strip())


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_labels(path: Path, labels: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=0, sort_keys=True) + "\n", encoding="utf-8")
