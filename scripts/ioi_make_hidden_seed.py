"""Make the secret seed for a hidden-split circuits challenge.

    python -m scripts.ioi_make_hidden_seed data/private/circuits/<slug>.json

Writes {"seed": <random int>} and refuses to overwrite, because a new seed means new
prompts and every score already on the board would stop being comparable. data/private/
is gitignored. Copy it to the server by hand.
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

from app.challenges.circuits import PUBLIC_SEED


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit(__doc__)
    path = Path(args[0])
    if path.exists():
        raise SystemExit(f"{path} exists. Delete it by hand if you really want new prompts.")
    seed = PUBLIC_SEED
    while abs(seed - PUBLIC_SEED) <= 1:
        seed = secrets.randbelow(2**31)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed}) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
