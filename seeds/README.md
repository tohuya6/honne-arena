# Seeds

Starter challenges, one JSON file per track. `scripts/seed.py` loads them into the
active season through the same validation as the admin API.

Each file is a JSON list of challenge objects with the admin API's fields:
`slug, title, kind, track, tier, description, config, points_initial, points_min,
points_decay, hidden`, plus optional `hints: [{"body": ..., "cost": ...}]`.

**This folder is public.** Nothing in it may give away an answer. Hidden material
(flags, test-set labels, hidden code tests) lives in `data/private/`, which is
gitignored, and seeds point at it by path:

- flag challenges: `"flag_file": "data/private/ctf/<slug>.txt"` instead of `"flag"`
- predictions: `config.labels_file`
- code: `config.tests_file` (hidden tests), with public `config.samples`

When deploying, copy `data/private/` to the server by hand.
