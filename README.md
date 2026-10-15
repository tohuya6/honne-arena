# Honne Arena

A competition and learning arena for NU AI Safety. *Honne* (本音) is Japanese for
what someone really thinks, as opposed to the face they show (*tatemae*). Finding a
model's honne behind its outputs is the game.

| Track | What you do | Scored by |
|---|---|---|
| **steering** | push a model's activations along a direction (from Steering Arena) | shift along the direction, on GPT-2 locally or a large model on NDIF |
| **lie-detection** | tell which statements or answers are lies (after Aletheia's Quest) | AUROC on hidden labels |
| **circuits** | find the attention heads GPT-2 uses for indirect object identification (after MIB) | faithfulness, computed live on GPT-2 |
| **cp** | solve programming problems | hidden tests, run in a locked-down Docker container |
| **ctf** | find the flag | a hash of the flag |

Every track has a `learn` tier (guided, small models, free hints) and a `compete` tier
(hidden test sets). Members sign in with their NU email. Teams, CTF-style decaying
points, a season leaderboard, all-time tiers per track, and a live feed tie it together.

Built on [Steering Arena](https://sohampadianeu-steering-arena.hf.space/) by Soham Padia.
See [CREDITS.md](CREDITS.md). Why things are the way they are: [DECISIONS.md](DECISIONS.md).
The HTTP API is in [API.md](API.md), deployment in [docs/DEPLOY.md](docs/DEPLOY.md).

## Run it locally

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-model.txt      # or requirements.txt for the tracks without models
cp .env.example .env                       # set ADMIN_TOKEN and IP_HASH_SALT
docker pull python:3.12-slim               # for the cp track
python scripts/seed.py --season "Fall 2026"
EMAIL_BACKEND=console DEV_EXPOSE_LINKS=true uvicorn app.main:create_app --factory --port 7860
```

With `DEV_EXPOSE_LINKS=true` the sign-in page shows the magic link instead of emailing it.
Hidden answers (flags, labels, hidden tests) live in `data/private/`, which is not in
git, so the compete challenges need those files copied in.

Tests: `pytest` (add `-m "not slow"` to skip the ones that load GPT-2). They never call
NDIF, and the Docker tests skip on machines without Docker.

## Add a challenge

Put it in a seed file under `seeds/` and run `scripts/seed.py`, or use the admin page,
or the admin API:

```bash
curl -X POST localhost:7860/api/admin/challenges -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H 'content-type: application/json' -d '{
    "slug": "warmup", "title": "Warmup", "kind": "flag", "track": "ctf", "tier": "learn",
    "flag": "nuais{hello}", "config": {"flag_format": "nuais{...}"}}'
```

A flag is hashed on arrival and never stored in plain text.

## NDIF

The compete steering challenge calls [NDIF](https://ndif.us), which is free for research
use but a small shared cluster. Model scoring there is **off** until `NDIF_ENABLED=true`,
and hard daily budgets apply when it is on. Everything else runs on this server. See
DECISIONS.md, "NDIF use".
