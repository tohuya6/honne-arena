# Honne Arena

A competition and learning arena for NU AI Safety. *Honne* (本音) is Japanese for
what someone really thinks, as opposed to the face they show (*tatemae*). Finding a
model's honne behind its outputs is the game. One site, several tracks:

- **steering**: push a model's activations along a direction (from Steering Arena)
- **lie-detection**: predict which transcripts are lies, scored by AUROC on hidden labels
- **circuits**: find the components that do a task (planned)
- **ctf**: classic flags

Each track has a `learn` tier (guided, small models) and a `compete` tier (hidden test
sets). Teams, CTF-style decaying points, and a season leaderboard tie them together.

Built on [Steering Arena](https://sohampadianeu-steering-arena.hf.space/) by Soham Padia.
See [CREDITS.md](CREDITS.md). Why things are the way they are: [DECISIONS.md](DECISIONS.md).

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # add requirements-model.txt for steering
cp .env.example .env                       # set ADMIN_TOKEN and IP_HASH_SALT
uvicorn app.main:create_app --factory --port 7860
```

Tests: `pytest`. They never call NDIF.

## Add a challenge

Create a season, then a challenge, with the admin token:

```bash
curl -X POST localhost:7860/api/admin/seasons -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H 'content-type: application/json' -d '{"name": "Fall 2026"}'

curl -X POST localhost:7860/api/admin/challenges -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H 'content-type: application/json' -d '{
    "slug": "warmup", "title": "Warmup", "kind": "flag", "track": "ctf", "tier": "learn",
    "flag": "nuais{hello}", "config": {"flag_format": "nuais{...}"}}'
```

The flag is hashed on arrival and never stored in plain text.

## API

| Route | What |
|---|---|
| `GET /api/challenges?track=&tier=` | open challenges with current points |
| `GET /api/challenges/{slug}` | one challenge and its board |
| `GET /api/leaderboard` | season standings by team |
| `POST /api/teams` | create a team, returns its join code |
| `POST /api/teams/join` | join with a code |
| `POST /api/submit` | `{handle, challenge, answer}` |
| `GET /api/docs` | interactive docs |

## NDIF

Steering challenges call [NDIF](https://ndif.us), which is free for research use but a
small shared cluster. Model scoring is **off** until `NDIF_ENABLED=true`, and hard daily
budgets apply when it is on. See DECISIONS.md, "NDIF use".
