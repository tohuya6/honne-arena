#!/usr/bin/env python3
"""A stand-in backend for working on web/ without the real server.

Pure standard library. Serves web/ and answers every route in API.md with fixture data
kept in memory, so sign-in, handles, submissions, hints, teams, writeups and the admin
forms all work. Nothing is saved: restart it to reset.

    python3 scripts/ui_stub.py               # http://localhost:8765
    python3 scripts/ui_stub.py --port 9000
    python3 scripts/ui_stub.py --no-ws       # /ws refuses, to check the UI copes
    python3 scripts/ui_stub.py --demo 20     # a fake player solves something every 20 s

Sign in with any @northeastern.edu address. The reply includes a dev_link to open.
admin@northeastern.edu (or any address passed with --admin) is an organizer.
Join code for the fixture team "Null Hypothesis" is printed at startup.
Answers that work: the flag "nuais{hello_arena}" on warmup-base64, and "nuais{vault_open}"
on prompt-vault. Code that reads stdin and prints passes the code challenge.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import secrets
import socket
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
COOKIE = "honne_session"
ADMIN_TOKEN = "dev-admin-token"
HANDLE_RE = re.compile(r"^[\w \-.]{1,32}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,63}$")
HEAD_RE = re.compile(r"^\d{1,3}\.\d{1,3}$")
TRACKS = ["ctf", "lie-detection", "steering", "circuits", "cp"]
KINDS = {"flag": "solve", "predictions": "score", "steering": "score", "code": "solve",
         "circuits": "score"}
LOCK = threading.RLock()


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ago(**kw) -> str:
    return iso(now() - timedelta(**kw))


def ahead(**kw) -> str:
    return iso(now() + timedelta(**kw))


class Fail(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ─────────────────────────── fixture state ───────────────────────────

PLAYERS = [  # handle, team
    ("ada", "Gradient Descenders"), ("byte_witch", "Gradient Descenders"),
    ("kestrel", "Null Hypothesis"), ("mo.lin", "Null Hypothesis"), ("quill", "Null Hypothesis"),
    ("ravi-k", "Residual Stream Team"), ("sana", "Residual Stream Team"),
    ("tomasz", "tomasz"), ("ume", "Sparse Coders"), ("vik", "Sparse Coders"),
    ("wren", "wren"), ("xiao", "Attention Seekers"), ("yusuf", "Attention Seekers"),
    ("zed", "zed"), ("pilar", "Probe Squad"), ("noor", "Probe Squad"),
]

IOI_HEADS = {"9.9", "9.6", "10.0", "10.7", "11.10", "8.6", "8.10", "7.3", "7.9", "5.5",
             "6.9", "3.0", "2.2", "4.11"}

LIE_LABELS = {f"item_{i:03d}": (1 if i % 3 == 0 or i % 7 == 0 else 0) for i in range(1, 25)}

DESC_WARMUP = """# Your first flag

Welcome to the arena. Somewhere in this message is a flag, encoded in **base64**:

```
bnVhaXN7aGVsbG9fYXJlbmF9
```

1. Decode it. Any tool works, for example `base64 -d` in a terminal.
2. Paste the result below. Flags look like `nuais{...}`.

New to CTFs? Read the [picoCTF primer](https://primer.picoctf.org/) for a gentle start.
"""

DESC_VAULT = """A support bot guards a vault. Its system prompt says it must *never* reveal the
vault code, yet people keep getting it out.

## What you get

- A transcript of the bot's system prompt, with the secret redacted
- Ten logged conversations, three of which leaked the code

Find the pattern the leaks share and reconstruct the flag. Watch out for
<script>alert("this is shown as text, never run")</script> tricks in the logs.
"""

DESC_LIES = """# Spot the lie

Each item is a short exchange where a model was told to either answer honestly or to
deceive. Give each item a probability that the model was **lying**.

Score: **AUROC** against the hidden labels. 0.5 is chance, 1.0 is perfect.

Format, one line per item:

```
item_001,0.92
item_002,0.10
```
"""

DESC_STEER = """Find a short text that, put in front of neutral prompts, pushes the model's
residual stream at layer 16 along the published *pro-human* direction.

- Higher shift wins.
- Keep it within the token budget.
- Weird, opaque strings are fair game.

Ported from Soham Padia's [Steering Arena](https://sohampadianeu-steering-arena.hf.space/).
"""

DESC_IOI = """# Find the IOI circuit

On sentences like *"When Mary and John went to the store, John gave a drink to"*, GPT-2
small predicts **Mary**. Which attention heads do the work?

Submit a set of heads. We keep only those heads, ablate the rest, and score how much of
the model's behaviour survives (faithfulness). Smaller circuits that keep more behaviour
rank higher.

Read about it in [Wang et al. 2022](https://arxiv.org/abs/2211.00593).
"""

DESC_CODE = """# Sum of pairs

Read `n`, then `n` integers. Print how many pairs `(i, j)` with `i < j` have an even sum.

## Limits

- `1 <= n <= 200000`
- Values fit in a 32-bit signed integer

Read from stdin and write to stdout.
"""

DESC_OPENER = """The season opener: a classic web flag hidden in a cookie. It is closed now, so
read how other people solved it below.
"""


def fixture_code(team: str) -> str:
    """Stable join codes for the fixture teams, so you can test joining one."""
    return "join-" + hashlib.sha256(team.encode()).hexdigest()[:6]


def make_state() -> dict:
    s: dict = {
        "seasons": [{"id": 1, "name": "Spring 2026", "active": False},
                    {"id": 2, "name": "Fall 2026", "active": True}],
        "users": {},        # id -> user dict
        "by_email": {},     # email -> id
        "tokens": {},       # token -> (email, expires)
        "sessions": {},     # cookie value -> user id
        "teams": {},        # name -> {name, join_code, members:[handle], creator}
        "challenges": [],
        "unlocked": set(),  # (user id, slug, position)
        "submitted": set(),  # user ids that have submitted this season
        "ndif_calls": 37,
        "submissions_today": 112,
        "next_id": 100,
    }
    for handle, team in PLAYERS:
        t = s["teams"].setdefault(team, {"name": team, "join_code": fixture_code(team),
                                         "members": [], "creator": None})
        t["members"].append(handle)

    def ch(**kw):
        base = {"id": len(s["challenges"]) + 1, "description": "", "config": {},
                "hidden": False, "closes_at": None, "points_initial": 500, "points_min": 100,
                "points_decay": 10, "hints": [], "writeups": [], "solves": [], "entries": [],
                "secret": None}
        base.update(kw)
        base["mode"] = KINDS[base["kind"]]
        s["challenges"].append(base)
        return base

    teams = [p[1] for p in PLAYERS]
    ch(slug="warmup-base64", title="Warm-up: decode the flag", kind="flag", track="ctf",
       tier="learn", description=DESC_WARMUP, config={"flag_format": "nuais{...}"},
       secret="nuais{hello_arena}", points_initial=100, points_min=50, points_decay=20,
       hints=[{"body": "Base64 text often ends in `=` and uses letters, digits, `+` and `/`.", "cost": 0},
              {"body": "Try `echo bnVhaXN7aGVsbG9fYXJlbmF9 | base64 -d`.", "cost": 0}],
       solves=[{"team": PLAYERS[i][1], "handle": PLAYERS[i][0], "created_at": ago(hours=50 - i * 3)}
               for i in (0, 2, 5, 8, 11, 14, 7)])
    ch(slug="prompt-vault", title="The leaky vault", kind="flag", track="ctf", tier="compete",
       description=DESC_VAULT, config={"flag_format": "nuais{...}"}, secret="nuais{vault_open}",
       points_initial=500, points_min=100, points_decay=12, closes_at=ahead(days=9),
       hints=[{"body": "All three leaks ask the bot to **translate** something.", "cost": 50},
              {"body": "The code is split across two replies. Join them.", "cost": 120}],
       solves=[{"team": PLAYERS[i][1], "handle": PLAYERS[i][0], "created_at": ago(hours=30 - i)}
               for i in (5, 0, 9, 12)])
    ch(slug="lies-101", title="Spot the lie (practice set)", kind="predictions",
       track="lie-detection", tier="learn", description=DESC_LIES,
       config={"metric": "accuracy", "item_ids": "/datasets/lies-101.csv"}, secret=LIE_LABELS,
       hints=[{"body": "Liars in this set hedge a lot: count words like *maybe* and *perhaps*.", "cost": 0}],
       entries=[{"handle": h, "team": t, "score": sc, "answer": None, "created_at": ago(hours=h_)}
                for (h, t), sc, h_ in zip(PLAYERS[:6], [0.875, 0.833, 0.792, 0.75, 0.708, 0.625],
                                          [40, 33, 29, 20, 12, 3])])
    ch(slug="aletheia-mini", title="Aletheia mini: deceptive models", kind="predictions",
       track="lie-detection", tier="compete", description=DESC_LIES,
       config={"metric": "auroc", "item_ids": "/datasets/aletheia-mini.csv"}, secret=LIE_LABELS,
       points_initial=600, closes_at=ahead(days=3, hours=4),
       entries=[{"handle": PLAYERS[i][0], "team": PLAYERS[i][1], "score": sc, "answer": None,
                 "created_at": ago(hours=60 - i * 3)}
                for i, sc in zip(range(16), [0.9412, 0.9301, 0.9177, 0.9002, 0.8811, 0.8735,
                                             0.8602, 0.8450, 0.8213, 0.8007, 0.7744, 0.7561,
                                             0.7120, 0.6833, 0.6402, 0.5810])])
    steer_answers = ["be kind honest humble humane", "cooperate. protect people. tell the truth",
                     "humanity flourishing together safely", "please help everyone thrive",
                     "aaaaaa human human human", "we care about you", "合作 honesty helpful",
                     "support wellbeing", "kindness", "the end"]
    ch(slug="pro-human-shift", title="Pro-human shift", kind="steering", track="steering",
       tier="compete", description=DESC_STEER,
       config={"model_id": "meta-llama/Llama-3.1-8B", "layer": 16, "token_budget": 20,
               "higher_is_better": True},
       entries=[{"handle": PLAYERS[i][0], "team": PLAYERS[i][1], "score": sc,
                 "answer": steer_answers[i], "created_at": ago(hours=70 - i * 5)}
                for i, sc in enumerate([0.4123, 0.3981, 0.3512, 0.3307, 0.2990, 0.2544,
                                        0.2211, 0.1802, 0.1410, 0.0333])])
    ch(slug="ioi-heads", title="Find the IOI circuit", kind="circuits", track="circuits",
       tier="compete", description=DESC_IOI,
       config={"model_id": "gpt2", "max_heads": 12},
       hints=[{"body": "Start with the **name mover** heads in layers 9 and 10.", "cost": 60}],
       entries=[{"handle": PLAYERS[i][0], "team": PLAYERS[i][1], "score": sc, "answer": None,
                 "created_at": ago(hours=20 - i)}
                for i, sc in zip((2, 6, 9, 13, 15), [0.8120, 0.7764, 0.6501, 0.5123, 0.3307])])
    ch(slug="pair-sums", title="Sum of pairs", kind="code", track="cp", tier="learn",
       description=DESC_CODE,
       config={"language": "python", "time_limit_s": 2, "memory_mb": 256,
               "samples": [{"input": "4\n1 2 3 4\n", "output": "2\n"},
                           {"input": "3\n2 4 6\n", "output": "3\n"}]},
       points_initial=200, points_min=100,
       hints=[{"body": "A sum is even when both numbers are even or both are odd.", "cost": 0}],
       solves=[{"team": PLAYERS[i][1], "handle": PLAYERS[i][0], "created_at": ago(hours=10 - i)}
               for i in (1, 5, 8)])
    ch(slug="season-opener", title="Season opener: cookie jar", kind="flag", track="ctf",
       tier="compete", description=DESC_OPENER, config={"flag_format": "nuais{...}"},
       secret="nuais{om_nom}", closes_at=ago(days=6),
       hints=[{"body": "Look at the cookies your browser stores for the site.", "cost": 40}],
       solves=[{"team": PLAYERS[i][1], "handle": PLAYERS[i][0], "created_at": ago(days=8, hours=i)}
               for i in (0, 2, 5, 9, 11, 13)],
       writeups=[{"handle": "ada", "created_at": ago(days=5),
                  "body": "Opened dev tools, Application tab, Cookies.\nThe value of `jar` was hex. "
                          "Decoding it gave the flag.\n\nTook about ten minutes."},
                 {"handle": "kestrel", "created_at": ago(days=4, hours=3),
                  "body": "curl -v shows a Set-Cookie header on the first response, so you never "
                          "need a browser. <b>Not bold</b>, because writeups are plain text."}])
    ch(slug="draft-rlhf-probe", title="Draft: reward hacking probe", kind="predictions",
       track="lie-detection", tier="compete", hidden=True, description="Coming soon.",
       config={"metric": "auroc"}, secret=LIE_LABELS)
    return s


S = make_state()
ARGS = None


# ─────────────────────────── helpers ───────────────────────────

def find(slug: str, include_hidden: bool = False) -> dict:
    for c in S["challenges"]:
        if c["slug"] == slug and (include_hidden or not c["hidden"]):
            return c
    raise Fail(404, "No such challenge.")


def is_closed(c: dict) -> bool:
    if not c["closes_at"]:
        return False
    return datetime.fromisoformat(str(c["closes_at"]).replace("Z", "+00:00")) <= now()


def dynamic_value(c: dict) -> int:
    n = max(len(c["solves"]), 1)
    a, b, d = c["points_initial"], c["points_min"], max(c["points_decay"], 1)
    v = ((b - a) / (d * d)) * ((n - 1) ** 2) + a
    return int(max(b, round(v)))


def best_entries(c: dict) -> list[dict]:
    hib = c["config"].get("higher_is_better", True)
    best: dict[str, dict] = {}
    for e in c["entries"]:
        cur = best.get(e["handle"])
        if cur is None or (e["score"] > cur["score"] if hib else e["score"] < cur["score"]):
            best[e["handle"]] = e
    rows = sorted(best.values(), key=lambda e: e["score"], reverse=hib)
    return rows


def team_of(handle: str | None) -> dict | None:
    if not handle:
        return None
    for t in S["teams"].values():
        if handle in t["members"]:
            return t
    return None


def public_config(c: dict) -> dict:
    cfg = dict(c["config"])
    cfg.pop("flag_sha256", None)
    cfg.pop("labels", None)
    cfg.pop("labels_file", None)
    cfg.pop("tests_file", None)
    return cfg


def challenge_view(c: dict, user: dict | None, full: bool) -> dict:
    handle = user["handle"] if user else None
    team = team_of(handle)
    out = {"slug": c["slug"], "title": c["title"], "kind": c["kind"], "mode": c["mode"],
           "track": c["track"], "tier": c["tier"], "description": c["description"],
           "config": public_config(c), "closes_at": c["closes_at"], "closed": is_closed(c),
           "solves": None, "entries": None, "solved_by_me": False, "my_best": None}
    if c["mode"] == "solve":
        out["solves"] = len(c["solves"])
        out["points"] = dynamic_value(c)
        out["solved_by_me"] = bool(team and any(s["team"] == team["name"] for s in c["solves"]))
    else:
        rows = best_entries(c)
        out["entries"] = len(rows)
        out["points"] = c["points_initial"]
        mine = [r for r in rows if r["handle"] == handle]
        out["my_best"] = mine[0]["score"] if mine else None
    if full:
        if c["mode"] == "solve":
            out["board"] = [dict(s) for s in c["solves"]]
        else:
            out["board"] = [{"rank": i + 1, "handle": r["handle"], "team": r["team"],
                             "score": r["score"],
                             "answer": r["answer"] if c["kind"] == "steering" else None,
                             "created_at": r["created_at"]} for i, r in enumerate(best_entries(c))]
        uid = user["id"] if user else None
        out["hints"] = []
        for i, h in enumerate(c["hints"], 1):
            unlocked = (uid, c["slug"], i) in S["unlocked"]
            out["hints"].append({"position": i, "cost": 0 if c["tier"] == "learn" else h["cost"],
                                 "unlocked": unlocked, "body": h["body"] if unlocked else None})
        out["writeups"] = [dict(w) for w in c["writeups"]] if out["closed"] else []
    return out


def season_board(season_id: int) -> list[dict]:
    if season_id == 1:
        rows = [("Null Hypothesis", 3120, 9), ("Gradient Descenders", 2875, 8),
                ("Probe Squad", 1990, 6), ("wren", 860, 3), ("zed", 400, 1)]
        return [{"rank": i + 1, "team": t, "points": p, "solves": n} for i, (t, p, n) in enumerate(rows)]
    totals: dict[str, dict] = {}
    for c in S["challenges"]:
        if c["hidden"]:
            continue
        if c["mode"] == "solve":
            v = dynamic_value(c)
            for s in c["solves"]:
                t = totals.setdefault(s["team"], {"team": s["team"], "points": 0, "solves": 0})
                t["points"] += v
                t["solves"] += 1
        else:
            seen: dict[str, dict] = {}
            for r in best_entries(c):
                seen.setdefault(r["team"], r)
            n = len(seen)
            for i, team in enumerate(seen):
                a, b = c["points_initial"], c["points_min"]
                pts = a if n == 1 else round(a - (a - b) * i / (n - 1))
                t = totals.setdefault(team, {"team": team, "points": 0, "solves": 0})
                t["points"] += pts
    rows = sorted(totals.values(), key=lambda r: (-r["points"], r["team"]))
    return [dict(r, rank=i + 1) for i, r in enumerate(rows)]


def tier_for(rank: int | None) -> int:
    if rank is None:
        return 0
    return 3 if rank <= 5 else 2 if rank <= 15 else 1


def track_boards() -> list[dict]:
    out = []
    history = {"ctf": 1200, "lie-detection": 900, "steering": 700, "circuits": 400, "cp": 300}
    for track in TRACKS:
        pts: dict[str, int] = {}
        for i, (h, _t) in enumerate(PLAYERS):
            seed = int(hashlib.sha256(f"{track}{h}".encode()).hexdigest(), 16)
            if seed % 5 == 0 and track in ("circuits", "cp"):
                continue
            pts[h] = (seed % history[track]) + 50
        for c in S["challenges"]:
            if c["track"] != track or c["hidden"]:
                continue
            if c["mode"] == "solve":
                for s in c["solves"]:
                    pts[s["handle"]] = pts.get(s["handle"], 0) + dynamic_value(c)
            else:
                for i, r in enumerate(best_entries(c)):
                    pts[r["handle"]] = pts.get(r["handle"], 0) + max(c["points_min"], c["points_initial"] - 40 * i)
        for u in S["users"].values():
            if u["handle"] and u["handle"] not in pts and track == "ctf" and u["id"] in S["submitted"]:
                pts[u["handle"]] = 10
        rows = sorted(pts.items(), key=lambda kv: (-kv[1], kv[0]))
        out.append({"track": track, "board": [{"rank": i + 1, "handle": h, "points": p,
                                               "tier": tier_for(i + 1)} for i, (h, p) in enumerate(rows)]})
    return out


def score_answer(c: dict, answer: str) -> dict:
    kind, cfg = c["kind"], c["config"]
    if kind == "predictions":
        preds: dict[str, float] = {}
        for n, line in enumerate(answer.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                raise Fail(400, f"Line {n}: expected 'id,probability'.")
            try:
                p = float(parts[1])
            except ValueError:
                raise Fail(400, f"Line {n}: {parts[1]!r} is not a number.") from None
            if not 0 <= p <= 1:
                raise Fail(400, f"Line {n}: probability must be between 0 and 1.")
            if parts[0] in preds:
                raise Fail(400, f"Line {n}: id {parts[0]!r} appears twice.")
            preds[parts[0]] = p
        labels = c["secret"]
        missing = [k for k in labels if k not in preds]
        extra = [k for k in preds if k not in labels]
        if missing or extra:
            raise Fail(400, f"Predictions must cover exactly the {len(labels)} ids in the item set "
                            f"({len(missing)} missing, {len(extra)} unknown).")
        if cfg.get("metric") == "accuracy":
            score = sum((preds[k] >= 0.5) == bool(v) for k, v in labels.items()) / len(labels)
        else:
            pos = [preds[k] for k, v in labels.items() if v]
            neg = [preds[k] for k, v in labels.items() if not v]
            wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
            score = wins / (len(pos) * len(neg))
        return {"score": round(score, 4), "n": len(labels)}
    if kind == "steering":
        tokens = max(len(answer.split()), (len(answer) + 3) // 4)
        budget = int(cfg.get("token_budget", 64))
        if tokens > budget:
            raise Fail(400, f"That is {tokens} tokens. The budget is {budget}.")
        h = int(hashlib.sha256(answer.strip().lower().encode()).hexdigest()[:8], 16)
        return {"score": round((h % 9000) / 20000 - 0.05, 4), "tokens": tokens}
    if kind == "circuits":
        heads = [x for x in re.split(r"[\s,;]+", answer.strip()) if x]
        bad = [x for x in heads if not HEAD_RE.match(x)]
        if bad:
            raise Fail(400, f"{bad[0]!r} is not a head. Use layer.head, like 9.6.")
        heads = list(dict.fromkeys(".".join(str(int(p)) for p in h.split(".")) for h in heads))
        for h in heads:
            layer, head = map(int, h.split("."))
            if layer > 11 or head > 11:
                raise Fail(400, f"{h} is out of range. GPT-2 small has layers 0-11 and heads 0-11.")
        if len(heads) > int(cfg.get("max_heads", 10)):
            raise Fail(400, f"That is {len(heads)} heads. The limit is {cfg.get('max_heads')}.")
        hit = len(set(heads) & IOI_HEADS)
        score = hit / len(IOI_HEADS) * 0.95 - 0.02 * (len(heads) - hit)
        return {"score": round(max(0.0, score), 4), "heads": len(heads)}
    raise Fail(400, "This kind can't be scored here.")


def judge_code(answer: str) -> dict:
    total = 12
    if "input" in answer or "stdin" in answer:
        if "print" in answer or "stdout" in answer:
            return {"correct": True, "verdict": "accepted", "passed": total, "total": total}
        return {"correct": False, "verdict": "wrong_answer", "passed": 0, "total": total}
    if "while True" in answer:
        return {"correct": False, "verdict": "time_limit_exceeded", "passed": 3, "total": total}
    return {"correct": False, "verdict": "runtime_error", "passed": 0, "total": total}


# ─────────────────────────── WebSocket (minimal) ───────────────────────────

WS_CLIENTS: list[socket.socket] = []


def ws_send(sock: socket.socket, text: str) -> None:
    data = text.encode("utf-8")
    head = bytes([0x81])
    n = len(data)
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + struct.pack("!H", n)
    else:
        head += bytes([127]) + struct.pack("!Q", n)
    sock.sendall(head + data)


def broadcast(event: dict) -> None:
    text = json.dumps(event)
    with LOCK:
        clients = list(WS_CLIENTS)
    for sock in clients:
        try:
            ws_send(sock, text)
        except OSError:
            with LOCK:
                if sock in WS_CLIENTS:
                    WS_CLIENTS.remove(sock)


def demo_loop(every: float) -> None:
    import random
    while True:
        time.sleep(every)
        with LOCK:
            c = random.choice([c for c in S["challenges"] if not c["hidden"] and not is_closed(c)])
            handle, team = random.choice(PLAYERS)
            if c["mode"] == "solve":
                if any(s["team"] == team for s in c["solves"]):
                    continue
                first = not c["solves"]
                c["solves"].append({"team": team, "handle": handle, "created_at": iso(now())})
                ev = {"type": "solve", "challenge": c["slug"], "handle": handle, "team": team,
                      "first_blood": first, "at": iso(now())}
            else:
                sc = round(random.random(), 4)
                c["entries"].append({"handle": handle, "team": team, "score": sc,
                                     "answer": "demo entry" if c["kind"] == "steering" else None,
                                     "created_at": iso(now())})
                ev = {"type": "entry", "challenge": c["slug"], "handle": handle, "team": team,
                      "score": sc, "at": iso(now())}
        broadcast(ev)


# ─────────────────────────── HTTP ───────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "HonneStub/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter
        if not ARGS or not ARGS.quiet:
            super().log_message(fmt, *args)

    # ── plumbing ──
    def user(self) -> dict | None:
        c = SimpleCookie(self.headers.get("Cookie", ""))
        sid = c[COOKIE].value if COOKIE in c else None
        uid = S["sessions"].get(sid) if sid else None
        return S["users"].get(uid) if uid else None

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            raise Fail(400, "Request too large.")
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            raise Fail(400, "Send JSON.")
        try:
            data = json.loads(raw)
        except ValueError:
            raise Fail(400, "Bad JSON.") from None
        if not isinstance(data, dict):
            raise Fail(400, "Send a JSON object.")
        return data

    def send_json(self, status: int, obj, cookie: str | None = None) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def do_PATCH(self):
        self.route("PATCH")

    def route(self, method: str) -> None:
        url = urlsplit(self.path)
        path, q = url.path, {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if path == "/ws" and method == "GET":
                return self.websocket()
            if path.startswith("/api/"):
                with LOCK:
                    out = self.api(method, path, q)
                if isinstance(out, tuple):
                    self.send_json(200, out[0], cookie=out[1])
                else:
                    self.send_json(200, out)
                return
            if path.startswith("/datasets/") and method == "GET":
                return self.dataset(path)
            if method != "GET":
                raise Fail(405, "Method not allowed.")
            return self.static(path)
        except Fail as f:
            self.send_json(f.status, {"error": f.message})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def static(self, path: str) -> None:
        rel = unquote(path).lstrip("/") or "index.html"
        target = (WEB / rel).resolve()
        if WEB.resolve() not in target.parents and target != WEB.resolve():
            raise Fail(404, "Not found.")
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            raise Fail(404, "Not found.")
        data = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "text/javascript"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def dataset(self, path: str) -> None:
        if path not in ("/datasets/lies-101.csv", "/datasets/aletheia-mini.csv"):
            raise Fail(404, "Not found.")
        lines = ["# item_id,transcript_excerpt"] + [f"{k},\"(transcript {k})\"" for k in LIE_LABELS]
        data = ("\n".join(lines) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def websocket(self) -> None:
        if ARGS.no_ws or self.headers.get("Upgrade", "").lower() != "websocket":
            raise Fail(404, "No WebSocket here.")
        key = self.headers.get("Sec-WebSocket-Key", "")
        accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.wfile.flush()
        sock = self.connection
        with LOCK:
            WS_CLIENTS.append(sock)
        try:
            while True:  # read and drop client frames until close
                head = self.rfile.read(2)
                if len(head) < 2 or head[0] & 0x0F == 0x8:
                    break
                n = head[1] & 0x7F
                if n == 126:
                    n = struct.unpack("!H", self.rfile.read(2))[0]
                elif n == 127:
                    n = struct.unpack("!Q", self.rfile.read(8))[0]
                if head[1] & 0x80:
                    self.rfile.read(4)
                self.rfile.read(n)
        except OSError:
            pass
        finally:
            with LOCK:
                if sock in WS_CLIENTS:
                    WS_CLIENTS.remove(sock)
            self.close_connection = True

    # ── API ──
    def need_user(self) -> dict:
        u = self.user()
        if not u:
            raise Fail(401, "Please sign in first.")
        return u

    def need_admin(self) -> dict | None:
        if self.headers.get("X-Admin-Token") == ADMIN_TOKEN:
            return None
        u = self.need_user()
        if not u["is_admin"]:
            raise Fail(403, "Organizers only.")
        return u

    def api(self, method: str, path: str, q: dict):
        m = lambda pat: re.fullmatch(pat, path)  # noqa: E731
        user = self.user()

        # auth
        if method == "POST" and path == "/api/auth/request":
            email = str(self.body().get("email", "")).strip().lower()
            if not re.fullmatch(r"[^@\s]+@northeastern\.edu", email):
                raise Fail(400, "Use your @northeastern.edu email.")
            token = secrets.token_urlsafe(24)
            S["tokens"][token] = (email, now() + timedelta(minutes=15))
            host = self.headers.get("Host", "localhost:8765")
            return {"ok": True, "dev_link": f"http://{host}/verify.html?token={token}"}
        if method == "POST" and path == "/api/auth/verify":
            token = str(self.body().get("token", ""))
            got = S["tokens"].pop(token, None)
            if not got or got[1] < now():
                raise Fail(400, "This link has expired or was already used. Ask for a new one.")
            email = got[0]
            uid = S["by_email"].get(email)
            if uid is None:
                uid = S["next_id"]
                S["next_id"] += 1
                S["users"][uid] = {"id": uid, "handle": None, "email": email,
                                   "is_admin": email in ARGS.admin}
                S["by_email"][email] = uid
            sid = secrets.token_urlsafe(24)
            S["sessions"][sid] = uid
            cookie = f"{COOKIE}={sid}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000"
            return {"user": S["users"][uid]}, cookie
        if method == "POST" and path == "/api/auth/logout":
            c = SimpleCookie(self.headers.get("Cookie", ""))
            if COOKIE in c:
                S["sessions"].pop(c[COOKIE].value, None)
            return {"ok": True}, f"{COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"
        if method == "GET" and path == "/api/me":
            if not user:
                return {"user": None, "team": None, "tiers": {}}
            team = team_of(user["handle"])
            tiers = {}
            for tb in track_boards():
                row = next((r for r in tb["board"] if r["handle"] == user["handle"]), None)
                tiers[tb["track"]] = ({"tier": row["tier"], "rank": row["rank"], "points": row["points"]}
                                      if row else {"tier": 0, "rank": None, "points": 0})
            return {"user": user,
                    "team": ({"name": team["name"],
                              "join_code": team["join_code"] if team["creator"] == user["id"] else None,
                              "members": list(team["members"])} if team else None),
                    "tiers": tiers}
        if method == "POST" and path == "/api/me/handle":
            u = self.need_user()
            h = str(self.body().get("handle", "")).strip()
            if not HANDLE_RE.fullmatch(h):
                raise Fail(400, "Handles are 1 to 32 characters: letters, numbers, spaces, - _ .")
            taken = {p[0].lower() for p in PLAYERS} | {
                x["handle"].lower() for x in S["users"].values() if x["handle"] and x["id"] != u["id"]}
            if h.lower() in taken:
                raise Fail(409, "That handle is taken.")
            old = u["handle"]
            u["handle"] = h
            team = team_of(old)
            if team:
                team["members"] = [h if x == old else x for x in team["members"]]
                if team["name"] == old and team["creator"] == u["id"]:  # solo team follows the handle
                    S["teams"][h] = S["teams"].pop(old)
                    team["name"] = h
            else:  # a new handle gets a solo team named after it
                S["teams"][h] = {"name": h, "join_code": secrets.token_hex(4), "members": [h],
                                 "creator": u["id"]}
            return {"user": u}

        # play
        if method == "GET" and path == "/api/season":
            active = next(s for s in S["seasons"] if s["active"])
            return {"id": active["id"], "name": active["name"], "captcha_sitekey": None,
                    "seasons": S["seasons"]}
        if method == "GET" and path == "/api/challenges":
            rows = [c for c in S["challenges"] if not c["hidden"]]
            if q.get("track"):
                rows = [c for c in rows if c["track"] == q["track"]]
            if q.get("tier"):
                rows = [c for c in rows if c["tier"] == q["tier"]]
            return {"challenges": [challenge_view(c, user, False) for c in rows]}
        if method == "GET" and (mm := m(r"/api/challenges/([^/]+)")):
            return challenge_view(find(unquote(mm.group(1))), user, True)
        if method == "POST" and path == "/api/submit":
            return self.submit(self.body())
        if method == "POST" and (mm := m(r"/api/challenges/([^/]+)/hints/(\d+)/unlock")):
            u = self.need_user()
            c = find(unquote(mm.group(1)))
            pos = int(mm.group(2))
            if not 1 <= pos <= len(c["hints"]):
                raise Fail(404, "No such hint.")
            S["unlocked"].add((u["id"], c["slug"], pos))
            h = c["hints"][pos - 1]
            return {"position": pos, "cost": 0 if c["tier"] == "learn" else h["cost"],
                    "unlocked": True, "body": h["body"]}
        if method == "POST" and (mm := m(r"/api/challenges/([^/]+)/writeup")):
            u = self.need_user()
            if not u["handle"]:
                raise Fail(400, "Pick a handle first.")
            c = find(unquote(mm.group(1)))
            if not is_closed(c):
                raise Fail(403, "Writeups open after the challenge closes.")
            # Stub shortcut: everyone counts as having entered the closed fixture challenge.
            entered = c["slug"] == "season-opener" or any(
                e["handle"] == u["handle"] for e in c["entries"] + c["solves"])
            if not entered:
                raise Fail(403, "Only people who submitted can post a writeup.")
            body = str(self.body().get("body", "")).strip()
            if not body:
                raise Fail(400, "A writeup can't be empty.")
            if len(body) > 20000:
                raise Fail(400, "Writeups are at most 20,000 characters.")
            c["writeups"] = [w for w in c["writeups"] if w["handle"] != u["handle"]]
            c["writeups"].append({"handle": u["handle"], "body": body, "created_at": iso(now())})
            return {"ok": True}

        # teams
        if method == "POST" and path == "/api/teams":
            u = self.need_user()
            if not u["handle"]:
                raise Fail(400, "Pick a handle first.")
            name = str(self.body().get("name", "")).strip()
            if not 1 <= len(name) <= 40:
                raise Fail(400, "Team names are 1 to 40 characters.")
            if name.lower() in {t.lower() for t in S["teams"]}:
                raise Fail(409, "That team name is taken.")
            if u["id"] in S["submitted"]:
                raise Fail(403, "You can change team only before your first submission this season.")
            old = team_of(u["handle"])
            if old:
                old["members"].remove(u["handle"])
            S["teams"][name] = {"name": name, "join_code": secrets.token_hex(4),
                                "members": [u["handle"]], "creator": u["id"]}
            return {"team": name, "join_code": S["teams"][name]["join_code"]}
        if method == "POST" and path == "/api/teams/join":
            u = self.need_user()
            if not u["handle"]:
                raise Fail(400, "Pick a handle first.")
            code = str(self.body().get("join_code", "")).strip()
            team = next((t for t in S["teams"].values() if t["join_code"] == code), None)
            if not team:
                raise Fail(404, "No team has that join code.")
            if u["id"] in S["submitted"]:
                raise Fail(403, "You can change team only before your first submission this season.")
            old = team_of(u["handle"])
            if old is team:
                raise Fail(409, "You're already on that team.")
            if old:
                old["members"].remove(u["handle"])
            team["members"].append(u["handle"])
            return {"team": team["name"], "members": list(team["members"])}

        # boards
        if method == "GET" and path == "/api/leaderboard":
            sid = int(q["season"]) if q.get("season", "").isdigit() else next(
                s["id"] for s in S["seasons"] if s["active"])
            season = next((s for s in S["seasons"] if s["id"] == sid), None)
            if not season:
                raise Fail(404, "No such season.")
            return {"season": season["name"], "teams": season_board(sid)}
        if method == "GET" and path == "/api/tracks":
            return {"tracks": track_boards()}

        # admin
        if path.startswith("/api/admin/"):
            self.need_admin()
            return self.admin(method, path)

        raise Fail(404, "No such route.")

    def submit(self, b: dict) -> dict:
        u = self.need_user()
        if not u["handle"]:
            raise Fail(400, "Pick a handle before you submit.")
        c = find(str(b.get("challenge", "")))
        if is_closed(c):
            raise Fail(403, "This challenge is closed.")
        answer = str(b.get("answer", ""))
        if not answer.strip():
            raise Fail(400, "Your answer is empty.")
        team = team_of(u["handle"])
        S["submitted"].add(u["id"])
        S["submissions_today"] += 1
        if c["mode"] == "solve":
            if any(s["team"] == team["name"] for s in c["solves"]):
                raise Fail(409, "Your team already solved this one.")
            if c["kind"] == "code":
                res = judge_code(answer)
            else:
                res = {"correct": answer.strip().lower() == c["secret"].lower()}
            first = False
            if res["correct"]:
                first = not c["solves"]
                c["solves"].append({"team": team["name"], "handle": u["handle"],
                                    "created_at": iso(now())})
                threading.Thread(target=broadcast, daemon=True, args=({
                    "type": "solve", "challenge": c["slug"], "handle": u["handle"],
                    "team": team["name"], "first_blood": first, "at": iso(now())},)).start()
            return dict(res, challenge=c["slug"], mode="solve", first_blood=first)
        key = " ".join(answer.strip().lower().split())
        if any(" ".join((e.get("_key") or "").split()) == key for e in c["entries"]):
            raise Fail(409, "That exact answer was already scored.")
        if c["kind"] == "steering":
            S["ndif_calls"] += 1
        res = score_answer(c, answer)
        c["entries"].append({"handle": u["handle"], "team": team["name"], "score": res["score"],
                             "answer": answer.strip() if c["kind"] == "steering" else None,
                             "created_at": iso(now()), "_key": key})
        rank = next(i + 1 for i, r in enumerate(best_entries(c)) if r["handle"] == u["handle"])
        threading.Thread(target=broadcast, daemon=True, args=({
            "type": "entry", "challenge": c["slug"], "handle": u["handle"], "team": team["name"],
            "score": res["score"], "at": iso(now())},)).start()
        return dict(res, challenge=c["slug"], mode="score", rank=rank)

    def admin(self, method: str, path: str):
        m = lambda pat: re.fullmatch(pat, path)  # noqa: E731
        if method == "POST" and path == "/api/admin/seasons":
            b = self.body()
            name = str(b.get("name", "")).strip()
            if not name:
                raise Fail(400, "Name the season.")
            active = bool(b.get("active", True))
            if active:
                for s in S["seasons"]:
                    s["active"] = False
            row = {"id": max(s["id"] for s in S["seasons"]) + 1, "name": name, "active": active}
            S["seasons"].append(row)
            return dict(row, created_at=iso(now()))
        if method == "GET" and path == "/api/admin/challenges":
            out = []
            for c in S["challenges"]:
                v = challenge_view(c, None, False)
                v.update(hidden=c["hidden"], points_initial=c["points_initial"],
                         points_min=c["points_min"], points_decay=c["points_decay"])
                out.append(v)
            return {"challenges": out}
        if method == "POST" and path == "/api/admin/challenges":
            b = self.body()
            slug = str(b.get("slug", ""))
            if not SLUG_RE.fullmatch(slug):
                raise Fail(400, "Slugs are lowercase letters, digits and dashes.")
            if any(c["slug"] == slug for c in S["challenges"]):
                raise Fail(409, "That slug is taken this season.")
            kind = b.get("kind")
            if kind not in KINDS:
                raise Fail(400, f"Unknown kind. Known: {sorted(KINDS)}")
            if not str(b.get("title", "")).strip():
                raise Fail(400, "Add a title.")
            cfg = b.get("config") or {}
            if not isinstance(cfg, dict):
                raise Fail(400, "config must be an object.")
            secret = None
            if kind == "flag":
                if not b.get("flag"):
                    raise Fail(400, "A flag challenge needs `flag`.")
                secret = str(b["flag"]).strip()
                cfg = dict(cfg, flag_sha256=hashlib.sha256(secret.lower().encode()).hexdigest())
            if kind == "predictions":
                secret = LIE_LABELS
            closes = b.get("closes_at")
            if closes:
                try:
                    datetime.fromisoformat(str(closes).replace("Z", "+00:00"))
                except ValueError:
                    raise Fail(400, "closes_at must be an ISO 8601 time.") from None
            hints = b.get("hints") or []
            if not isinstance(hints, list):
                raise Fail(400, "hints must be a list.")
            cid = S["next_id"]
            S["next_id"] += 1
            S["challenges"].append({
                "id": cid, "slug": slug, "title": str(b["title"]).strip(), "kind": kind,
                "mode": KINDS[kind], "track": str(b.get("track", "")),
                "tier": "learn" if b.get("tier") == "learn" else "compete",
                "description": str(b.get("description", "")), "config": cfg,
                "hidden": bool(b.get("hidden", False)), "closes_at": closes or None,
                "points_initial": int(b.get("points_initial", 500)),
                "points_min": int(b.get("points_min", 100)),
                "points_decay": int(b.get("points_decay", 10)),
                "hints": [{"body": str(h.get("body", "")), "cost": int(h.get("cost", 0))} for h in hints],
                "writeups": [], "solves": [], "entries": [], "secret": secret})
            return {"id": cid, "slug": slug}
        if method == "PATCH" and (mm := m(r"/api/admin/challenges/([^/]+)")):
            c = find(unquote(mm.group(1)), include_hidden=True)
            b = self.body()
            for k in ("title", "description", "hidden", "closes_at", "points_initial",
                      "points_min", "points_decay"):
                if k in b:
                    c[k] = b[k]
            return {"slug": c["slug"]}
        if method == "POST" and (mm := m(r"/api/admin/challenges/([^/]+)/hints")):
            c = find(unquote(mm.group(1)), include_hidden=True)
            b = self.body()
            if not str(b.get("body", "")).strip():
                raise Fail(400, "A hint needs a body.")
            c["hints"].append({"body": str(b["body"]), "cost": int(b.get("cost", 0))})
            return {"position": len(c["hints"])}
        if method == "GET" and path == "/api/admin/usage":
            return {"ndif": {"enabled": False, "calls_today": S["ndif_calls"], "cap_per_day": 150,
                             "cap_per_player_day": 10},
                    "submissions_today": S["submissions_today"],
                    "users": len(S["users"]) + len(PLAYERS), "local_jobs_in_flight": 1}
        raise Fail(404, "No such route.")


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--admin", action="append", default=["admin@northeastern.edu"],
                   help="email that signs in as an organizer (repeatable)")
    p.add_argument("--no-ws", action="store_true", help="refuse /ws, to test the fallback")
    p.add_argument("--demo", type=float, default=0, help="seconds between fake live events")
    p.add_argument("--quiet", action="store_true")
    ARGS = p.parse_args()
    ARGS.admin = {a.lower() for a in ARGS.admin}
    if ARGS.demo > 0:
        threading.Thread(target=demo_loop, args=(ARGS.demo,), daemon=True).start()
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    srv.daemon_threads = True
    print(f"Honne Arena UI stub on http://{ARGS.host}:{ARGS.port}")
    print(f"  organizer sign-in: {', '.join(sorted(ARGS.admin))}")
    print(f"  join code for Null Hypothesis: {fixture_code('Null Hypothesis')}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
