# API contract

The backend (`core` branch) implements this and the frontend (`ui` branch) is built
against it. Change it only by agreement, and update this file in the same commit.

All JSON. Errors are `{"error": "<message safe to show>"}` with a status code:
400 bad input, 401 not signed in, 403 not allowed, 404 missing, 409 duplicate,
429 rate limited, 503 scoring unavailable, 504 backend timeout.

Signed-in state is an HttpOnly cookie `honne_session` (SameSite=Lax). The browser never
sees the token. Every POST body is JSON, which with SameSite=Lax blocks cross-site
form posts.

## Types

```
User      {id, handle: str|null, email: str, is_admin: bool}     # email only ever sent to its owner
Tier      {tier: 0|1|2|3, rank: int|null, points: int}           # 0 = not on the board yet
Hint      {position: int, cost: int, unlocked: bool, body: str|null}
BoardRowSolve  {team, handle, created_at}                        # first solve per team, first blood first
BoardRowScore  {rank, handle, team, score, answer: str|null, created_at}  # best per player; answer only for steering
Challenge {slug, title, kind, mode: "solve"|"score", track, tier: "learn"|"compete",
           description,            # markdown, render escape-first
           config,                 # public config only (flag_format, samples, metric, token_budget, ...)
           points: int,            # current value (flags decay)
           solves: int|null,       # solve mode
           entries: int|null,      # score mode
           closes_at: str|null, closed: bool,
           solved_by_me: bool,     # I personally solved it (solve mode)
           solved_by_team: bool,   # anyone on my team solved it (solve mode)
           submitted_by_me: bool,  # I have any submission on it (gates the writeup form)
           my_best: float|null}
```

Kinds today: `flag` (solve), `code` (solve), `predictions` (score), `steering` (score),
`circuits` (score). Circuits' public config includes `n_layers`, `n_heads`, `max_heads`.
Tracks: `ctf`, `lie-detection`, `steering`, `circuits`, `cp`.

## Auth

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/auth/request` | `{email}` | `{ok: true}`. 400 if not an allowed domain (`@northeastern.edu`). Dev only: `{ok, dev_link}` when `EMAIL_BACKEND=console` and `DEV_EXPOSE_LINKS=true` |
| POST | `/api/auth/verify` | `{token}` | `{user: User}` and sets the cookie. Tokens are single use, 15 minutes |
| POST | `/api/auth/logout` | `{}` | `{ok: true}`, clears the cookie |
| GET | `/api/me` | | `{user: User|null, team: {name, join_code, members: [handle]}|null, tiers: {<track>: Tier}}`. Any member sees the join code |
| POST | `/api/me/handle` | `{handle}` | `{user: User}`. 1-32 chars `[\w \-.]`, unique. Required before submitting |

The emailed link is `/verify.html?token=...`. That page must not consume the token on
load (Outlook Safe Links prefetches every link). It shows a "Sign in" button that POSTs
`/api/auth/verify`.

## Play

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/season` | | `{id, name, captcha_sitekey: str|null, seasons: [{id, name, active}]}` |
| GET | `/api/challenges?track=&tier=` | | `{challenges: [Challenge]}` |
| GET | `/api/challenges/{slug}` | | Hidden challenges are 404 except to admins. `Challenge` plus `{board: [BoardRow...], hints: [Hint], writeups: [{handle, body, created_at}]}` (writeups only after close) |
| POST | `/api/submit` | `{challenge, answer, turnstile_token?}` | solve: `{challenge, mode, correct, first_blood: bool}`. score: `{challenge, mode, score, ...detail}`. code adds `{verdict, passed, total}`. 401 if signed out, 400 if no handle yet, 409 duplicate, 403 if closed |
| POST | `/api/challenges/{slug}/hints/{position}/unlock` | `{}` | `Hint` with body. Costs points except on `learn` challenges |
| POST | `/api/challenges/{slug}/writeup` | `{body}` | `{ok: true}`. Only after close, only by someone who submitted. Plain text, max 20k chars, one per person (re-posting replaces) |

## Teams

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/teams` | `{name}` | `{team, join_code}` |
| POST | `/api/teams/join` | `{join_code}` | `{team, members: [handle]}`. Only before your first submission this season |

## Boards

| Method | Path | Returns |
|---|---|---|
| GET | `/api/leaderboard?season=` | `{season: {id, name}, teams: [{rank, team, points, solves}]}` (default active season) |
| GET | `/api/tracks` | `{tracks: [{track, board: [{rank, handle, points, tier}]}]}`, all-time across seasons |
| WS | `/ws` | server pushes `{type: "solve", challenge, handle, team, first_blood, at}` and `{type: "entry", challenge, handle, team, score, at}`. Wrong answers are never broadcast. Clients refetch what changed |

Tiers per track, all-time, from fixed ranks on `/api/tracks`: t3 = rank 1-5,
t2 = rank 6-15, t1 = any other ranked entry.

## Static

- `/` serves `web/`. `/datasets/...` serves `data/public/` (public challenge files such
  as unlabeled test items).

## Admin

Allowed for a signed-in user whose email is in `ADMIN_EMAILS`, or with header
`X-Admin-Token`.

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/admin/seasons` | `{name, active}` | season row |
| GET | `/api/admin/challenges` | | `{challenges: [Challenge + {hidden, points_initial, points_min, points_decay}]}` incl. hidden, never secrets |
| POST | `/api/admin/challenges` | challenge fields + `flag?` + `hints?: [{body, cost}]` + `closes_at?` | `{id, slug}` |
| PATCH | `/api/admin/challenges/{slug}` | any of `{title, description, hidden, closes_at, points_initial, points_min, points_decay}`. `closes_at: null` reopens | `{slug}` |
| POST | `/api/admin/challenges/{slug}/hints` | `{body, cost}` | `{position}` |
| GET | `/api/admin/usage` | | `{ndif: {enabled, calls_today, cap_per_day, cap_per_player_day}, submissions_today, users, local_jobs_in_flight}` |
