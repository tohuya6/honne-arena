# Decisions log

A running record of what was decided, why, and what is still open. Newest entries go at
the bottom of each section. Dates are when the decision was made.

## Goal (2026-10-14)

One arena for NU AI Safety members to use for both competition and learning. It hosts
accessible replicas of three existing contests, plus classic CTF flags:

| Track | Replica of | How an answer is scored | Model call at submit time? |
|---|---|---|---|
| `steering` | Steering Arena (Soham Padia) | shift of the model's residual along a direction | yes, budgeted |
| `lie-detection` | Aletheia's Quest (Cadenza Labs + NDIF) | AUROC of predictions against hidden labels | no |
| `circuits` | MIB, circuit localization (Mueller et al., ICML 2025) | faithfulness of the submitted heads on GPT-2 small | local only, no NDIF |
| `ctf` | classic CTF | SHA-256 of the flag | no |
| `cp` | competitive programming | sandboxed run against hidden tests | local Docker, no NDIF |

"Accessible" means every track has two tiers:
- `learn`: guided, small models, hints free, open to everyone, good for the weekly talks.
- `compete`: hidden test sets, counts fully toward the season board.

## Base code (2026-10-14)

- Started from Steering Arena's public HF Space. The first commit is Soham's code as
  published, with his authorship kept, so the history shows exactly what came from him.
- Kept: the bounded NDIF queue (`app/queue.py`), rate limits, CAPTCHA, the NDIF residual
  reader, the steering scorer, the retro web UI, and the tests for those parts.
- Dropped: the research record (`data/analysis`, `_falsifier`, writeups), the live
  generation demo, Supabase, and the single-challenge `main.py`/`db.py`/`submission.py`,
  which were rewritten for multiple challenges.

## License and credit (2026-10-14)

- Soham is in the club and is fine with us using it. His Space had no license, so only
  he could license his code. LICENSE is MIT naming both authors.
- 2026-10-14: Soham OK'd MIT. Repo made public as tohuya6/honne-arena.
- Credit for the UI and the steering scorer goes in CREDITS.md and the site footer.

## Name (2026-10-14)

- 2026-10-14: **Honne Arena**, repo `honne-arena`. *Honne* (本音) means one's true
  feelings, as opposed to *tatemae*, the face shown in public. That split is what the
  tracks look for in a model. Two syllables, and no collision with the contests replicated.
- Avoid "Aletheia's Arena": Aletheia's Quest is Cadenza Labs and NDIF's contest, and
  the name would read as an official NDIF spin-off. Revisit if NDIF wants to affiliate.

## Storage (2026-10-14)

- SQLite (`data/arena.db`). A club is tens of players and a few hundred submissions a
  week, so one file is plenty, needs no hosted service, and `:memory:` keeps tests
  self-contained. The db methods are query-shaped so Postgres could replace it later.

## Identity and teams (2026-10-14)

- Superseded 2026-10-14 by NU email sign-in (see "Sign-in and accounts"). Originally:
  handles per season, no accounts. A new handle gets a solo team named after it.
- Teams have a join code that only the creator sees. Players may change team only
  before their first submission, so one player's solves cannot count for two teams.
- Open: handles can be impersonated. Fine among club members. Before opening to other
  schools, add sign-in (Steering Arena's Supabase auth is a ready reference).

## Points (2026-10-14)

- Flags use CTFd's dynamic scoring: full value for the first solve, then a parabola
  down to `points_min` after `points_decay` solves. Every solver holds the current
  value. `app/points.py`.
- Scored challenges (steering, predictions) rank teams by best entry and pay linearly
  from `points_initial` (rank 1) to `points_min` (last).
- One flag solve per team. One scored row per distinct answer, ever (dedup).
  Changed 2026-10-14: every player can solve a flag themselves (personal tiers need
  it), the team earns it once, and decay counts distinct solving teams.

## NDIF use (2026-10-14)

NDIF is free for research use (ndif.us) and nobody is billed. It is a small shared
cluster (one 8xH200 node, six 4xA40 nodes), so the rule is: spend it on purpose, never
by accident.

- `NDIF_ENABLED=false` by default. While off, steering challenges answer 503 and make
  zero calls. Everything else keeps working.
- Budgets, counted from the `ndif_calls` ledger so they survive restarts:
  150 remote calls per day globally, 10 per player per day, 2 per player per minute.
- One submission is one batched forward. The probe baseline is one forward per
  challenge, cached to `data/cache/`, and logged with no player so it counts against
  the global cap only.
- A repeated answer returns the stored score and is never re-scored.
- One remote job in flight at a time, with a 120 s deadline (Steering Arena's gate).
- Tests and local dev never touch NDIF. Tests use a fake scorer. `SCORING_BACKEND=local`
  runs a tiny local model.
- Prefer scoring designs that need no model call at submit time (predictions, lookups).
- To do: ask the NDIF team whether a club leaderboard is in scope for research use and
  whether it should have its own key.

## App structure (2026-10-14)

- `create_app()` factory, run with `uvicorn app.main:create_app --factory`. Importing
  the module has no side effects, so tests build isolated apps.
- Challenge kinds are subclasses of `app.challenges.base.Challenge`, registered in
  `app/challenges/__init__.py`. A kind decides how to normalize and judge an answer.
- Heavy model deps (nnsight, torch, transformers) moved to `requirements-model.txt`.
  The base install runs every non-model track.

## Tiers and identity (2026-10-14)

- Each arena (track) gives a tier from the all-time leaderboard, using fixed ranks:
  t3 = top 5, t2 = top 15, t1 = any ranked entry.
- Tiers are all-time. Reaching t3 in a track stays after a season resets, so a player
  must be one identity across seasons, not a per-season handle.
- Sign-in is by NU email (magic link to @northeastern.edu). An open sign-in such as
  GitHub would let anyone, or any bot, farm the boards. Handles become display names on
  a verified account.
- Tiers are not tied to club membership. That may change later; it is not official.

## Compute modes (2026-10-14)

- Every challenge says where judging runs, via `compute_mode()`:
  `inline` (hash compare, AUROC), `local` (sandboxed code, small local models), or
  `ndif` (remote model). `local` and `ndif` each get their own bounded queue, so a burst
  of code runs cannot delay NDIF scoring and the other way round.
- Only `ndif` touches the NDIF budget. A steering challenge with `config.backend =
  "local"` scores on a small model on this server, which gives the steering track a
  learn tier that costs no NDIF at all.
- Circuits changed from "lookup table" to live local scoring: GPT-2 small runs on CPU
  in about a second, so the score can be the real faithfulness of whatever head set is
  submitted, not an approximation from a table.

## Replica notes (2026-10-14)

- Aletheia's Quest is over. It ran Summer 2026 (Cadenza Labs, NDIF, funded by Schmidt
  Sciences), with invited teams meeting in Boston on Aug 25. Final ranking was mean
  AUROC over ten held-out test datasets, in white-box and black-box tracks. Weekly
  prizes used balanced accuracy with AUROC as tie-break. Our lie-detection track uses
  the same metrics. Source: aletheias-quest.github.io/winners.html.

## Hidden material (2026-10-14)

- The repo is public, so nothing committed may give away an answer. Flags, test-set
  labels, and hidden code tests live in `data/private/` (gitignored) and seeds point at
  them by path. See `seeds/README.md`. Deploying means copying `data/private/` by hand.
- Admin emails and the sign-in sender address live in the server's `.env`, never in
  the repo. For now both are the organizer's NU address.

## Build plan (2026-10-14)

Built the same day in parallel branches, merged into `main` after review:
- `runner`: sandboxed code runner (Docker) and the `code` challenge kind for the `cp`
  track.
- `local-models`: GPT-2 small loader, the `circuits` kind, and the local steering
  backend.
- `lie-data`: learn and compete datasets for the lie-detection track.
- `main`: NU email sign-in, all-time identity and tiers, hints, writeups, live
  leaderboard, admin API, then the new UI.

## Code runner (2026-10-14)

- One fresh Docker container per test case: no network, read-only root, a small
  noexec /tmp, memory and swap capped together, one CPU, 64 processes, every
  capability dropped, uid 65534, `--pull never`, and no container logs. Submissions are
  bind-mounted read-only from `/var/tmp/honne-run-*` (a snap-installed Docker daemon
  cannot see the host /tmp).
- A small trusted harness runs as PID 1 inside the container. It times the program,
  reads OOM kills from the cgroup, kills every other process, then writes one status
  line, so a submission cannot forge its own verdict. The host keeps a backstop of the
  time limit plus 10 s and removes the exact container by name.
- `--security-opt no-new-privileges` is probed once per process and dropped when the
  daemon cannot run with it (true of snap Docker on the dev box). With the bounding set
  empty a setuid binary still gains nothing. Production should use a regular Docker
  install so the flag stays on.
- Output is capped at 1 MiB per stream (over is OLE). `wall_ms` is program time
  measured inside the container. Container start costs about 0.7 s per test.
- A failing sample shows expected and actual output (500 chars). A failing hidden test
  shows only the verdict, counts, and which test number failed.
- Known limits: the kernel is shared (runc, default seccomp and AppArmor, no gVisor),
  timing is noisy on a busy machine, and a verdict leaks about one bit per submission,
  which only rate limits bound. Python only for now.
- Seeds: `sum-a-list` (learn) and `range-sums`, `pair-count` (compete). A naive
  solution gets TLE on both compete problems. Hidden tests and reference solutions are
  in `data/private/cp/`.

## Local models: circuits and steering (2026-10-14)

- GPT-2 small runs on this server's CPU. "Layer L" means the output of block
  `transformer.h[L]` before the final norm, the same as Steering Arena's NDIF reader.
  Left-padded batches pass explicit position ids, because GPT-2 does not derive them
  from the attention mask. One lock per model, since hooks are shared state.
- Circuits replicates MIB's circuit localization on IOI. Nodes are attention heads at
  every position. MLPs and embeddings are never ablated. Every other head is replaced
  by its position-wise mean over an ABC distribution, per template. Faithfulness is
  (LD_circuit - LD_ablated) / (LD_full - LD_ablated) on the prompt-mean logit
  difference.
- Checked numbers on the 180 public prompts: all heads 1.00, empty set 0.00, the
  26-head circuit from Wang et al. (2022) 0.868 (the paper reports 87%), and the best
  of 30 random 26-head sets 0.28. A submission takes about 2 s.
- Scoring: the learn tier ranks by raw faithfulness (higher wins), where a score above
  1 is itself a lesson about negative name movers. The compete tier ranks by distance
  `|1 - faithfulness|` (lower wins, like MIB's CMD), because raw faithfulness lets a
  set that drops the negative name movers (1.11) beat the real circuit.
- The hidden split reads a seed file from `data/private/` and excludes every public
  prompt. Templates and the generator are public, so the hidden split stops tuning to
  exact prompts, not to the distribution.
- Local steering (learn tier) uses a GPT-2 sentiment direction at layer 9 (a normalized
  mean difference over contrast pairs, script in `scripts/`). Aligned prefixes score
  about +0.026 and opposite ones about -0.024. The vector is in the public repo, like
  Steering Arena's. The compete steering challenge keeps Steering Arena's OLMo-3-32B
  setup on NDIF and stays hidden while NDIF is off.
- Torch threads are now capped, see "Local compute share".

## Sign-in and accounts (2026-10-14)

- Magic link to an `@northeastern.edu` address. The domain must match exactly, so
  subdomains are rejected, and addresses are lowercased. Only sha256 hashes of link and
  session tokens are stored. A link works once for 15 minutes. A session lasts 30 days.
- Only a POST consumes a link. The emailed page shows a "Sign in" button, because
  Outlook Safe Links fetches every link in an email before the person clicks it.
- Session cookie `honne_session`: HttpOnly, SameSite=Lax, and Secure when
  `COOKIE_SECURE=true` (required in production). Every POST body is JSON, which with
  SameSite=Lax blocks cross-site form posts.
- Rate limits on links: 3 per email per 15 minutes, 10 per connection per hour. The
  response never says whether an account exists.
- Email goes through `EMAIL_BACKEND=console` (dev, logs the link) or `smtp`.
  `DEV_EXPOSE_LINKS` returns the link in the response, console backend only.
- Handles are display names, unique ignoring case, changeable, and required before
  submitting. Players are per (season, user). A user gets a solo team on their first
  action in a season. Joining a team is allowed only before your first submission.

## Hints, writeups, tiers, live feed (2026-10-14)

- Hints: each player pays for the hints they unlock. A team pays each distinct hint
  once. Hints on learn challenges are free.
- A challenge can close (`closes_at`). After close, submissions get 403 and anyone who
  submitted may post one plain-text writeup, shown to everyone.
- Track boards are all-time across seasons. Tie-break: more points, then whoever
  reached their total first, then lower user id. A solve or a scored entry puts you on
  the board, and wrong answers don't. Hidden challenges count on no board.
- `/ws` broadcasts solves and scored entries, never wrong answers. Each connection
  has a bounded queue and slow clients are dropped.
- Admin is a signed-in user listed in `ADMIN_EMAILS`, or the `X-Admin-Token` header.
  The admin listing strips secret-looking config keys and keeps `*_file` paths.
- `scripts/seed.py` loads `seeds/*.json` idempotently through the same validation as
  the admin API.
- Known limits at club scale: track boards are recomputed per request, and there is
  one SQLite connection without multi-statement transactions (unique indexes catch
  the races).

## Local compute share (2026-10-14)

- Torch uses at most `LOCAL_THREADS=8` CPU threads by default. A circuits submission
  takes about 2.9 s at 8 threads against 1.7 s at 24, with the same score, and two
  jobs no longer saturate a shared machine.

## Lie-detection data (2026-10-14)

- Learn tier, `true-or-false`: city statements in the style of Marks and Tegmark
  ("The city of Paris is in France."), 237 well-known cities in 46 countries, each
  giving one true and one false statement. False countries are a shuffle of the true
  ones, so the country name carries no signal. No city is in both splits. Label 1 =
  true. A GPT-2 probe at layer 7 scores 0.952 AUROC, and random scores about 0.49.
- Compete tier, `spot-the-liar`: Qwen2.5-0.5B-Instruct answers factual questions under
  a hidden system prompt to answer honestly or to lie, as in Aletheia's Quest. Label
  1 = lie. Kept only lies that name a specific wrong answer from a closed list and
  honest answers that are correct, balanced per category: 240 items, 98 train
  (labeled) and 142 test. Baselines on test: TF-IDF 0.571, Qwen as its own judge
  0.828, a white-box probe on Qwen's activations 0.906, random about 0.50.
- The exact lie and honest prompts and both random seeds stay in `data/private/lie/`.
  With the exact prompts a player could compare an answer's likelihood under each and
  read off the label, and a public seed would rebuild the answer key. The READMEs
  describe the prompts in plain words.
- Both use AUROC. `balanced_accuracy` (threshold 0.5) is available too, as in
  Aletheia's weekly boards.
- Known limits: in the cities set each test city appears once true and once false, a
  shortcut that is fine for learning but not for competing. 142 test items give an
  AUROC standard deviation of about 0.05, so close ranks are noise.

## Open / next

- UI: `web/` is still Steering Arena's single-challenge page and calls the old routes.
  Rework it for the challenge list, teams, and season board. Style direction: plainer
  and more readable (simple white text), keeping the arcade feel.
- Admin page for writing and scheduling challenges (today: the admin API).
