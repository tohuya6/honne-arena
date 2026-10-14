-- Arena schema (SQLite). Applied by Database.migrate(); every statement is idempotent.
-- Timestamps are ISO-8601 UTC strings written by the app, so rate-limit windows compare
-- as plain strings. Do not add SQL-side defaults in a different format.

-- One row per person, across every season. Sign-in is by NU email, so the email is the
-- identity and the handle is only a display name.
create table if not exists users (
  id          integer primary key,
  email       text    not null unique,            -- always stored lowercased
  -- Null until the user picks one. Unique ignoring case. SQLite allows many nulls.
  handle      text    unique collate nocase,
  created_at  text    not null
);

-- Magic-link tokens. Only the sha256 of the token is stored, so a leaked database
-- cannot be used to sign in. Rows double as the sign-in rate-limit ledger.
create table if not exists login_tokens (
  id            integer primary key,
  token_sha256  text    not null unique,
  email         text    not null,
  ip_hash       text,
  expires_at    text    not null,
  used_at       text,                             -- set once, on the single use
  created_at    text    not null
);
create index if not exists login_tokens_email_idx on login_tokens (email, created_at);
create index if not exists login_tokens_ip_idx on login_tokens (ip_hash, created_at);

-- Signed-in sessions, keyed by the sha256 of the cookie value.
create table if not exists sessions (
  id            integer primary key,
  token_sha256  text    not null unique,
  user_id       integer not null references users(id),
  expires_at    text    not null,
  created_at    text    not null
);
create index if not exists sessions_user_idx on sessions (user_id);

create table if not exists seasons (
  id          integer primary key,
  name        text    not null,
  active      integer not null default 0,
  created_at  text    not null
);
-- At most one live season. Partial index: any number of rows may be inactive.
create unique index if not exists seasons_one_active on seasons (active) where active = 1;

create table if not exists teams (
  id          integer primary key,
  season_id   integer not null references seasons(id),
  name        text    not null collate nocase,
  join_code   text    not null unique,
  created_at  text    not null,
  unique (season_id, name)
);

-- A user's presence in one season. The display handle comes from users.
create table if not exists players (
  id          integer primary key,
  season_id   integer not null references seasons(id),
  user_id     integer not null references users(id),
  team_id     integer not null references teams(id),
  created_at  text    not null,
  unique (season_id, user_id)
);
create index if not exists players_team_idx on players (team_id);
create index if not exists players_user_idx on players (user_id);

create table if not exists challenges (
  id              integer primary key,
  season_id       integer not null references seasons(id),
  slug            text    not null,
  title           text    not null,
  kind            text    not null,               -- key into app.challenges.KINDS
  -- Which contest family it belongs to, e.g. steering, lie-detection, circuits, ctf.
  track           text    not null default '',
  -- "learn": guided, hints free, small models, open to everyone.
  -- "compete": hidden test set, counts fully toward the season board.
  tier            text    not null default 'compete',
  description     text    not null default '',
  config          text    not null default '{}',  -- JSON, server-side only (holds flag hashes)
  points_initial  integer not null default 500,
  points_min      integer not null default 100,
  points_decay    integer not null default 10,    -- solving teams until a flag hits points_min
  hidden          integer not null default 0,
  closes_at       text,                           -- ISO UTC, null = never closes
  created_at      text    not null,
  unique (season_id, slug)
);

-- One row per attempt. Flag and code challenges set `correct` (0/1) and leave `score`
-- null. Scored challenges (steering, predictions) leave `correct` null and set `score`.
create table if not exists submissions (
  id            integer primary key,
  challenge_id  integer not null references challenges(id),
  player_id     integer not null references players(id),
  team_id       integer not null references teams(id),
  answer        text    not null,
  answer_key    text    not null,                 -- normalized form for dedup
  correct       integer,
  score         real,
  detail        text,                             -- JSON
  ip_hash       text,
  created_at    text    not null
);
create index if not exists submissions_challenge_idx on submissions (challenge_id, correct);
create index if not exists submissions_ip_time_idx on submissions (ip_hash, created_at);
create index if not exists submissions_time_idx on submissions (created_at);
create index if not exists submissions_player_idx on submissions (player_id);
-- Each player solves a flag once. A team still earns the points only once, which the
-- standings code handles. Also closes the race where one player submits twice at once.
create unique index if not exists submissions_one_solve_per_player
  on submissions (challenge_id, player_id) where correct = 1;
-- A scored answer is scored once, ever: a repeat would spend a remote model call.
create unique index if not exists submissions_scored_once
  on submissions (challenge_id, answer_key) where correct is null;

-- Hints, numbered from 1 within a challenge. Unlocking one costs `cost` points, except
-- on learn-tier challenges, where hints are free whatever is stored here.
create table if not exists hints (
  id            integer primary key,
  challenge_id  integer not null references challenges(id),
  position      integer not null,
  body          text    not null,
  cost          integer not null default 0,
  created_at    text    not null,
  unique (challenge_id, position)
);

create table if not exists hint_unlocks (
  id          integer primary key,
  hint_id     integer not null references hints(id),
  player_id   integer not null references players(id),
  created_at  text    not null,
  unique (hint_id, player_id)
);
create index if not exists hint_unlocks_player_idx on hint_unlocks (player_id);

-- Writeups, one per user per challenge, shown once the challenge has closed.
create table if not exists writeups (
  id            integer primary key,
  challenge_id  integer not null references challenges(id),
  user_id       integer not null references users(id),
  body          text    not null,
  created_at    text    not null,
  updated_at    text    not null,
  unique (challenge_id, user_id)
);

-- Every remote model call, including probe-baseline builds (player_id null).
-- This is what the NDIF budget counts, so it survives restarts.
create table if not exists ndif_calls (
  id            integer primary key,
  player_id     integer,
  challenge_id  integer,
  ok            integer not null,
  created_at    text    not null
);
create index if not exists ndif_calls_time_idx on ndif_calls (created_at);
create index if not exists ndif_calls_player_idx on ndif_calls (player_id, created_at);
