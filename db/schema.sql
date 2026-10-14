-- Arena schema (SQLite). Applied by Database.migrate(); every statement is idempotent.
-- Timestamps are ISO-8601 UTC strings written by the app, so rate-limit windows compare
-- as plain strings. Do not add SQL-side defaults in a different format.

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
  name        text    not null,
  join_code   text    not null unique,
  created_at  text    not null,
  unique (season_id, name)
);

create table if not exists players (
  id          integer primary key,
  season_id   integer not null references seasons(id),
  handle      text    not null,
  team_id     integer not null references teams(id),
  created_at  text    not null,
  unique (season_id, handle)
);

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
  points_decay    integer not null default 10,    -- solves until a flag hits points_min
  hidden          integer not null default 0,
  created_at      text    not null,
  unique (season_id, slug)
);

-- One row per attempt. Flag challenges set `correct` (0/1) and leave `score` null.
-- Scored challenges (steering) leave `correct` null and set `score`.
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
-- A team solves a flag once. Also closes the race where two teammates submit at once.
create unique index if not exists submissions_one_solve_per_team
  on submissions (challenge_id, team_id) where correct = 1;
-- A scored answer is scored once, ever: a repeat would spend a remote model call.
create unique index if not exists submissions_scored_once
  on submissions (challenge_id, answer_key) where correct is null;

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
