"""Central config, read from env / .env. Anything here can be overridden by an
environment variable of the same name (case-insensitive). Secrets stay server-side.

The NDIF block is deliberately conservative. NDIF is free for research use but it is a
small shared cluster, so every remote call this app makes is one some researcher could
not. Scoring through it is OFF until `NDIF_ENABLED=true`, and the budgets below cap it
even when it is on. See DECISIONS.md, "NDIF use".
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `model_*` names would collide with pydantic's protected namespace, so disable it.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Storage ──
    db_path: str = "data/arena.db"   # SQLite file. ":memory:" for throwaway runs.

    # Public challenge files (unlabeled test items and the like), served at /datasets.
    public_dir: str = "data/public"

    # ── Admin ──
    # Sent as the X-Admin-Token header. Empty disables token access to /admin routes.
    admin_token: str = ""
    # Comma-separated emails. A signed-in user with one of these is an admin.
    admin_emails: str = ""

    # ── Sign-in (magic link by email) ──
    # Comma-separated. The email's domain must equal one of these exactly, so
    # subdomains such as cs.northeastern.edu are not accepted.
    allowed_email_domains: str = "northeastern.edu"
    login_token_minutes: int = 15      # how long an emailed link works (single use)
    session_days: int = 30             # how long a sign-in lasts
    login_per_email: int = 3           # sign-in emails per address per 15 minutes
    login_per_ip_hour: int = 10        # sign-in emails per client IP per hour
    # Builds the emailed link: <public_base_url>/verify.html?token=...
    public_base_url: str = "http://localhost:7860"
    # Set true in production (HTTPS). False lets the cookie work on plain http locally.
    cookie_secure: bool = False

    # ── Email ──
    # "console" logs the link instead of sending it (local dev). "smtp" sends it.
    email_backend: str = "console"
    # Dev only: also return the link in the /api/auth/request response. Has no effect
    # unless email_backend is "console".
    dev_expose_links: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    smtp_timeout_s: int = 10
    email_from: str = ""

    # ── Submissions (all challenge kinds) ──
    rate_per_min: int = 10
    rate_per_day: int = 200
    global_per_day: int = 5000
    leaderboard_max: int = 200
    ip_hash_salt: str = ""

    # ── NDIF (steering challenges only) ──
    ndif_enabled: bool = False
    ndif_api_key: str = ""
    scoring_backend: str = "ndif"      # "ndif", or "local" for a small local model in dev
    local_model_id: str = "EleutherAI/pythia-70m"
    prepend_bos: bool = True
    ndif_timeout_s: int = 120          # deadline for one remote call, queue wait included
    ndif_concurrency: int = 1          # one job in flight at a time
    ndif_calls_per_day: int = 150      # hard global budget of remote calls per rolling day
    ndif_player_per_day: int = 10
    ndif_player_per_min: int = 2
    cache_dir: str = "data/cache"      # probe baselines are computed once and kept here

    # ── Local jobs: sandboxed code runs and small models on this server ──
    local_concurrency: int = 2         # jobs in flight at once
    local_timeout_s: int = 90          # backstop deadline per job, queue wait included
    # "docker" runs submitted code in a locked-down container. "unsafe-subprocess" runs
    # it as a plain child process with rlimits only. It exists for tests on machines
    # without Docker and must never face real users.
    runner_backend: str = "docker"
    runner_image: str = "python:3.12-slim"
    local_device: str = "cpu"          # torch device for small local models
    # CPU threads torch may use for local models, shared by every job in this process.
    # The default leaves most of a shared machine free. 0 keeps torch's own default.
    local_threads: int = 8

    # ── CAPTCHA (Cloudflare Turnstile), active only when the secret is set ──
    turnstile_secret: str = ""
    turnstile_sitekey: str = ""

    # ── Server ──
    port: int = 7860
    # Empty means same-origin only, which is all the bundled frontend needs.
    allowed_origin: str = ""
    # Trusted reverse-proxy hops in front of the app (for spoof-resistant client IPs).
    trusted_proxy_hops: int = 1

    # ── Live feed (/ws) ──
    ws_max_clients: int = 500          # open sockets at once, more are refused
    ws_queue_size: int = 100           # events buffered per client before it is dropped

    @property
    def admin_email_set(self) -> set[str]:
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}

    @property
    def allowed_domain_set(self) -> set[str]:
        return {d.strip().lower() for d in self.allowed_email_domains.split(",")
                if d.strip()}


settings = Settings()
