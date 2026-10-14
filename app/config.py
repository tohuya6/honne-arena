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

    # ── Admin ──
    # Sent as the X-Admin-Token header. Empty disables every /admin route.
    admin_token: str = ""

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

    # ── CAPTCHA (Cloudflare Turnstile), active only when the secret is set ──
    turnstile_secret: str = ""
    turnstile_sitekey: str = ""

    # ── Server ──
    port: int = 7860
    # Empty means same-origin only, which is all the bundled frontend needs.
    allowed_origin: str = ""
    # Trusted reverse-proxy hops in front of the app (for spoof-resistant client IPs).
    trusted_proxy_hops: int = 1


settings = Settings()
