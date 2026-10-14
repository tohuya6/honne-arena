# Deploying Honne Arena

The arena is one Python process plus a SQLite file. The code track runs submissions in
throwaway Docker containers on the same machine, so the app runs **directly on a Linux
host that has Docker**, not inside a container of its own (giving a container the Docker
socket would hand it root on the host).

```
internet -> Caddy (HTTPS, port 443) -> uvicorn on 127.0.0.1:7860 -> data/arena.db
                                              \-> docker run (one per code test)
```

## What you need

- A Linux VM with 2+ cores, 4 GB RAM, 10 GB disk, and a public DNS name.
  GPT-2 small for the circuits and local steering tracks runs on the CPU.
- Docker, Python 3.12, Caddy.
- SMTP credentials for sign-in links (any provider: Resend, Postmark, SES, or a
  Google Workspace app password). Northeastern's Microsoft 365 often blocks SMTP sign-in
  for student accounts, so a transactional provider is the safer choice.

## Install

```bash
sudo adduser --system --group --home /srv/honne honne
sudo usermod -aG docker honne                 # lets the app start sandbox containers
sudo -u honne git clone https://github.com/tohuya6/honne-arena /srv/honne/app
cd /srv/honne/app
sudo -u honne python3 -m venv .venv
sudo -u honne .venv/bin/pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
sudo -u honne .venv/bin/pip install --no-cache-dir -r requirements-model.txt
sudo docker pull python:3.12-slim             # the sandbox image, pulled once
```

Copy the hidden material (flags, labels, hidden tests) that is not in git:

```bash
rsync -a data/private/ honne@<host>:/srv/honne/app/data/private/
```

Create `/srv/honne/app/.env` from `.env.example`, mode 600, owned by `honne`. At least:

```
ADMIN_TOKEN=<random>
IP_HASH_SALT=<random>
ADMIN_EMAILS=<organizer emails>
PUBLIC_BASE_URL=https://<host>
COOKIE_SECURE=true
EMAIL_BACKEND=smtp
SMTP_HOST=...
SMTP_USER=...
SMTP_PASSWORD=...
EMAIL_FROM=...
HF_HOME=/srv/honne/hf_cache
```

Leave `NDIF_ENABLED=false` until the NDIF team has said a club leaderboard is in scope.

## Run

```bash
sudo cp deploy/honne-arena.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now honne-arena
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile   # edit the host name first
sudo systemctl reload caddy
```

Load the first season:

```bash
sudo -u honne .venv/bin/python scripts/seed.py --season "Fall 2026"
```

## Backups

`deploy/backup.sh` takes a consistent SQLite snapshot with `.backup` (safe while the app
is running). Run it nightly from cron and copy the snapshots off the machine.

```
15 3 * * * /srv/honne/app/deploy/backup.sh /srv/honne/backups
```

## Updating

```bash
cd /srv/honne/app && sudo -u honne git pull && sudo systemctl restart honne-arena
```

The schema is applied idempotently at startup.
