#!/bin/sh
# Consistent SQLite snapshot, safe while the app is running. Keeps the last 14.
# Usage: deploy/backup.sh <backup dir> [db path]
set -eu
dest="${1:?backup dir}"
db="${2:-/srv/honne/app/data/arena.db}"
mkdir -p "$dest"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
python3 - "$db" "$dest/arena-$stamp.db" <<'EOF'
import sqlite3, sys
src = sqlite3.connect(sys.argv[1])
out = sqlite3.connect(sys.argv[2])
src.backup(out)
out.close()
src.close()
EOF
ls -1t "$dest"/arena-*.db | tail -n +15 | xargs -r rm -f
echo "backed up to $dest/arena-$stamp.db"
