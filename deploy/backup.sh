#!/bin/sh
#
# Nightly pg_dump, run inside the compose network by the `backup` service.
#
#   backup.sh           dump once and exit  (what restore-testing uses)
#   backup.sh --loop    dump daily at ~03:15 UTC  (what compose runs)
#
# Custom format (-Fc), not plain SQL: it is compressed, and pg_restore can
# read a single table out of it. A 2 GB plain-text dump you have to load in
# full to recover one row is a backup you will not use.
#
# WHAT THIS DOES NOT DO: leave the machine. A backup on the same disk as the
# database survives a bad migration and nothing else — not a failed volume,
# not a deleted VM, not ransomware. DEPLOY.md's off-site step is the half
# that makes this a real backup, and it is not optional.

set -eu

BACKUP_DIR="${BACKUP_DIR:-/backups}"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-14}"

dump() {
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    out="$BACKUP_DIR/cubearena_${ts}.dump"
    mkdir -p "$BACKUP_DIR"

    echo "[backup] $(date -u +%FT%TZ) dumping ${PGDATABASE} -> $(basename "$out")"
    # Write to .partial first and rename on success. A dump interrupted
    # halfway is not a backup, and the one thing worse than no backup is a
    # truncated one that looks fine in `ls`.
    if pg_dump -Fc -Z6 --no-owner --no-privileges -f "${out}.partial"; then
        mv "${out}.partial" "$out"
        echo "[backup] ok: $(du -h "$out" | cut -f1)"
    else
        rm -f "${out}.partial"
        echo "[backup] FAILED" >&2
        return 1
    fi

    # Prune old dumps, but only ever after a SUCCESSFUL new one — otherwise
    # a week of failing backups silently deletes the last good copy.
    find "$BACKUP_DIR" -name 'cubearena_*.dump' -type f -mtime "+${KEEP_DAYS}" -print -delete

    n=$(find "$BACKUP_DIR" -name 'cubearena_*.dump' -type f | wc -l)
    echo "[backup] ${n} dump(s) retained, keeping ${KEEP_DAYS} days"
}

if [ "${1:-}" = "--loop" ]; then
    # Dump once at startup so a fresh deploy has a restore point immediately
    # rather than after the first night.
    dump || true
    while true; do
        # Seconds until the next 03:15 UTC, by ARITHMETIC ONLY.
        #
        # No `date -d`: this runs on postgres:17-alpine, whose busybox date
        # does not parse "YYYY-MM-DD HH:MM:SS" the way GNU date does. It
        # fails silently and returns the current time, which would turn this
        # into a hot loop dumping the database continuously.
        #
        # The Unix epoch starts at midnight UTC, so `now % 86400` IS the
        # seconds since midnight, with no timezone reasoning needed.
        # 11700 = 03:15.
        now=$(date -u +%s)
        delta=$(( (11700 - (now % 86400) + 86400) % 86400 ))
        [ "$delta" -eq 0 ] && delta=86400
        sleep "$delta"
        dump || true
    done
else
    dump
fi
