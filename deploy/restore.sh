#!/usr/bin/env bash
#
# Restore a dump, or — far more importantly — TEST that one restores.
#
#     ./deploy/restore.sh --list
#     ./deploy/restore.sh --verify              # newest dump -> scratch DB
#     ./deploy/restore.sh --verify <file>
#     ./deploy/restore.sh --restore <file>      # DESTRUCTIVE, asks first
#
# `--verify` is the one to run on a schedule. It restores into a throwaway
# database beside the real one, counts the rows that matter, and drops it.
# Nothing about production is touched, so there is no reason not to run it,
# and it converts "we have backups" from a belief into a measurement.
#
# `--restore` overwrites the live database. It requires typing the database
# name to confirm, because the moment you need this you will be moving fast
# and that is exactly when a wrong argument is unrecoverable.

set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)

# shellcheck disable=SC1091
[ -f .env ] && set -a && . ./.env && set +a
DB="${POSTGRES_DB:-cubearena}"
USER_="${POSTGRES_USER:-cubearena}"

psql_() { "${COMPOSE[@]}" exec -T db psql -U "$USER_" -v ON_ERROR_STOP=1 "$@"; }

newest() {
    "${COMPOSE[@]}" exec -T backup sh -c \
        'ls -1t /backups/cubearena_*.dump 2>/dev/null | head -1'
}

case "${1:---list}" in

--list)
    echo "Dumps on the server:"
    "${COMPOSE[@]}" exec -T backup sh -c 'ls -lht /backups/ 2>/dev/null || echo "  (none yet)"'
    ;;

--verify)
    FILE="${2:-$(newest)}"
    [ -n "$FILE" ] || { echo "no dumps found"; exit 1; }
    SCRATCH="restore_check_$(date -u +%s)"
    echo "==> verifying $FILE into scratch database $SCRATCH"

    psql_ -d postgres -c "CREATE DATABASE \"$SCRATCH\";"
    # shellcheck disable=SC2064
    trap "echo '==> dropping $SCRATCH'; psql_ -d postgres -c 'DROP DATABASE IF EXISTS \"$SCRATCH\";' >/dev/null" EXIT

    "${COMPOSE[@]}" exec -T backup sh -c \
        "pg_restore --no-owner --no-privileges -d '$SCRATCH' -h db -U '$USER_' '$FILE'" \
        || { echo "RESTORE FAILED — this backup is not usable"; exit 1; }

    echo
    echo "==> row counts in the restored copy:"
    psql_ -d "$SCRATCH" -c "
        SELECT 'users'     AS table, count(*) FROM auth_user
        UNION ALL SELECT 'profiles',  count(*) FROM core_profile
        UNION ALL SELECT 'solves',    count(*) FROM core_solve
        UNION ALL SELECT 'scrambles', count(*) FROM core_scramble
        UNION ALL SELECT 'waitlist',  count(*) FROM core_waitlistsignup;"
    echo
    echo "==> RESTORE VERIFIED. Compare the numbers above with production:"
    echo "    ./deploy/restore.sh --counts"
    ;;

--counts)
    psql_ -d "$DB" -c "
        SELECT 'users'     AS table, count(*) FROM auth_user
        UNION ALL SELECT 'profiles',  count(*) FROM core_profile
        UNION ALL SELECT 'solves',    count(*) FROM core_solve
        UNION ALL SELECT 'scrambles', count(*) FROM core_scramble
        UNION ALL SELECT 'waitlist',  count(*) FROM core_waitlistsignup;"
    ;;

--restore)
    FILE="${2:-}"
    [ -n "$FILE" ] || { echo "usage: $0 --restore /backups/cubearena_....dump"; exit 1; }
    echo "This OVERWRITES the live database '$DB'. Everything since that dump is lost."
    read -r -p "Type the database name to confirm: " confirm
    [ "$confirm" = "$DB" ] || { echo "aborted"; exit 1; }

    echo "==> stopping the app so nothing writes mid-restore"
    "${COMPOSE[@]}" stop django next

    # --clean --if-exists drops each object before recreating it, so this
    # works against a populated database rather than only an empty one.
    "${COMPOSE[@]}" exec -T backup sh -c \
        "pg_restore --clean --if-exists --no-owner --no-privileges -d '$DB' -h db -U '$USER_' '$FILE'"

    echo "==> restarting"
    "${COMPOSE[@]}" start django next
    echo "==> done. Check https://\${DOMAIN}/api/health/"
    ;;

*)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
