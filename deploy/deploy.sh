#!/usr/bin/env bash
#
# Deploy, or roll forward to the latest commit. Run on the server, as the
# deploy user, from the repo root:
#
#     ./deploy/deploy.sh
#     ./deploy/deploy.sh --no-pull      # rebuild what is already checked out
#
# Exists so the compose invocation is never typed by hand. Both -f flags are
# load-bearing: naming them explicitly is what stops compose auto-loading
# docker-compose.override.yml and publishing dev's plaintext :8080 next to
# the real site.

set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)
log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
[ -f .env ] || die ".env missing. cp deploy/.env.prod.example .env && chmod 600 .env"

perms=$(stat -c '%a' .env)
[ "$perms" = "600" ] || die ".env is mode ${perms}; it holds the database password
     and the key that signs scrambles.  chmod 600 .env"

# CRLF in .env is a silent disaster and worth its own check. `.` sources it,
# so every value picks up a trailing carriage return: DOMAIN becomes
# "example.com\r", Caddy requests a certificate for a host that does not
# exist, and the ALLOWED_HOSTS comparison below fails against a value that
# looks perfectly correct in any editor. .gitattributes keeps the committed
# example clean, but this file is created by hand and may have been edited on
# Windows or pasted through one.
if grep -q $'\r' .env 2>/dev/null; then
    die ".env has Windows line endings (CRLF).
     Every value would carry a trailing carriage return.  Fix with:
         sed -i 's/\\r\$//' .env"
fi

# shellcheck disable=SC1091
set -a; . ./.env; set +a
for v in DOMAIN ACME_EMAIL DJANGO_SECRET_KEY POSTGRES_PASSWORD; do
    [ -n "${!v:-}" ] || die "$v is empty in .env"
done
[ "${DJANGO_DEBUG:-0}" = "0" ] || die "DJANGO_DEBUG is ${DJANGO_DEBUG}. Never deploy with it on —
     it disables secure cookies, HSTS and the HTTPS redirect, and serves
     tracebacks containing settings to anyone who triggers a 500."

case "${DJANGO_ALLOWED_HOSTS:-}" in
    *"$DOMAIN"*) ;;
    *) die "DJANGO_ALLOWED_HOSTS (${DJANGO_ALLOWED_HOSTS:-unset}) does not contain ${DOMAIN}.
     Django will reject every request with a 400 that reads like a proxy bug." ;;
esac
case "${DJANGO_CSRF_TRUSTED_ORIGINS:-}" in
    *"https://$DOMAIN"*) ;;
    *) die "DJANGO_CSRF_TRUSTED_ORIGINS must contain https://${DOMAIN}, or every
     login and signup POST returns a bare 403." ;;
esac

# --- pull ------------------------------------------------------------------
if [ "${1:-}" != "--no-pull" ]; then
    log "Fetching"
    git pull --ff-only
fi

log "Building"
"${COMPOSE[@]}" build --pull

# --- back up before migrating ---------------------------------------------
# Migrations are the most likely thing in a deploy to be unrecoverable. Take
# a dump first, every time, so rolling back is possible at all.
if "${COMPOSE[@]}" ps --status running --services 2>/dev/null | grep -qx db; then
    log "Backing up before migrating"
    "${COMPOSE[@]}" exec -T backup /usr/local/bin/backup.sh || \
        die "pre-deploy backup failed — refusing to migrate without a restore point"
fi

log "Starting"
"${COMPOSE[@]}" up -d --remove-orphans

# --- verify ----------------------------------------------------------------
log "Waiting for health"
for i in $(seq 1 30); do
    if curl -fsS --max-time 4 "https://${DOMAIN}/api/health/" >/dev/null 2>&1; then
        log "Healthy: https://${DOMAIN}/api/health/"
        "${COMPOSE[@]}" ps
        log "Pruning old images"
        docker image prune -f >/dev/null
        exit 0
    fi
    sleep 4
done

# A first deploy legitimately fails this check while Caddy is still getting
# its certificate, so say what to look at rather than implying disaster.
printf '\n\033[1;33mWARNING: not answering on https://%s/api/health/ yet.\033[0m\n' "$DOMAIN"
cat <<EOF

  On a FIRST deploy this is usually ACME still running. Check with:

      docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f caddy

  The usual causes, in order of likelihood:
    * DNS A record does not point at this VM yet (Let's Encrypt checks it)
    * port 80 blocked, so the HTTP-01 challenge cannot complete
    * DOMAIN in .env does not match the DNS name
EOF
exit 1
