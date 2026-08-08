# CubeArena API (Django + DRF)

The backend. Started 2026-08-05; implements SYSTEM_DESIGN S0 and the timing
half of S2.

## The one architectural thing to know

**Frontend and backend are not hosted separately.** One box, one domain, one
`docker compose up`. Caddy routes `/api/*` here and everything else to Next:

```
                    :8080  Caddy
                      |
        /api/*  ------+------  /*
           |                    |
     Django (gunicorn)      Next (node)
```

Same scheme + host + port means the browser sees **one origin**, so there is
no CORS anywhere in this project, session cookies are first-party, and there
is one TLS cert. If you ever need CORS headers to make something work, the
routing is wrong — fix `Caddyfile`, don't punch a hole in the API.

Neither app publishes a port of its own. That is deliberate: a directly
reachable Django is a second origin and a way around the proxy.

## Run it

Locally, without Docker or Postgres:

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py test          # 38 tests; most of them are the attacks
python manage.py runserver 8001
```

That path uses SQLite, which is a **dev convenience only** — settings refuse
to start with `DJANGO_DEBUG=0` and no `POSTGRES_HOST`, because SQLite's
table-wide write lock deadlocks concurrent writes under more than one
gunicorn worker.

The whole stack:

```bash
cp server/.env.example .env    # then set DJANGO_SECRET_KEY + POSTGRES_PASSWORD
docker compose up --build      # from the repo root
# -> http://localhost:8080
```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/health/` | Liveness, plus the live anticheat thresholds |
| `POST` | `/api/waitlist/` | Landing-page signup. Honeypot + dedupe |
| `GET`/`POST` | `/api/waitlist/unsubscribe/<token>/` | One-click unsubscribe |
| `POST` | `/api/scrambles/` | Issues a signed, single-use scramble |
| `GET` | `/api/solves/` | The caller's own solves |
| `POST` | `/api/solves/` | Submit a solve; **the server** decides the verdict |
| — | `/api/auth/browser/v1/*` | allauth headless — signup, login, verify, reset |
| — | `/api/admin/` | Django admin — the human-review queue |

## Auth

**allauth in headless mode**, because the frontend is Next and the backend is
a REST API. Headless exposes login/signup/verify/reset as JSON under
`/api/auth/` and lets Next own every screen — the normal allauth flow would
render its own Django templates and put two different UIs on one product.

Sessions, not tokens: Caddy serves both halves from one origin, so the
session cookie is first-party and just works. A JWT scheme would be extra
machinery bought for a cross-origin problem this deployment does not have,
and would need its own refresh, revocation and storage story.

Email verification is **mandatory**. A leaderboard where an unverified
address can hold a top slot is one where a banned user re-registers in
seconds.

### CSRF — read this before writing the auth UI

The headless *browser* client is session-based, so **every unsafe request
needs a CSRF token**. Miss it and you get a bare `403` with an HTML error
page, which looks nothing like an auth problem. Found the hard way: the
endpoints tested green because Django's test client disables CSRF checking
by default, and the same call over real HTTP returned 403.

```js
// once, on load — this sets the csrftoken cookie
await fetch('/api/auth/browser/v1/config', { credentials: 'include' });

// then on every POST/PUT/DELETE
await fetch('/api/auth/browser/v1/auth/signup', {
  method: 'POST',
  credentials: 'include',                    // send + receive cookies
  headers: {
    'content-type': 'application/json',
    'X-CSRFToken': getCookie('csrftoken'),   // required
  },
  body: JSON.stringify({ email, password }),
});
```

`AllauthCsrfTests` pins this with `enforce_csrf_checks=True` so it cannot
regress silently again.

A successful signup returns **401 with `verify_email: is_pending`**, not 200
— the account exists but is not logged in until the address is confirmed.
That is the correct response for mandatory verification, not an error.

> `Authorization: Device <alphanumeric>` still works and is a **stub**
> (`auth.py`) — it mints a user for any string it is shown. It exists so the
> solve endpoints were buildable before the auth UI. **Delete it before
> launch.** The Next-side auth screens are not built yet, which is the only
> reason it is still here.

## Waitlist and sending

The `WaitlistSignup` table **is** the mailing list — no provider, no webhook
mirror, decided 2026-08-05 to cut complexity. `manage.py send_waitlist` is
what mails it:

```bash
# who would get it
python manage.py send_waitlist --campaign launch --subject "We're live" \
    --body launch.txt --dry-run

# yourself first, every time
python manage.py send_waitlist --campaign launch --subject "We're live" \
    --body launch.txt --only you@example.com

# the real thing
python manage.py send_waitlist --campaign launch --subject "We're live" \
    --body launch.txt
```

It is **campaign-keyed and resumable**: each row records the last campaign it
received, so a crash or Ctrl-C halfway through is recovered by re-running.
Without that, the recovery action is "mail everyone twice", which is how a
launch announcement becomes a spam report.

Every message carries a per-recipient unsubscribe link plus
`List-Unsubscribe` / `List-Unsubscribe-Post` headers — that is what Gmail and
Outlook actually read, and it is legally required besides.

**Point `EMAIL_HOST` at a transactional relay** (Postmark, SES, SendGrid,
Mailgun), not a self-hosted SMTP server. That is not the same as handing over
your list — the table stays the source of truth and the relay only carries
the message. It matters because bulk mail from an unknown IP largely lands in
spam, and SPF/DKIM/DMARC alignment plus IP reputation are what fix that.
Unset `EMAIL_HOST` prints to the console, which is what dry-run development
wants.

## The part that matters: never trust the client's clock

The obvious attack is `POST {"time": 0.01, "verified": true}`. The rule, and
`core/timing.py` exists to enforce it:

> No number a client sends is ever a fact. Every value a verdict rests on is
> either stamped by the server's clock or derived from evidence the server
> re-analyzes.

Three bounds, none of which the client controls. None is sufficient alone —
the point is that a forgery must satisfy all three at once:

1. **Server wall-clock window.** `received_at − issued_at`, both server
   stamps. Honest about its limits: this is an *upper* bound only, and does
   nothing against claiming *faster* than reality — which is what an attacker
   actually wants. Easy to mistake for a defence; it isn't one.
2. **Evidence-derived duration.** `frame_count / fps` is authoritative. The
   client's number is stored as `claimed_seconds` purely so a disagreement
   can be flagged.
3. **Move-rate plausibility — the one that closes it.** The timing attack
   *collapses into the move-count attack already solved*: to claim a fast
   time you must ship a short bundle, which leaves you either below the move
   floor (32 QTM — rejected as substitution or solver-following) or at an
   impossible turn rate. There is no gap between those two.

Demonstrated against a running server:

```
POST "it took 0.01s"                   -> rejected | claimed_time_contradicts_evidence
...so shrink the bundle instead        -> rejected | duration_below_human_floor
...keep moves up to clear the floor    -> rejected | implied_tps_impossible:40.0>20
...inflate fps to shorten it           -> rejected | implausible_fps:100000
swap in a solved cube (0 moves)        -> rejected | too_few_moves:0<32
follow a solver's 20-move solution     -> rejected | too_few_moves:20<32
solve after stopping the timer         -> rejected | moves_after_timer_stop:45>12
elite 7.5 TPS solve                    -> review   | too_fast_to_separate
dark room, count reads low             -> review   | lighting_unreadable
replayed bundle                        -> HTTP 409 | scramble consumed
```

The last two are the discipline that makes the rest trustworthy: evidence
that is *unreadable* abstains, it does not accuse. Bad light destroys the
onset detector (−40 pts) and the move count *is* that detector's output, so
rejecting there would manufacture a cheat out of a model failure.

Thresholds are imported from `ble/move_detector/anticheat_gate.py` rather
than restated — a second copy would drift, and for a number that decides
whether someone is called a cheat that is the worst possible failure.

## What this does NOT do yet

- **Frame counts are reported, not measured.** Bound 2 is currently an
  integrity check on a self-report. It becomes a real measurement when the
  re-verification worker lands (S3); the code shape does not change. This is
  the biggest remaining gap in the whole verdict path.
- **Fabricated frames pass everything here.** Every bound reads the evidence
  rather than authenticating its origin. That is camera injection, and it
  needs challenge-response — ANTICHEAT.md §4.
- **The Postgres path has not been run end to end.** Settings switching,
  the `DJANGO_DEBUG=0` guard and `check --deploy` are verified, and the
  migration contains nothing backend-specific (plain `CreateModel`/`AddIndex`,
  no raw SQL) — but the machine this was built on has neither Docker nor a
  local Postgres, so the 38 tests ran on SQLite. `docker compose up` is the
  first real exercise of it.
- **No auth UI.** The allauth endpoints are mounted and tested; the Next
  screens they redirect to (`/auth/verify-email/{key}` etc., listed in
  `HEADLESS_FRONTEND_URLS`) do not exist yet. Those keys must stay in sync
  with what Next serves or verification links 404.
- **`DeviceKeyAuthentication` is still enabled.** Delete it before launch.
