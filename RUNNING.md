# Running CubeArena locally

No Docker needed. Two terminals.

---

## How the two halves fit together

You had this right: **Django is headless.** It renders no pages. It is a
program that listens on a port, receives HTTP requests, and answers with
JSON. Next is the only thing that produces anything a human looks at.

The wrinkle is that they are two separate programs on two separate ports,
and a browser treats `localhost:3000` and `localhost:8000` as two different
**origins**. Browsers deliberately make cross-origin requests painful: the
server has to send explicit CORS permission headers, and cookies (which is
how login works) need extra configuration on both ends. All of that is real
work you would then have to undo.

So we don't do it. **The browser only ever talks to one port.**

```
                    browser
                       |
                       |  everything, one origin
                       v
              Next.js  :3000
                 |            \
    pages, landing page        \   anything starting with /api/
    (Next answers itself)       \  is quietly forwarded
                                 v
                            Django  :8000
                                 |
                                 v
                          database (SQLite locally,
                                    Postgres in prod)
```

The forwarding is four lines in `web/next.config.ts` (`rewrites`). The
browser asks `:3000` for `/api/health/`, Next passes it to Django, and hands
the answer back. As far as the browser is concerned there is one server. No
CORS, and cookies just work.

**This is the same shape as production.** There, Caddy does the forwarding
instead of Next, from one real domain. Same rule (`/api/*` → Django,
everything else → the frontend), so local and production behave identically
and there is no class of bug that only appears after deploying.

---

## Run it

**Terminal 1 — Django (the API):**

```bash
cd server
../.venv/Scripts/python manage.py migrate      # first time only
../.venv/Scripts/python manage.py runserver 8000
```

**Terminal 2 — Next (the site):**

```bash
cd web
npm install                                    # first time only
npm run dev
```

Then open **<http://localhost:3000>**. That is the only URL you need — do
not open `:8000` in the browser, it is not meant to be looked at.

`http://127.0.0.1:3000` works too, but only because `allowedDevOrigins` in
`web/next.config.ts` lists it. Next 16 blocks its dev endpoints from origins
that are not on that list, and the failure mode is nasty: the page renders
but React never hydrates, so no effect and no click handler ever runs. It
looks like the app is hung, not like a dev-server problem. The only clue is a
"Blocked cross-origin request" warning in terminal 2. Add any other host you
browse from to that list.

| What | Where |
|---|---|
| Landing page + waitlist form | <http://localhost:3000> |
| API health + live anticheat thresholds | <http://localhost:3000/api/health/> |
| Django admin (solve review queue) | <http://localhost:3000/api/admin/> |
| Auth endpoints (JSON, no UI yet) | `/api/auth/browser/v1/*` |

For the admin you need a login:

```bash
cd server && ../.venv/Scripts/python manage.py createsuperuser
```

### Demo players

`seed_demo` creates 15 **real** accounts — real password hashes, real
verified email rows, real solves tied to them by foreign key. Not fixtures:
the leaderboard reads the same tables the app does, so you can sign in as any
of them and see the product from their side.

```bash
cd server
../.venv/Scripts/python manage.py seed_demo          # create or top up
../.venv/Scripts/python manage.py seed_demo --reset  # start clean
```

Sign in with `<username>@demo.cubearena.local` / `cubearena123` — e.g.
`feliks_zx@demo.cubearena.local`. It refuses to run with `DJANGO_DEBUG` off,
since every account shares one printed password.

### What you can actually test right now

- **The waitlist form**, end to end: scroll to the bottom of the landing
  page, enter an address, submit. It writes a row to the database. Check it
  in the admin under *Waitlist signups*.
- **The anticheat API**, with curl (there is no UI for it yet):

  ```bash
  # get a scramble
  curl -X POST http://localhost:3000/api/scrambles/ \
      -H "Authorization: Device mydevice" -H "content-type: application/json" -d '{}'

  # submit a solve against its nonce
  curl -X POST http://localhost:3000/api/solves/ \
      -H "Authorization: Device mydevice" -H "content-type: application/json" \
      -d '{"nonce":"<paste>","frame_count":660,"fps":30,"observed_moves":58,
           "post_stop_seconds":10,"lighting_ok":true}'
  ```

  Then look at the same solve in the admin. Try `"observed_moves": 20` to
  watch it get rejected as solver-following.

### What is NOT wired yet

`/compete`, `/compete/play` and `/solve/[id]` still render **mock data**.
Everything else is wired: `/home`, `/leaderboard`, `/settings` and the
profile pages all read the API (`/api/me/`, `/api/leaderboard/`,
`/api/users/<username>/`). See `SETTINGS_AUDIT.md` for what each settings
control does and what is still deliberately inert.

Auth *is* wired, end to end. Every page under `app/(app)/` sits behind
`RequireAuth`, which bounces a signed-out visitor to
`/auth/login?next=<where they were going>`. The screens are:

| Route | What it does |
|---|---|
| `/auth/login` | Sign in, and sign up via the same form (`?mode=signup`) |
| `/auth/signup` | Redirects into the above — allauth's mail links here |
| `/auth/verify-email/<key>` | Landing page for a confirmation email |
| `/auth/reset-password` | Request a reset link |
| `/auth/reset-password/<key>` | Set the new password |

The API examples above still use the `Authorization: Device <anything>` stub
because the anticheat endpoints have not been moved onto session auth yet.

### Signing up locally

`ACCOUNT_EMAIL_VERIFICATION` is `mandatory` and, with no `EMAIL_HOST` set,
mail goes to the **Django console**. So signing up prints the confirmation
link into terminal 1 and the account stays unusable until you open it. Copy
the `/auth/verify-email/...` URL out of that terminal and paste it in.

To skip that while iterating on the UI:

```bash
cd server
ACCOUNT_EMAIL_VERIFICATION=optional ../.venv/Scripts/python manage.py runserver 8000
```

A fresh signup then lands straight in the app. The override only applies when
`DJANGO_DEBUG` is on — production always verifies.

Two allauth behaviours look like bugs and are not: **signup** and **password
reset** both answer `401` on SUCCESS (`meta.is_authenticated: false`),
because neither one signs you in. `lib/auth.tsx` translates both; if you add
another allauth flow, check its success status before treating a 401 as a
failure.

---

## Two gotchas already fixed (so you don't rediscover them)

1. **Trailing slashes.** Django's URLs end in `/` and it redirects to add
   one; Next strips them by default. Left alone, the two bounce a request
   between each other forever and every API call returns `301`.
   `skipTrailingSlashRedirect` plus an explicit slash on the rewrite
   destination settles it.
2. **allauth is the exception.** Its routes do *not* end in a slash, so it
   gets its own rewrite rule ahead of the general one. Without that the
   whole auth surface 404s, which reads like "allauth isn't installed"
   rather than a routing problem.

---

## Docker (later, not now)

`docker compose up --build` runs all four pieces — Caddy, Next, Django,
Postgres — on <http://localhost:8080> with one command. It needs Docker
Desktop installed and a `.env` (copy `server/.env.example`). Worth doing
before deploying, since it is the first thing that exercises Postgres; the
two-terminal setup above uses SQLite and is fine for everything else.
