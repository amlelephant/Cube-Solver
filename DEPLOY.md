# Deploying CubeArena

One VM, one `docker compose`, one domain. This is the runbook; the reasoning
behind each choice is inline where it is not obvious.

**Read `RUNNING.md` first if you only want it on your laptop.** This document
is about the public internet, where the failure modes are different.

---

## 0. The shape

```
                        internet
                            │
                     :80 :443 only          ufw drops everything else
                            ▼
                   ┌─────────────────┐
                   │ caddy           │  TLS, security headers, admin allowlist
                   └────────┬────────┘
              /api/*        │        /*
           ┌────────────────┴───────────────┐
           ▼                                ▼
    ┌─────────────┐                  ┌─────────────┐
    │ django      │                  │ next        │   both read-only rootfs,
    │ gunicorn    │                  │ node        │   no capabilities, no ports
    └──────┬──────┘                  └─────────────┘
           ▼
    ┌─────────────┐        ┌─────────────┐
    │ postgres    │◄───────│ backup      │  nightly pg_dump, 14 days
    └─────────────┘        └─────────────┘
```

Only Caddy publishes a port. Django, Next and Postgres are reachable only by
service name inside the compose network — a directly reachable Django is a
second origin that bypasses every header Caddy sets, and a directly reachable
Postgres is just a bad idea.

**Two compose files, never three.** `docker-compose.override.yml` is dev-only
and compose loads it *automatically*, so production names its files
explicitly. `deploy/deploy.sh` does that for you; if you ever run compose by
hand in production, both `-f` flags are mandatory or you will publish dev's
plaintext `:8080` next to the real site.

---

## 1. Pick a provider

Any provider that rents a plain Ubuntu VM works — nothing here is
provider-specific. What matters: 2 vCPU, 4 GB RAM, ~40 GB SSD, and a static
IPv4.

| Provider | Plan | ~cost | Notes |
|---|---|---|---|
| **Hetzner Cloud** | CPX21 (3 vCPU, 4 GB) | **~€8/mo** | Best value by a wide margin. DE/FI/US locations. **Recommended.** |
| DigitalOcean | Basic 2 vCPU / 4 GB | ~$24/mo | Easiest console, best docs, most tutorials when something breaks. |
| Vultr / Linode | equivalent | ~$24/mo | Fine. No particular advantage. |
| AWS Lightsail | 2 vCPU / 4 GB | ~$24/mo | Only if you are already in AWS. Plain EC2 is more moving parts than this needs. |

Why 4 GB and not 2: Postgres, two application runtimes and the build all have
to fit. `docker compose build` on a 2 GB box gets the Next build OOM-killed,
which presents as a confusing `exit code 137` rather than "out of memory".

**Do not** reach for Kubernetes, ECS, or a managed Postgres yet. At this scale
they add failure modes and bills without removing any work — the whole stack
is four containers that fit on one machine, and `SYSTEM_DESIGN.md` §4.1 sizes
the closed beta at well under this VM.

Take the provider's **automatic snapshot/backup** option if offered (usually
~20% of instance cost). It is a different kind of backup from the pg_dumps —
it restores the whole machine, not just the data.

---

## 2. DNS

Point the domain at the VM **before** deploying. Let's Encrypt validates by
connecting to the name over port 80; if DNS is not live, certificate issuance
fails and the site never comes up.

```
A     cubearena.example       -> <VM IPv4>
A     www.cubearena.example   -> <VM IPv4>
AAAA  cubearena.example       -> <VM IPv6>   (if the provider gives you one)
```

Wait for it to actually resolve — `dig +short cubearena.example` from your
laptop — before step 4. TTL on a fresh record is usually minutes.

---

## 3. Harden the VM

Once, on a fresh Ubuntu 24.04 image, from your laptop:

```bash
ssh-copy-id root@<ip>                              # if you have no key there yet
ssh root@<ip> 'bash -s' < deploy/bootstrap.sh
```

That script patches the box and turns on automatic security updates, creates
a `deploy` user, locks SSH to key-only with no root login, sets `ufw` to deny
everything except 22/80/443, installs fail2ban on SSH, and installs Docker
from Docker's own repository.

**It refuses to disable password authentication unless a key is already
installed for `deploy`.** That check is why it is a script: "disable password
auth, then discover the key never copied" costs you the server.

Verify before closing the root session — that shell is your only way back in
if SSH is broken:

```bash
ssh deploy@<ip> 'docker ps && sudo ufw status'
```

---

## 4. Configure and deploy

```bash
ssh deploy@<ip>
git clone <your-repo-url> ~/cubearena && cd ~/cubearena

cp deploy/.env.prod.example .env
chmod 600 .env                       # deploy.sh refuses to run otherwise
$EDITOR .env                         # every REQUIRED field; see the comments

./deploy/deploy.sh
```

`deploy.sh` refuses to start on the mistakes that are painful to diagnose
later: `DJANGO_DEBUG=1`, a `.env` readable by other users, a domain missing
from `DJANGO_ALLOWED_HOSTS` (which surfaces as a 400 that reads like a proxy
bug), or a missing `https://` entry in `DJANGO_CSRF_TRUSTED_ORIGINS` (which
surfaces as every login returning a bare 403).

It also takes a database dump before running migrations, every time, so a bad
migration is recoverable rather than terminal.

First run takes a few minutes — two images build, and Caddy fetches a
certificate. If health does not come up, it is almost always DNS or port 80;
`deploy.sh` prints the check list.

### Create your admin user

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec django python manage.py createsuperuser
```

Then set `ADMIN_ALLOW_CIDR` in `.env` to your own address
(`curl -s https://ifconfig.me` gives it) and re-run `./deploy/deploy.sh`.
Until you do, `/api/admin/` is reachable from anywhere and a password is the
only thing in front of every row in the database. With it set, Caddy answers
**404** to everyone else, so the admin does not appear to exist at all.

---

## 5. What is actually defended, and what is not

Being specific, because "secure" on its own is not a claim anyone can check.

### Closed

| Attack | What stops it |
|---|---|
| SSH brute force | key-only auth, no root, fail2ban, `MaxAuthTries 3` |
| Unpatched-service compromise | unattended-upgrades with automatic reboot |
| Direct hits on Postgres / Django / Next | they publish no ports; only Caddy does |
| Password sniffing, downgrade | HSTS, `SECURE_SSL_REDIRECT`, secure cookies |
| CSRF | Django's middleware + `CSRF_TRUSTED_ORIGINS`; one origin so no CORS exists to get wrong |
| XSS via stored content | React escapes by default; `nosniff` + CSP-adjacent headers; no `dangerouslySetInnerHTML` on user data |
| Clickjacking | `X-Frame-Options: DENY` at both Caddy and Django |
| SQL injection | Django ORM parameterises everything; no raw SQL on user input |
| Login brute force | allauth `ACCOUNT_RATE_LIMITS` — 5 failures / 5 min / IP *and* / account |
| Signup + reset spam | allauth: 10 signups/h/IP, 5 resets/h/IP |
| API flooding | DRF `ScopedRateThrottle` on waitlist / scramble / solve |
| Account enumeration | reset and email-change return identical answers either way |
| Open redirect after login | `?next=` restricted to same-site paths |
| Username impersonation | `core/moderation.py` skeleton matching + rename rate limits |
| Container escape via privilege gain | `no-new-privileges`, all capabilities dropped |
| Webshell persistence | read-only root filesystems; writable paths are tmpfs |
| One service OOMing the box | per-container `mem_limit` |
| Disk full from logs | capped json-file logs, in compose *and* in the daemon |
| Bad migration / accidental delete | pre-deploy dump + nightly dumps, restore-tested |
| Admin credential stuffing | `ADMIN_ALLOW_CIDR` — 404 to everyone else |
| Large-POST memory exhaustion | 2 MB cap at Caddy and at Django |

### Not closed — know these

1. **Volumetric DDoS.** One VM cannot absorb it. The answer is a proxy in
   front, not a bigger box — see §7.
2. **Camera injection / virtual camera.** The anticheat's own open problem,
   not a hosting one. `ANTICHEAT.md` §4.
3. **Enough-moves-then-swap.** `ANTICHEAT.md` §3; the count gate is
   structurally blind to it.
4. **Supply chain.** `npm ci` and `pip install` pull what the lockfiles say.
   Lockfiles are committed, which pins versions, but nobody is auditing the
   dependencies themselves.
5. **A single machine is a single point of failure.** Provider outage, disk
   failure, or a bad `docker compose down -v` takes the site down. Backups
   make it recoverable, not continuous.
6. **Secrets live in `.env` on the box.** Anyone with the `deploy` account or
   root has them. Acceptable for one operator; revisit if the team grows.

---

## 6. Operating it

```bash
cd ~/cubearena
C="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

$C ps                      # what is running
$C logs -f --tail=100      # everything
$C logs -f caddy           # TLS and routing problems live here
./deploy/deploy.sh         # roll forward to the latest commit
$C restart django          # after an .env change that only Django reads
```

### Backups

Nightly at 03:15 UTC, 14 days retained, plus one before every deploy.

```bash
./deploy/restore.sh --list       # what exists
./deploy/restore.sh --verify     # restore the newest into a scratch DB and count rows
./deploy/restore.sh --counts     # the same counts from production, to compare
./deploy/restore.sh --restore /backups/cubearena_....dump    # destructive
```

**Run `--verify` monthly.** A backup you have never restored is a hypothesis.
It touches nothing in production, so there is no reason not to.

**They are on the same disk as the database**, which means they survive a bad
migration and nothing else — not a failed volume, not a deleted VM. Get them
off the box; from your laptop, on a schedule:

```bash
scp -r deploy@<ip>:/var/lib/docker/volumes/cubearena_backups/_data/ ./offsite/
```

(or `rclone` to object storage, or the provider's snapshot feature, which
also covers the machine itself).

### Email

`ACCOUNT_EMAIL_VERIFICATION` is mandatory, so with `EMAIL_HOST` unset **mail
goes to the container log and nobody can finish signing up.** Point it at a
transactional relay — Postmark, SES, Mailgun, SendGrid — and set SPF, DKIM
and DMARC on the sending domain. Mail from an unknown IP mostly lands in
spam, and those three records plus the relay's reputation are what get
verification mail delivered at all. Test with a real address before launch.

---

## 7. When one VM stops being enough

In the order the pressure actually arrives:

1. **Put Cloudflare in front** (free tier). DDoS absorption, a basic WAF, and
   it hides the origin IP. Then tighten `ufw` to accept 80/443 from
   Cloudflare's ranges only, so nobody can bypass it by hitting the IP. This
   is the single highest-value addition and it is an afternoon.
2. **Move Postgres to a managed instance** when backups, failover or
   point-in-time recovery start mattering more than the bill.
3. **Split the VM** only when a real metric says so — `SYSTEM_DESIGN.md` §4
   sizes this, and it is a long way past the closed beta.

Do these when something forces them, not before.
