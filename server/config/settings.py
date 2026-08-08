"""
Django settings for the CubeArena API.

Deployment shape (SYSTEM_DESIGN.md §2.2): ONE box, ONE domain, ONE
`docker compose up`. Caddy terminates TLS and routes `/api/*` here and
everything else to Next. Because both are served from the same origin there
is **no CORS anywhere in this project** — if you ever find yourself adding
django-cors-headers to make something work, the routing is wrong, not the
CORS config.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


DEBUG = _env_bool("DJANGO_DEBUG", True)

# Fail loudly in production rather than silently running on a known key.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "dev-only-not-a-secret-change-me"
    else:
        raise RuntimeError(
            "DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is off. It signs "
            "scrambles — a leaked or default key lets anyone mint their own."
        )

ALLOWED_HOSTS = [h for h in os.environ.get(
    "DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if h]

# Caddy terminates TLS and forwards over plain HTTP, so Django needs telling
# that the original request was secure — otherwise secure cookies never set
# and every redirect downgrades to http://.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
CSRF_TRUSTED_ORIGINS = [o for o in os.environ.get(
    "DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o]

# Local dev needs the Next origin trusted explicitly, and the reason is not
# obvious. Next proxies /api/* to 127.0.0.1:8000, so Django sees
# `Host: 127.0.0.1:8000` while the browser sends
# `Origin: http://localhost:3000`. Django compares the two, finds they differ,
# and rejects every POST with
#
#   Forbidden (Origin checking failed - http://localhost:3000
#              does not match any trusted origins.)
#
# which surfaces in the UI as a bare 403 and looks like a CSRF-token bug
# rather than an origin-trust one. Production does not hit this — Caddy passes
# the real Host through, one origin end to end — so these entries are DEV
# ONLY and are not added when DEBUG is off.
if DEBUG:
    CSRF_TRUSTED_ORIGINS += [
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8080", "http://127.0.0.1:8080",
    ]

# Cookie and transport security, on whenever DEBUG is off. These become
# load-bearing the moment allauth issues real sessions: a session cookie sent
# over plain HTTP is one sniffed request away from account takeover, which
# would make the whole auth story decorative.
#
# SECURE_SSL_REDIRECT is safe behind Caddy *because* SECURE_PROXY_SSL_HEADER
# is set above — without that pairing Django cannot tell the proxy already
# terminated TLS and every request becomes a redirect loop.
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = True
    # One year, and only meaningful once the domain really is HTTPS-only.
    # HSTS is hard to walk back — browsers cache it — so this deliberately
    # does NOT include preload or subdomains until the domain is settled.
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = False
    SECURE_HSTS_PRELOAD = False
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"   # one origin, so Lax is sufficient and
                                  # strictly safer than None
CSRF_COOKIE_HTTPONLY = False      # the frontend must READ this one to echo
                                  # it back as X-CSRFToken (lib/api.ts)
CSRF_COOKIE_SAMESITE = "Lax"

# Caddy sets these too. Both layers do it on purpose: the proxy protects
# whatever it serves, and these protect Django even if it is ever reached
# some other way — a misrouted rule, a future second entrypoint, a `runserver`
# someone starts to debug something.
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
X_FRAME_OPTIONS = "DENY"          # nothing here is legitimately embeddable,
                                  # and clickjacking a "verify my solve"
                                  # button is a real attack

# Uploads are memory-buffered below this and spooled to disk above it. The
# evidence bundle (TODO 1) is the only large upload this API will ever take
# and it does not exist yet, so a low ceiling costs nothing today and closes
# the cheapest memory-exhaustion attack: a large POST that gunicorn buffers.
# RAISE THIS with the frame-upload endpoint, not before.
DATA_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024      # 2 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024
# Cheap defence against a POST with 50k fields, which is a DoS on the
# parser rather than on bandwidth.
DATA_UPLOAD_MAX_NUMBER_FIELDS = 200

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",          # required by allauth
    "rest_framework",
    "allauth",
    "allauth.account",
    "allauth.headless",              # see the ACCOUNT block below
    "core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    # AFTER AuthenticationMiddleware (it reads request.user) and after
    # allauth's (it wraps an allauth endpoint). See core/middleware.py for
    # why the password limit lives here rather than in a view.
    "core.middleware.PasswordChangeRateLimit",
]

SITE_ID = 1
AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",      # admin login
    "allauth.account.auth_backends.AuthenticationBackend",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# Postgres when POSTGRES_HOST is set (compose, and every deploy), SQLite
# otherwise so `manage.py test` and a bare `runserver` work with no services
# running. The fallback is a developer convenience ONLY: SQLite takes a
# database-wide write lock, so concurrent solve submissions under more than
# one gunicorn worker throw "database is locked".
#
# Discrete variables rather than a DATABASE_URL: it avoids a dependency, and
# it means a password containing "@" or "/" cannot silently corrupt the
# parse — which is a genuinely annoying failure to debug.
if os.environ.get("POSTGRES_HOST"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("POSTGRES_DB", "cubearena"),
            "USER": os.environ.get("POSTGRES_USER", "cubearena"),
            "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
            "HOST": os.environ["POSTGRES_HOST"],
            "PORT": os.environ.get("POSTGRES_PORT", "5432"),
            # Reuse connections across requests. Postgres connection setup is
            # not free and this endpoint set is chatty.
            "CONN_MAX_AGE": int(os.environ.get("POSTGRES_CONN_MAX_AGE", "60")),
            "CONN_HEALTH_CHECKS": True,
        }
    }
elif not DEBUG:
    raise RuntimeError(
        "POSTGRES_HOST must be set when DJANGO_DEBUG is off. The SQLite "
        "fallback is a dev convenience and will deadlock under concurrent "
        "writes with more than one gunicorn worker."
    )
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": os.environ.get("DJANGO_DB_PATH", BASE_DIR / "db.sqlite3"),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
# UTC everywhere, non-negotiable: every anticheat bound in core/timing.py is
# a difference between two server timestamps, and a local-time database would
# make those differences jump by an hour twice a year.
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- allauth --------------------------------------------------------------
#
# HEADLESS mode, because the frontend is Next and the backend is a REST API.
# allauth's normal flow renders its own Django templates for login/signup,
# which would mean two different UIs on one product. Headless exposes the
# same flows as JSON under /api/auth/ and lets Next own every screen.
#
# Sessions, not tokens: Caddy serves both halves from ONE origin
# (SYSTEM_DESIGN §2.2), so the session cookie is first-party and just works.
# A JWT scheme here would be additional machinery bought for a cross-origin
# problem this deployment does not have — and would need its own refresh,
# revocation and storage story.
HEADLESS_ONLY = True
HEADLESS_FRONTEND_URLS = {
    # Where allauth points people in the emails it sends. These are Next
    # routes and do not exist yet — they are part of the auth UI still to be
    # built, and the keys must stay in sync with whatever Next actually
    # serves or the links in verification mail 404.
    "account_confirm_email": "/auth/verify-email/{key}",
    "account_reset_password": "/auth/reset-password",
    "account_reset_password_from_key": "/auth/reset-password/{key}",
    "account_signup": "/auth/signup",
}

# Without this, allauth prefixes every subject with the django.contrib.sites
# record, which ships as "example.com" — so verification mail would go out
# reading "[example.com] Please Confirm Your Email Address", which looks
# exactly like phishing and is the fastest way to train users to ignore it.
# Set explicitly rather than relying on the Site row being fixed up, so a
# fresh database cannot send a wrong-looking email even once.
ACCOUNT_EMAIL_SUBJECT_PREFIX = os.environ.get(
    "ACCOUNT_EMAIL_SUBJECT_PREFIX", "[CubeArena] ")

ACCOUNT_LOGIN_METHODS = {"email"}
ACCOUNT_SIGNUP_FIELDS = ["email*", "password1*", "password2*"]
ACCOUNT_UNIQUE_EMAIL = True
# Mandatory verification, deliberately. A leaderboard where an unverified
# address can hold a top slot is one where a banned user re-registers in
# seconds, and account recovery for a wrong address is impossible.
#
# Overridable ONLY as a local-development convenience. With no EMAIL_HOST set,
# mail goes to the Django console — so a mandatory flow means signing up
# prints a link into the terminal and the account is unusable until you go
# and fetch it. That is correct behaviour and miserable to iterate against.
#
#     ACCOUNT_EMAIL_VERIFICATION=optional ../.venv/Scripts/python manage.py runserver 8000
#
# lets a fresh signup land straight in the app. The default is unchanged, and
# the override is ignored entirely when DEBUG is off — production cannot be
# talked out of verifying addresses by an environment variable.
ACCOUNT_EMAIL_VERIFICATION = "mandatory"
if DEBUG:
    _v = os.environ.get("ACCOUNT_EMAIL_VERIFICATION", "").strip().lower()
    if _v in ("mandatory", "optional", "none"):
        ACCOUNT_EMAIL_VERIFICATION = _v
ACCOUNT_RATE_LIMITS = {
    "login_failed": "5/5m/ip,5/5m/key",
    "signup": "10/h/ip",
    "reset_password": "5/h/ip",
}

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        # Session first: this is how a real logged-in user arrives once the
        # auth UI exists. DeviceKey stays below it as a build-time stub and
        # must be deleted before launch (core/auth.py).
        "rest_framework.authentication.SessionAuthentication",
        "core.auth.DeviceKeyAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        # The waitlist is the one unauthenticated write endpoint on a public
        # page, so it is the one that gets scraped/flooded first.
        "waitlist": "10/hour",
        "scramble": "120/hour",
        "solve": "120/hour",
    },
}

# --- Email ----------------------------------------------------------------
#
# The waitlist is a Postgres table and `manage.py send_waitlist` is what
# mails it. No list-management SaaS, no webhook, no second copy of the
# addresses — decided 2026-08-05 to cut complexity.
#
# But point EMAIL_HOST at a TRANSACTIONAL RELAY (Postmark, SES, SendGrid,
# Mailgun), not at a self-hosted SMTP server. That is not the same thing as
# handing your list to a provider: the table stays the source of truth and
# the relay only carries the message. It matters because bulk mail from an
# unknown IP largely lands in spam — SPF, DKIM, DMARC alignment and IP
# reputation are what get a launch announcement actually delivered, and none
# of them are things you win by running your own postfix.
#
# Unset EMAIL_HOST prints mail to the console, which is what `--dry-run`
# development wants.
EMAIL_BACKEND = ("django.core.mail.backends.smtp.EmailBackend"
                 if os.environ.get("EMAIL_HOST")
                 else "django.core.mail.backends.console.EmailBackend")
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "587"))
EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "")
EMAIL_USE_TLS = _env_bool("EMAIL_USE_TLS", True)
EMAIL_TIMEOUT = 20
DEFAULT_FROM_EMAIL = os.environ.get("DEFAULT_FROM_EMAIL",
                                    "CubeArena <hello@localhost>")

#: Public origin, used to build unsubscribe links in waitlist mail. Must be
#: how a recipient actually reaches the site, not an internal hostname.
SITE_URL = os.environ.get("SITE_URL", "http://localhost:8080")
