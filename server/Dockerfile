FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# /repo/server, not /app. core/timing.py imports the anticheat thresholds
# from ble/move_detector/ by walking up two directories, and compose mounts
# the real ./ble at /repo/ble — so the app has to sit at /repo/server for
# that relative walk to land on it. Moving this breaks the import.
WORKDIR /repo/server

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DJANGO_DEBUG=1 only for this build step: collectstatic imports settings,
# and settings now (correctly) refuses to start without POSTGRES_HOST when
# DEBUG is off — which no database is, at image-build time.
RUN DJANGO_DEBUG=1 DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput

# Drop root. A remote-code bug in Django is bad; a remote-code bug running as
# root inside a container sharing the host kernel is how it becomes a host
# compromise. `app` owns nothing it does not need — the code is copied as
# root and stays read-only to this user, which is correct: the application
# has no business rewriting its own source at runtime.
#
# Ownership of staticfiles is granted explicitly because collectstatic ran
# before this line, as root.
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /repo/server/staticfiles
USER app

EXPOSE 8000

# Compose gates startup on Postgres being healthy; this is the app's own
# answer to "is it serving", which compose and any external monitor read.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health/', timeout=4).status==200 else 1)"

# Postgres, so more than one worker is safe now — this was pinned to 1 while
# the database was SQLite and its table-wide write lock turned concurrent
# solve submissions into "database is locked".
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers ${GUNICORN_WORKERS:-3} --timeout 60"]
