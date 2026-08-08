"""
send_waitlist — mail the waitlist from the Postgres table.

Replaces the email-provider integration entirely (decided 2026-08-05): the
`WaitlistSignup` table is the list, and this is the thing that sends to it.

    # see who would get it, send nothing
    python manage.py send_waitlist --campaign launch --subject "We're live" \\
        --body launch.txt --dry-run

    # send to yourself first. Do this every time.
    python manage.py send_waitlist --campaign launch --subject "We're live" \\
        --body launch.txt --only you@example.com

    # the real thing
    python manage.py send_waitlist --campaign launch --subject "We're live" \\
        --body launch.txt

DESIGN NOTES, because bulk mail has sharp edges that are not obvious until
after you have hit them and cannot un-hit them:

* **Campaign-keyed, resumable.** A row records the last campaign it received.
  Re-running the same campaign skips everyone already sent, so a crash, a
  timeout or a Ctrl-C halfway through is recoverable by just running it
  again. Without this, the recovery action after a partial send is "mail
  everyone twice", which is how a launch announcement becomes a spam report.
* **One message per recipient, never a shared BCC.** BCC leaks nothing but it
  also cannot carry a per-recipient unsubscribe link, and a single bad
  address fails the whole batch.
* **Unsubscribe is not optional.** Legally required (CAN-SPAM, GDPR) and the
  main thing keeping the sending domain out of spam folders. Both a footer
  link and a `List-Unsubscribe` header, which is what Gmail and Outlook
  actually read.
* **Failures are recorded, not fatal.** One dead address must not abort a run
  of a thousand.

The send is deliberately serial with a small delay. It is slower than it
could be and that is correct: relays rate-limit, and being throttled or
blocked mid-launch costs far more than the minutes saved.
"""

import time
from pathlib import Path

from django.conf import settings
from django.core.mail import EmailMultiAlternatives, get_connection
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import WaitlistSignup


class Command(BaseCommand):
    help = "Send an email to the waitlist (campaign-keyed and resumable)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--campaign", required=True,
            help="Short key, e.g. 'launch'. Recorded per recipient so a "
                 "re-run skips anyone already sent. Reusing a key after "
                 "changing the body means those people never get the new "
                 "one — use a fresh key instead.")
        parser.add_argument("--subject", required=True)
        parser.add_argument(
            "--body", required=True,
            help="Path to a plain-text file. '{unsubscribe}' in it is "
                 "replaced with the recipient's link; if absent, a footer "
                 "is appended.")
        parser.add_argument("--dry-run", action="store_true",
                            help="List recipients and send nothing.")
        parser.add_argument("--only", nargs="+", default=None,
                            help="Restrict to these addresses. Use it to "
                                 "send yourself a real one first.")
        parser.add_argument("--limit", type=int, default=0,
                            help="Stop after N recipients (staged rollout).")
        parser.add_argument("--delay", type=float, default=0.2,
                            help="Seconds between messages (default 0.2).")
        parser.add_argument("--resend", action="store_true",
                            help="Ignore the already-sent check for this "
                                 "campaign. Mails people a second time — "
                                 "there is no undo.")

    def handle(self, *args, **o):
        # Resolve and report the ABSOLUTE path on failure. `--body launch.txt`
        # is relative to the shell's cwd, not to manage.py, so running this
        # from the repo root while the file sits in server/ fails with a bare
        # "No such file or directory: 'launch.txt'" that says nothing about
        # where it actually looked.
        body_path = Path(o["body"]).expanduser()
        try:
            body_template = body_path.read_text(encoding="utf-8")
        except OSError as e:
            raise CommandError(
                f"could not read --body {body_path.resolve()}\n"
                f"  ({e.strerror})\n"
                f"  cwd is {Path.cwd()}\n"
                f"  (the path is relative to your shell, not to manage.py)")
        if not body_template.strip():
            raise CommandError(f"--body {body_path.resolve()} is empty")

        qs = WaitlistSignup.objects.filter(unsubscribed_at__isnull=True)
        if o["only"]:
            qs = qs.filter(email__in=[e.lower() for e in o["only"]])
        if not o["resend"]:
            qs = qs.exclude(last_campaign=o["campaign"])
        qs = qs.order_by("created_at")
        if o["limit"]:
            qs = qs[:o["limit"]]

        recipients = list(qs)
        total_sub = WaitlistSignup.objects.filter(
            unsubscribed_at__isnull=True).count()
        already = WaitlistSignup.objects.filter(
            last_campaign=o["campaign"]).count()

        self.stdout.write(
            f"campaign '{o['campaign']}': {len(recipients)} to send  "
            f"({total_sub} subscribed, {already} already got this campaign)")
        if settings.EMAIL_BACKEND.endswith("console.EmailBackend"):
            self.stdout.write(self.style.WARNING(
                "  EMAIL_HOST is unset -> console backend. Nothing will "
                "actually be delivered."))
        if not recipients:
            self.stdout.write("nothing to do")
            return

        if o["dry_run"]:
            for r in recipients[:50]:
                self.stdout.write(f"  would send -> {r.email}")
            if len(recipients) > 50:
                self.stdout.write(f"  ... and {len(recipients) - 50} more")
            self.stdout.write(self.style.WARNING("dry run: sent nothing"))
            return

        # One connection for the whole run rather than one per message —
        # relays dislike connection churn and some count it against you.
        connection = get_connection()
        sent = failed = 0
        try:
            connection.open()
            for r in recipients:
                url = f"{settings.SITE_URL.rstrip('/')}/api/waitlist/unsubscribe/{r.unsubscribe_token}/"
                if "{unsubscribe}" in body_template:
                    body = body_template.replace("{unsubscribe}", url)
                else:
                    body = (f"{body_template.rstrip()}\n\n--\n"
                            f"You are receiving this because you joined the "
                            f"CubeArena waitlist.\nUnsubscribe: {url}\n")
                msg = EmailMultiAlternatives(
                    subject=o["subject"], body=body,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[r.email], connection=connection,
                    headers={
                        # What Gmail/Outlook actually parse to show their own
                        # unsubscribe button. The One-Click header requires
                        # the URL to accept POST, which the view does.
                        "List-Unsubscribe": f"<{url}>",
                        "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                    },
                )
                try:
                    msg.send(fail_silently=False)
                except Exception as e:            # noqa: BLE001 - one bad
                    # address must not abort a run of a thousand
                    failed += 1
                    r.send_failures += 1
                    r.save(update_fields=["send_failures"])
                    self.stderr.write(f"  FAILED {r.email}: {e}")
                    continue
                r.last_campaign = o["campaign"]
                r.last_sent_at = timezone.now()
                r.save(update_fields=["last_campaign", "last_sent_at"])
                sent += 1
                if sent % 25 == 0:
                    self.stdout.write(f"  {sent}/{len(recipients)}")
                time.sleep(o["delay"])
        finally:
            connection.close()

        self.stdout.write(self.style.SUCCESS(f"sent {sent}"))
        if failed:
            self.stdout.write(self.style.ERROR(
                f"failed {failed} (recorded in send_failures; re-running the "
                f"same campaign retries only these)"))
