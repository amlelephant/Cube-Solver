"""
Data model for the CubeArena API.

One rule runs through all of it and is worth stating before the code: **no
field a client sends is ever trusted as fact.** Client-supplied values are
stored as `claimed_*` and exist only so the server can compare them against
what it derived itself. Everything a verdict rests on is either stamped by
the server clock or computed from evidence the server re-analyzes.
"""

import secrets
from datetime import timedelta

from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models
from django.utils import timezone


def _new_nonce() -> str:
    """URL-safe random token. Used for scramble nonces and unsubscribe links
    alike — both need to be unguessable, and `secrets` is the only source
    here that is."""
    return secrets.token_urlsafe(24)


#: How often each self-service account change is allowed. Username is the
#: long one on purpose: it is the handle other people know you by, it is the
#: field impersonation attacks target, and `moderation.check_username`
#: refuses names that merely LOOK like an existing account — a limit is what
#: stops someone cycling candidates until one slips past that check.
USERNAME_CHANGE_EVERY = timedelta(days=7)
EMAIL_CHANGE_EVERY = timedelta(days=1)
PASSWORD_CHANGE_EVERY = timedelta(days=1)


class Profile(models.Model):
    """Per-account data that is not authentication.

    WHY THIS IS NOT COLUMNS ON `auth.User`. Django lets you swap
    AUTH_USER_MODEL for a custom class, but only cleanly on an empty
    database — this one already has accounts, and allauth rows point at them.
    A OneToOne extension is the supported way to add fields afterwards, and it
    behaves identically from the app's side (`user.profile.country`).

    `user_id` is untouched by any of this: Django's AutoField already numbers
    accounts from 1 upward, which is the "which number user am I" the
    settings page shows.
    """

    AVATAR_PENDING = "pending"
    AVATAR_APPROVED = "approved"
    AVATAR_REJECTED = "rejected"
    AVATAR_NONE = "none"
    AVATAR_STATUSES = [
        (AVATAR_NONE, "No upload"),
        (AVATAR_PENDING, "Awaiting review"),
        (AVATAR_APPROVED, "Approved"),
        (AVATAR_REJECTED, "Rejected"),
    ]

    user = models.OneToOneField(settings.AUTH_USER_MODEL,
                                on_delete=models.CASCADE,
                                related_name="profile")

    #: ISO 3166-1 alpha-2, or "" for none. FREELY CHOSEN — never inferred
    #: from IP or locale. You rep whoever you want to rep, wherever you are.
    country = models.CharField(max_length=2, blank=True, db_index=True)

    #: Elo. Server-owned; the client has never been able to write it.
    rating = models.IntegerField(default=1000, db_index=True)
    #: Best rank ever HELD, which is what unlocks wreath cosmetics — so a
    #: past podium keeps its laurel after slipping down the table.
    best_rank = models.PositiveIntegerField(null=True, blank=True)
    #: Hand-granted only. Lived in client-side mock data before, where anyone
    #: could mint one from devtools (lib/cosmetics.tsx said as much).
    is_founder = models.BooleanField(default=False)

    #: The paid Coach tier (LAUNCH_ROADMAP §1). Server-owned and, for now,
    #: hand-granted — there is no billing integration, so nothing flips this
    #: but an admin. When billing lands it writes here and the gates below do
    #: not change.
    #:
    #: WHAT IT GATES: how much of your own analysis you can see at once.
    #: Free sees its most recent analysed solve; paid sees every solve and
    #: therefore averages and per-solve history. Deliberately NOT public —
    #: `is_founder` is a badge others see, this is billing state, and
    #: `public_profile_json` leaves it out on purpose.
    is_premium = models.BooleanField(default=False, db_index=True)

    #: Rate-limit clocks. Null means "never changed", which allows the first
    #: change immediately rather than making a new account wait a week.
    username_changed_at = models.DateTimeField(null=True, blank=True)
    email_changed_at = models.DateTimeField(null=True, blank=True)
    password_changed_at = models.DateTimeField(null=True, blank=True)

    #: Every name this account has held. Someone renaming right after a
    #: report is a signal, and you cannot investigate what you did not
    #: record. List of {"name", "at"}.
    previous_usernames = models.JSONField(default=list, blank=True)
    #: Non-blocking moderation signals from `moderation.review_flags`.
    name_flags = models.JSONField(default=list, blank=True)
    needs_name_review = models.BooleanField(default=False, db_index=True)

    #: Avatar pipeline. No uploads are accepted yet (SETTINGS_AUDIT.md §3
    #: recommends launching without them) — these columns exist so turning
    #: them on is not a migration on a live table.
    avatar_status = models.CharField(max_length=16, choices=AVATAR_STATUSES,
                                     default=AVATAR_NONE, db_index=True)
    avatar_path = models.CharField(max_length=300, blank=True)
    avatar_updated_at = models.DateTimeField(null=True, blank=True)

    #: The chosen preset avatar — a key from `core.avatars.AVATAR_PRESETS`,
    #: or "" for none. SEPARATE from `avatar_path` on purpose: that column
    #: means "a file this account uploaded", and overloading it with a
    #: preset key would make the upload pipeline's first query ambiguous
    #: the day it ships. A preset needs no moderation because there is no
    #: user-supplied imagery — only a key the server validated.
    avatar_preset = models.CharField(max_length=32, blank=True)

    # Notification preferences. Stored so the choice survives, but NOTHING
    # SENDS YET — there is no scheduler and no live-match feature. The
    # settings UI says so rather than implying mail is on its way.
    notify_invites = models.BooleanField(default=True)
    notify_recap = models.BooleanField(default=False)
    notify_pb = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} (#{self.user_id})"

    # -- rate limiting -----------------------------------------------------
    #
    # Returns seconds remaining rather than a bool so the UI can say "in 4
    # days" instead of a bare "not allowed", which reads like a bug.

    @staticmethod
    def _retry_after(last, period: timedelta) -> int:
        if last is None:
            return 0
        elapsed = timezone.now() - last
        return max(0, int((period - elapsed).total_seconds()))

    def username_retry_after(self) -> int:
        return self._retry_after(self.username_changed_at, USERNAME_CHANGE_EVERY)

    def email_retry_after(self) -> int:
        return self._retry_after(self.email_changed_at, EMAIL_CHANGE_EVERY)

    def password_retry_after(self) -> int:
        return self._retry_after(self.password_changed_at, PASSWORD_CHANGE_EVERY)


class WaitlistSignup(models.Model):
    """Landing-page signups. This table IS the mailing list.

    Decided 2026-08-05: no list-management SaaS and no webhook mirror. One
    table, and `manage.py send_waitlist` mails it. Two copies of a mailing
    list is two places for it to be wrong, and the provider bought us
    nothing we cannot do in ~150 lines.

    What that decision does NOT let us skip is unsubscribe. It is a legal
    requirement (CAN-SPAM, GDPR) and it is also what keeps the domain out of
    spam folders, so `unsubscribe_token` exists from the first migration
    rather than being retrofitted after the first send.
    """

    email = models.EmailField(unique=True, max_length=254)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    referrer = models.CharField(max_length=500, blank=True)
    # Hashed, never raw — see core.views.hash_ip. Blank when no salt is set,
    # which is the deliberate default: no salt means store nothing rather
    # than a trivially-reversible hash of a small input space.
    ip_hash = models.CharField(max_length=64, blank=True)

    #: Which campaign this address has already received. Send state lives on
    #: the row so a crashed or half-finished run can be resumed without
    #: re-mailing anyone — the failure mode that makes people unsubscribe.
    last_campaign = models.CharField(max_length=64, blank=True, db_index=True)
    last_sent_at = models.DateTimeField(null=True, blank=True)
    send_failures = models.PositiveIntegerField(default=0)

    unsubscribed_at = models.DateTimeField(null=True, blank=True)
    unsubscribe_token = models.CharField(max_length=64, unique=True,
                                         default=_new_nonce, db_index=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [models.Index(fields=["last_campaign", "unsubscribed_at"])]

    def __str__(self):
        return self.email

    @property
    def is_subscribed(self) -> bool:
        return self.unsubscribed_at is None


class Scramble(models.Model):
    """A server-issued, single-use scramble.

    Three properties matter and each closes a specific attack:

    * **Server-generated.** The client never chooses its own scramble, or it
      would choose one it has a prepared solution for.
    * **Single-use** (`consumed_at`). Otherwise you record one genuine solve
      and replay its evidence bundle forever.
    * **Server-timestamped** (`issued_at`). This is one end of the wall-clock
      window in `core.timing`; a client-supplied start time would make that
      whole bound meaningless.

    The signature is over the scramble text plus the nonce, so a tampered
    scramble fails verification even if the row is somehow reachable.
    """

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="scrambles", null=True, blank=True)
    moves = models.CharField(max_length=200)
    nonce = models.CharField(max_length=64, unique=True, default=_new_nonce,
                             db_index=True)
    signature = models.CharField(max_length=128)
    issued_at = models.DateTimeField(auto_now_add=True, db_index=True)
    expires_at = models.DateTimeField()
    consumed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-issued_at",)

    def __str__(self):
        return f"{self.nonce[:8]}... {self.moves[:32]}"

    @property
    def is_consumed(self) -> bool:
        return self.consumed_at is not None

    @property
    def is_expired(self) -> bool:
        return timezone.now() > self.expires_at


class Solve(models.Model):
    """A submitted solve attempt and the verdict the server reached on it."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    REVIEW = "review"
    VERDICTS = [(VERIFIED, "Verified"), (REJECTED, "Rejected"),
                (REVIEW, "Needs review")]

    SOLO = "solo"
    WIN = "win"
    LOSS = "loss"
    RESULTS = [(SOLO, "Solo trial"), (WIN, "Win"), (LOSS, "Loss")]

    #: EVERY solve belongs to exactly one account, and the history table on a
    #: profile is this FK filtered by user — which is what makes a solve
    #: someone's rather than a row in a shared list. Still nullable because
    #: the `Authorization: Device <anything>` stub predates login and its
    #: rows have no user; that stub is scheduled for deletion (core/auth.py)
    #: and this should become non-null with it.
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="solves", null=True, blank=True,
                             db_index=True)
    scramble = models.OneToOneField(Scramble, on_delete=models.PROTECT,
                                    related_name="solve")

    #: Who this was raced against, if anyone. Self-referential through the
    #: user table rather than a name string, so an opponent's profile is
    #: reachable from the row and a rename does not orphan the history.
    opponent = models.ForeignKey(settings.AUTH_USER_MODEL,
                                 on_delete=models.SET_NULL,
                                 related_name="solves_against",
                                 null=True, blank=True)
    result = models.CharField(max_length=8, choices=RESULTS, default=SOLO,
                              db_index=True)
    #: Elo change this solve caused. Null for solo trials, which are unrated.
    rating_delta = models.IntegerField(null=True, blank=True)

    # -- server-stamped ----------------------------------------------------
    received_at = models.DateTimeField(auto_now_add=True, db_index=True)
    #: frames / fps. THE authoritative duration. Never taken from the client.
    derived_seconds = models.FloatField(validators=[MinValueValidator(0.0)])

    # -- evidence the server counted for itself ----------------------------
    frame_count = models.PositiveIntegerField()
    fps = models.FloatField()
    observed_moves = models.PositiveIntegerField()
    observed_moves_after_stop = models.PositiveIntegerField(default=0)
    post_stop_seconds = models.FloatField(null=True, blank=True)
    evidence_sha256 = models.CharField(max_length=64, blank=True, db_index=True)

    # -- what the client SAID, kept only to compare against the above ------
    claimed_seconds = models.FloatField(null=True, blank=True)
    #: derived_seconds - claimed_seconds. A solve whose client disagrees with
    #: the evidence is interesting whether or not it was rejected for it.
    claim_discrepancy = models.FloatField(null=True, blank=True)

    # -- verdict -----------------------------------------------------------
    verdict = models.CharField(max_length=16, choices=VERDICTS, db_index=True)
    reject_reasons = models.JSONField(default=list, blank=True)
    review_reasons = models.JSONField(default=list, blank=True)
    detail = models.JSONField(default=dict, blank=True)

    #: The coach payload — `ble/move_detector/coach/report.py`'s
    #: `solve_report()` output, stored verbatim. Empty `{}` when no analysis
    #: has been run for this solve, which is the normal state for a solve
    #: submitted through `POST /api/solves/` today.
    #:
    #: WHY THIS IS NOT COLUMNS. The registry in `coach/report.py` IS the
    #: gate on what may be reported, and it moves when `metric_robustness.py`
    #: re-measures. Mirroring it as columns here would make that a migration
    #: every time, and would put a second, silently-diverging copy of the
    #: gate in a codebase that already treats having one as the point. The
    #: payload carries each metric's label, unit, confidence and measured
    #: error alongside its value, so nothing downstream has to re-derive
    #: them.
    #:
    #: THIS NEVER TOUCHES A VERDICT (TODO §2B). Analytics reads `analysis`;
    #: the gate reads `observed_moves`. Keep that boundary in code.
    analysis = models.JSONField(default=dict, blank=True)
    #: Set once a worker has re-run the CV pipeline on the uploaded frames,
    #: as opposed to scoring what the client reported. Until that exists,
    #: every row here is `False` and no verdict is stronger than the client's
    #: honesty about its own frame count — see TODO.md.
    reverified = models.BooleanField(default=False)

    class Meta:
        ordering = ("-received_at",)
        indexes = [
            models.Index(fields=["verdict", "-received_at"]),
            # The profile history query: this user's solves, newest first.
            models.Index(fields=["user", "-received_at"]),
            # The leaderboard's best-time column: verified solves by duration.
            models.Index(fields=["user", "verdict", "derived_seconds"]),
        ]

    def __str__(self):
        return f"{self.derived_seconds:.2f}s {self.verdict}"

    @property
    def tps(self) -> float | None:
        if not self.derived_seconds:
            return None
        return self.observed_moves / self.derived_seconds
