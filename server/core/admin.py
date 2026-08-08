"""
Admin — the human-review backstop (LAUNCH_ROADMAP C3).

The `review` verdict is not a diagnostic, it is a queue: every solve the
gate abstained on has to be looked at by a person, and this is where that
happens. Keep it cheap to scan — the top leaderboard slice is the only part
that ever needs the attention.
"""

from django.contrib import admin
from django.utils import timezone

from .models import Scramble, Solve, WaitlistSignup


@admin.register(WaitlistSignup)
class WaitlistSignupAdmin(admin.ModelAdmin):
    """The mailing list itself. `manage.py send_waitlist` is what sends to
    it; this is for looking, and for unsubscribing someone who asked by
    hand."""

    list_display = ("email", "created_at", "last_campaign", "last_sent_at",
                    "subscribed", "send_failures")
    list_filter = ("last_campaign", "unsubscribed_at", "created_at")
    search_fields = ("email",)
    readonly_fields = ("created_at", "ip_hash", "unsubscribe_token",
                       "last_sent_at", "send_failures")
    actions = ("mark_unsubscribed",)

    @admin.display(boolean=True, description="subscribed")
    def subscribed(self, obj):
        return obj.is_subscribed

    @admin.action(description="Unsubscribe selected")
    def mark_unsubscribed(self, request, queryset):
        n = queryset.filter(unsubscribed_at__isnull=True).update(
            unsubscribed_at=timezone.now())
        self.message_user(request, f"Unsubscribed {n}.")


@admin.register(Scramble)
class ScrambleAdmin(admin.ModelAdmin):
    list_display = ("nonce_short", "moves", "user", "issued_at", "consumed_at")
    list_filter = ("issued_at",)
    search_fields = ("nonce", "moves")
    readonly_fields = ("nonce", "signature", "issued_at")

    @admin.display(description="nonce")
    def nonce_short(self, obj):
        return obj.nonce[:10]


@admin.register(Solve)
class SolveAdmin(admin.ModelAdmin):
    list_display = ("id", "verdict", "derived_seconds", "observed_moves",
                    "tps_display", "claim_discrepancy", "reverified",
                    "received_at")
    # `verdict` first: the review queue is the reason this page exists.
    list_filter = ("verdict", "reverified", "received_at")
    readonly_fields = [f.name for f in Solve._meta.fields]

    @admin.display(description="TPS")
    def tps_display(self, obj):
        return f"{obj.tps:.2f}" if obj.tps else "-"

    def has_add_permission(self, request):
        # Solves are only ever created by the submission endpoint, which is
        # what runs the verdict logic. A hand-added row would carry a verdict
        # nothing actually adjudicated.
        return False
