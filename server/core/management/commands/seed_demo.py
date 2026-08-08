"""
Populate the database with demo players you can actually sign in as.

These are REAL accounts — real rows in auth_user, real verified email
addresses, real password hashes, real Solve rows tied to them by foreign key.
Not fixtures for a mock layer. That is the point: the leaderboard reads the
same tables the app does, so clicking a name opens a profile whose history is
that account's actual solves, and you can sign in as any of them to see the
product from their side.

    python manage.py seed_demo                 # create/refresh, keep existing
    python manage.py seed_demo --reset         # delete demo accounts first
    python manage.py seed_demo --password foo  # set a different shared password

DEV ONLY. Refuses to run when DJANGO_DEBUG is off — these accounts have a
shared, printed password and would be an open door in production.
"""

import random
from datetime import timedelta

from allauth.account.models import EmailAddress
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from core.models import Profile, Scramble, Solve
from core.scrambles import generate, sign

User = get_user_model()

DEFAULT_PASSWORD = "cubearena123"

#: Marks an account as ours to manage, so --reset cannot touch a real one.
DEMO_DOMAIN = "demo.cubearena.local"

# (username, country, rating, best_rank, is_founder, target_best_seconds)
PLAYERS = [
    ("feliks_zx",     "AU", 2340, 1, False, 5.51),
    ("mira_cubes",    "JP", 2198, 1, False, 4.89),
    ("jperm_fan42",   "US", 2107, 2, False, 6.20),
    ("speedy_dan",    "GB", 1972, 3, False, 5.12),
    ("cubewizard",    "KR", 1888, 2, False, 5.44),
    ("torch_bearer",  "DE", 1804, 4, False, 7.61),
    ("algcrunch",     "CA", 1755, 5, False, 6.01),
    ("olltwist",      "FR", 1690, 3, False, 6.71),
    ("f2l_finn",      "SE", 1622, 7, False, 6.33),
    ("crossmaster",   "BR", 1560, 6, False, 7.02),
    ("roux_ruby",     "IN", 1498, 9, False, 7.44),
    ("zbll_zoe",      "PL", 1445, 11, False, 7.88),
    ("sub10_sam",     "NG", 1390, 14, False, 8.12),
    ("megaminx_mo",   "MX", 1322, 18, False, 8.90),
    ("turnspeed",     "ES", 1265, 22, False, 9.35),
]


class Command(BaseCommand):
    help = "Create demo players with real logins, profiles and solve history."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true",
                            help="Delete existing demo accounts first.")
        parser.add_argument("--password", default=DEFAULT_PASSWORD,
                            help="Shared password for every demo account.")
        parser.add_argument("--solves", type=int, default=24,
                            help="Solves to generate per player.")
        parser.add_argument("--force", action="store_true",
                            help="Bypass the DEBUG guard. For the test suite, "
                                 "which runs with DEBUG off by design.")

    def handle(self, *args, **opts):
        if not settings.DEBUG and not opts["force"]:
            raise CommandError(
                "seed_demo refuses to run with DJANGO_DEBUG off. Every account "
                "it creates shares one printed password. Pass --force if you "
                "genuinely mean it (the test suite does).")

        password = opts["password"]
        per_player = max(1, opts["solves"])
        rng = random.Random(20260805)

        if opts["reset"]:
            qs = User.objects.filter(email__endswith=f"@{DEMO_DOMAIN}")
            n = qs.count()
            # Solves and scrambles cascade from the user, EXCEPT that Solve
            # protects its Scramble — so clear solves first, then scrambles.
            Solve.objects.filter(user__in=qs).delete()
            Scramble.objects.filter(user__in=qs).delete()
            qs.delete()
            self.stdout.write(f"Removed {n} demo accounts.")

        created = 0
        for (username, country, rating, best_rank, founder, target) in PLAYERS:
            email = f"{username}@{DEMO_DOMAIN}"
            with transaction.atomic():
                user, is_new = User.objects.get_or_create(
                    username=username, defaults={"email": email})
                if is_new:
                    created += 1
                user.email = email
                user.set_password(password)
                user.save()

                # allauth authenticates against EmailAddress, not user.email.
                # Without a VERIFIED row here these accounts exist but cannot
                # sign in — ACCOUNT_EMAIL_VERIFICATION is mandatory, and the
                # rejection reads as a wrong password.
                EmailAddress.objects.update_or_create(
                    user=user, email=email,
                    defaults={"verified": True, "primary": True})

                profile, _ = Profile.objects.get_or_create(user=user)
                profile.country = country
                profile.rating = rating
                profile.best_rank = best_rank
                profile.is_founder = founder
                profile.save()

                have = Solve.objects.filter(user=user).count()
                for i in range(max(0, per_player - have)):
                    self._make_solve(user, target, rng, i)

        self._link_opponents(rng)

        total = User.objects.filter(email__endswith=f"@{DEMO_DOMAIN}").count()
        self.stdout.write(self.style.SUCCESS(
            f"\n{total} demo players ready ({created} new), "
            f"{Solve.objects.count()} solves total."))
        self.stdout.write(
            f"\n  Sign in as any of them:\n"
            f"    email:    <username>@{DEMO_DOMAIN}\n"
            f"    password: {password}\n\n"
            f"  e.g. feliks_zx@{DEMO_DOMAIN} / {password}\n")

    def _make_solve(self, user, target: float, rng: random.Random, i: int):
        """One solve, with a consumed scramble behind it.

        Times are lognormal-ish around the player's target: a long right tail
        (everyone has a bad solve) and a floor near their real best, rather
        than a symmetric jitter that produces implausible personal records.
        """
        seconds = round(max(target, target * rng.lognormvariate(0.10, 0.16)), 2)
        # Occasionally a genuine PB, so `best_seconds` is not always the floor.
        if rng.random() < 0.08:
            seconds = round(target * rng.uniform(0.97, 1.0), 2)

        fps = 30.0
        frames = int(seconds * fps)
        moves = max(20, int(seconds * rng.uniform(5.5, 8.5)))
        when = timezone.now() - timedelta(
            days=rng.randint(0, 120), hours=rng.randint(0, 23))

        scramble = Scramble(
            user=user, moves=generate(),
            expires_at=when + timedelta(minutes=30),
            consumed_at=when,
        )
        scramble.signature = sign(scramble.moves, scramble.nonce)
        scramble.save()
        # auto_now_add ignores an assigned value, so backdate after insert.
        Scramble.objects.filter(pk=scramble.pk).update(issued_at=when)

        solve = Solve.objects.create(
            user=user, scramble=scramble,
            derived_seconds=seconds, frame_count=frames, fps=fps,
            observed_moves=moves, observed_moves_after_stop=0,
            post_stop_seconds=rng.uniform(2, 12),
            claimed_seconds=round(seconds + rng.uniform(-0.05, 0.05), 2),
            verdict=Solve.VERIFIED, result=Solve.SOLO,
            detail={"seeded": True},
        )
        Solve.objects.filter(pk=solve.pk).update(received_at=when)

    def _link_opponents(self, rng: random.Random):
        """Turn some solo trials into head-to-head results.

        Done as a second pass because it needs every player to exist first —
        and it is what gives the history table an opponent to link through to.
        """
        users = list(User.objects.filter(email__endswith=f"@{DEMO_DOMAIN}"))
        if len(users) < 2:
            return
        for solve in Solve.objects.filter(opponent__isnull=True,
                                          result=Solve.SOLO):
            if rng.random() > 0.55:
                continue
            others = [u for u in users if u.pk != solve.user_id]
            if not others:
                continue
            won = rng.random() < 0.5
            solve.opponent = rng.choice(others)
            solve.result = Solve.WIN if won else Solve.LOSS
            solve.rating_delta = rng.randint(8, 24) * (1 if won else -1)
            solve.save(update_fields=["opponent", "result", "rating_delta"])
