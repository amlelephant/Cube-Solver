"""
Account, profile and name-moderation tests.

Same spirit as `tests.py`: these are the ATTACKS written down. The moderation
cases in particular are all real evasions that defeat a blocklist applied to
the raw string — each one is the reason `moderation.py` folds to a skeleton
before it matches anything.

    python manage.py test
"""

from datetime import timedelta
from io import StringIO

from allauth.account.models import EmailAddress
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from . import scrambles as scr
from .models import Profile, Scramble, Solve
from .avatars import AVATAR_PRESETS
from .moderation import NameRejected, check_username, review_flags, skeletons

FPS = 30.0
User = get_user_model()

# Written as escapes on purpose — the whole point is that these are NOT the
# ASCII characters they look like, and a literal would be indistinguishable
# in a diff.
CYRILLIC_A = "а"
CYRILLIC_E = "е"
CYRILLIC_GHE = "г"
CYRILLIC_ES = "с"
CYRILLIC_I = "і"
ZWSP = "​"


def make_user(username, email=None, verified=True):
    user = User.objects.create_user(
        username=username, email=email or f"{username}@example.com",
        password="pw-Long-123!")
    if verified:
        EmailAddress.objects.create(user=user, email=user.email,
                                    verified=True, primary=True)
    return user


class ModerationTests(TestCase):
    def test_plain_slur_is_blocked(self):
        with self.assertRaises(NameRejected):
            check_username("nigger")

    def test_homoglyph_evasion_is_blocked(self):
        """Cyrillic substitutions and leet digits produce zero ASCII hits."""
        for evasion in [
            "ni" + CYRILLIC_GHE + CYRILLIC_GHE + "er",
            "n1gg3r",
            "nigg" + CYRILLIC_E + "r",
        ]:
            with self.assertRaises(NameRejected, msg=evasion):
                check_username(evasion)

    def test_repeat_padding_is_blocked(self):
        with self.assertRaises(NameRejected):
            check_username("fuuuuuck")

    def test_zero_width_padding_is_blocked(self):
        with self.assertRaises(NameRejected):
            check_username("f" + ZWSP + "uc" + ZWSP + "k")

    def test_reserved_names_blocked_including_lookalikes(self):
        for name in ["admin", "Admin", CYRILLIC_A + "dmin", "4dm1n", "support"]:
            with self.assertRaises(NameRejected, msg=name):
                check_username(name)

    def test_scunthorpe_problem_does_not_block_real_names(self):
        """Word-boundary matching plus an allow-list exists exactly for these.

        Every name here contains a blocked substring and every one is
        something a real person could legitimately want.
        """
        for name in ["scunthorpe", "Penistone", "assassin", "classic",
                     "compass", "analyst", "Cockburn", "shiitake", "Dickens"]:
            self.assertEqual(check_username(name), name)

    def test_ordinary_names_pass(self):
        for name in ["mira_cubes", "jperm.fan42", "sub10-sam", "Aiden"]:
            self.assertEqual(check_username(name), name)

    def test_structural_rules(self):
        for bad in ["ab", "_leading", "trailing_", "has space", "x" * 30, ""]:
            with self.assertRaises(NameRejected, msg=repr(bad)):
                check_username(bad)

    def test_non_ascii_usernames_are_refused_outright(self):
        """The structural rule is ASCII-only, which kills the entire
        cross-script impersonation class before the skeleton check is even
        reached. Cheaper and more certain than detecting it."""
        with self.assertRaises(NameRejected):
            check_username("m" + CYRILLIC_I + "ra_cubes")

    def test_lookalike_of_existing_account_is_refused(self):
        """Leet impersonation, which IS spellable in ASCII: `m1ra_cubes`
        folds onto `mira_cubes` and must not be claimable by someone else."""
        taken = {sk: 1 for sk in skeletons("mira_cubes")}
        with self.assertRaises(NameRejected):
            check_username("m1ra_cubes", taken_skeletons=taken)
        # The owner renaming to their own lookalike is fine.
        self.assertEqual(
            check_username("m1ra_cubes", taken_skeletons=taken, allow_id=1),
            "m1ra_cubes")

    def test_mixed_script_is_flagged_not_blocked(self):
        self.assertIn("mixed_script", review_flags("mira" + CYRILLIC_ES + "ubes"))
        self.assertEqual(review_flags("mira_cubes"), [])


class AccountApiTests(TestCase):
    def setUp(self):
        self.user = make_user("tester")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_every_user_gets_a_profile_and_a_counting_id(self):
        self.assertTrue(Profile.objects.filter(user=self.user).exists())
        self.assertGreaterEqual(self.user.id, 1)
        r = self.client.get("/api/me/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], self.user.id)

    def test_country_is_free_choice_and_validated(self):
        r = self.client.patch("/api/me/", {"country": "JP"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["country"], "JP")

        r = self.client.patch("/api/me/", {"country": "ZQ"}, format="json")
        self.assertEqual(r.status_code, 400)

        r = self.client.patch("/api/me/", {"country": ""}, format="json")
        self.assertIsNone(r.json()["country"])

    def test_username_change_then_rate_limited_for_a_week(self):
        r = self.client.post("/api/me/username/", {"username": "renamed_one"},
                             format="json")
        self.assertEqual(r.status_code, 200, r.content[:200])
        self.user.refresh_from_db()
        self.assertEqual(self.user.username, "renamed_one")

        r = self.client.post("/api/me/username/", {"username": "again_now"},
                             format="json")
        self.assertEqual(r.status_code, 429)
        self.assertGreater(r.json()["retry_after"], 6 * 24 * 3600)

        p = Profile.objects.get(user=self.user)
        p.username_changed_at = timezone.now() - timedelta(days=7, minutes=1)
        p.save()
        r = self.client.post("/api/me/username/", {"username": "again_now"},
                             format="json")
        self.assertEqual(r.status_code, 200, r.content[:200])

    def test_username_history_is_recorded(self):
        self.client.post("/api/me/username/", {"username": "renamed_one"},
                         format="json")
        p = Profile.objects.get(user=self.user)
        self.assertEqual([h["name"] for h in p.previous_usernames], ["tester"])

    def test_rejected_name_does_not_burn_the_weekly_allowance(self):
        r = self.client.post("/api/me/username/", {"username": "admin"},
                             format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIsNone(Profile.objects.get(user=self.user).username_changed_at)

    def test_email_change_is_pending_until_verified_and_limited_daily(self):
        r = self.client.post("/api/me/email/", {"email": "new@example.com"},
                             format="json")
        self.assertEqual(r.status_code, 200, r.content[:200])

        # Until its owner proves they can read it, the account must not move.
        new = EmailAddress.objects.get(user=self.user, email="new@example.com")
        self.assertFalse(new.verified)
        self.assertFalse(new.primary)
        self.user.refresh_from_db()
        self.assertEqual(self.user.email, "tester@example.com")

        r = self.client.post("/api/me/email/", {"email": "third@example.com"},
                             format="json")
        self.assertEqual(r.status_code, 429)
        self.assertGreater(r.json()["retry_after"], 23 * 3600)

    def test_email_change_does_not_reveal_existing_accounts(self):
        make_user("other", email="taken@example.com")
        r = self.client.post("/api/me/email/", {"email": "taken@example.com"},
                             format="json")
        self.assertEqual(r.status_code, 400)
        # Identical wording to a malformed address: no enumeration oracle.
        self.assertEqual(r.json()["error"], "That address can't be used.")

    def test_notification_prefs_persist(self):
        r = self.client.patch("/api/me/", {"notify_recap": True,
                                           "notify_pb": False}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["notify_recap"])
        self.assertFalse(r.json()["notify_pb"])

    def test_me_requires_a_session(self):
        self.assertEqual(APIClient().get("/api/me/").status_code, 403)


class PasswordChangeLimitTests(TestCase):
    """The limit on an endpoint allauth owns — enforced in middleware."""

    def setUp(self):
        self.user = make_user("pwuser")
        self.client = APIClient()
        self.client.force_login(self.user)

    def _change(self, current, new):
        return self.client.post(
            "/api/auth/browser/v1/account/password/change",
            {"current_password": current, "new_password": new}, format="json")

    def test_second_change_within_a_day_is_refused(self):
        r = self._change("pw-Long-123!", "pw-Other-456!")
        self.assertIn(r.status_code, (200, 401), r.content[:300])

        r = self._change("pw-Other-456!", "pw-Third-789!")
        self.assertEqual(r.status_code, 429)
        self.assertGreater(r.json()["retry_after"], 23 * 3600)

    def test_a_failed_attempt_does_not_burn_the_allowance(self):
        """One typo must not lock someone out of their password for a day."""
        r = self._change("wrong-password", "pw-Other-456!")
        self.assertGreaterEqual(r.status_code, 400)
        self.assertLess(r.status_code, 500)
        self.assertIsNone(Profile.objects.get(user=self.user).password_changed_at)


class PublicProfileTests(TestCase):
    """Every profile is public. That is a product decision, so it is a test."""

    def setUp(self):
        self.user = make_user("publicguy")
        Profile.objects.filter(user=self.user).update(country="SE", rating=1500)

    def test_profile_readable_while_signed_out(self):
        r = APIClient().get("/api/users/publicguy/")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["username"], "publicguy")
        self.assertEqual(body["country"], "SE")
        self.assertIn("solves", body)

    def test_profile_never_leaks_email_or_rate_limits(self):
        body = APIClient().get("/api/users/publicguy/").json()
        self.assertNotIn("email", body)
        self.assertNotIn("limits", body)

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(APIClient().get("/api/users/PublicGuy/").status_code, 200)

    def test_unknown_player_is_404_not_500(self):
        self.assertEqual(APIClient().get("/api/users/ghost/").status_code, 404)

    def test_leaderboard_is_public_and_ranked(self):
        other = make_user("higher")
        Profile.objects.filter(user=other).update(rating=2000)
        r = APIClient().get("/api/leaderboard/")
        self.assertEqual(r.status_code, 200)
        rows = r.json()["results"]
        ratings = [row["rating"] for row in rows]
        self.assertEqual(ratings, sorted(ratings, reverse=True))
        self.assertEqual(rows[0]["rank"], 1)
        # Every row carries what linking to that player needs.
        self.assertTrue(all(row.get("username") for row in rows))


class SolveOwnershipTests(TestCase):
    """Solves belong to accounts; a profile's history IS that foreign key."""

    def setUp(self):
        self.a = make_user("playera")
        self.b = make_user("playerb")

    def _solve(self, user, seconds, opponent=None, result=Solve.SOLO):
        s = Scramble(user=user, moves=scr.generate(),
                     expires_at=timezone.now() + timedelta(minutes=30))
        s.signature = scr.sign(s.moves, s.nonce)
        s.save()
        return Solve.objects.create(
            user=user, scramble=s, derived_seconds=seconds,
            frame_count=int(seconds * FPS), fps=FPS, observed_moves=50,
            verdict=Solve.VERIFIED, result=result, opponent=opponent)

    def test_history_is_scoped_to_its_owner(self):
        self._solve(self.a, 12.0)
        self._solve(self.a, 13.0)
        self._solve(self.b, 9.0)
        body = APIClient().get("/api/users/playera/").json()
        self.assertEqual(len(body["solves"]), 2)
        self.assertEqual(body["total_solves"], 2)
        self.assertEqual(body["best_seconds"], 12.0)

    def test_opponent_is_a_reference_so_a_rename_cannot_orphan_it(self):
        self._solve(self.a, 11.0, opponent=self.b, result=Solve.WIN)
        self.b.username = "renamed_b"
        self.b.save()
        row = APIClient().get("/api/users/playera/").json()["solves"][0]
        self.assertEqual(row["opponent"], "renamed_b")
        self.assertEqual(row["result"], "win")


class SeedDemoTests(TestCase):
    def test_seeded_players_can_actually_sign_in(self):
        """The reason these are real accounts and not fixtures."""
        call_command("seed_demo", "--solves", "2", "--force", stdout=StringIO())
        c = APIClient()
        c.get("/api/auth/browser/v1/config")
        r = c.post("/api/auth/browser/v1/auth/login",
                   {"email": "feliks_zx@demo.cubearena.local",
                    "password": "cubearena123"}, format="json")
        self.assertEqual(r.status_code, 200, r.content[:300])
        self.assertTrue(r.json()["meta"]["is_authenticated"])

    def test_seeded_solves_belong_to_their_player(self):
        call_command("seed_demo", "--solves", "3", "--force", stdout=StringIO())
        user = User.objects.get(username="mira_cubes")
        self.assertEqual(Solve.objects.filter(user=user).count(), 3)
        self.assertFalse(Solve.objects.filter(user__isnull=True).exists())

    def test_seeded_names_survive_our_own_moderation(self):
        """A demo player the rename endpoint would reject is a bad demo."""
        from .management.commands.seed_demo import PLAYERS
        for (username, *_rest) in PLAYERS:
            self.assertEqual(check_username(username), username)


class PremiumGateTests(TestCase):
    """The paid tier, written down as what a FREE account must not receive.

    These assert on the RESPONSE, not on the UI, because that is where the
    gate lives: a client that ignores `is_premium` still cannot average
    solves it was never sent.
    """

    def setUp(self):
        self.user = make_user("gated")
        self.other = make_user("someoneelse")
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def _analysed(self, user, seconds, span=None):
        s = Scramble(user=user, moves=scr.generate(),
                     expires_at=timezone.now() + timedelta(minutes=30))
        s.signature = scr.sign(s.moves, s.nonce)
        s.save()
        return Solve.objects.create(
            user=user, scramble=s, derived_seconds=seconds,
            frame_count=int(seconds * FPS), fps=FPS, observed_moves=50,
            verdict=Solve.VERIFIED,
            analysis={
                "usable": True, "regime": "daytime", "suppressed": [],
                "metrics": {"span_seconds": {
                    "value": span if span is not None else seconds,
                    "label": "Solve time", "unit": "s",
                    "confidence": "high", "accuracy_pct": 0.1}},
            })

    def _set_premium(self, on):
        # Written THROUGH `self.user.profile`, not via a fresh
        # `Profile.objects.get`. The signal's `get_or_create(user=instance)`
        # populates the reverse one-to-one cache on `self.user`, and
        # `force_authenticate` hands that same object to every request — so
        # updating a different Python object for the same row leaves the
        # request reading a stale `is_premium`. Real requests reload the user
        # from the session each time and never see this.
        self.user.profile.is_premium = on
        self.user.profile.save()

    def test_free_account_is_served_only_its_latest_analysis(self):
        self._analysed(self.user, 30.0)
        newest = self._analysed(self.user, 20.0)
        body = self.client.get("/api/solves/analysis/").json()
        self.assertEqual(len(body["solves"]), 1)
        self.assertEqual(body["solves"][0]["id"], str(newest.id))
        self.assertTrue(body["truncated"])
        self.assertEqual(body["total"], 2)
        self.assertFalse(body["is_premium"])

    def test_premium_account_is_served_all_of_them_oldest_first(self):
        older = self._analysed(self.user, 30.0)
        newer = self._analysed(self.user, 20.0)
        self._set_premium(True)
        body = self.client.get("/api/solves/analysis/").json()
        self.assertEqual([s["id"] for s in body["solves"]],
                         [str(older.id), str(newer.id)])
        self.assertFalse(body["truncated"])

    def test_free_account_cannot_open_an_older_solve(self):
        older = self._analysed(self.user, 30.0)
        self._analysed(self.user, 20.0)
        r = self.client.get(f"/api/solves/{older.id}/analysis/")
        self.assertEqual(r.status_code, 402)
        self.assertTrue(r.json()["upgrade_required"])

    def test_free_account_can_open_its_latest_solve(self):
        self._analysed(self.user, 30.0)
        newest = self._analysed(self.user, 20.0)
        r = self.client.get(f"/api/solves/{newest.id}/analysis/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], str(newest.id))

    def test_someone_elses_solve_is_404_not_402(self):
        """404, never 402: whether a solve id exists on another account is
        not something this endpoint should confirm."""
        theirs = self._analysed(self.other, 8.0)
        self._set_premium(True)
        r = self.client.get(f"/api/solves/{theirs.id}/analysis/")
        self.assertEqual(r.status_code, 404)

    def test_solves_without_an_analysis_are_omitted(self):
        s = Scramble(user=self.user, moves=scr.generate(),
                     expires_at=timezone.now() + timedelta(minutes=30))
        s.signature = scr.sign(s.moves, s.nonce)
        s.save()
        Solve.objects.create(user=self.user, scramble=s, derived_seconds=9.0,
                             frame_count=270, fps=FPS, observed_moves=50,
                             verdict=Solve.VERIFIED)
        self.assertEqual(self.client.get("/api/solves/analysis/").json()["solves"], [])

    def test_premium_is_private(self):
        self._set_premium(True)
        self.assertNotIn("is_premium", APIClient().get("/api/users/gated/").json())
        self.assertTrue(self.client.get("/api/me/").json()["is_premium"])


class AvatarPresetTests(TestCase):
    """Presets exist so there is no user-supplied imagery to moderate. That
    only holds if the server refuses anything off the allowlist."""

    def setUp(self):
        self.user = make_user("painter")
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def test_a_known_preset_is_stored_and_is_public(self):
        r = self.client.patch("/api/me/", {"avatar": AVATAR_PRESETS[0]},
                              format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["avatar"], AVATAR_PRESETS[0])
        self.assertEqual(
            APIClient().get("/api/users/painter/").json()["avatar"],
            AVATAR_PRESETS[0])

    def test_an_unknown_key_is_refused(self):
        for bogus in ("../../etc/passwd", "<script>x</script>",
                      "https://evil.example/a.png", "cube-nope"):
            r = self.client.patch("/api/me/", {"avatar": bogus}, format="json")
            self.assertEqual(r.status_code, 400, bogus)
        self.assertEqual(Profile.objects.get(user=self.user).avatar_preset, "")

    def test_empty_clears_the_choice(self):
        self.client.patch("/api/me/", {"avatar": AVATAR_PRESETS[1]}, format="json")
        r = self.client.patch("/api/me/", {"avatar": ""}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["avatar"])

    def test_preset_never_touches_the_upload_column(self):
        """`avatar_path` means 'a file this account uploaded'. Overloading it
        would make the upload pipeline's first query ambiguous."""
        self.client.patch("/api/me/", {"avatar": AVATAR_PRESETS[2]}, format="json")
        p = Profile.objects.get(user=self.user)
        self.assertEqual(p.avatar_path, "")
        self.assertEqual(p.avatar_status, Profile.AVATAR_NONE)


class BestSolveLinkTests(TestCase):
    """`best_solve_id` is what makes "Current Best" a link."""

    def setUp(self):
        self.user = make_user("pber")

    def _solve(self, seconds, verdict=Solve.VERIFIED):
        s = Scramble(user=self.user, moves=scr.generate(),
                     expires_at=timezone.now() + timedelta(minutes=30))
        s.signature = scr.sign(s.moves, s.nonce)
        s.save()
        return Solve.objects.create(
            user=self.user, scramble=s, derived_seconds=seconds,
            frame_count=int(seconds * FPS), fps=FPS, observed_moves=50,
            verdict=verdict)

    def test_points_at_the_fastest_verified_solve(self):
        self._solve(20.0)
        best = self._solve(9.0)
        self._solve(15.0)
        body = APIClient().get("/api/users/pber/").json()
        self.assertEqual(body["best_solve_id"], best.id)
        self.assertEqual(body["best_seconds"], 9.0)

    def test_a_faster_rejected_solve_is_not_the_best(self):
        good = self._solve(20.0)
        self._solve(3.0, verdict=Solve.REJECTED)
        self.assertEqual(
            APIClient().get("/api/users/pber/").json()["best_solve_id"], good.id)

    def test_none_when_there_is_nothing_to_link_to(self):
        body = APIClient().get("/api/users/pber/").json()
        self.assertIsNone(body["best_solve_id"])
        self.assertIsNone(body["best_seconds"])
