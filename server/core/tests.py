"""
Tests. These are the ATTACKS, written down.

The point of this file is not coverage — it is that every claim made in
`timing.py`'s docstring has an executable counterexample here, so a future
change that quietly re-opens one of these fails loudly instead.

    python manage.py test
"""

import tempfile
from datetime import timedelta
from io import StringIO
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core import mail
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from .models import Scramble, Solve, WaitlistSignup
from . import scrambles as scr

FPS = 30.0


def frames_for(seconds: float) -> int:
    return int(round(seconds * FPS))


class ScrambleTests(TestCase):
    def test_generated_scramble_is_well_formed(self):
        for _ in range(200):
            moves = scr.generate().split()
            self.assertEqual(len(moves), 20)
            faces = [m[0] for m in moves]
            for a, b in zip(faces, faces[1:]):
                self.assertNotEqual(a, b, "consecutive turns of the same face")

    def test_scrambles_are_unpredictable(self):
        """Not a randomness test — just that we are not handing out the same
        scramble twice, which would let one prepared solution be reused."""
        seen = {scr.generate() for _ in range(200)}
        self.assertEqual(len(seen), 200)

    def test_signature_binds_moves_to_nonce(self):
        sig = scr.sign("R U R'", "nonce-a")
        self.assertTrue(scr.verify("R U R'", "nonce-a", sig))
        # Same signature must not validate a different scramble...
        self.assertFalse(scr.verify("L D L'", "nonce-a", sig))
        # ...nor the same scramble under a different nonce.
        self.assertFalse(scr.verify("R U R'", "nonce-b", sig))


class ApiTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION="Device testdevicekey")

    def new_scramble(self) -> dict:
        r = self.client.post("/api/scrambles/", {}, format="json")
        self.assertEqual(r.status_code, 201, r.content)
        return r.data

    def submit(self, nonce, seconds, moves, **extra):
        """Submit a solve described by a plausible-looking bundle."""
        payload = {
            "nonce": nonce,
            "frame_count": frames_for(seconds),
            "fps": FPS,
            "observed_moves": moves,
            "observed_moves_after_stop": 0,
            "post_stop_seconds": 10.0,
            "lighting_ok": True,
        }
        payload.update(extra)
        return self.client.post("/api/solves/", payload, format="json")


class HealthTests(ApiTestCase):
    def test_health_reports_live_thresholds(self):
        r = self.client.get("/api/health/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "ok")
        # The QTM correction must survive to the wire: a floor at or below
        # God's number in HTM (20) would catch nothing.
        self.assertGreater(r.data["anticheat"]["move_floor"], 26)


class TimingAttackTests(ApiTestCase):
    """The attack the whole module exists for: POST a fabricated fast time."""

    def test_honest_solve_verifies(self):
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=22.0, moves=58,
                        claimed_seconds=22.0)
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.data["verdict"], "verified", r.data)

    def test_claimed_time_is_ignored_evidence_wins(self):
        """The headline attack. Claiming 0.01s while shipping a bundle that
        shows 22 seconds must not produce a 0.01s result."""
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=22.0, moves=58,
                        claimed_seconds=0.01)
        self.assertEqual(r.status_code, 201)
        # The stored time is the DERIVED one, not the claim.
        self.assertAlmostEqual(r.data["seconds"], 22.0, places=1)
        self.assertEqual(r.data["verdict"], "rejected")
        self.assertTrue(any("claimed_time_contradicts_evidence" in x
                            for x in r.data["reject_reasons"]), r.data)

    def test_fabricating_a_short_bundle_hits_the_move_floor(self):
        """If instead you shrink the BUNDLE to match a fast claim, you fall
        through the move-count floor — the timing attack collapsing into the
        count attack."""
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=0.5, moves=2, claimed_seconds=0.5)
        self.assertEqual(r.data["verdict"], "rejected")
        joined = " ".join(r.data["reject_reasons"])
        self.assertIn("too_few_moves", joined, r.data)

    def test_short_bundle_with_many_moves_is_physically_impossible(self):
        """And if you keep the move count up to clear the floor, the implied
        turn rate becomes impossible. There is no gap between the two."""
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=1.5, moves=60, claimed_seconds=1.5)
        self.assertEqual(r.data["verdict"], "rejected")
        joined = " ".join(r.data["reject_reasons"])
        self.assertIn("implied_tps_impossible", joined, r.data)

    def test_duration_cannot_exceed_the_server_window(self):
        """Bound 1: you cannot have solved for 10 minutes inside a window the
        server knows was 30 seconds."""
        s = self.new_scramble()
        Scramble.objects.filter(nonce=s["nonce"]).update(
            issued_at=timezone.now() - timedelta(seconds=30))
        r = self.submit(s["nonce"], seconds=600.0, moves=900,
                        claimed_seconds=600.0)
        joined = " ".join(r.data["reject_reasons"])
        self.assertIn("duration_exceeds_server_window", joined, r.data)

    def test_absurd_fps_is_rejected(self):
        """Manufacturing a short duration out of long footage by inflating
        the declared frame rate."""
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=22.0, moves=58, fps=100000.0)
        joined = " ".join(r.data["reject_reasons"])
        self.assertIn("implausible_fps", joined, r.data)


class ReplayTests(ApiTestCase):
    def test_scramble_is_single_use(self):
        """Record one genuine solve, replay its bundle forever — closed by
        consuming the scramble."""
        s = self.new_scramble()
        first = self.submit(s["nonce"], seconds=22.0, moves=58)
        self.assertEqual(first.status_code, 201)
        second = self.submit(s["nonce"], seconds=22.0, moves=58)
        self.assertEqual(second.status_code, 409, second.content)

    def test_rejected_attempt_still_burns_the_scramble(self):
        """Otherwise an attacker retries against one scramble until a forgery
        happens to pass."""
        s = self.new_scramble()
        bad = self.submit(s["nonce"], seconds=0.5, moves=2)
        self.assertEqual(bad.data["verdict"], "rejected")
        retry = self.submit(s["nonce"], seconds=22.0, moves=58)
        self.assertEqual(retry.status_code, 409)

    def test_expired_scramble_is_rejected(self):
        s = self.new_scramble()
        Scramble.objects.filter(nonce=s["nonce"]).update(
            expires_at=timezone.now() - timedelta(seconds=1))
        r = self.submit(s["nonce"], seconds=22.0, moves=58)
        self.assertIn("scramble_expired", r.data["reject_reasons"])

    def test_unknown_nonce_is_rejected(self):
        r = self.submit("not-a-real-nonce", seconds=22.0, moves=58)
        self.assertEqual(r.status_code, 400)


class AbstainTests(ApiTestCase):
    """Unreadable evidence must abstain, never accuse."""

    def test_bad_light_reviews_rather_than_rejects(self):
        """Low light costs the onset detector 40+ points, so the count reads
        low. Rejecting on that would manufacture a cheat out of a model
        failure."""
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=22.0, moves=12, lighting_ok=False)
        self.assertEqual(r.data["verdict"], "review", r.data)
        self.assertEqual(r.data["reject_reasons"], [])

    def test_very_fast_solve_reviews_rather_than_rejects(self):
        """Past the separation limit the legit and cheat bands overlap and no
        threshold distinguishes them, so the gate must abstain.

        The speed is DERIVED from `separation_tps_limit()` rather than
        hardcoded. It was hardcoded at 7.5 TPS, and when speed augmentation
        moved the limit from 7.11 to 9.62 this test started failing for a
        change that was entirely intended — a test that has to be edited
        every time a measurement improves is a test that will eventually be
        edited without being read.
        """
        from anticheat_gate import separation_tps_limit
        limit = separation_tps_limit()
        moves = 60
        seconds = moves / (limit + 2.0)      # comfortably past the limit
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=seconds, moves=moves)
        self.assertEqual(r.data["verdict"], "review", r.data)
        self.assertTrue(any("too_fast_to_separate" in x
                            for x in r.data["review_reasons"]), r.data)

    def test_solve_just_under_the_limit_still_verifies(self):
        """The other side of the same boundary — and the thing the speed
        augmentation bought. A solve just inside the separation limit must
        VERIFY, not get punted to human review."""
        from anticheat_gate import separation_tps_limit
        limit = separation_tps_limit()
        moves = 60
        seconds = moves / (limit - 1.0)      # comfortably inside
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=seconds, moves=moves)
        self.assertEqual(r.data["verdict"], "verified", r.data)


class PostStopTests(ApiTestCase):
    """Stop the timer unsolved, solve it, then scan."""

    def test_hidden_solve_after_timer_stop_is_rejected(self):
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=22.0, moves=58,
                        observed_moves_after_stop=45, post_stop_seconds=30.0)
        joined = " ".join(r.data["reject_reasons"])
        self.assertIn("moves_after_timer_stop", joined, r.data)

    def test_phantom_moves_after_stop_are_tolerated(self):
        """The allowance scales with window length — phantoms accumulate."""
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=22.0, moves=58,
                        observed_moves_after_stop=6, post_stop_seconds=30.0)
        self.assertEqual(r.data["verdict"], "verified", r.data)

    def test_overlong_scan_window_abstains(self):
        """Past ~60s the phantom allowance could hide a whole solve, so the
        test has no power and must say so instead of passing."""
        s = self.new_scramble()
        r = self.submit(s["nonce"], seconds=22.0, moves=58,
                        observed_moves_after_stop=0, post_stop_seconds=300.0)
        self.assertEqual(r.data["verdict"], "review", r.data)
        self.assertTrue(any("post_stop_window_too_long" in x
                            for x in r.data["review_reasons"]), r.data)


class WaitlistTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_signup_and_dedupe(self):
        r = self.client.post("/api/waitlist/", {"email": "A@Example.com "},
                             format="json")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.data["duplicate"])
        again = self.client.post("/api/waitlist/", {"email": "a@example.com"},
                                 format="json")
        self.assertTrue(again.data["duplicate"])
        self.assertEqual(WaitlistSignup.objects.count(), 1)

    def test_bad_email_rejected(self):
        r = self.client.post("/api/waitlist/", {"email": "nope"}, format="json")
        self.assertEqual(r.status_code, 400)

    def test_honeypot_silently_accepts_and_stores_nothing(self):
        r = self.client.post("/api/waitlist/",
                             {"email": "bot@spam.com", "company": "Acme"},
                             format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(WaitlistSignup.objects.count(), 0)

    def test_no_ip_stored_without_a_salt(self):
        self.client.post("/api/waitlist/", {"email": "x@example.com"},
                         format="json")
        self.assertEqual(WaitlistSignup.objects.get().ip_hash, "")


class UnsubscribeTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.client.post("/api/waitlist/", {"email": "a@example.com"},
                         format="json")
        self.row = WaitlistSignup.objects.get()

    def url(self, token=None):
        return f"/api/waitlist/unsubscribe/{token or self.row.unsubscribe_token}/"

    def test_get_unsubscribes(self):
        r = self.client.get(self.url())
        self.assertEqual(r.status_code, 200)
        self.row.refresh_from_db()
        self.assertFalse(self.row.is_subscribed)

    def test_post_unsubscribes(self):
        """Gmail/Outlook POST here for one-click unsubscribe. A GET-only
        endpoint makes their button silently fail."""
        r = self.client.post(self.url())
        self.assertEqual(r.status_code, 200)
        self.row.refresh_from_db()
        self.assertFalse(self.row.is_subscribed)

    def test_unknown_token_does_not_leak_membership(self):
        r = self.client.get(self.url("totally-made-up"))
        self.assertEqual(r.status_code, 200)

    def test_resignup_resubscribes_and_clears_campaign(self):
        """Signing up again is an explicit act by the address owner. The
        campaign key must clear too, or they rejoin and then silently never
        receive the campaign they just asked to be part of."""
        self.client.get(self.url())
        WaitlistSignup.objects.update(last_campaign="launch")
        self.client.post("/api/waitlist/", {"email": "a@example.com"},
                         format="json")
        self.row.refresh_from_db()
        self.assertTrue(self.row.is_subscribed)
        self.assertEqual(self.row.last_campaign, "")


class SendWaitlistTests(TestCase):
    """`manage.py send_waitlist` — the thing that replaced the email
    provider. Its failure modes all end in 'mailed someone twice'."""

    def setUp(self):
        for i in range(5):
            WaitlistSignup.objects.create(email=f"u{i}@example.com")
        self.body = Path(tempfile.gettempdir()) / "wl_body.txt"
        self.body.write_text("Hello from CubeArena.", encoding="utf-8")

    def run_cmd(self, **kw):
        out = StringIO()
        opts = {"campaign": "launch", "subject": "Hi",
                "body": str(self.body), "delay": 0, "stdout": out}
        opts.update(kw)
        call_command("send_waitlist", **opts)
        return out.getvalue()

    def test_sends_to_everyone_once(self):
        self.run_cmd()
        self.assertEqual(len(mail.outbox), 5)
        self.assertEqual(
            WaitlistSignup.objects.filter(last_campaign="launch").count(), 5)

    def test_rerunning_sends_to_nobody(self):
        """The resumability property: a crash halfway through is recovered by
        running it again, NOT by mailing everyone twice."""
        self.run_cmd()
        mail.outbox.clear()
        self.run_cmd()
        self.assertEqual(len(mail.outbox), 0)

    def test_partial_run_resumes(self):
        self.run_cmd(limit=2)
        self.assertEqual(len(mail.outbox), 2)
        mail.outbox.clear()
        self.run_cmd()
        self.assertEqual(len(mail.outbox), 3)

    def test_dry_run_sends_nothing(self):
        out = self.run_cmd(dry_run=True)
        self.assertEqual(len(mail.outbox), 0)
        self.assertIn("would send", out)
        self.assertEqual(
            WaitlistSignup.objects.filter(last_campaign="launch").count(), 0)

    def test_unsubscribed_are_skipped(self):
        WaitlistSignup.objects.filter(email="u0@example.com").update(
            unsubscribed_at=timezone.now())
        self.run_cmd()
        self.assertEqual(len(mail.outbox), 4)
        self.assertNotIn("u0@example.com",
                         [a for m in mail.outbox for a in m.to])

    def test_every_message_carries_a_unique_unsubscribe(self):
        """Per-recipient, which is why this cannot be one BCC'd message."""
        self.run_cmd()
        links = set()
        for m in mail.outbox:
            self.assertIn("List-Unsubscribe", m.extra_headers)
            self.assertIn("One-Click", m.extra_headers["List-Unsubscribe-Post"])
            self.assertIn("/unsubscribe/", m.body)
            links.add(m.extra_headers["List-Unsubscribe"])
        self.assertEqual(len(links), 5)

    def test_unsubscribe_link_in_mail_actually_works(self):
        """End to end: the token the command emits must be the one the view
        accepts. A mismatch here is invisible until after a real send."""
        self.run_cmd(only=["u0@example.com"])
        body = mail.outbox[0].body
        url = [w for w in body.split() if "/unsubscribe/" in w][0]
        path = url[url.index("/api/"):]
        self.assertEqual(APIClient().get(path).status_code, 200)
        self.assertFalse(
            WaitlistSignup.objects.get(email="u0@example.com").is_subscribed)

    def test_only_restricts_recipients(self):
        self.run_cmd(only=["u1@example.com"])
        self.assertEqual([m.to for m in mail.outbox], [["u1@example.com"]])


class AllauthTests(TestCase):
    """allauth headless — the launch auth path. These assert the endpoints
    exist and are wired, not that allauth works."""

    def setUp(self):
        self.client = APIClient()

    def test_headless_config_endpoint_is_mounted(self):
        r = self.client.get("/api/auth/browser/v1/config")
        self.assertEqual(r.status_code, 200, r.content[:200])

    def test_session_endpoint_reports_unauthenticated(self):
        r = self.client.get("/api/auth/browser/v1/auth/session")
        # 401 is allauth's "no session", which is the correct answer here.
        self.assertIn(r.status_code, (401, 410))

    def test_signup_creates_an_unverified_user(self):
        r = self.client.post(
            "/api/auth/browser/v1/auth/signup",
            {"email": "new@example.com", "password": "a-Long-Passw0rd!"},
            format="json")
        self.assertIn(r.status_code, (200, 401), r.content[:300])
        self.assertTrue(
            get_user_model().objects.filter(email="new@example.com").exists())
        # Verification is mandatory, so signing up must send mail rather than
        # hand out a usable account.
        self.assertEqual(len(mail.outbox), 1)

    def test_verification_subject_is_not_the_default_site(self):
        """A fresh django.contrib.sites row is "example.com", and allauth
        prefixes subjects with it — so without an explicit prefix the first
        email a real user ever gets reads "[example.com] Please Confirm Your
        Email Address", which looks precisely like phishing."""
        self.client.post(
            "/api/auth/browser/v1/auth/signup",
            {"email": "subj@example.com", "password": "a-Long-Passw0rd!"},
            format="json")
        self.assertEqual(len(mail.outbox), 1)
        self.assertNotIn("[example.com]", mail.outbox[0].subject)
        self.assertIn("CubeArena", mail.outbox[0].subject)


class AllauthCsrfTests(TestCase):
    """CSRF, with enforcement ON.

    This class exists because of a bug the ordinary tests could not see: the
    Django test client disables CSRF checking by default, so every headless
    POST above passes whether or not CSRF works. Live over HTTP, signup
    returned **403** — allauth's browser client is session-based and Django
    demands the token. The Next auth UI must therefore fetch a CSRF cookie
    and send `X-CSRFToken` on every unsafe request, and that requirement now
    has a test instead of being discovered again by hand.
    """

    def setUp(self):
        self.client = APIClient(enforce_csrf_checks=True)

    def test_signup_without_csrf_token_is_rejected(self):
        r = self.client.post(
            "/api/auth/browser/v1/auth/signup",
            {"email": "nocsrf@example.com", "password": "a-Long-Passw0rd!"},
            format="json")
        self.assertEqual(r.status_code, 403, r.content[:200])
        self.assertFalse(get_user_model().objects
                         .filter(email="nocsrf@example.com").exists())

    def test_signup_with_csrf_token_succeeds(self):
        # GET anything that sets the cookie, then echo it back in the header.
        self.client.get("/api/auth/browser/v1/config")
        token = self.client.cookies["csrftoken"].value
        r = self.client.post(
            "/api/auth/browser/v1/auth/signup",
            {"email": "withcsrf@example.com", "password": "a-Long-Passw0rd!"},
            format="json", HTTP_X_CSRFTOKEN=token)
        self.assertIn(r.status_code, (200, 401), r.content[:300])
        self.assertTrue(get_user_model().objects
                        .filter(email="withcsrf@example.com").exists())

    def test_solve_submission_is_exempt_via_device_key(self):
        """The solve endpoints must stay usable by a non-cookie client while
        the device-key stub is what the capture tool authenticates with —
        DRF only enforces CSRF for SessionAuthentication."""
        c = APIClient(enforce_csrf_checks=True)
        c.credentials(HTTP_AUTHORIZATION="Device csrfprobe")
        r = c.post("/api/scrambles/", {}, format="json")
        self.assertEqual(r.status_code, 201, r.content[:200])
