"""
API endpoints.

Route map (all under /api/, see config/urls.py):

    GET  health/            liveness + what this build believes
    POST waitlist/          landing-page signup
    POST scrambles/         issue a signed, single-use scramble
    GET  solves/            the caller's own solves
    POST solves/            submit a solve; the SERVER decides the verdict
    GET  solves/analysis/   the caller's coach analyses, oldest first
    GET  solves/<id>/analysis/  one solve's analysis (paid, or your latest)
"""

import hashlib
import os
from datetime import timedelta

from django.db import IntegrityError, transaction
from django.http import HttpResponse
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import (api_view, authentication_classes,
                                       permission_classes, throttle_classes)
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from . import scrambles as scr
from .models import Scramble, Solve, WaitlistSignup
from .throttles import ScrambleThrottle, SolveThrottle, WaitlistThrottle
from .timing import TimingFacts, evaluate

#: How long a scramble stays usable. Long enough to walk away and come back,
#: short enough that a stockpile of unused scrambles is not worth farming.
SCRAMBLE_TTL = timedelta(minutes=30)


def hash_ip(request) -> str:
    """Salted hash of the caller's IP, or "" when no salt is configured.

    No salt means store NOTHING. An unsalted hash of an IP address is not
    anonymisation — the space is small enough to enumerate exhaustively in
    seconds — so the choice is a real salt or no record at all.
    """
    salt = os.environ.get("WAITLIST_IP_SALT", "")
    if not salt:
        return ""
    fwd = request.headers.get("X-Forwarded-For", "")
    ip = fwd.split(",")[0].strip() or request.META.get("REMOTE_ADDR", "")
    if not ip:
        return ""
    return hashlib.sha256(f"{salt}{ip}".encode()).hexdigest()[:32]


@api_view(["GET"])
def health(request):
    """Liveness probe, and a place to read the live anticheat constants.

    Exposing the thresholds is deliberate: they are not secrets (a cheat
    learns the floor by bisection in a dozen attempts regardless), and having
    them readable means a client can show honest guidance instead of guessing
    at what the server will accept.
    """
    from anticheat_gate import (MIN_OBSERVED_MOVES, SOLVER_CEILING_QTM,
                                separation_tps_limit)

    from .timing import MAX_HUMAN_TPS, MIN_PLAUSIBLE_SECONDS
    return Response({
        "status": "ok",
        "time": timezone.now().isoformat(),
        "anticheat": {
            "move_floor": MIN_OBSERVED_MOVES,
            "solver_ceiling_qtm": SOLVER_CEILING_QTM,
            "abstain_above_tps": separation_tps_limit(),
            "max_human_tps": MAX_HUMAN_TPS,
            "min_seconds": MIN_PLAUSIBLE_SECONDS,
        },
    })


@api_view(["POST"])
@throttle_classes([WaitlistThrottle])
def waitlist(request):
    """Landing-page signup. Mirrors the Next route handler it replaces."""
    data = request.data if isinstance(request.data, dict) else {}

    # Honeypot: hidden from sighted users, screen readers and autofill, so
    # anything in it is a bot. Answer 200 — a bot told it was caught simply
    # retries without the field.
    if str(data.get("company", "")).strip():
        return Response({"ok": True})

    email = str(data.get("email", "")).strip().lower()
    if not _looks_like_email(email):
        return Response({"error": "That does not look like an email address."},
                        status=status.HTTP_400_BAD_REQUEST)

    row, created = WaitlistSignup.objects.get_or_create(
        email=email,
        defaults={
            "referrer": str(data.get("ref") or "")[:500],
            "ip_hash": hash_ip(request),
        },
    )

    # Signing up again after unsubscribing is a re-subscribe. It is an
    # explicit act by the address owner, so honouring it is correct — but it
    # must clear the campaign key too, or they rejoin and then silently never
    # receive the campaign they just asked to be part of.
    if not created and row.unsubscribed_at is not None:
        row.unsubscribed_at = None
        row.last_campaign = ""
        row.save(update_fields=["unsubscribed_at", "last_campaign"])

    return Response({"ok": True, "duplicate": not created})


@api_view(["GET", "POST"])
@permission_classes([AllowAny])
@authentication_classes([])
def unsubscribe(request, token: str):
    """One-click unsubscribe.

    GET so a link in the footer works, POST because that is what
    `List-Unsubscribe-Post: List-Unsubscribe=One-Click` requires — Gmail and
    Outlook POST here when someone uses their built-in unsubscribe button,
    and a GET-only endpoint makes that button silently fail.

    Unknown tokens answer 200, not 404: the response must not reveal whether
    an address is on the list, and someone clicking an old link should see
    "you're unsubscribed" either way rather than an error.
    """
    row = WaitlistSignup.objects.filter(unsubscribe_token=token).first()
    if row and row.unsubscribed_at is None:
        row.unsubscribed_at = timezone.now()
        row.save(update_fields=["unsubscribed_at"])
    if request.method == "POST":
        return Response({"ok": True})
    return HttpResponse(
        "<!doctype html><meta charset=utf-8>"
        "<title>Unsubscribed</title>"
        "<div style=\"font:16px/1.6 system-ui;max-width:32rem;margin:20vh auto;"
        "padding:0 1rem;text-align:center\">"
        "<h1 style=\"font-size:1.3rem\">You're unsubscribed.</h1>"
        "<p style=\"color:#666\">You won't get any more email from "
        "CubeArena.</p></div>",
        content_type="text/html; charset=utf-8")


def _looks_like_email(email: str) -> bool:
    """Permissive on purpose — see the Next route this replaces. A rejected
    real address is lost forever; a bounced one costs nothing."""
    if not (5 <= len(email) <= 254) or " " in email or email.count("@") != 1:
        return False
    local, _, domain = email.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") \
        and not domain.endswith(".")


@api_view(["POST"])
@throttle_classes([ScrambleThrottle])
def issue_scramble(request):
    """Issue a signed, single-use scramble bound to the caller."""
    user = request.user if getattr(request.user, "is_authenticated", False) else None
    row = Scramble(
        user=user,
        moves=scr.generate(),
        expires_at=timezone.now() + SCRAMBLE_TTL,
    )
    row.signature = scr.sign(row.moves, row.nonce)
    row.save()
    return Response({
        "nonce": row.nonce,
        "moves": row.moves,
        "signature": row.signature,
        "issued_at": row.issued_at.isoformat(),
        "expires_at": row.expires_at.isoformat(),
    }, status=status.HTTP_201_CREATED)


@api_view(["GET", "POST"])
@throttle_classes([SolveThrottle])
def solves(request):
    if request.method == "GET":
        user = request.user if getattr(request.user, "is_authenticated", False) else None
        qs = Solve.objects.filter(user=user) if user else Solve.objects.none()
        return Response([_solve_json(s) for s in qs[:50]])
    return _submit_solve(request)


def _submit_solve(request):
    data = request.data if isinstance(request.data, dict) else {}

    try:
        scramble = Scramble.objects.get(nonce=str(data.get("nonce", "")).strip())
    except Scramble.DoesNotExist:
        return Response({"error": "Unknown scramble nonce."},
                        status=status.HTTP_400_BAD_REQUEST)

    if not scr.verify(scramble.moves, scramble.nonce, scramble.signature):
        # The stored row itself does not verify — tampering at the DB layer,
        # or the signing key changed. Either way it cannot anchor a verdict.
        return Response({"error": "Scramble signature invalid."},
                        status=status.HTTP_400_BAD_REQUEST)

    # Evidence properties. These are REPORTED for now; the re-verification
    # worker will measure them from the uploaded frames instead. See
    # timing.py's "what this does not close".
    try:
        frame_count = int(data.get("frame_count", 0))
        fps = float(data.get("fps", 0) or 0)
        observed_moves = int(data.get("observed_moves", 0))
        after_stop = int(data.get("observed_moves_after_stop", 0) or 0)
    except (TypeError, ValueError):
        return Response(
            {"error": "frame_count, fps and observed_moves must be numbers."},
            status=status.HTTP_400_BAD_REQUEST)

    claimed = data.get("claimed_seconds")
    claimed = float(claimed) if isinstance(claimed, (int, float)) else None
    post_stop = data.get("post_stop_seconds")
    post_stop = float(post_stop) if isinstance(post_stop, (int, float)) else None

    facts = TimingFacts(
        server_window_seconds=(timezone.now() - scramble.issued_at).total_seconds(),
        frame_count=max(frame_count, 0),
        fps=fps,
        observed_moves=max(observed_moves, 0),
        claimed_seconds=claimed,
        observed_moves_after_stop=max(after_stop, 0),
        post_stop_seconds=post_stop,
        lighting_ok=(data.get("lighting_ok")
                     if isinstance(data.get("lighting_ok"), bool) else None),
        continuity=(data.get("continuity")
                    if isinstance(data.get("continuity"), dict) else None),
        scramble_consumed=scramble.is_consumed,
        scramble_expired=scramble.is_expired,
    )
    result = evaluate(facts)

    # Consume the scramble WHATEVER the verdict. A rejected attempt still
    # burns it — otherwise an attacker retries against one scramble until a
    # forgery happens to pass, which is exactly the search the single-use
    # rule exists to prevent.
    user = request.user if getattr(request.user, "is_authenticated", False) else None
    try:
        with transaction.atomic():
            if not scramble.is_consumed:
                scramble.consumed_at = timezone.now()
                scramble.save(update_fields=["consumed_at"])
            solve = Solve.objects.create(
                user=user,
                scramble=scramble,
                derived_seconds=facts.derived_seconds,
                frame_count=facts.frame_count,
                fps=max(fps, 0.0),
                observed_moves=facts.observed_moves,
                observed_moves_after_stop=facts.observed_moves_after_stop,
                post_stop_seconds=post_stop,
                evidence_sha256=str(data.get("evidence_sha256", ""))[:64],
                claimed_seconds=claimed,
                claim_discrepancy=result["claim_discrepancy"],
                verdict=result["verdict"],
                reject_reasons=result["reject_reasons"],
                review_reasons=result["review_reasons"],
                detail=result,
            )
    except IntegrityError:
        # OneToOne on scramble: a second submission raced the first, or is
        # a deliberate replay. The first wins.
        return Response({"error": "A solve for this scramble already exists."},
                        status=status.HTTP_409_CONFLICT)

    return Response(_solve_json(solve), status=status.HTTP_201_CREATED)


#: Face keys in URFDLB order, so `face_share` always answers with all six
#: even for the faces a solve never turned. Omitting them would leave the
#: client to decide whether a missing face means zero or unmeasured.
FACES = ("U", "R", "F", "D", "L", "B")

#: Newest-first is the API's default everywhere else, but a trend is only a
#: trend read forwards, and the client should not have to know to reverse it.
ANALYSIS_LIMIT = 200


def _analysed(user):
    """This user's usable analyses, oldest first. One definition, because
    the list view, the detail view's permission check and the free-tier cut
    must all agree on which solve is "your latest"."""
    return [s for s in (Solve.objects.filter(user=user)
                        .exclude(analysis={})
                        .order_by("received_at")[:ANALYSIS_LIMIT])
            if s.analysis.get("usable")]


def _is_premium(user) -> bool:
    profile = getattr(user, "profile", None)
    return bool(profile and profile.is_premium)


@api_view(["GET"])
def solve_analysis(request):
    """The caller's own analysed solves, oldest first.

    Shape matches `coach/report.py`'s `solve_report()` payload one-to-one —
    see `web/lib/analytics.ts`, which is written against it. Nothing is
    recomputed here: the gate that decided which metrics may be shown, and
    the measured error beside each one, were applied when the analysis was
    stored, and re-deriving either in this layer would put a second copy of
    the registry somewhere it could quietly disagree.

    THE PAID GATE IS HERE, not in the client. A free account is served its
    most recent analysed solve and nothing else — so averages, trends and
    per-solve history are not merely hidden by the UI, they are absent from
    the response. A client that ignores `is_premium` still cannot compute an
    average it was never sent. `truncated` tells an honest UI to offer the
    upgrade instead of drawing a one-point chart.

    Solves without an analysis are omitted rather than returned empty. A
    solve submitted through `POST solves/` has none until a worker runs the
    decode (TODO §2B), and a row of dashes reads as a broken metric rather
    than as work not yet done.
    """
    if not getattr(request.user, "is_authenticated", False):
        return Response({"detail": "Authentication required."},
                        status=status.HTTP_403_FORBIDDEN)

    rows = _analysed(request.user)
    premium = _is_premium(request.user)
    shown = rows if premium else rows[-1:]
    return Response({
        "solves": [_analysis_json(s) for s in shown],
        "is_premium": premium,
        # Whether anything was withheld, so the free tier can say "3 more
        # solves analysed" rather than implying this is all there is.
        "truncated": len(shown) < len(rows),
        "total": len(rows),
    })


@api_view(["GET"])
def solve_analysis_detail(request, solve_id: int):
    """One solve's analysis.

    A free account may open its most recent analysed solve — the one the
    `/analytics` preview already shows — and nothing older. Paid opens any
    of its own. Someone else's is 404, never 403: whether a given solve id
    exists on another account is not something this endpoint should confirm.
    """
    if not getattr(request.user, "is_authenticated", False):
        return Response({"detail": "Authentication required."},
                        status=status.HTTP_403_FORBIDDEN)

    rows = _analysed(request.user)
    match = next((s for s in rows if s.id == solve_id), None)
    if match is None:
        return Response({"error": "No analysis for that solve."},
                        status=status.HTTP_404_NOT_FOUND)

    if not _is_premium(request.user) and match.id != rows[-1].id:
        return Response(
            {"error": "Per-solve analysis is part of Coach.",
             "upgrade_required": True},
            status=status.HTTP_402_PAYMENT_REQUIRED)

    # The headline facts travel with the analysis so the detail page renders
    # from one request. These come from the SERVER-derived columns, not from
    # the coach payload — `derived_seconds` is what the verdict rests on and
    # `span_seconds` is not the same number (it excludes setup and the
    # post-solve scan).
    return Response({
        **_analysis_json(match),
        "is_premium": _is_premium(request.user),
        "solve": _solve_json(match),
        "result": match.result,
        "opponent": (match.opponent.username if match.opponent_id else None),
        "rating_delta": match.rating_delta,
        # Safe to return: it is the caller's own consumed, single-use
        # scramble. Never expose one that is still unconsumed on another
        # account — that is a solution someone could prepare against.
        "scramble": match.scramble.moves,
    })


def _analysis_json(s: Solve) -> dict:
    metrics = s.analysis.get("metrics", {})
    values = {k: m["value"] for k, m in metrics.items()
              if not isinstance(m.get("value"), dict)}
    share = metrics.get("face_share", {}).get("value") or {}
    return {
        "id": str(s.id),
        # Date only: the trend's x axis is a day, and the exact minute a
        # solve was recorded is not something a chart should leak.
        "date": s.received_at.date().isoformat(),
        "regime": s.analysis.get("regime", "unknown"),
        "values": values,
        "faceShare": {f: round(float(share.get(f, 0.0)), 4) for f in FACES},
        "suppressed": s.analysis.get("suppressed", []),
    }


def _solve_json(s: Solve) -> dict:
    return {
        "id": s.id,
        "verdict": s.verdict,
        "seconds": round(s.derived_seconds, 3),
        "claimed_seconds": s.claimed_seconds,
        "claim_discrepancy": s.claim_discrepancy,
        "observed_moves": s.observed_moves,
        "tps": round(s.tps, 2) if s.tps else None,
        "reject_reasons": s.reject_reasons,
        "review_reasons": s.review_reasons,
        "reverified": s.reverified,
        "received_at": s.received_at.isoformat(),
    }
