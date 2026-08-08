"""
Account and profile endpoints.

    GET   me/                 the signed-in account + profile
    PATCH me/                 country, preset avatar, notification prefs
    POST  me/username/        rename, rate limited to once a week
    POST  me/email/           change address, once a day, re-verification required
    GET   users/<username>/   ANY profile — every profile is public
    GET   leaderboard/        the ranked table

Password change is NOT here: allauth already owns it at
`/api/auth/browser/v1/account/password/change`, and a second implementation
would be a second place for the session-invalidation rules to be wrong. The
rate limit for it is enforced in `middleware.py` instead, which is the only
way to bound an endpoint we do not own.
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.db.models import Count, Min, Q
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from . import moderation
from .avatars import is_valid_avatar
from .countries import is_valid_country
from .models import Profile, Solve

User = get_user_model()

#: Cap on how many rows the public leaderboard will hand out at once.
LEADERBOARD_MAX = 100


def _require_profile(user) -> Profile:
    """Profiles come from a signal, but a row predating it would 500 every
    request for that account. Heal instead."""
    profile, _ = Profile.objects.get_or_create(user=user)
    return profile


def solve_stats(user) -> dict:
    """Aggregate a user's verified solves in ONE query.

    Deliberately not a loop over solves: this runs per row on the
    leaderboard, and the N+1 version is what turns a 10-row table into 30
    queries.
    """
    agg = Solve.objects.filter(user=user, verdict=Solve.VERIFIED).aggregate(
        best=Min("derived_seconds"),
        total=Count("id"),
    )
    return {
        "best_seconds": round(agg["best"], 2) if agg["best"] is not None else None,
        "total_solves": agg["total"] or 0,
        "best_solve_id": best_solve_id(user),
    }


def best_solve_id(user) -> int | None:
    """The id of the fastest verified solve, so "Current Best" can be a link.

    A second query rather than folding into the aggregate above: SQLite has
    no `arg_min`, and the alternatives (window function, or ordering the
    whole set) cost more than one indexed lookup. `Solve`'s
    (user, verdict, derived_seconds) index is exactly this query.

    NOT derived on the client from the history rows — that list is capped at
    50, so on an account with more solves than that the client's "best" would
    silently be the best of the most recent 50.
    """
    row = (Solve.objects.filter(user=user, verdict=Solve.VERIFIED)
           .order_by("derived_seconds")
           .values_list("id", flat=True)
           .first())
    return row


def public_profile_json(user, *, stats: dict | None = None,
                        rank: int | None = None) -> dict:
    """The shape every profile is served in.

    ONE function, used by the leaderboard, the public profile page and your
    own settings page, so the three cannot disagree about what a player looks
    like. It contains no email and no rate-limit clocks — everything here is
    safe for anyone to read, which is what makes "all profiles are public"
    a statement about this function rather than a policy someone has to
    remember.
    """
    p = _require_profile(user)
    data = {
        "id": user.id,
        "username": user.username,
        "country": p.country or None,
        "rating": p.rating,
        "best_rank": p.best_rank,
        "is_founder": p.is_founder,
        # Public: an avatar is meant to be seen. Safe to expose because it
        # is a key from a fixed set, not anything this account authored.
        "avatar": p.avatar_preset or None,
        "joined": user.date_joined.isoformat(),
    }
    data.update(stats if stats is not None else solve_stats(user))
    if rank is not None:
        data["rank"] = rank
    return data


@api_view(["GET", "PATCH"])
@permission_classes([IsAuthenticated])
def me(request):
    """Your own account. Adds the private fields `public_profile_json` omits."""
    user = request.user
    p = _require_profile(user)

    if request.method == "PATCH":
        data = request.data if isinstance(request.data, dict) else {}

        if "country" in data:
            country = str(data.get("country") or "").strip().upper()
            # "" clears it. Anything else must be a real ISO code — but note
            # it is never checked against where the request came FROM. Repping
            # a country you do not live in is the point.
            if country and not is_valid_country(country):
                return Response({"error": "Unknown country code."},
                                status=status.HTTP_400_BAD_REQUEST)
            p.country = country

        if "avatar" in data:
            # "" / null clears it. Anything else must be a key this server
            # knows — the allowlist in `core.avatars` IS the gate, so a
            # client offering an unknown key is refused rather than having
            # its string stored and rendered back at everyone.
            avatar = str(data.get("avatar") or "").strip()
            if not is_valid_avatar(avatar):
                return Response({"error": "Unknown avatar."},
                                status=status.HTTP_400_BAD_REQUEST)
            if avatar != p.avatar_preset:
                p.avatar_preset = avatar
                p.avatar_updated_at = timezone.now()

        for field in ("notify_invites", "notify_recap", "notify_pb"):
            if field in data and isinstance(data[field], bool):
                setattr(p, field, data[field])

        p.save()

    stats = solve_stats(user)
    out = public_profile_json(user, stats=stats)
    out.update({
        # Same shape and same source as a public profile's history, so the
        # home page's "recent solves" and a profile's table cannot disagree.
        "solves": [
            _history_json(s) for s in
            Solve.objects.filter(user=user).select_related("opponent")[:50]
        ],
        "email": user.email,
        # Private, unlike `is_founder`: this is billing state, not a badge.
        "is_premium": p.is_premium,
        "notify_invites": p.notify_invites,
        "notify_recap": p.notify_recap,
        "notify_pb": p.notify_pb,
        "rank": _rank_of(p),
        # Seconds until each change is allowed again. 0 means "now".
        "limits": {
            "username": p.username_retry_after(),
            "email": p.email_retry_after(),
            "password": p.password_retry_after(),
        },
    })
    return Response(out)


def _rank_of(profile: Profile) -> int:
    """1-based position by rating. Ties break toward the older account."""
    better = Profile.objects.filter(
        Q(rating__gt=profile.rating)
        | Q(rating=profile.rating, user_id__lt=profile.user_id)
    ).count()
    return better + 1


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def change_username(request):
    """Rename, at most once every 7 days.

    The rate limit is not anti-spam decoration — `check_username` refuses
    names that merely LOOK like an existing account, and without a limit an
    attacker just cycles candidates until one gets past that check.
    """
    user = request.user
    p = _require_profile(user)

    retry = p.username_retry_after()
    if retry:
        return Response(
            {"error": "You can change your username again later.",
             "retry_after": retry},
            status=status.HTTP_429_TOO_MANY_REQUESTS)

    requested = str((request.data or {}).get("username", ""))

    if requested.strip() == user.username:
        return Response({"error": "That's already your username."},
                        status=status.HTTP_400_BAD_REQUEST)

    # Skeleton -> user id for every OTHER account, so a lookalike is caught.
    # Fine at this size; becomes a stored `username_skeleton` column with an
    # index the moment the user table is large enough to care.
    taken = {}
    for uid, uname in User.objects.exclude(pk=user.pk).values_list("id", "username"):
        for sk in moderation.skeletons(uname):
            taken.setdefault(sk, uid)

    try:
        clean = moderation.check_username(requested, taken_skeletons=taken,
                                          allow_id=user.pk)
    except moderation.NameRejected as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    now = timezone.now()
    history = list(p.previous_usernames or [])
    history.append({"name": user.username, "at": now.isoformat()})

    flags = moderation.review_flags(clean)

    try:
        with transaction.atomic():
            user.username = clean
            user.save(update_fields=["username"])
            p.username_changed_at = now
            p.previous_usernames = history[-20:]
            p.name_flags = flags
            p.needs_name_review = bool(flags)
            p.save(update_fields=["username_changed_at", "previous_usernames",
                                  "name_flags", "needs_name_review"])
    except IntegrityError:
        # Exact-match collision; the skeleton check above catches lookalikes
        # but a straight race for the same name lands here.
        return Response({"error": "That username was just taken."},
                        status=status.HTTP_409_CONFLICT)

    return Response({"username": user.username,
                     "retry_after": p.username_retry_after()})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def change_email(request):
    """Change the address on the account, at most once a day.

    Goes through allauth's EmailAddress table rather than writing
    `user.email` directly, because allauth treats that table as the source of
    truth for verification and login. Writing only `user.email` produces an
    account that can no longer sign in — the login lookup misses, and the
    "wrong password" it reports sends you hunting in entirely the wrong place.
    """
    from allauth.account.models import EmailAddress

    user = request.user
    p = _require_profile(user)

    retry = p.email_retry_after()
    if retry:
        return Response(
            {"error": "You can change your email again later.",
             "retry_after": retry},
            status=status.HTTP_429_TOO_MANY_REQUESTS)

    email = str((request.data or {}).get("email", "")).strip().lower()
    if not _looks_like_email(email):
        return Response({"error": "That does not look like an email address."},
                        status=status.HTTP_400_BAD_REQUEST)
    if email == (user.email or "").lower():
        return Response({"error": "That's already your address."},
                        status=status.HTTP_400_BAD_REQUEST)
    if EmailAddress.objects.filter(email__iexact=email).exclude(user=user).exists():
        # Deliberately the same wording a malformed address gets. "Already in
        # use" would confirm which addresses have accounts here, which is the
        # enumeration leak the reset form is careful to avoid.
        return Response({"error": "That address can't be used."},
                        status=status.HTTP_400_BAD_REQUEST)

    now = timezone.now()
    with transaction.atomic():
        # Unverified and NOT primary yet: the new address only takes over once
        # its owner proves they can read it. Swapping first would let a typo —
        # or someone on a borrowed session — move the account to an inbox its
        # owner cannot reach, taking password reset with it.
        addr, _ = EmailAddress.objects.get_or_create(
            user=user, email=email,
            defaults={"verified": False, "primary": False})
        p.email_changed_at = now
        p.save(update_fields=["email_changed_at"])

    # `EmailAddress.send_confirmation` is the allauth 65 API; the older
    # module-level `send_email_confirmation` helper no longer exists.
    addr.send_confirmation(request, signup=False)

    return Response({
        "pending_email": email,
        "message": "Check that inbox — the change applies once you confirm it.",
        "retry_after": p.email_retry_after(),
    })


def _looks_like_email(email: str) -> bool:
    if not (5 <= len(email) <= 254) or " " in email or email.count("@") != 1:
        return False
    local, _, domain = email.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") \
        and not domain.endswith(".")


@api_view(["GET"])
@permission_classes([AllowAny])
def public_profile(request, username: str):
    """Anyone's profile, by username. No auth required — every profile is
    public, which is the decision recorded in SETTINGS_AUDIT.md §1."""
    user = User.objects.filter(username__iexact=username, is_active=True).first()
    if user is None:
        return Response({"error": "No such player."},
                        status=status.HTTP_404_NOT_FOUND)

    p = _require_profile(user)
    data = public_profile_json(user, rank=_rank_of(p))
    data["solves"] = [
        _history_json(s) for s in
        Solve.objects.filter(user=user)
        .select_related("opponent")[:50]
    ]
    return Response(data)


def _history_json(s: Solve) -> dict:
    return {
        "id": s.id,
        "seconds": round(s.derived_seconds, 2),
        "result": s.result,
        "opponent": s.opponent.username if s.opponent_id else None,
        "verdict": s.verdict,
        "rating_delta": s.rating_delta,
        "received_at": s.received_at.isoformat(),
    }


@api_view(["GET"])
@permission_classes([AllowAny])
def leaderboard(request):
    """The ranked table, with every row carrying the username needed to link
    through to that player's profile."""
    try:
        limit = min(int(request.query_params.get("limit", 25)), LEADERBOARD_MAX)
    except (TypeError, ValueError):
        limit = 25

    # One query for the page of profiles, one for everyone's solve
    # aggregates, then joined in memory — rather than `solve_stats` per row.
    profiles = list(
        Profile.objects.select_related("user")
        .filter(user__is_active=True)
        .order_by("-rating", "user_id")[:limit]
    )
    user_ids = [p.user_id for p in profiles]
    agg = {
        row["user"]: row for row in
        Solve.objects.filter(user_id__in=user_ids, verdict=Solve.VERIFIED)
        .values("user").annotate(best=Min("derived_seconds"), total=Count("id"))
    }

    rows = []
    for i, p in enumerate(profiles, start=1):
        a = agg.get(p.user_id, {})
        rows.append(public_profile_json(
            p.user,
            stats={
                "best_seconds": round(a["best"], 2) if a.get("best") is not None else None,
                "total_solves": a.get("total", 0),
            },
            rank=i,
        ))
    return Response({"results": rows})
