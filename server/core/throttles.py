"""
Rate limits.

`ScopedRateThrottle` reads its scope off the *view*, which function-based
views wrapped by `@api_view` cannot conveniently set — so each scope gets a
tiny `UserRateThrottle` subclass with the scope baked in instead. That base
class already does the right keying: authenticated callers are limited per
user, anonymous ones per IP. Rates live in settings.DEFAULT_THROTTLE_RATES.

These are anti-abuse, not anti-cheat. A rate limit slows a scripted attacker
down; it never decides whether a solve is real.
"""

from rest_framework.throttling import UserRateThrottle


class WaitlistThrottle(UserRateThrottle):
    """The one unauthenticated write endpoint on a public page, so the first
    thing that gets flooded."""
    scope = "waitlist"


class ScrambleThrottle(UserRateThrottle):
    """Bounds scramble farming — stockpiling issued scrambles to shop for an
    easy one. The single-use rule is the real defence; this just makes the
    search slow."""
    scope = "scramble"


class SolveThrottle(UserRateThrottle):
    scope = "solve"
