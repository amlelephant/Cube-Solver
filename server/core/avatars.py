"""
The preset avatar set.

WHY PRESETS AND NOT UPLOADS. An uploaded avatar is user-generated imagery on
a public profile, which needs moderation — review queue, reporting, takedown,
and someone to run it. `Profile.avatar_status`/`avatar_path` exist for that
day (SETTINGS_AUDIT.md §3 recommends launching without it) and are untouched
here. A preset carries none of that risk: the server only ever stores a key
from the list below, and the artwork is drawn client-side, so there is no
image to moderate, no file to serve and no upload endpoint to abuse.

KEEP IN SYNC with `web/lib/avatars.tsx`, which holds the same keys and the
drawings for them. This module is the gate — a key absent here is rejected
even if the client offers it — but a key present here with no drawing on the
client renders as nothing, so the two lists move together.
"""

#: Every avatar a profile may hold. Ordered as the picker shows them.
#: Names are cosmetic-neutral on purpose: they are cube patterns, not
#: characters, so none of them can read as impersonating a real person.
AVATAR_PRESETS: tuple[str, ...] = (
    "cube-classic",
    "cube-checker",
    "cube-cross",
    "cube-stripe",
    "cube-corners",
    "cube-spiral",
    "cube-sunset",
    "cube-ocean",
    "cube-forest",
    "cube-mono",
    "cube-neon",
    "cube-ember",
)

#: What an account with no choice made shows. Empty rather than a default
#: pattern, so "never picked one" and "picked the first one" stay distinct.
AVATAR_NONE = ""


def is_valid_avatar(key: str) -> bool:
    """True for a key this server will store. `""` clears the choice."""
    return key == AVATAR_NONE or key in AVATAR_PRESETS
