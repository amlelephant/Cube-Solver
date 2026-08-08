"""
Username / display-name moderation.

THE ORDER MATTERS AND IT IS THE WHOLE POINT: normalise first, match second.
Running a blocklist against the raw string does not work, because the attacker
picks the encoding. All three of these defeat a naive filter completely:

    Ν1ggеr      Greek capital Nu, digit one, Cyrillic e
    a<ZWNJ>dmin zero-width non-joiners between every letter
    𝗮𝗱𝗺𝗶𝗻       mathematical bold, a different codepoint per character

So every check below runs against a *skeleton* — an aggressively folded form
where anything that looks alike compares equal. This is the approach Unicode
standardises in TR39 (Security Mechanisms); `str.casefold()` plus NFKD gets
most of the way there without pulling in ICU, and the explicit CONFUSABLES map
covers the Cyrillic/Greek lookalikes that NFKD deliberately leaves alone
(they are distinct letters, not compatibility variants).

If this ever needs to be airtight, the upgrade is PyICU's `SpoofChecker`,
which ships the full ~6,500-entry confusables table. That is a deliberate
non-dependency for now: it needs a C build toolchain, which on Windows is the
same fight `kociemba` lost (see CLAUDE.md).

See SETTINGS_AUDIT.md §4 for why each layer exists.
"""

from __future__ import annotations

import re
import unicodedata

#: Characters that are visually confusable with ASCII but survive NFKD,
#: because they are genuinely different letters rather than compatibility
#: forms. Cyrillic and Greek are the ones that matter in practice.
CONFUSABLES = {
    "а": "a", "ᴀ": "a", "α": "a", "@": "a",
    "ь": "b", "в": "b", "β": "b", "ᛒ": "b",
    "с": "c", "ϲ": "c", "ς": "c", "¢": "c",
    "ԁ": "d", "ԃ": "d",
    "е": "e", "ё": "e", "є": "e", "ε": "e", "℮": "e",
    "ғ": "f", "ϝ": "f",
    "ɡ": "g", "ց": "g",
    "һ": "h", "н": "h", "ħ": "h",
    "і": "i", "ı": "i", "ї": "i", "ι": "i", "|": "i", "!": "i",
    "ј": "j", "ʝ": "j",
    "к": "k", "κ": "k",
    "ӏ": "l", "ł": "l", "ľ": "l",
    "м": "m", "μ": "m",
    "п": "n", "ɴ": "n", "η": "n", "ν": "n",
    "о": "o", "ο": "o", "σ": "o", "ø": "o", "ө": "o", "°": "o",
    "р": "p", "ρ": "p", "þ": "p",
    "ԛ": "q",
    "г": "r", "я": "r", "ʀ": "r",
    "ѕ": "s", "ș": "s", "$": "s", "§": "s",
    "т": "t", "τ": "t", "†": "t",
    "и": "u", "ц": "u", "υ": "u", "µ": "u",
    "ѵ": "v", "ν": "v",
    "ш": "w", "ω": "w", "щ": "w",
    "х": "x", "χ": "x", "×": "x",
    "у": "y", "ү": "y", "γ": "y",
    "з": "z", "ʐ": "z",
}

#: Leetspeak, applied after confusable folding.
#:
#: `1` IS AMBIGUOUS and that ambiguity is load-bearing: it stands in for `l`
#: in `he1lo` and for `i` in `4dm1n` / `n1gg3r`. A single table has to pick
#: one, and whichever it picks, the other spelling walks straight through —
#: which is exactly the bug the tests caught. So there are two tables and
#: `skeletons()` checks both readings.
LEET_BASE = {"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g"}
LEET_L = {**LEET_BASE, "1": "l"}
LEET_I = {**LEET_BASE, "1": "i", "!": "i"}

#: Zero-width and bidi controls. U+202E (RIGHT-TO-LEFT OVERRIDE) reverses
#: display order and is a classic spoofing primitive, so these are stripped
#: outright rather than folded.
INVISIBLE = re.compile(
    r"[­᠎​-‏‪-‮⁠-⁤⁪-⁯﻿ﾠ]"
)

#: Matched anywhere, including inside another word. Reserved for terms with
#: no innocent embedding — keep this list SHORT, because every entry here is
#: a Scunthorpe waiting to happen.
SUBSTRING_BLOCKED = frozenset({
    "nigger", "nigga", "faggot", "kike", "chink", "spic", "tranny",
    "raped", "rapist", "pedo", "pedophile", "childporn",
    "hitler", "nazi", "kkk",
})

#: Matched only as a whole word (or the whole name). These have legitimate
#: embeddings — "class" contains one, "assassin" and "bass" contain another,
#: "Scunthorpe" and "Penistone" are real places.
WORD_BLOCKED = frozenset({
    "ass", "cunt", "fuck", "shit", "bitch", "whore", "slut", "dick",
    "cock", "penis", "vagina", "porn", "sex", "anal", "cum", "wank",
    # Short enough that substring matching would be a disaster ("cpu",
    # "recap"), so it is whole-word only.
    "cp",
})

#: Never available to anyone, at any casing or spelling. Impersonating staff
#: is the highest-value name attack on any platform with a support channel.
RESERVED = frozenset({
    "admin", "administrator", "root", "system", "staff", "support",
    "moderator", "mod", "official", "help", "security", "billing",
    "cubearena", "cube arena", "team", "owner", "founder", "everyone",
    "here", "null", "undefined", "anonymous", "deleted", "me", "you",
})

#: Real words that CONTAIN something in SUBSTRING_BLOCKED and must survive.
#: This is the Scunthorpe allow-list. Checked against the skeleton.
ALLOW = frozenset({
    "scunthorpe", "penistone", "lightwater", "cockburn", "cockfosters",
    "assassin", "assassins", "class", "classic", "bass", "brass", "grass",
    "pass", "compass", "mass", "massive", "glass", "cassette", "analysis",
    "analyst", "analytics", "canal", "banal", "cumulative", "cumberland",
    "circumstance", "document", "accumulate", "shiitake", "titan", "titans",
    "dickens", "dickinson", "essex", "sussex", "middlesex", "cpu",
})

USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{1,22})[A-Za-z0-9]$")
MIN_LEN, MAX_LEN = 3, 24


class NameRejected(ValueError):
    """Raised with a message intended to be shown to the person renaming."""


def strip_invisible(s: str) -> str:
    return INVISIBLE.sub("", s)


def skeleton(name: str, *, leet: dict | None = None, collapse: bool = True) -> str:
    """Fold a name to a comparison form where lookalikes collapse together.

    NFKD first (handles fullwidth, mathematical alphanumerics, circled
    letters, accents), then drop combining marks, then case-fold, then map
    the confusables NFKD deliberately preserves, then optionally leet, then
    drop everything that is not a letter or digit and collapse runs of the
    same character.

    Collapsing runs is what makes `fuuuuck` and `fuck` compare equal. It also
    makes `aaa` and `a` equal, which is why this is a MATCHING form only and
    never anything we store as identity.
    """
    s = strip_invisible(name)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = "".join(CONFUSABLES.get(c, c) for c in s)
    if leet:
        s = "".join(leet.get(c, c) for c in s)
    s = "".join(c for c in s if c.isalnum())
    if collapse:
        # fuuuck -> fuck
        s = re.sub(r"(.)\1+", r"\1", s)
    return s


def skeletons(name: str) -> set[str]:
    """Every reading of a name a filter has to consider.

    Two axes, because an evasion only has to survive ONE reading to get in.

    **Leet**, three ways, since `1` is genuinely ambiguous:

        no leet   `he1lo` stays `he1lo`
        1 as l    `he1lo` -> `helo`
        1 as i    `4dm1n` -> `admin`, `n1gg3r` -> `niger`

    **Repeat collapsing**, both ways. Collapsed catches `fuuuck`. Uncollapsed
    has to be kept because collapsing is DESTRUCTIVE for any term whose
    repetition is the content: `kkk` collapses to `k`, and a pattern of `k`
    matches every name containing the letter — which is precisely the bug
    that blocked `feliks_zx`. Patterns are therefore built uncollapsed (see
    `_patterns`) and matched against both readings of the input.
    """
    out = set()
    for leet in (None, LEET_L, LEET_I):
        for collapse in (True, False):
            out.add(skeleton(name, leet=leet, collapse=collapse))
    return out


def _patterns(terms) -> set[str]:
    """Blocklist terms as skeletons, NEVER repeat-collapsed.

    See `skeletons` for why. The length guard is a second belt: a pattern
    short enough to appear inside ordinary words would over-block everything,
    so it is dropped rather than trusted.
    """
    out = set()
    for t in terms:
        sk = skeleton(t, collapse=False)
        if len(sk) >= 3:
            out.add(sk)
    return out


def skeletons_of_all(terms) -> set[str]:
    """Union of every reading of every term. Used for the allow-list, which
    has to match however the input happened to fold."""
    out: set[str] = set()
    for t in terms:
        out |= skeletons(t)
    return out


def _words(name: str) -> set[str]:
    """Token-ish pieces of a name, in every folding, for whole-word matching.

    Splits on separators AND on camelCase, so `BigAssRock` yields `ass`
    rather than hiding it inside one long token.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", strip_invisible(name))
    parts = re.split(r"[^A-Za-z0-9]+", spaced)
    out: set[str] = set()
    for p in parts:
        if p:
            out |= skeletons(p)
    return out


def check_username(name: str, *, taken_skeletons: dict[str, str] | None = None,
                   allow_id: int | None = None) -> str:
    """Validate and return the name to store, or raise `NameRejected`.

    `taken_skeletons` maps skeleton -> username for existing accounts, so a
    new name that merely LOOKS like an existing one is refused. That check is
    the reason the rate limit on renames matters: without a limit, someone
    simply cycles candidates until one slips through.
    """
    raw = strip_invisible(name).strip()

    if not raw:
        raise NameRejected("Pick a username.")
    if len(raw) < MIN_LEN:
        raise NameRejected(f"Usernames need at least {MIN_LEN} characters.")
    if len(raw) > MAX_LEN:
        raise NameRejected(f"Usernames can be at most {MAX_LEN} characters.")
    if not USERNAME_RE.match(raw):
        raise NameRejected(
            "Use letters, numbers, dot, dash or underscore — and start and "
            "end with a letter or number.")

    skels = skeletons(raw)

    if skels & _patterns(RESERVED):
        raise NameRejected("That username is reserved.")

    words = _words(raw)
    allowed = skeletons_of_all(ALLOW)

    # Whole-word matches, minus anything on the Scunthorpe allow-list.
    if (words & _patterns(WORD_BLOCKED)) and not (words & allowed):
        raise NameRejected("That username isn't available. Try another.")

    # Substring matches, checked against the skeleton so encoding tricks do
    # not help. Skipped when the whole name is an allow-listed word.
    if not (skels & allowed):
        subs = _patterns(SUBSTRING_BLOCKED)
        for sk in skels:
            if any(b in sk for b in subs):
                raise NameRejected("That username isn't available. Try another.")

    if taken_skeletons:
        for sk in skels:
            owner = taken_skeletons.get(sk)
            if owner is not None and owner != allow_id:
                raise NameRejected(
                    "That username is too close to one already in use.")

    return raw


def review_flags(name: str) -> list[str]:
    """Non-blocking signals worth a human look.

    Deliberately separate from `check_username`: a pure block/allow decision
    either over-blocks real people or under-blocks determined ones. These
    names are ACCEPTED and queued.
    """
    flags = []
    raw = strip_invisible(name)

    scripts = set()
    for ch in raw:
        if not ch.isalpha():
            continue
        try:
            block = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        scripts.add("LATIN" if block == "LATIN" else block)
    if len(scripts) > 1:
        flags.append("mixed_script")

    if raw != name:
        flags.append("invisible_characters")
    if any(ord(c) > 0x2FFF for c in raw):
        flags.append("unusual_codepoints")
    return flags
