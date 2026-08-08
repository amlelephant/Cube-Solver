# Settings audit — what works, what doesn't

Written 2026-08-05. Every control on `/settings`, plus the surrounding
account/profile surface, and what each one actually does today.

The short version: **one control out of ten was real.** Theme worked. The
other nine were `useState` — they moved when clicked, kept their position
until you navigated away, and were connected to nothing. Three of the four
sections carried a "Coming soon" tag, which was honest; the Privacy section's
toggles read as functioning privacy controls, which was not.

---

## 1. The audit

### Appearance — REAL

| Control | State | Notes |
|---|---|---|
| Theme (light/dark) | **Works** | `lib/theme.tsx`, persisted to `localStorage` under `cubearena-theme`, with a no-flash inline script in `app/layout.tsx`. Per-device, not per-account — which is defensible, but means it does not follow you to another machine. |
| Preview | **Works** | Reflects the above. Display only. |

### Account — WAS 100% NON-FUNCTIONAL

| Control | Was | Now |
|---|---|---|
| Username | `<input disabled value="Aiden">` — hardcoded string, not your account | **Wired**, once per 7 days |
| Email | `<input disabled>` with a placeholder and no value | **Wired**, once per 24h, re-verification required |
| Password | `<button disabled>` | **Wired** via allauth `/account/password/change` |
| User ID | did not exist | **Added** — shows `#7` etc. |

"Aiden" was a literal in the JSX. Signed in as anyone, it said Aiden.

### Notifications — NOT FUNCTIONAL, and nothing to connect to

| Control | State |
|---|---|
| Match invites | `useState(true)`. No backing column, no notification system, no live-match feature to notify about. |
| Weekly recap email | `useState(false)`. No scheduler, no job runner. `manage.py send_waitlist` is the only mail path and it is for the waitlist, not accounts. |
| New personal best alerts | `useState(true)`. No PB tracking existed. |

**Left as "Coming soon" deliberately.** These are not one-line wirings — each
needs a delivery mechanism that does not exist. Persisting the toggle without
the mechanism is worse than not having it: it tells someone they will be
emailed, and then they are not. The toggles now persist to the database so the
preference survives, but the section still says clearly that nothing sends
yet.

### Privacy — REMOVED / REPLACED

| Control | Was | Now |
|---|---|---|
| Public profile | `useState(true)`. Nothing read it. There was no profile route to gate. | **Removed.** All profiles are public — your instruction, and it matches the product: a competitive ladder with hidden competitors is not a ladder. |
| Show country flag | `useState(true)`. Boolean, and no country was ever stored. | **Replaced with a country picker.** Free choice from the full ISO 3166-1 list — you rep whatever country you want, and it is never inferred from IP or locale. |

---

## 2. Everything beyond settings that was also fake

Worth stating, because these are what the settings controls were supposed to
be editing.

- **`lib/mockData.ts`** was the entire data layer of the signed-in app. The
  leaderboard's ten players, your rank (#47), rating (1240), the nine-row
  solve history, the Elo chart, the six-month heat map, and every per-solve
  breakdown were generated in the browser from seeded PRNGs.
- **Names were not clickable anywhere.** The leaderboard listed ten players
  and none of them went anywhere, because there was no profile route to go to
  and no account behind the name.
- **`/profile` was your own only** and hardcoded to "Aiden", "US", rank #47.
- **Solves had no home.** The `Solve` model existed server-side with a real
  `user` FK and full anticheat evidence — and the app never once read it. The
  history table came from the array above. `Solve.objects.count()` was 0.
- **`isFounder` and `bestRank` were client-side literals**, and the code
  comment in `lib/cosmetics.tsx` already flagged this: a founder's wreath was
  mintable by anyone with devtools.

---

## 3. Profile pictures — the architecture

Not built yet, deliberately, but the shape is decided and the model has the
columns.

**Storage.** Never serve a user upload from the app origin as-is. The pipeline
is: accept → **quarantine** (private bucket, not publicly reachable) →
validate → **re-encode** → publish to a separate origin/CDN.

Re-encoding is the load-bearing step and is not about file size:

- It kills **polyglot files** — a valid GIF that is also valid HTML/JS, which
  a browser may sniff and execute. Serving that from your own origin is stored
  XSS.
- It strips **EXIF**, which routinely carries GPS coordinates. A speedcuber
  uploading a phone photo should not be publishing their home address.
- It normalises the format, so `image/svg+xml` (which is a script container)
  never reaches a viewer.

Plus: `Content-Type` from the decoder, never from the client; a hard byte
ceiling checked before decode; decode-bomb limits (a 200MB PNG that expands to
40GB is a trivial DoS); and serve from a cookieless domain so a hosted file
cannot ride the session.

**Moderation.** See below — images are the harder half.

---

## 4. Name and image moderation — how this is actually done

You said you did not know how large platforms handle this. Here is what they
do, and what applies at your size.

### Names: normalise first, match second

The single most common mistake is running a blocklist against the raw string.
It does not work, because the attacker controls the encoding:

- `Ν1ggеr` — Greek capital Nu, digit one, Cyrillic `е`. Zero blocklist hits.
- `a‌d‌m‌i‌n` — with zero-width non-joiners between every letter.
- `𝗮𝗱𝗺𝗶𝗻` — mathematical bold, a different codepoint per character.

The industry answer is **Unicode TR39 (Security Mechanisms)** and its
*skeleton* algorithm: NFD-normalise, case-fold, map every character through
`confusables.txt` (~6,500 entries) to a representative, re-normalise. Two
strings that look alike produce the same skeleton. You then match the
**skeleton**, never the input. Rust uses exactly this to warn on confusable
identifiers; ICU ships it as `SpoofChecker`.

Layered on top:

1. **Strip invisibles** — zero-width, bidi controls (`U+202E` reverses
   display order and is a classic spoof), variation selectors.
2. **Mixed-script detection** — a name that is Latin except for one Cyrillic
   character is almost never innocent. Whole-script confusables (an *entirely*
   Cyrillic name that reads as Latin) need the skeleton check to catch.
3. **Leet folding** — `4→a`, `3→e`, `0→o`, `$→s`, and collapse runs (`fuuuck`).
4. **The Scunthorpe problem.** Substring matching breaks real words — the
   town of Scunthorpe famously could not register with AOL, and "Penistone",
   "Lightwater", "assassin", "class" all trip naive filters. The fix is
   **word-boundary matching plus an allow-list** of legitimate words that
   contain a blocked substring, with only a small, deliberate set of terms
   (slurs) matched as substrings anywhere.
5. **Impersonation** — reserve `admin`, `staff`, `support`, `official`,
   `mod`, `cubearena`, and check new names for **skeleton collision with
   existing accounts**. This is why the rate limit matters: without it,
   someone cycles names until one lands.
6. **Tiers, not a binary.** Block outright / allow-but-queue-for-review /
   allow. A pure blocklist either over-blocks real people or under-blocks
   determined ones.
7. **Keep the history.** Store previous names with timestamps. Someone who
   changes name right after a report is a signal, and you cannot investigate
   what you did not record.

Implemented in `server/core/moderation.py` — skeleton folding, leet folding,
invisible stripping, word-boundary matching with an allow-list, reserved
names, and skeleton-collision against existing accounts. The blocklist there
is deliberately small and slur-focused; it is a starting point, not a finished
policy.

### Images: hashes, then classifiers, then humans

Three layers, in cost order:

1. **Perceptual hash matching against known-bad.** This is what catches the
   overwhelming majority of real CSAM reports at large platforms. Microsoft
   **PhotoDNA** is the standard and is free to qualifying platforms; NCMEC
   distributes the hash lists. Cheap, deterministic, near-zero false
   positives, and catches re-uploads of previously-actioned content.
2. **ML classifiers** for nudity/violence/hate symbols on everything else —
   AWS Rekognition Content Moderation, Google Cloud Vision SafeSearch, Hive,
   or Sightengine. Fractions of a cent per image at your volume.
3. **Human review queue** for the middle band. Automation decides the
   confident cases; a person decides the rest. The `Profile.avatar_status`
   column (`pending` / `approved` / `rejected`) plus a Django admin queue is
   this layer's skeleton.

**The legal part, which is not optional.** If you operate in the US, once you
obtain **actual knowledge** of apparent CSAM on your platform, 18 U.S.C.
§ 2258A requires you to report it to the NCMEC CyberTipline "as soon as
reasonably possible." You must then **preserve** the material and associated
records for 90 days (extendable to 180 on law-enforcement request). Knowing
and willful failure to report carries fines up to $150,000 for a first
offense and $300,000 thereafter.

Two things follow that are easy to get wrong:

- The law does **not** impose a general duty to proactively scan. It attaches
  to knowledge. So "we don't scan, therefore we never know" is not a legal
  violation — but it is a terrible product position, and the moment a user
  reports something you have knowledge and the clock starts.
- **Do not delete it.** The instinct on discovering CSAM is to purge it
  immediately. That destroys evidence you are legally required to preserve.
  Take it out of public reach, preserve the record, report, and follow
  NCMEC's instructions.

**Recommendation for launch:** ship with no user-uploaded images at all. The
generated `Avatar` plus the earned wreaths already carry the identity, and
avatars are not what makes a verification product worth using. Add uploads
when there is a reason to, and when there is someone to staff a review queue —
the reporting obligation above is real and it does not scale down.

---

## Sources

- [18 U.S.C. § 2258A — Reporting requirements of providers](https://uscode.house.gov/view.xhtml?req=granuleid%3AUSC-prelim-title18-section2258A&num=0&edition=prelim)
- [NCMEC — Reporting by online platforms](https://www.globalchildexploitationpolicy.org/content/gpp-ncmec/us/en/policy-advocacy/reporting-by-online-platforms.html)
- [Unicode confusables and the skeleton algorithm](https://unicodefyi.com/guide/unicode-confusables-guide/)
- [Confusable detection: skeletons and mixed-script checks](https://www.namesilo.com/blog/en/brand-protection/confusable-detection-101-unicode-skeletons-and-mixed-script-checks)
- [Building a profanity filter that handles Scunthorpe](https://doogal.dev/how-to-build-a-profanity-filter-that-actually-works)
