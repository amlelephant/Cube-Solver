"""
timing.py — establishing how long a solve took without believing the client.

THE ATTACK
----------
    POST /api/solves/ {"time": 0.01, "verified": true}

Trivial, and fatal if the API takes the client's word for anything. The rule
this module enforces:

    **No number a client sends is ever a fact. Every value a verdict rests
    on is either stamped by the server's own clock or derived from evidence
    the server re-analyzes.**

Client-supplied durations are kept only as `claimed_seconds`, purely so a
disagreement with the derived value can be flagged.

THREE INDEPENDENT BOUNDS
------------------------
None of these is sufficient alone; the point is that a forgery has to
satisfy all three simultaneously, and they constrain each other.

1. **Server wall-clock window.** The server stamps `issued_at` when it hands
   out the scramble and `received_at` when the submission lands. The solve
   physically happened inside that window, so

       derived_seconds <= (received_at - issued_at) + slack

   Note what this does and does not do. It is an UPPER bound: it catches a
   claimed 8-second solve submitted 3 seconds after the scramble was issued.
   It does **nothing** against claiming *faster* than reality, which is what
   an attacker actually wants. On its own this bound is close to useless
   against the real threat, and it would be easy to mistake it for a defence.

2. **Evidence-derived duration.** `frame_count / fps` is the authoritative
   duration. It comes from footage the server can re-analyze, and the same
   frames are what the move stream is decoded from — so the duration and the
   move count cannot be forged independently of each other.

3. **Move-rate plausibility — the one that actually closes it.** The timing
   attack *collapses into the move-count attack already solved*. To claim
   0.01s you must supply a bundle with almost no frames, and then:

       * few frames, few moves  -> below the count floor (32), rejected as
         substitution or solver-following;
       * few frames, many moves -> absurd implied TPS, rejected as
         physically impossible.

   There is no gap between those. A fast time is only believable when it
   comes with enough moves at a humanly achievable rate, and the count gate
   already owns that judgement.

WHAT THIS DOES NOT CLOSE
------------------------
A client that fabricates the frames themselves — renders a synthetic solve,
or feeds a virtual camera — satisfies every bound here, because every bound
here reads the evidence rather than authenticating its origin. That is camera
injection (ANTICHEAT.md §4) and it needs challenge-response: the server
demanding a specific face at a random instant that no pre-recording can
anticipate. Nothing in this file substitutes for that.

Also, until the re-verification worker exists, `frame_count` is *reported* by
the client rather than counted by the server. Bound 2 is therefore currently
an integrity check on a self-report, not a measurement. It becomes a real
measurement the moment the worker lands; the code shape does not change.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the anticheat gate rather than restating its constants. The floor,
# the abstain band and the post-stop allowance were derived and measured in
# ble/move_detector/anticheat_gate.py, and a second copy of those numbers
# here would drift from them silently — which for a threshold that decides
# whether someone is called a cheat is the worst possible failure.
_GATE = Path(__file__).resolve().parents[2] / "ble" / "move_detector"
if str(_GATE) not in sys.path:
    sys.path.insert(0, str(_GATE))

from anticheat_gate import (  # noqa: E402
    MIN_OBSERVED_MOVES, SOLVER_CEILING_QTM, SolveEvidence, adjudicate,
    separation_tps_limit,
)

#: Slack on bound 1. Covers network latency, the user reading the scramble,
#: inspection, and clock jitter between the two server stamps. Generous on
#: purpose — this bound is not where the security comes from, and a tight
#: value here would reject honest users on a slow connection for no gain.
WALL_CLOCK_SLACK_S = 120.0

#: Hard ceiling on implied turns per second. The world record is ~9-11 TPS;
#: anything past this is not a fast human, it is a bad frame count. Set well
#: above the record because this is a physical-impossibility check, not a
#: skill judgement — the "too fast to read" abstain band inside the gate
#: (~7.1 TPS) is what handles genuinely elite solves, and it abstains rather
#: than rejecting.
MAX_HUMAN_TPS = 20.0

#: A solve cannot be shorter than this. The official 3x3 single record is
#: ~3.1s; below a second is not a solve under any interpretation.
MIN_PLAUSIBLE_SECONDS = 1.0

#: How far the client's claimed time may differ from the derived time before
#: it is treated as a lie rather than rounding. Timer-start latency and frame
#: boundaries account for a few hundred ms honestly.
CLAIM_TOLERANCE_S = 1.0

#: Sanity bounds on the declared frame rate. A client claiming 1000 fps is
#: manufacturing a short duration out of a long recording.
FPS_RANGE = (5.0, 480.0)


@dataclass
class TimingFacts:
    """Everything needed to establish a duration, split by who it came from.

    The split is the point: `server_*` fields are stamped by us, `claimed_*`
    are the client's assertions, and `frame_count`/`fps` are evidence
    properties that a re-verification worker will eventually measure rather
    than accept.
    """

    server_window_seconds: float          # received_at - issued_at
    frame_count: int
    fps: float
    observed_moves: int
    claimed_seconds: float | None = None
    observed_moves_after_stop: int = 0
    post_stop_seconds: float | None = None
    lighting_ok: bool | None = None
    continuity: dict | None = None
    scramble_consumed: bool = False
    scramble_expired: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def derived_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def evaluate(f: TimingFacts) -> dict:
    """TimingFacts -> verdict dict.

    Pure function over plain data, deliberately: the browser can run the same
    logic for instant feedback while the server runs it for the verdict, and
    the two agree by construction rather than by two implementations
    happening to match. Client-side is UX; only this call site counts.
    """
    reject: list[str] = []
    review: list[str] = []
    derived = f.derived_seconds

    # -- replay and staleness. Checked first: a reused or expired scramble
    #    makes every timestamp below meaningless, because the footage may be
    #    from an entirely different sitting.
    if f.scramble_consumed:
        reject.append("scramble_already_used")
    if f.scramble_expired:
        reject.append("scramble_expired")

    # -- evidence sanity. These are malformed-bundle checks, not accusations
    #    of cheating, but they must reject rather than abstain: an unusable
    #    bundle cannot be verified, and letting it through as "review" would
    #    fill the review queue with garbage.
    if not (FPS_RANGE[0] <= f.fps <= FPS_RANGE[1]):
        reject.append(f"implausible_fps:{f.fps:g}")
    if f.frame_count <= 0:
        reject.append("no_frames")

    # -- bound 1: the server's own clock.
    if derived > f.server_window_seconds + WALL_CLOCK_SLACK_S:
        # Claiming to have solved for longer than the scramble has existed.
        reject.append(
            f"duration_exceeds_server_window:{derived:.2f}s>"
            f"{f.server_window_seconds:.2f}s+{WALL_CLOCK_SLACK_S:g}s")

    # -- bound 2: the client's claim vs the evidence. Never authoritative,
    #    always interesting.
    if f.claimed_seconds is not None:
        gap = abs(derived - f.claimed_seconds)
        if gap > CLAIM_TOLERANCE_S:
            reject.append(
                f"claimed_time_contradicts_evidence:claimed "
                f"{f.claimed_seconds:.2f}s vs derived {derived:.2f}s")

    # -- bound 3: physical plausibility of the derived duration itself.
    if 0 < derived < MIN_PLAUSIBLE_SECONDS:
        reject.append(f"duration_below_human_floor:{derived:.2f}s")
    tps = (f.observed_moves / derived) if derived > 0 else None
    if tps is not None and tps > MAX_HUMAN_TPS:
        reject.append(f"implied_tps_impossible:{tps:.1f}>{MAX_HUMAN_TPS:g}")

    # -- and then the move-count gate, which is where a short claimed time
    #    actually gets caught: too few moves to be a solve at all.
    gate = adjudicate(SolveEvidence(
        session=f"solve",
        solve_seconds=derived or None,
        observed_moves=f.observed_moves,
        observed_moves_after_stop=f.observed_moves_after_stop,
        post_stop_seconds=f.post_stop_seconds,
        continuity=f.continuity,
        lighting_ok=f.lighting_ok,
    ))
    reject += gate["reject_reasons"]
    review += gate["review_reasons"]

    verdict = "rejected" if reject else ("review" if review else "verified")
    return {
        "verdict": verdict,
        "reject_reasons": reject,
        "review_reasons": review,
        "derived_seconds": round(derived, 3),
        "claimed_seconds": f.claimed_seconds,
        "claim_discrepancy": (None if f.claimed_seconds is None
                              else round(derived - f.claimed_seconds, 3)),
        "server_window_seconds": round(f.server_window_seconds, 3),
        "tps": round(tps, 2) if tps is not None else None,
        "move_floor": MIN_OBSERVED_MOVES,
        "solver_ceiling_qtm": SOLVER_CEILING_QTM,
        "abstain_above_tps": separation_tps_limit(),
        "gate": gate,
    }
