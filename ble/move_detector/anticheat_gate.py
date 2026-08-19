"""
anticheat_gate.py — the single place a VERIFIED verdict is decided.

ANTICHEAT.md issue 1. `count_check.py` measured that the move model counts
well; `speed_sim.py` measured where counting breaks down as solves get
faster. This file turns those two measurements into the actual gate, and
adds the boundary that neither of them tested: a FLOOR ABOVE GOD'S NUMBER.

Why a floor above God's number
------------------------------
Counting alone defeats substitution (a swapped-in solved cube needs ~0
moves). It does not, by itself, defeat the cleverer attack: run the scramble
through a solver, read off the solution, and execute it on camera. Every
move genuinely happens, the cube genuinely goes from scrambled to solved,
and the continuity guard sees one cube throughout. Nothing about that
session is fake except the thinking.

But it is bounded. God's number is the maximum, over all 43 quintillion
states, of the optimal solution length — so ANY solver-derived solution is
at most that long, whatever scramble it was given. A human method is not
bounded that way and is not close to it: CFOP, Roux and ZZ all spend moves
on structure a solver never needs. So "did this solve take more moves than
any solver would have needed" separates the two, and it separates them by a
margin rather than at a boundary.

  THE METRIC MATTERS AND IS EASY TO GET WRONG. God's number is 20 in the
  HALF-turn metric (HTM), where R2 counts as one move. This model's
  alphabet is 12 classes — 6 faces x 2 directions (dataset.WCA_FLIP_PERM) —
  so R2 decodes as R,R and everything here is in the QUARTER-turn metric.
  God's number in QTM is 26, not 20. A floor set at 20-something QTM would
  sit BELOW the cheat ceiling and catch nothing. If the R2-as-its-own-class
  change ever lands (see the speed-ceiling work), this constant must move
  back to 20 and MIN_OBSERVED_MOVES must be recalibrated with it.

Calibrating the floor against what we actually observe
------------------------------------------------------
The gate sees PREDICTED moves, not true ones, and the model under-counts —
more so the faster the solve. Under-counting cuts both ways, and only one
way is dangerous:

  * a cheat's count is pushed DOWN, further below the floor — harmless;
  * a legit solve's count is pushed DOWN, toward the floor — this is the
    false-DQ risk, and it is what the floor must be calibrated against.

Measured worst-case retention (predicted/true, WORST session, not mean —
a no-false-DQ gate is set by its worst case), from
speed_sim_move_ctc_aug44_s0_blur.json:

    <=2 TPS  0.98   |  5 TPS  0.84   |   8 TPS  0.63
      3 TPS  0.91   |  6 TPS  0.81   |  10 TPS  0.50
      4 TPS  0.90   |

Take 45 QTM as the legit floor: that is a very efficient human solve (elite
Roux is ~45-48 HTM, which is MORE in QTM, so this is conservative in the
safe direction). Then the lowest count a legit solve can present as:

    at 6 TPS   45 x 0.81 = 36 observed
    at 8 TPS   45 x 0.63 = 28 observed
    at 10 TPS  45 x 0.50 = 22 observed   <-- below the 26 cheat ceiling

So MIN_OBSERVED_MOVES = 30 sits above every solver-derived solve (<=26 true,
and under-counting only lowers that) and below every legit solve up to about
8 TPS. Above ~8 TPS the two bands genuinely overlap and no threshold
separates them — the gate ABSTAINS there rather than guessing (see
SEPARATION_TPS_LIMIT). That band is roughly sub-8-second solves: the top
sliver of the leaderboard, which is exactly the slice that gets human review
anyway (LAUNCH_ROADMAP C3).

Three-tier, unlike the continuity guard
---------------------------------------
`cv/detection/continuity_guard.py` is deliberately two-tier (pass/dq): every
signal it has is swap-indicative, so there is nothing to be unsure about.
This gate is three-tier (VERIFIED / REJECTED / REVIEW) because it has two
conditions that make the evidence UNREADABLE rather than incriminating:

  * the solve is too fast for counting to separate the bands;
  * the lighting is bad, which collapses the onset detector (measured
    2026-08-05: the detector loses 40+ points in low light while the
    classifier loses ~5 — see TODO.md item 7).

Both would otherwise manufacture a false DQ out of a model failure. An
honest system says "I cannot read this", not "you cheated". REJECTED is
reserved for evidence that is legible AND incriminating.

WHAT THIS FILE DOES NOT ESTABLISH
---------------------------------
The count floor is calibrated entirely on LEGIT sessions and on a frame-drop
simulation of speed. That bounds FALSE POSITIVES. It does not show that a
real solver-following attempt or a real fast solve lands where the
simulation says — `speed_sim.py`'s own caveat (dropped frames are sharper
than genuinely fast footage) applies to every number above. Record the
attacks (cv/labeling/record_attack.py --type solver_follow) and real fast
solves before treating these thresholds as validated rather than derived.

Run from inside ble/move_detector/:
    python anticheat_gate.py explain
    python anticheat_gate.py score --ctc checkpoints/move_ctc_aug44_s0.pt
    python anticheat_gate.py calibrate-poststop --ctc checkpoints/move_ctc_aug44_s0.pt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SESSION_ROOT = Path("../training_data")

# --- the move-count boundary ---------------------------------------------

GODS_NUMBER_QTM = 26      # max optimal solution length, quarter-turn metric.
                          #   The ceiling on an OPTIMAL solver's output. See
                          #   the metric warning in the module docstring.
SOLVER_CEILING_QTM = 30   # the ceiling that actually matters. A cheat will
                          #   not run an optimal solver — those are slow and
                          #   nobody bothers. They will run two-phase
                          #   (Kociemba), which returns ~19-22 HTM; with the
                          #   ~25-30% of its moves that are half turns
                          #   costing two each, that is ~25-30 QTM. So the
                          #   realistic attack reaches HIGHER than God's
                          #   number, and a floor set at God's number + a
                          #   couple would be inside the attack's range.
                          #   Corroborated by the recorded scramble sessions,
                          #   which are the same object (a machine-generated
                          #   ~20-HTM sequence executed by hand) and measure
                          #   20-28 QTM true, median 22.
MIN_OBSERVED_MOVES = 32   # floor on the DECODED count over the timed window.
                          #   Above SOLVER_CEILING_QTM, and under-counting
                          #   only pushes a cheat further below it. Not yet
                          #   validated against a recorded solver-following
                          #   attempt — the scramble sessions are a proxy,
                          #   not the real thing.
LEGIT_FLOOR_QTM = 45      # assumed true move count of the most efficient
                          #   realistic human solve (elite Roux is ~45-48
                          #   HTM, which is MORE in QTM, so this is
                          #   conservative in the safe direction). The
                          #   margin rests entirely on this; lower it and the
                          #   gate gets more false DQs, not more catches.
                          #   For scale, the lowest count on any RECORDED
                          #   legit solve is 66 observed — 34 clear of the
                          #   floor. 45 is a hypothetical, not a corpus fact.

# Retention floor by turns-per-second: predicted/true move count on the WORST
# held-out session, blur variant, and then the WORSE OF THE TWO SEEDS at each
# level. Two levels of worst-casing, both deliberate — this number decides
# where the gate stops trusting itself, and a gate that must never false-DQ
# cannot be calibrated on the luckier seed.
#
# THESE DESCRIBE THE SPEED-AUGMENTED CHECKPOINTS (move_ctc_spd_s0/s1,
# 2026-08-05). If verification ever runs a different checkpoint, re-measure
# with speed_sim.py --blur and put ITS numbers here — a gate calibrated on a
# model that is not the one running is worse than no calibration, because it
# looks calibrated.
#
# For the record, what speed augmentation bought (aug44 -> spd, worst
# session, both seeds re-run fresh on identical input):
#
#     TPS      seed 0            seed 1
#      5    0.836 -> 0.967    0.757 -> 0.875
#      6    0.809 -> 0.954    0.763 -> 0.842
#      8    0.632 -> 0.882    0.612 -> 0.796
#     10    0.500 -> 0.770    0.428 -> 0.691
#
# Every level improved on both seeds. The abstain band moves 7.11 -> 9.62 TPS
# (see separation_tps_limit), i.e. a ~4.7s 45-move solve against a ~3.1s
# world record — the "too fast to separate" case now covers only the very
# fastest humans alive rather than everyone under 6.3 seconds.
RETENTION_FLOOR = {2: 0.941, 3: 0.934, 4: 0.928, 5: 0.875,
                   6: 0.842, 8: 0.796, 10: 0.691}

# --- the post-timer window (TODO.md item 6) ------------------------------

# The obvious specification of this test — "require ZERO moves between
# timer-stop and scan-complete" — does not survive measurement, and the way
# it fails is instructive. Phantom onsets ACCUMULATE WITH WINDOW LENGTH, so
# a fixed small constant is a test whose false-positive rate depends on how
# long the user takes to present their cube to the camera. Measured on
# move-free session tails (calibrate-poststop, seed 0, held-out): mean 0.45
# phantoms per 10s, worst 3.62 per 10s. Over a 40s scan that worst rate is
# ~14 phantom moves — a constant limit of 2 would reject almost everyone.
#
# So the test is rate-based, and it is really the SAME discriminant as the
# count floor above, pointed the other way: an attacker solving the cube in
# this window must make a full solve's worth of moves (>=45 QTM true, >=36
# observed at worst retention), while an honest user makes none and accrues
# only phantoms.
POST_STOP_PHANTOM_RATE = 4.0   # allowed phantom moves per 10s. Above the
                               #   worst held-out measurement (3.62/10s) so
                               #   the worst-behaved honest session still
                               #   passes.
POST_STOP_MIN_LIMIT = 2        # floor for very short windows, where the
                               #   rate times a small number rounds to ~0
                               #   and one phantom would reject.
POST_STOP_MAX_WINDOW_S = 60.0  # A PRODUCT CONSTRAINT THAT FALLS OUT OF THE
                               #   ABOVE, not a tuning knob. The allowance
                               #   grows at 0.4 moves/s and a real solve
                               #   reads >=36, so beyond ~90s the allowance
                               #   swallows an entire hidden solve and the
                               #   test has no power left. Capping the scan
                               #   window at 60s keeps the allowance at 24
                               #   against a real solve's 36 — a 12-move
                               #   margin. If the UI ever lets the post-timer
                               #   scan run longer than this, THIS TEST
                               #   SILENTLY STOPS WORKING.
POST_STOP_CALIBRATED = False   # the tails available are 1-3s and the real
                               #   window is tens of seconds; the rate is an
                               #   extrapolation off short samples. Needs
                               #   real timer-stop-to-scan-complete footage.


def post_stop_limit(window_s: float | None) -> int:
    """Allowed decoded moves in a post-timer window of `window_s` seconds."""
    if window_s is None:
        return POST_STOP_MIN_LIMIT
    return max(POST_STOP_MIN_LIMIT,
               int(math.ceil(POST_STOP_PHANTOM_RATE * window_s / 10.0)))


@dataclass
class SolveEvidence:
    """Everything the gate is allowed to look at. Deliberately plain data:
    the verdict must be reproducible server-side from a stored bundle, so
    nothing here may be a live model, camera or file handle."""

    session: str = "unknown"
    solve_seconds: float | None = None
    observed_moves: int | None = None          # decoded, timed window only
    observed_moves_after_stop: int | None = None   # timer-stop -> scan done
    post_stop_seconds: float | None = None     # length of that window
    continuity: dict | None = None             # ContinuityGuard.report()
    swap_jump: float | None = None             # swap_check.session_jump()[0]
    swap_threshold: float | None = None        # from swap_check_baseline.npz
    lighting_ok: bool | None = None            # None = not assessed
    # solved_check.solved_at() over the frames straddling the timer stop.
    # {"solved": True|False|None, ...}; None means the window was unreadable.
    solved_at_stop: dict | None = None
    # ContinuityGuard.report() over the POST-STOP window alone — not the same
    # object as `continuity`, which covers the whole attempt. See adjudicate().
    post_stop_continuity: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def tps(self) -> float | None:
        if not self.solve_seconds or self.observed_moves is None:
            return None
        return self.observed_moves / self.solve_seconds


def separation_tps_limit() -> float:
    """Fastest solve at which counting still separates the two bands.

    DERIVED, not chosen, so that moving MIN_OBSERVED_MOVES moves the abstain
    band with it instead of silently leaving a gap where the gate rejects
    legitimate fast solves. The condition is simply that the slowest-reading
    legit solve still clears the floor:

        LEGIT_FLOOR_QTM * retention(tps) >= MIN_OBSERVED_MOVES

    Above the returned TPS that fails and the gate must abstain.
    """
    lo, hi = min(RETENTION_FLOOR), max(RETENTION_FLOOR)
    if LEGIT_FLOOR_QTM * retention_at(lo) < MIN_OBSERVED_MOVES:
        return 0.0                      # floor is too high for ANY speed
    if LEGIT_FLOOR_QTM * retention_at(hi) >= MIN_OBSERVED_MOVES:
        return float(hi)
    for _ in range(40):                 # bisect; retention is monotone in tps
        mid = (lo + hi) / 2
        if LEGIT_FLOOR_QTM * retention_at(mid) >= MIN_OBSERVED_MOVES:
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


def retention_at(tps: float) -> float:
    """Worst-case predicted/true retention at `tps`, linearly interpolated
    between the measured levels and clamped outside them."""
    ks = sorted(RETENTION_FLOOR)
    if tps <= ks[0]:
        return RETENTION_FLOOR[ks[0]]
    if tps >= ks[-1]:
        return RETENTION_FLOOR[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= tps <= b:
            f = (tps - a) / (b - a)
            return RETENTION_FLOOR[a] + f * (RETENTION_FLOOR[b] - RETENTION_FLOOR[a])
    return RETENTION_FLOOR[ks[-1]]


def adjudicate(ev: SolveEvidence) -> dict:
    """SolveEvidence -> verdict dict.

    Pure function, no I/O, so the server can re-run it on a stored bundle
    and get the same answer the client did — which is the whole point of
    re-verification.

    Precedence, and it is not arbitrary: ABSTAIN conditions are checked
    BEFORE reject conditions, because an unreadable solve must never be
    rejected on the strength of evidence we have just declared unreadable.
    The one exception is continuity/appearance evidence, which is
    independent of the move model and therefore still legible when the move
    model is not — a cube that visibly left the frame and came back
    different is incriminating regardless of how fast or how dark it was.
    """
    reasons: list[str] = []
    abstain: list[str] = []

    # -- 1. continuity + appearance. Independent of the move model, so this
    #       is evaluated first and can reject even when counting cannot.
    if ev.continuity is not None and ev.continuity.get("verdict") == "dq":
        reasons += [f"continuity:{r}" for r in ev.continuity.get("dq_reasons", [])]
    if (ev.swap_jump is not None and ev.swap_threshold is not None
            and ev.swap_jump > ev.swap_threshold):
        reasons.append(f"appearance_jump:{ev.swap_jump:.4f}>"
                       f"{ev.swap_threshold:.4f}")

    # -- 1b. the enough-moves-then-swap attack: a full solve's worth of real
    #        moves on a cube that never got solved, then a solved cube
    #        substituted before the scan. The count is genuine so §1 passes
    #        it, and appearance cannot see it because the two cubes are
    #        chosen identical. What it cannot fake is the cube ON CAMERA AT
    #        THE STOP, which is still scrambled — the substitution has not
    #        happened yet.
    #
    #        The two halves are one test and only work together:
    #          * solved-at-stop says the cube was solved when the clock
    #            stopped;
    #          * post-stop custody says it is still the SAME object when the
    #            scan validates it.
    #        Either alone leaves the swap a window to happen in. Custody is
    #        demanded only over the post-stop window, not the whole attempt,
    #        because that window is short and deliberate — the cube is being
    #        presented, not manipulated — whereas whole-session continuity at
    #        DQ strength is what drives the known false-DQ rate.
    if ev.solved_at_stop is not None:
        st = ev.solved_at_stop.get("solved")
        if st is False:
            # NOT exempt from the lighting abstention, unlike continuity and
            # appearance. Those read geometry and are independent of the move
            # model; this reads COLOUR, and its measured failure mode is
            # specifically a lighting one — in warm evening light the
            # background classifies as a solid region and the statistic
            # breaks. Letting a colour test reject in the light it is known
            # to fail in is how a false DQ ships.
            if ev.lighting_ok is False:
                abstain.append("solved_at_stop_unreadable:lighting")
            else:
                reasons.append(
                    f"not_solved_at_timer_stop:"
                    f"{ev.solved_at_stop.get('n_regions')}regions>"
                    f"{ev.solved_at_stop.get('threshold')}")
        elif st is None:
            # Occlusion at the stop is ordinary — hands are still on the cube
            # — and it pushes the statistic toward "scrambled", so an
            # unreadable window must abstain. This is also the evasion path
            # (present the cube badly on purpose), and review is the right
            # answer to it.
            abstain.append("solved_at_stop_unreadable")
    if (ev.post_stop_continuity is not None
            and ev.post_stop_continuity.get("verdict") == "dq"):
        reasons += [f"post_stop_custody:{r}"
                    for r in ev.post_stop_continuity.get("dq_reasons", [])]

    # -- 2. conditions that make the MOVE COUNT unreadable.
    if ev.observed_moves is None:
        abstain.append("no_move_count")
    if ev.lighting_ok is False:
        # The detector, not the classifier, is what low light destroys; the
        # count is exactly the detector's output. Counting under bad light
        # under-reads and would fabricate a too_few_moves DQ.
        abstain.append("lighting_unreadable")
    tps = ev.tps
    if tps is not None and tps > separation_tps_limit():
        abstain.append(f"too_fast_to_separate:{tps:.1f}tps")

    # -- 3. the count tests, only where the count is readable.
    if not abstain and ev.observed_moves is not None:
        if ev.observed_moves < MIN_OBSERVED_MOVES:
            # Which attack this is depends on how far below the floor it
            # landed, and saying so makes the verdict reviewable rather
            # than oracular.
            if ev.observed_moves <= SOLVER_CEILING_QTM // 2:
                kind = "substitution_or_partial_solve"
            else:
                kind = "solver_following"
            reasons.append(f"too_few_moves:{ev.observed_moves}<"
                           f"{MIN_OBSERVED_MOVES} ({kind})")

    if ev.observed_moves_after_stop is None:
        abstain.append("post_stop_window_not_analyzed")
    else:
        # TODO.md item 6: stop the timer on an unsolved cube, then solve it,
        # then scan. No substitution happens, so nothing above catches it.
        if (ev.post_stop_seconds is not None
                and ev.post_stop_seconds > POST_STOP_MAX_WINDOW_S):
            # The allowance would by now be wide enough to hide a whole
            # solve. Refusing to verify is the only honest answer; passing
            # would be asserting something this test can no longer see.
            abstain.append(f"post_stop_window_too_long:"
                           f"{ev.post_stop_seconds:.0f}s>"
                           f"{POST_STOP_MAX_WINDOW_S:.0f}s")
        limit = post_stop_limit(ev.post_stop_seconds)
        if ev.observed_moves_after_stop > limit:
            reasons.append(f"moves_after_timer_stop:"
                           f"{ev.observed_moves_after_stop}>{limit}")

    if reasons:
        verdict = "rejected"
    elif abstain:
        verdict = "review"
    else:
        verdict = "verified"

    out = {
        "session": ev.session,
        "verdict": verdict,
        "reject_reasons": reasons,
        "review_reasons": abstain,
        "observed_moves": ev.observed_moves,
        "floor": MIN_OBSERVED_MOVES,
        "solve_seconds": ev.solve_seconds,
        "tps": round(tps, 2) if tps is not None else None,
    }
    if tps is not None and ev.observed_moves is not None:
        r = retention_at(tps)
        # How many moves this solve had to spare before the floor, and what
        # true count the floor implies at this speed. Both are what a human
        # reviewer needs to judge a borderline case.
        out["headroom_moves"] = ev.observed_moves - MIN_OBSERVED_MOVES
        out["retention_floor_at_tps"] = round(r, 3)
        out["implied_true_floor"] = int(math.ceil(MIN_OBSERVED_MOVES / r))
    if not POST_STOP_CALIBRATED:
        out.setdefault("caveats", []).append(
            "post-stop phantom rate extrapolated off 1-3s tails; needs real "
            "timer-stop-to-scan-complete footage")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_explain(args):
    """Print the separation table the thresholds come from. This is the
    argument for the numbers, kept runnable so it cannot drift from them."""
    lim = separation_tps_limit()
    print(f"God's number (QTM, this alphabet)  : {GODS_NUMBER_QTM}")
    print(f"realistic solver ceiling (2-phase) : {SOLVER_CEILING_QTM}")
    print(f"observed-count floor               : {MIN_OBSERVED_MOVES}")
    print(f"assumed legit true floor           : {LEGIT_FLOOR_QTM} QTM\n")
    print(f"{'TPS':>5} {'retention':>10} {'legit floor reads as':>21} "
          f"{'cheat ceiling reads as':>23}  separates?")
    for tps in sorted(RETENTION_FLOOR):
        r = RETENTION_FLOOR[tps]
        legit = LEGIT_FLOOR_QTM * r
        # under-counting only lowers a cheat's count, so its ceiling as
        # observed is its true ceiling
        ok = legit >= MIN_OBSERVED_MOVES > SOLVER_CEILING_QTM
        print(f"{tps:5.1f} {r:10.3f} {legit:21.1f} {SOLVER_CEILING_QTM:23d}"
              f"  {'yes' if ok else 'NO - abstain'}")
    print(f"\nabstain above {lim:.2f} TPS "
          f"(~{LEGIT_FLOOR_QTM / lim:.0f}s for a {LEGIT_FLOOR_QTM}-move "
          f"solve - the human-review slice). DERIVED from the floor, so "
          f"changing MIN_OBSERVED_MOVES moves this with it.")
    print("\nRetention is the WORST session at each level, from "
          "speed_sim_*_blur.json, not the mean.")


def _load_model(ckpt_path):
    import torch
    from model import build_joint_from_ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_joint_from_ckpt(ck, device)
    model.eval()
    return ck, model, device


def _decode_count(model, stream, device, beam=0):
    import numpy as np
    from ctc_decode import greedy_decode, prefix_beam_decode
    from model import score_stream_joint
    _, cp, _ = score_stream_joint(model, stream, device)
    lp = np.log(np.maximum(cp, 1e-12))
    labels, _ = (prefix_beam_decode(lp, beam=beam) if beam
                 else greedy_decode(lp))
    return labels


def cmd_score(args):
    """Run the gate over the recorded corpus, which splits into two halves
    that measure the two OPPOSITE errors — and the second half is free.

      *_solve    sessions are legit solves. Every non-`verified` verdict on
                 one is a FALSE DQ (or an abstention). This is the number
                 the launch go/no-go gate asks for (<1% false DQ).
      *_scramble sessions are a machine-generated ~20-HTM sequence executed
                 by hand on camera. That is STRUCTURALLY IDENTICAL to the
                 solver-following attack — the cheat is precisely "execute a
                 machine-generated short sequence and present the result".
                 They were recorded as scrambles, not as attacks, and were
                 never designed for this, which makes them a genuinely
                 uncontaminated proxy. Every non-`rejected` verdict on one
                 is a MISS.

    The proxy is not the real attack in one respect worth stating: a
    scramble takes a solved cube to a scrambled one, and an attacker runs
    the reverse. The move COUNT — the only thing this gate reads — is
    identical in both directions, so the substitution is sound for this
    measurement and would not be for an appearance-based one.
    """
    import numpy as np
    from dataset import JointSessionStream

    ck, model, device = _load_model(args.ctc)
    trained = set(ck.get("train_session_names") or [])

    sessions = sorted(d for d in SESSION_ROOT.iterdir()
                      if (d / "detector_stream_color.npz").is_file())
    if args.limit:
        sessions = sessions[:args.limit]

    rows = []
    for i, d in enumerate(sessions):
        s = JointSessionStream(d / "detector_stream_color.npz")
        labels = _decode_count(model, s, device, args.beam)
        raw = np.load(d / "detector_stream_color.npz", allow_pickle=True)
        secs = len(raw["frames"]) / float(raw["fps"])
        ev = SolveEvidence(
            session=d.name,
            solve_seconds=secs,
            observed_moves=len(labels),
            # The corpus has no separate post-stop window and no lighting
            # assessment wired in yet; leaving them None makes every session
            # abstain, which would hide the count result. Pass them as
            # satisfied so this command measures the COUNT gate specifically.
            observed_moves_after_stop=0,
            lighting_ok=True,
        )
        v = adjudicate(ev)
        v["seen_in_training"] = d.name in trained
        v["role"] = "scramble_proxy" if d.name.endswith("_scramble") else "solve"
        rows.append(v)
        print(f"[{i+1}/{len(sessions)}] {v['verdict']:8s} {d.name:38s} "
              f"moves={v['observed_moves']:4d} {v['tps']:.2f} TPS "
              f"headroom={v.get('headroom_moves', 0):+4d} "
              f"{'(trained)' if v['seen_in_training'] else 'HELD-OUT'}")
        if v["reject_reasons"]:
            print(f"          reject: {v['reject_reasons']}")
        if v["review_reasons"]:
            print(f"          review: {v['review_reasons']}")

    solves = [r for r in rows if r["role"] == "solve"]
    proxies = [r for r in rows if r["role"] == "scramble_proxy"]

    print(f"\n{'=' * 66}\nFALSE-DQ SIDE - legit solves (want: all verified)")
    for label, sel in (("HELD-OUT (honest)", [r for r in solves if not r["seen_in_training"]]),
                       ("trained-on", [r for r in solves if r["seen_in_training"]])):
        if not sel:
            continue
        n = len(sel)
        bad = [r for r in sel if r["verdict"] == "rejected"]
        rev = [r for r in sel if r["verdict"] == "review"]
        head = [r["headroom_moves"] for r in sel if "headroom_moves" in r]
        print(f"\n  --- {label}: {n} legit solves ---")
        print(f"    verified : {n - len(bad) - len(rev)}/{n}")
        print(f"    FALSE DQ : {len(bad)}/{n}"
              + (f"  {[r['session'] for r in bad]}" if bad else ""))
        print(f"    review   : {len(rev)}/{n}")
        if head:
            print(f"    headroom above the floor: min {min(head):+d} "
                  f"median {int(np.median(head)):+d} max {max(head):+d}")
            print(f"      -> the min is how close a legit solve came to "
                  f"being falsely rejected.")

    if proxies:
        print(f"\n{'=' * 66}\nCATCH SIDE - scramble sessions as solver-following "
              f"proxies (want: all rejected)")
        for label, sel in (("HELD-OUT (honest)", [r for r in proxies if not r["seen_in_training"]]),
                           ("trained-on", [r for r in proxies if r["seen_in_training"]])):
            if not sel:
                continue
            n = len(sel)
            caught = [r for r in sel if r["verdict"] == "rejected"]
            miss = [r for r in sel if r["verdict"] != "rejected"]
            m = [r["observed_moves"] for r in sel]
            print(f"\n  --- {label}: {n} proxy attacks ---")
            print(f"    caught   : {len(caught)}/{n}")
            print(f"    MISSED   : {len(miss)}/{n}"
                  + (f"  {[r['session'] for r in miss]}" if miss else ""))
            print(f"    observed count: min {min(m)} median "
                  f"{int(np.median(m))} max {max(m)}  (floor "
                  f"{MIN_OBSERVED_MOVES})")
            print(f"    margin below the floor: {MIN_OBSERVED_MOVES - max(m)} "
                  f"moves at the worst proxy.")

    if solves and proxies:
        lo_legit = min(r["observed_moves"] for r in solves)
        hi_att = max(r["observed_moves"] for r in proxies)
        print(f"\n{'=' * 66}\nSEPARATION: lowest legit solve reads {lo_legit}; "
              f"highest proxy attack reads {hi_att}; gap {lo_legit - hi_att} "
              f"moves, floor at {MIN_OBSERVED_MOVES} sits inside it.")

    out = Path(f"anticheat_gate_{Path(args.ctc).stem}.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {out}")
    print("\nThe catch side is a PROXY, not a recorded attack. It shows the "
          "count separates a short machine sequence from a human solve; it "
          "does not show that someone deliberately trying to beat this gate "
          "fails. Record those (cv/labeling/record_attack.py) before "
          "treating the catch rate as validated.")


def cmd_calibrate_poststop(args):
    """Measure the phantom rate over a window with no true moves in it.

    Method: take the tail of each session AFTER its last true onset, which
    is genuinely move-free footage of a cube being held/put down — the same
    content as the post-timer window — and count what the model decodes
    there. POST_STOP_MOVE_LIMIT must sit above the worst of these or the
    honest solves it was built to protect get rejected by it.
    """
    import numpy as np
    from dataset import JointArrayStream

    ck, model, device = _load_model(args.ctc)
    trained = set(ck.get("train_session_names") or [])

    rows = []
    for d in sorted(SESSION_ROOT.iterdir()):
        p = d / "detector_stream_color.npz"
        if not p.is_file():
            continue
        raw = np.load(p, allow_pickle=True)
        onsets, fps = raw["onset_idx"].astype(int), float(raw["fps"])
        if len(onsets) == 0:
            continue
        start = int(onsets.max()) + int(round(args.guard_s * fps))
        tail = raw["frames"][start:]
        secs = len(tail) / fps
        if secs < args.min_s:
            continue
        st = JointArrayStream(tail, name=d.name, fps=fps,
                              onset_idx=np.array([], dtype=int),
                              onset_class=np.array([], dtype=int))
        n = len(_decode_count(model, st, device, args.beam))
        rows.append({"session": d.name, "tail_s": round(secs, 1),
                     "phantoms": n, "per_10s": round(10 * n / secs, 2),
                     "seen": d.name in trained})
        print(f"{d.name:38s} tail {secs:5.1f}s  phantoms {n:3d}  "
              f"{10 * n / secs:5.2f}/10s  "
              f"{'(trained)' if d.name in trained else 'HELD-OUT'}")

    if not rows:
        raise SystemExit(f"no session had >={args.min_s}s of move-free tail; "
                         "lower --min-s or record dedicated post-stop clips")
    for label, sel in (("HELD-OUT (honest)", [r for r in rows if not r["seen"]]),
                       ("trained-on", [r for r in rows if r["seen"]])):
        if not sel:
            continue
        per10 = np.array([r["per_10s"] for r in sel])
        raw_n = np.array([r["phantoms"] for r in sel])
        print(f"\n=== {label}: {len(sel)} tails ===")
        print(f"  phantoms per 10s: mean {per10.mean():.2f} "
              f"p90 {np.percentile(per10, 90):.2f} max {per10.max():.2f}")
        print(f"  worst absolute count on a move-free tail: {raw_n.max()}")
        # The limit is a RATE, not a constant (see POST_STOP_PHANTOM_RATE), so
        # what has to clear the worst measurement is the rate itself.
        print(f"  -> POST_STOP_PHANTOM_RATE must exceed {per10.max():.2f} "
              f"per 10s, currently {POST_STOP_PHANTOM_RATE:.2f}"
              + ("" if POST_STOP_PHANTOM_RATE > per10.max()
                 else "   <-- DOES NOT; this rate would false-DQ"))
        print(f"     over a {POST_STOP_MAX_WINDOW_S:.0f}s window that allows "
              f"{post_stop_limit(POST_STOP_MAX_WINDOW_S)} moves against a "
              f"real solve's >={MIN_OBSERVED_MOVES}")

    out = Path(f"poststop_phantoms_{Path(args.ctc).stem}.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {out}")
    print("\nCaveat: a session tail is the cuber ADMIRING a solved cube. The "
          "real post-stop window includes handing the cube toward the camera "
          "for the 6-face scan, which moves more. Treat this as a LOWER "
          "bound on the phantom rate and re-measure on real scan footage.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("explain", help="show the separation table behind the thresholds")

    s = sub.add_parser("score", help="run the gate over the legit corpus")
    s.add_argument("--ctc", required=True)
    s.add_argument("--beam", type=int, default=0)
    s.add_argument("--limit", type=int, default=0)

    c = sub.add_parser("calibrate-poststop",
                       help="measure the phantom rate on move-free footage")
    c.add_argument("--ctc", required=True)
    c.add_argument("--beam", type=int, default=0)
    c.add_argument("--guard-s", type=float, default=1.0,
                   help="skip this much after the last true onset")
    c.add_argument("--min-s", type=float, default=3.0,
                   help="ignore tails shorter than this")

    args = ap.parse_args()
    {"explain": cmd_explain, "score": cmd_score,
     "calibrate-poststop": cmd_calibrate_poststop}[args.cmd](args)


if __name__ == "__main__":
    main()
