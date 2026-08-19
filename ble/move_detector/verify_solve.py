"""
verify_solve.py

The whole pipeline, end to end, stated as the product claim:

    "This person was handed a cube in THIS state, and this video shows
     them solving it" — decided from webcam frames alone.

Everything upstream measures a component. This measures the claim:

    detector      when a move happened          (checkpoints/move_detector.pt)
    classifier    which move it was             (move_classifier_*.pt)
    reconstruct   which sequence is CONSISTENT with the cube going from
                  the known start state to the known end state
    this file     does the resulting verdict actually mean anything?

That last question is the reason this file exists rather than being one
more flag on live_detect.py. A verifier that says "solved" is worthless
until you know what it says to a video that should be REJECTED, and the
decoder will happily repair a wrong claim if you hand it enough budget —
every cube state is at most 20 moves from solved, so with a large enough
insertion allowance literally every video "verifies". So this test always
runs the same video against deliberately wrong claims too, reports the
margin separating the true claim from the nearest false one, and — because
a rejection can come from the cost model or merely from the beam giving up
— prints a constructive upper bound on what each wrong claim's best story
would cost, so the two cannot be confused. A pass with no margin is a
fail, and a rejection the beam produced is not the same result as a
rejection the cost model produced.

Two phases, one sitting
-----------------------
PHASE 1 - SCRAMBLE (measurement).  Start from a SOLVED cube and perform a
prescribed scramble on camera.  Every move is known, so this yields honest
per-move numbers with no smart cube involved: detector recall, classifier
accuracy, and what the group-theoretic decode adds on top of them.  The
label comes from the instruction, not the hardware, so it measures the
cube you actually own.

PHASE 2 - SOLVE (verification).  Solve that cube on camera.  There is no
per-move ground truth here — that is the point, it is the real use case —
but the endpoints are known, so the decode either finds a consistent story
or it does not, and the falsifiability sweep says how much that is worth.

Phase 1 is not just a warm-up: it is what tells you whether phase 2's
claimed start state is real.  If the scramble was performed wrong, the
cube is not in the state phase 2 asserts it was, and a failed verification
is a human error rather than a pipeline error.  The report says so.

Ground truth from the smart cube (--ble)
----------------------------------------
Both phases assume something they cannot check on their own: phase 1 that
you performed the printed scramble correctly, phase 2 that the pipeline's
mistakes can be inferred from a verdict.  Neither survives contact with a
60-move solve nobody can recall afterwards.  With `--ble` the cube logs
every turn it feels, on the same wall clock as the frames, and the report
gains the thing that makes a failure debuggable — WHICH move was misread:

  * the cube is checked to actually BE solved before phase 1 starts;
  * phase 1 compares what you performed against what was printed, and
    verifies against the former, so a mis-performed scramble stops being
    indistinguishable from a pipeline error;
  * phase 2 — the free solve, which has no other possible source of truth —
    gets a full per-move breakdown: detector recall, classifier accuracy,
    end to end, and what the decode recovered;
  * the cube is asked whether it is really solved at the end, which the
    decode can never establish on its own.

The BLE stream is a LABEL, never an input.  The detector, classifier and
decode never see it, so the product claim is unchanged: still decided from
webcam frames alone.  Drop `--ble` and every number is still produced, just
against the weaker assumption.

    python verify_solve.py --ble --front blue --top yellow

One caveat the report cannot state for you: the truth is blind to whole-cube
rotations.  The cube does not report x/y/z as BLE events and the IMU stream
is disabled, so rotating mid-take leaves the label frame where it was
calibrated and scores a correct classifier wrong.  It is distinguishable
after the fact — a rotation corrupts EVERY following label, so it appears as
one unbroken run of errors, while classifier errors come isolated.

Keeping the take (--save)
-------------------------
    python verify_solve.py --ble --front blue --top yellow --save

writes both takes as session directories in record_training.py's layout, so
postprocess_session.py / prepare_data.py / train_move_classifier.py /
--session all read them unchanged.  A live take is the only source of data
for the regime that breaks this pipeline, and the failures are the ones
worth keeping.

Rehearsal without a camera
--------------------------
    python verify_solve.py --session ../training_data/solve_20260721_103149/

replays a recorded BLE session through the identical verification path,
using the BLE move list as ground truth.  Same code, same report, no
hardware — use it to check the wiring before spending a live take, and to
re-measure after retraining either model.

    python verify_solve.py                      # live, 20-move scramble
    python verify_solve.py --scramble 25 --seed 7
    python verify_solve.py --skip-scramble      # cube already scrambled...
    python verify_solve.py --claim "R U R' U'"  # ...by this word

Which model reads the video (--joint, --ctc)
--------------------------------------------
Three arms are wired here, all producing the same moves list and all
decoded by the same reconstruct.py, so the only thing that differs between
them is how a video becomes a move sequence:

    (default)   the deployed detector + classifier pair — two models, a
                window cut around each detected onset
    --joint     Stage A's single joint onset+class model, peak-picked off
                its own posteriorgram (train_joint.py, joint_decode.py)
    --ctc       the same trunk trained with a CTC objective and decoded by
                a prefix beam search over FRAMES (train_ctc.py,
                ctc_decode.py). No onset threshold and no min_sep: whether
                a move exists at all is settled inside the search instead
                of by peak-picking beforehand, which is where every
                phantom and every merged-onset miss used to be created.

--ctc is the best-measured of the three offline — held out, replicated on
both seeds: move error rate 8.5% -> 5.8%, phantoms down 64%, verified
sessions 1/8 -> 3/8 — but it has never been run on a live take, which is
the regime that has broken every previous offline result in this project.
That is what this script is for. The default arm is unchanged so the two
can be compared on the same sitting.

    python verify_solve.py --ble --front blue --top yellow --ctc --save

WHICH CTC CHECKPOINT, AND WHY IT MATTERS HERE
---------------------------------------------
--ctc-model now defaults to `checkpoints/move_ctc_spd_s0.pt`, the
SPEED-AUGMENTED model (2026-08-05, `--speed-aug 0.5`). It is not better on
the offline val split — 5.4% MER against aug44's 5.26%, a tie — and that is
the point: the val split is slow footage, so it cannot see what this model
was trained for.

What it IS better at is counting moves when the solve is fast, which is the
regime a live take actually reaches and the offline corpus does not (~2.4
TPS median). Worst held-out session retention, predicted/true move count,
both seeds:

    TPS      aug44 -> spd (s0)     aug44 -> spd (s1)
     6      0.809 -> 0.954        0.763 -> 0.842
     8      0.632 -> 0.882        0.612 -> 0.796
    10      0.500 -> 0.770        0.428 -> 0.691

That is what pushed the anticheat gate's abstain band from 7.11 to 9.62 TPS
(anticheat_gate.RETENTION_FLOOR). **Those constants are calibrated on THIS
checkpoint.** Passing a different --ctc-model without re-running
`speed_sim.py --blur` and updating RETENTION_FLOOR leaves the gate believing
a retention curve the running model does not have.

Caveat that a live take is exactly the right way to attack: the speed gains
were measured by SIMULATING speed (dropping frames, blur-approximating the
longer exposure). Real fast footage has motion blur those frames only
approximate. Solve fast on camera and see whether the count holds.

--lm additionally fuses move_lm.py's n-gram prior into the CTC beam. It is
regime-dependent and measured as such: a gain cross-day, a loss on the
same-day holdout, on both seeds. A live take is cross-day, so it should
help — a prediction, not a measurement, and the reason it is a flag.

Run from inside move_detector/, same convention as the rest of the repo.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# win_compat has to run before anything pulls in cv2 or pywin32, or bleak's
# WinRT callbacks silently stop firing (see ble/win_compat.py). live_detect
# imports cv2, so the BLE-side bootstrap has to come first even though the
# cube is optional — ble_truth.py's own header import is what does it.
_BLE_DIR = Path(__file__).resolve().parents[1]
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))
import win_compat  # noqa: F401,E402

# The anticheat half of this script needs the continuity guard and the
# solved-at-stop check, which live in cv/detection (CLAUDE.md's cross-topic
# bootstrap convention — same one live_anticheat.py uses).
_DETECTION_DIR = Path(__file__).resolve().parents[2] / "cv" / "detection"
if str(_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTION_DIR))

import numpy as np  # noqa: E402

import live_detect as LD  # noqa: E402
import reconstruct as RC  # noqa: E402
from anticheat_gate import (  # noqa: E402
    POST_STOP_MAX_WINDOW_S as AG_POST_STOP_MAX_WINDOW_S)
from decode import MIN_SEP  # noqa: E402

# How far off a claimed start state has to be before the verifier rejects
# it. Swept, not assumed — see falsifiability_sweep.
DECOY_DISTANCES = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# One claim: "this video takes the cube from A to B"
# ---------------------------------------------------------------------------

def _decode_one(start_state, end_state, cost_rows, beam, args, tables, kw):
    """Single-pass decode() (default) or D3 --bidir, one shared call site."""
    if not getattr(args, "bidir", False):
        return RC.decode_between(start_state, end_state, cost_rows,
                                 beam=beam, **kw)
    if kw.get("rotations"):
        sys.exit("--bidir does not support --rotations")
    kw = {k: v for k, v in kw.items() if k != "rotations"}
    if getattr(args, "meet_sweep", False):
        return RC.decode_bidirectional_sweep(start_state, end_state,
                                             cost_rows, beam=beam, **kw)
    return RC.decode_bidirectional(start_state, end_state, cost_rows,
                                   beam=beam, meet=getattr(args, "meet", None),
                                   **kw)


def decode_claim(start_state, end_state, cost_rows, del_costs, args, tables,
                 beam=None, retry=True):
    """
    Decode one claim, optionally retrying once at a wider beam if it fails.

    The decoys in the falsifiability sweep are run with retry=False, at the
    same beam that decided the true claim. Retrying them would be both slow
    and unfair in the wrong direction: a decoy rejected at beam B and a
    decoy rejected at beam 4B are different statements, and the comparison
    only means something if every claim got the same search effort.

    --bidir (D3, PATH_TO_VERIFICATION.md §5) swaps in the bidirectional
    meet-in-the-middle decoder — decode_bidirectional already takes
    (start_state, end_state) natively, so unlike decode_between it needs
    no frame-shift; same beam/retry contract either way.
    """
    # slices/c_slice/slice_rows travel with every claim, true and decoy
    # alike. A cost model that is more generous to the true claim than to
    # its decoys inflates every verdict without making any of them more
    # true, which would silently gut the whole point of the sweep.
    kw = dict(c_del=args.del_cost, c_ins=args.ins_cost, c_rot=args.rot_cost,
              max_end_ins=args.max_end_ins, rel_weight=args.rel_weight,
              del_costs=del_costs, rotations=args.rotations, tables=tables,
              slices=bool(getattr(args, "slices", False)),
              c_slice=float(getattr(args, "c_slice", RC.C_SLICE)),
              slice_rows=getattr(args, "slice_rows", None))
    beam = beam or args.beam
    res = _decode_one(start_state, end_state, cost_rows, beam, args, tables, kw)
    if retry and not res["solved"] and args.retry_beam > beam:
        res = _decode_one(start_state, end_state, cost_rows,
                          args.retry_beam, args, tables, kw)
        res["retried"] = True
    res["beam_used"] = args.retry_beam if res.get("retried") else beam
    return res


def verify_claim(label, start_state, end_state, moves, threshold, args,
                 tables, truth=None):
    """
    Run and report one claim.  `truth` is the true move word when it is
    known (phase 1's prescribed scramble, or a session's BLE move list) and
    None when it is not (phase 2 — the real use case).
    """
    pred_names, cost_rows, del_costs = RC.costs_from_moves(
        moves, threshold, args.blend_inv, args.blend_unif, args.del_cost,
        args.candidate_threshold, args.del_floor, args.blend_adj)

    # Which onsets may be read as a middle-slice turn. Computed once here and
    # stashed on args so decode_claim uses the SAME mask for the true claim
    # and for every falsifiability decoy — see decode_claim's kw comment.
    args.slice_rows = (RC.slice_rows_from_moves(moves, args.slice_gate)
                       if getattr(args, "slices", False) else None)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # What the raw pipeline says on its own, before any group theory. This
    # is the honest baseline the decode has to beat — and on a clean take
    # it can already be enough, in which case the decode is confirming
    # rather than repairing.
    raw_ok = bool((RC.compose_seq(start_state, pred_names) ==
                   end_state).all())
    print(f"\n  raw pipeline: {len(pred_names)} moves detected+named, "
          f"consistent with the claim: {'YES' if raw_ok else 'no'}")

    out = {"label": label, "raw_ok": raw_ok, "n_onsets": len(pred_names),
           "pred_names": pred_names, "cost_rows": cost_rows,
           "del_costs": del_costs, "raw_acc": None, "acc": None}

    # Scored before the decode, and reported whether or not the decode
    # succeeds — a failed verification still has to say how noisy its
    # input was, or there is no way to tell a search failure from a
    # pipeline that never had a chance.
    raw = None
    if truth is not None:
        raw = RC.score_vs_gt(truth, pred_names)
        out["raw_acc"] = raw["acc"]
        out["raw_errors"] = raw["sub"] + raw["miss"] + raw["phantom"]

    t0 = time.time()
    res = decode_claim(start_state, end_state, cost_rows, del_costs,
                       args, tables)
    out["res"] = res

    if not res["solved"]:
        if raw is not None:
            print(f"  raw vs truth: {raw['acc']*100:5.1f}% per-move   "
                  f"({raw['sub']} named wrong, {raw['miss']} missed, "
                  f"{raw['phantom']} phantom of {len(truth)} true moves)")
        print(f"  decode:       NOT VERIFIED  — no story within the edit "
              f"budget explains this video")
        print(f"                closest end state is {res['min_h_final']} "
              f"quarter turns away ({time.time()-t0:.1f}s"
              f"{', retried at beam ' + str(args.retry_beam) if res.get('retried') else ''})")
        print(f"                raise --beam if the true story is expensive, "
              f"--max-end-ins if\n                the camera missed the last "
              f"few moves — but see the caveat below.")
        return out

    ops = res["op_counts"]
    print(f"  decode:       VERIFIED  — cost {res['cost']:.2f}, "
          f"{len(res['moves'])} moves  ({res['decode_seconds']:.1f}s)")
    print(f"                edits: {ops}")
    _print_word("reconstructed sequence", res["moves"])

    if raw is not None:
        rec = RC.score_vs_gt(truth, res["moves"])
        gtc = RC.gt_path_cost(truth, pred_names, cost_rows, del_costs,
                              args.ins_cost)
        gap = res["cost"] - gtc
        side = ("identical to the true path" if abs(gap) < 1e-6 else
                f"the model preferred a CHEAPER wrong story (gap {gap:+.2f}) "
                f"— calibration, not search" if gap < 0 else
                f"the beam lost the true path (gap {gap:+.2f}) — raise --beam")
        print(f"\n  vs ground truth ({len(truth)} true moves):")
        print(f"    raw     {raw['acc']*100:5.1f}% per-move   "
              f"({raw['sub']} named wrong, {raw['miss']} missed, "
              f"{raw['phantom']} phantom)")
        print(f"    decoded {rec['acc']*100:5.1f}% per-move   "
              f"({rec['sub']} named wrong, {rec['miss']} missed, "
              f"{rec['phantom']} phantom)   "
              f"{'EXACT MATCH' if rec['exact'] else ''}")
        print(f"    decode cost {res['cost']:.2f} vs true path {gtc:.2f}  "
              f"[{side}]")
        out.update({"acc": rec["acc"], "exact": rec["exact"],
                    "gt_cost": gtc})
    return out


def _print_word(label, moves, per_row=18):
    print(f"\n  {label} ({len(moves)} moves):")
    for i in range(0, len(moves), per_row):
        print(f"    {' '.join(f'{m:<2}' for m in moves[i:i+per_row])}")


# ---------------------------------------------------------------------------
# Falsifiability — what does the same video say to a claim that is WRONG?
# ---------------------------------------------------------------------------

def _offset_state(state, d, tables, rng, tries=40):
    """
    A cube state exactly d quarter turns from `state`.

    Exactly, not approximately: a random d-move word can collapse to
    something shorter, and a decoy that is secretly 2 moves away would
    report as a 4-move decoy and flatter the verifier. The minimal length
    is checked with the same exact IDDFS the decoder finishes with.
    """
    for _ in range(tries):
        word = [RC.WCA12[k] for k in RC._random_gt(rng, d)]
        cand = RC.compose_seq(state, word)
        rel = RC.compose(RC.inverse(state), cand)
        exact = RC._completion(rel, d, tables)
        if exact is not None and len(exact) == d:
            return cand, word
    return None, None


def falsifiability_sweep(true_start, end_state, cost_rows, del_costs,
                         args, tables, seed=0, beam=None):
    """
    Decode the same onsets against start states that are deliberately
    wrong, and report which ones the verifier accepts anyway.

    This is the number that decides whether a "VERIFIED" verdict is
    evidence.  The decoder's job is to repair noise, and a wrong claim is
    indistinguishable from noise if the repair budget covers the
    difference: a claim d quarter turns off the truth costs roughly
    d * C_INS extra to explain, so it is ACCEPTED for small d and rejected
    once d exceeds what the budget can buy.  Two conclusions follow, and
    both belong in any honest write-up of this pipeline:

      * the true claim must come out CHEAPEST.  Verification here is a MAP
        decision among claims, not a yes/no test, and "cheapest" is the
        part that carries information.
      * the verifier cannot separate claims closer together than the edit
        budget.  That is a property of the cost model, not a bug, and it
        is why --max-end-ins is small.  Report the measured separation
        distance rather than the bare verdict.

    The last two decoys are the frauds that actually matter: a cube that
    was never scrambled at all, and a cube scrambled by something else
    entirely.

    The decoys perturb the START state, which is the fraud with teeth ("I
    was handed a different cube"), and they are harder to repair than an
    equally wrong END state would be: fixing a wrong start needs moves
    inserted at the FRONT, where MAX_CHAIN caps a gap at two insertions,
    while a wrong end is absorbed by the tail's exact completion up to
    --max-end-ins. So the separation distance measured here is a property
    of start-claims specifically. It is still the right thing to measure —
    the start state is the half a verifier is told rather than observes.
    """
    rng = np.random.default_rng(seed)
    claims = [("true claim", true_start, 0)]
    for d in DECOY_DISTANCES:
        cand, _ = _offset_state(true_start, d, tables, rng)
        if cand is not None:
            claims.append((f"start state {d} move{'s' if d > 1 else ''} off",
                           cand, d))
    claims.append(("cube was never scrambled", RC.SOLVED.copy(), None))
    far = RC.compose_seq(RC.SOLVED.copy(),
                         [RC.WCA12[k] for k in RC._random_gt(rng, 25)])
    claims.append(("scrambled by something else", far, None))

    beam = beam or args.beam
    print(f"\n{'-'*70}")
    print(f"  FALSIFIABILITY — the same video decoded against wrong claims")
    print(f"  (every claim searched at beam {beam}, no retries — same "
          f"effort each)")
    print(f"{'-'*70}")
    print(f"    {'claimed start state':<30} {'verdict':<9} {'cost':>7} "
          f"{'vs true':>8}  {'story exists at':>15}")

    rows, true_cost = [], None
    for name, start, d in claims:
        res = decode_claim(start, end_state, cost_rows, del_costs,
                           args, tables, beam=beam, retry=False)
        cost = res["cost"] if res["solved"] else None
        if d == 0:
            true_cost = cost
        # Constructive upper bound on this decoy's TRUE optimum, and the
        # reason the table has a fifth column. If the true claim decoded to
        # the word W, then a claim d quarter turns away is solved by
        # w^-1 . W for the d-move word w that separates them — the same
        # accepts, plus d insertions. So a valid story provably exists at
        # <= true_cost + d*C_INS whether or not the beam can find it, and
        # "rejected" on such a row is a statement about the SEARCH, not
        # about the cost model. Without this column the sweep would read
        # as much stronger evidence than it is.
        bound = None if (d in (0, None) or true_cost is None) \
            else true_cost + d * args.ins_cost
        cost_s = "-" if cost is None else f"{cost:.2f}"
        delta = ("" if cost is None or true_cost is None else
                 ("baseline" if d == 0 else f"{cost - true_cost:+.2f}"))
        bound_s = "" if bound is None else f"<= {bound:.2f}"
        verdict = "ACCEPTED" if res["solved"] else "rejected"
        print(f"    {name:<30} {verdict:<9} {cost_s:>7} {delta:>8}  "
              f"{bound_s:>15}")
        rows.append({"name": name, "distance": d, "verified": res["solved"],
                     "cost": cost, "bound": bound})

    if true_cost is None:
        print(f"\n    The TRUE claim did not verify, so nothing here is "
              f"meaningful yet.")
        return rows

    cheaper = [r for r in rows if r["verified"] and r["distance"] != 0
               and r["cost"] < true_cost - 1e-6]
    sep = min((r["distance"] for r in rows
               if r["distance"] and not r["verified"]), default=None)
    search_rejected = [r for r in rows if not r["verified"]
                       and r["bound"] is not None]
    frauds = [r for r in rows if r["distance"] is None]

    print()
    if cheaper:
        print(f"    A WRONG claim explained this video more cheaply than the "
              f"true one\n    ({cheaper[0]['name']}). The verdict is not "
              f"evidence — the cost model is\n    mis-calibrated, or the "
              f"video is too noisy to identify the story.")
    else:
        near = [r for r in rows if r["verified"] and r["distance"]]
        print(f"    Of {len(rows)} claims tested, "
              f"{sum(r['verified'] for r in rows)} verified"
              f"{' — only the true one' if sum(r['verified'] for r in rows) == 1 else ''}, "
              f"and the true claim was the cheapest.")
        if near:
            m = min(r["cost"] - true_cost for r in near)
            print(f"    Nearest accepted wrong claim costs {m:+.2f} — that "
                  f"margin, {args.ins_cost:.1f} per\n    quarter turn of "
                  f"claim error, is the whole of what this verdict asserts.")
    if all(not r["verified"] for r in frauds) and frauds:
        print(f"    Both outright frauds — a cube never scrambled, and a cube "
              f"scrambled by\n    something else — were rejected. Those are "
              f"~20 quarter turns from the\n    truth, far outside any repair "
              f"budget, so their rejection is real.")
    if search_rejected:
        d0 = min(r["distance"] for r in search_rejected)
        print(f"    But read the near decoys carefully: a valid story exists "
              f"for every one of\n    them at the cost in the last column "
              f"(true story + one insertion per\n    quarter turn of error), "
              f"and the decoder still rejected them from "
              f"{d0}\n    move{'s' if d0 > 1 else ''} out. Those rejections "
              f"come from the SEARCH — beam width, and the\n    "
              f"{RC.MAX_CHAIN}-insertion cap on a single gap, which cannot "
              f"even express a repair\n    more than {RC.MAX_CHAIN} moves "
              f"long at the front — not from the cost model, whose\n    own "
              f"separation is just {args.ins_cost:.1f} per quarter turn. "
              f"Operationally the system does\n    reject them, which is "
              f"what a verifier has to do; but do not report it as\n    "
              f"proof that no cheap wrong story exists.")
    elif sep is not None:
        print(f"    Claims >= {sep} quarter turns from the truth are "
              f"rejected on cost. Closer\n    ones sit inside the "
              f"{args.max_end_ins}-move repair budget and cannot be "
              f"distinguished —\n    that bounds what a VERIFIED verdict "
              f"asserts.")
    else:
        print(f"    Every decoy tested was accepted — at this budget the "
              f"verdict carries no\n    information. Lower --max-end-ins, or "
              f"treat only the cost ranking as the\n    result.")
    return rows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_stack(args):
    import torch
    from model import build_model
    from crop_utils import load_detector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.detector, map_location=device)
    det_model = build_model(device)
    det_model.load_state_dict(ckpt["state_dict"])
    det_model.eval()
    threshold = args.threshold if args.threshold is not None \
        else ckpt.get("threshold", 0.5)
    min_sep = args.min_sep if args.min_sep is not None \
        else ckpt.get("min_sep", MIN_SEP)

    print(f"\n  Detector:   {args.detector}  (epoch {ckpt.get('epoch','?')}, "
          f"threshold {threshold}, min_sep {min_sep})")
    print(f"  Classifier: {args.classifier}")
    print(f"  Device:     "
          f"{torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    detector = load_detector()
    if detector is None:
        print(f"  WARNING: cube detector unavailable — falling back to "
              f"centered square crops.\n           Both models trained on "
              f"cube crops; expect degraded accuracy.")
    return detector, det_model, device, threshold, min_sep


def load_joint_stack(args):
    """
    --joint: load Stage A's single joint onset+class model instead of the
    deployed detector+classifier pair (MODEL_REWORK_PLAN.md). Returns the
    same 5-tuple shape as load_stack so run_live/_run_live need no
    branching beyond which loader they call.
    """
    import torch
    from model import build_joint_from_ckpt
    from crop_utils import load_detector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.joint_model, map_location=device)
    _check_model_type(ckpt, args.joint_model, want="joint")
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    threshold = args.threshold if args.threshold is not None \
        else ckpt.get("threshold", 0.5)
    min_sep = args.min_sep if args.min_sep is not None \
        else ckpt.get("min_sep", MIN_SEP)

    print(f"\n  Joint model: {args.joint_model}  (epoch {ckpt.get('epoch','?')}, "
          f"seed {ckpt.get('seed','?')}, threshold {threshold}, "
          f"min_sep {min_sep})")
    print(f"  Val (at save): F1 {ckpt.get('val_f1',0)*100:.1f}%, at_onset "
          f"{ckpt.get('val_at_onset',0)*100:.1f}%, held out "
          f"{ckpt.get('val_session_names', [])}")
    if ckpt.get("n_counts"):
        print(f"  Count head:  present — merged peaks may be split in two "
              f"(joint_decode.split_peak)")
    print(f"  Device:      "
          f"{torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    detector = load_detector()
    if detector is None:
        print(f"  WARNING: cube detector unavailable — falling back to "
              f"centered square crops.")
    return detector, model, device, threshold, min_sep


def _check_model_type(ckpt, path, want):
    """
    Refuse a checkpoint decoded by the wrong decoder.

    The two Stage A arms share a trunk and a state_dict shape, so a CTC
    checkpoint loads perfectly under --joint and a peak-picked checkpoint
    loads perfectly under --ctc. Neither fails loudly — they just produce
    nonsense, because the 13th column means "background" in one arm and
    CTC's blank in the other, and a blank column is ~0.9 on almost every
    frame. That would read as a catastrophic model regression rather than
    as the flag mistake it is, live, with a cube in your hand.
    """
    got = ckpt.get("model_type", "joint")
    if got != want:
        sys.exit(f"{path} is a '{got}' checkpoint but --{want} decodes it as "
                 f"'{want}'.\nThe two share a state_dict shape so this would "
                 f"load and silently produce garbage.\nUse --{got} "
                 f"--{got}-model {path} instead.")


def load_ctc_stack(args):
    """
    --ctc: load the CTC-trained trunk (train_ctc.py) plus, with --lm, the
    n-gram move prior fused into the prefix beam search. Same 5-tuple as
    load_stack/load_joint_stack; `threshold` still feeds
    costs_from_moves' deletion pricing downstream, `min_sep` is inert on
    this path (see joint_decode.analyse_ctc_live).

    The LM is fit on the CHECKPOINT'S OWN training sessions, read off the
    checkpoint rather than passed in. Fitting it on anything wider would
    leak held-out move sequences into their own decode; on a live take
    there is no holdout to leak, but the prior has to be the same object
    that was measured offline or the live number is not comparable to it.
    """
    import torch
    from model import build_joint_from_ckpt
    from crop_utils import load_detector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ctc_model, map_location=device)
    _check_model_type(ckpt, args.ctc_model, want="ctc")
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    threshold = args.threshold if args.threshold is not None \
        else ckpt.get("threshold", 0.5)

    print(f"\n  CTC model:  {args.ctc_model}  (epoch {ckpt.get('epoch','?')}, "
          f"seed {ckpt.get('seed','?')})")
    print(f"  Val (at save): MER {ckpt.get('val_mer', float('nan'))*100:.1f}% "
          f"greedy"
          + (f", {ckpt['val_mer_beam']*100:.1f}% at beam {ckpt.get('beam')}"
             if ckpt.get("val_mer_beam") is not None else "")
          + f"; held out {ckpt.get('val_session_names', [])}")
    print(f"  Decoder:    prefix beam {args.ctc_beam} over frames — no onset "
          f"threshold, no min_sep")

    args.lm = None
    if args.lm_fusion:
        from move_lm import MoveLM
        train_names = list(ckpt.get("train_session_names") or [])
        if not train_names:
            sys.exit("--lm needs the checkpoint's train_session_names to fit "
                     "the prior on;\nthis checkpoint has none.")
        args.lm = MoveLM.from_sessions(train_names, order=args.lm_order)
        print(f"  LM:         order {args.lm_order}, {args.lm.n_sequences} "
              f"sessions / {args.lm.n_moves} moves, alpha {args.lm_alpha}, "
              f"beta {args.lm_beta}")
        print(f"              regime-dependent: measured as a GAIN cross-day "
              f"and a LOSS on the\n              same-day holdout, on both "
              f"seeds. A live take today is cross-day.")
    print(f"  Device:     "
          f"{torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    detector = load_detector()
    if detector is None:
        print(f"  WARNING: cube detector unavailable — falling back to "
              f"centered square crops.")
    return detector, model, device, threshold, None


# ---------------------------------------------------------------------------
# Anticheat — the verdict the SERVER will reach, on the same take
# ---------------------------------------------------------------------------
#
# The reconstruction verdict above answers "is there a consistent story".
# That is a different question from "should this count", and the second one
# is `anticheat_gate.adjudicate()`. Wiring it in here rather than leaving it
# to `live_anticheat.py` costs one extra recording and buys the thing
# neither program had on its own: ONE sitting that produces both verdicts on
# the SAME frames, so a disagreement between them is attributable.
#
# adjudicate() is a pure function over plain data precisely so the client and
# the server can both run it and agree. Everything assembled below is
# therefore plain data: counts, seconds and dicts, no models and no handles.
#
# THE TWO INPUTS A CALLER MUST NOT OMIT are `solved_at_stop` and
# `post_stop_continuity`. They are the only arm covering the
# enough-moves-then-swap attack (make a full solve's worth of real moves
# without solving, then substitute a solved cube), and adjudicate() treats
# both as "not assessed" when absent — which passes silently. That is why
# this file records a post-timer scan window at all.

#: Frames either side of the timer stop fed to solved_check.solved_at.
#: ~0.7s a side at 30fps, matching the 1.5s window solved_check.py's
#: thresholds were measured over. Coupled to that calibration: change it and
#: `solved_check.py sweep` has to be re-run.
STOP_WINDOW_FRAMES = 20

#: Capture rate is a temporal scale factor on the move count (the model's
#: receptive field is measured in FRAMES), so a run far off 30fps is
#: measuring the mismatch rather than the solve. Same constants as
#: live_anticheat.py, deliberately.
TRAIN_FPS = 30.0
FPS_TOLERANCE = 0.25


def _guard_over(src, fw, fh, label):
    """ContinuityGuard report over one recorded window.

    Post-hoc rather than live, and that is not a shortcut: the guard is a
    pure function of (t, boxes, sig), so replaying buffered frames with
    their REAL capture timestamps reaches the verdict a live pass would
    have — and, unlike a live pass, the server can reproduce it from the
    stored bundle, which is the property the whole design rests on.
    """
    import continuity_guard as CG
    load_color, n, _fps, _window, ftimes = src
    print(f"  continuity ({label}): {n} frames...")
    guard = CG.run_guard(((i, load_color(i)) for i in range(n)),
                         fw, fh, times=ftimes)
    return guard.report()


def _solved_at_stop(solve_src, scan_src, detector):
    """Was the cube solved when the timer stopped? (solved_check.solved_at)

    The window STRADDLES the stop — tail of the timed take plus head of the
    scan — rather than starting at it. The stop frame itself is the worst
    one available (a hand is leaving the cube for the keyboard), and a
    one-sided window answers a subtly different question: before the stop is
    "did they finish", after it is "what are they presenting", and the
    attack lives exactly in the seam.
    """
    from solved_check import solved_at

    if detector is None:
        return {"solved": None, "reason": "no_detector", "n_regions": None}
    s_load, s_n, _f, _w, _t = solve_src
    c_load, c_n, _f2, _w2, _t2 = scan_src
    pre = [s_load(i) for i in range(max(0, s_n - STOP_WINDOW_FRAMES), s_n)]
    post = [c_load(i) for i in range(min(STOP_WINDOW_FRAMES, c_n))]
    frames = [f for f in pre + post if f is not None]
    if len(frames) < 8:
        return {"solved": None, "reason": "too_few_frames",
                "n_regions": None, "n_frames": len(frames)}

    from prepare_data import per_frame_boxes
    n = len(frames)
    boxes, _ = per_frame_boxes(detector,
                               lambda i: frames[i] if 0 <= i < n else None, n)
    return solved_at(list(zip(frames, boxes)))


def adjudicate_take(args, solve_src, scan_src, solve_moves, scan_moves,
                    detector, lighting_ok=None):
    """Assemble SolveEvidence from one sitting and run the gate."""
    from anticheat_gate import SolveEvidence, adjudicate

    s_load, s_n, s_fps, _w, s_t = solve_src
    solve_seconds = float(s_t[-1] - s_t[0]) if len(s_t) > 1 else None

    probe = s_load(0)
    fh, fw = probe.shape[:2]

    # A capture rate far off training makes the COUNT unreadable, not merely
    # noisier. It goes in as None — which abstains — rather than as a number
    # that would manufacture a too_few_moves rejection out of a frame-rate
    # problem.
    fps_bad = abs(s_fps - TRAIN_FPS) / TRAIN_FPS > FPS_TOLERANCE

    scan_seconds = None
    if scan_src is not None:
        _cl, _cn, _cf, _cw, c_t = scan_src
        scan_seconds = float(c_t[-1] - c_t[0]) if len(c_t) > 1 else 0.0

    ev = SolveEvidence(
        session=f"verify_{time.strftime('%Y%m%d_%H%M%S')}",
        solve_seconds=solve_seconds,
        observed_moves=None if fps_bad else len(solve_moves),
        observed_moves_after_stop=(None if (fps_bad or scan_src is None)
                                   else len(scan_moves)),
        post_stop_seconds=scan_seconds,
        continuity=_guard_over(solve_src, fw, fh, "solve"),
        # The appearance/swap meter is deliberately NOT fed in. Its
        # threshold is uncalibrated on the attack side (swap_check.py bounds
        # the legit ceiling only), and an uncalibrated bar in a verdict is
        # how a false DQ ships. Same call live_anticheat.py makes.
        swap_jump=None, swap_threshold=None,
        lighting_ok=lighting_ok,
        solved_at_stop=(_solved_at_stop(solve_src, scan_src, detector)
                        if scan_src is not None else None),
        post_stop_continuity=(_guard_over(scan_src, fw, fh, "post-stop")
                              if scan_src is not None else None),
    )
    verdict = adjudicate(ev)
    verdict["_detail"] = {
        "capture_fps": round(s_fps, 1),
        "count_suppressed_by_fps": fps_bad,
        "raw_move_count": len(solve_moves),
        "lighting_ok": lighting_ok,
        "solve_frames": s_n,
        "scan_frames": None if scan_src is None else scan_src[1],
    }
    return verdict


def print_anticheat(v):
    from anticheat_gate import (MIN_OBSERVED_MOVES, post_stop_limit,
                                separation_tps_limit)

    d = v["_detail"]
    print(f"\n{'=' * 70}")
    print(f"  ANTICHEAT VERDICT: {v['verdict'].upper()}")
    print(f"{'=' * 70}")
    for r in v["reject_reasons"]:
        print(f"    REJECTED: {r}")
    for r in v["review_reasons"]:
        print(f"    review:   {r}")
    if not v["reject_reasons"] and not v["review_reasons"]:
        print(f"    every test passed")
    print(f"\n    moves in the timed window   {v['observed_moves']} "
          f"(floor {MIN_OBSERVED_MOVES}"
          + (f", headroom {v['headroom_moves']:+d}"
             if "headroom_moves" in v else "") + ")")
    if v.get("solve_seconds"):
        print(f"    solve time                  {v['solve_seconds']:.2f}s"
              + (f"   {v['tps']:.2f} TPS (abstains above "
                 f"{separation_tps_limit():.2f})" if v.get("tps") else ""))
    if d["count_suppressed_by_fps"]:
        print(f"    capture rate {d['capture_fps']:.1f} fps suppressed the "
              f"count; it would have read {d['raw_move_count']}")
    if d["scan_frames"] is not None:
        print(f"    post-stop window            {d['scan_frames']} frames, "
              f"allowance {post_stop_limit(v.get('post_stop_seconds'))} moves")
    for c in v.get("caveats", []):
        print(f"    caveat: {c}")
    print(f"\n    This is the SERVER's question — should this count — and it "
          f"is decided\n    separately from whether a consistent story "
          f"exists. Both were computed\n    from the same frames.")


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def connect_truth(args):
    """
    Bring up the smart cube as a ground-truth logger, or return None.

    A failed connection is offered as a choice rather than an exit: the
    take is still a valid product test without truth, just a much worse
    debugging one, and the caller is standing there with a cube in hand.
    """
    if not args.ble:
        return None
    import ble_truth as BT

    if not args.front or not args.top:
        sys.exit("--ble needs the cube's orientation so BLE moves can be "
                 "named in the same\ncamera-relative frame the classifier "
                 "uses: pass --front <colour> --top <colour>\ndescribing how "
                 "you will hold it (e.g. --front blue --top yellow).")
    print(f"\n  Connecting to the cube (front={args.front}, top={args.top})...")
    truth = BT.BleTruth(args.front, args.top, address=args.address,
                        echo=not args.no_echo)
    if truth.start():
        print(f"  Cube connected"
              + (f" — battery {truth.battery}%" if truth.battery is not None
                 else ""))
        # Checked up front, not after a wasted sitting: a flat cube corrupts
        # the truth and the solved check together, and both failures look
        # like pipeline bugs from the outside.
        if truth.warn_if_unhealthy():
            if input("\n  Continue anyway? [y/N] ").strip().lower() \
                    not in ("y", "yes"):
                truth.stop()
                sys.exit("Aborted — charge the cube.")
        print(f"  Ground truth is a LABEL ONLY: the detector, classifier and "
              f"decode never\n  see it, so the verdict is still decided from "
              f"webcam frames alone.")
        return truth
    print(f"\n  Could not connect: {truth.error}")
    print(f"  Without it phase 2 has no per-move ground truth and a "
          f"mis-performed\n  scramble cannot be told from a pipeline error.")
    if input("  Continue without ground truth? [y/N] ").strip().lower() \
            not in ("y", "yes"):
        sys.exit("Aborted.")
    truth.stop()
    return None


def save_take(out_dir: Path, buf, stamps, truth, meta: dict) -> Path | None:
    """
    Persist one take as a standard session directory.

    Written in exactly `record_training.py`'s layout — frames/, frames.jsonl,
    moves.jsonl, config.json — so `postprocess_session.py`,
    `prepare_data.py`, `train_move_classifier.py` and `--session` all read it
    with no special-casing. A live take is the only source of data for the
    regime that actually breaks this pipeline (long solves, fast regrips,
    whatever room you are in), and until now every one of them was buffered
    in memory and thrown away at exit — including the failures, which are
    the ones worth keeping.

    The ORIGINAL frames and timestamps are saved, never the resampled
    ordering: resampling is an analysis choice, and baking it into the
    recording would make the saved session unable to reproduce anything but
    that one choice.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "frames.jsonl", "w") as fh:
        for i, (enc, ts) in enumerate(zip(buf, stamps)):
            name = f"frame_{i:06d}_{ts:.3f}.jpg"
            (frames_dir / name).write_bytes(enc.tobytes())
            fh.write(json.dumps({"idx": i, "ts": float(ts),
                                 "file": name}) + "\n")

    n_moves = 0
    if truth is not None:
        events = truth.moves_between(float(stamps[0]), float(stamps[-1]))
        with open(out_dir / "moves.jsonl", "w") as fh:
            for k, e in enumerate(events, 1):
                fh.write(json.dumps({
                    "move_num": k,
                    "timestamp": e["ts"],
                    "ble_color": e["ble_color"],
                    "ble_direction": e["ble_direction"],
                    "ble_raw": e.get("ble_raw"),
                    "wca_face": e.get("wca_face"),
                    "wca_notation": e["wca"],
                    "front_color": e.get("front_color"),
                    "top_color": e.get("top_color"),
                }) + "\n")
        n_moves = len(events)

    # ble_meta.json mirrors record_training.py's, plus the health snapshot.
    # Without it, "was the cube actually solved / was it lying" can only be
    # asked live, and by the time a take looks wrong the cube has moved on.
    if truth is not None:
        solved, wrong = truth.solved_report()
        # Named as the claim it is, not as ground truth — see the same
        # change in record_training.py for why. Nothing downstream reads
        # these; every decode targets the solved state and derives its
        # start state from the move word.
        with open(out_dir / "ble_meta.json", "w") as fh:
            json.dump({"cube_reported_end_state": truth.snapshot_state(),
                       "cube_reported_end_solved": solved,
                       "end_facelets_wrong": wrong,
                       "end_state_trusted": not (wrong and wrong > 24),
                       "end_state_note": (
                           "The cube's own state tracking, not an "
                           "observation; >24 facelets wrong means the cube "
                           "has drifted, since one quarter turn moves 6. "
                           "Check the move word with session_check.py."),
                       "battery": truth.battery, "move_count": n_moves,
                       "health": truth.health()}, fh, indent=2)

    with open(out_dir / "config.json", "w") as fh:
        json.dump({"session_id": out_dir.name,
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                               time.localtime(stamps[0])),
                   "front_color": getattr(truth, "front", None),
                   "top_color": getattr(truth, "top", None),
                   "frame_skip": 1, "jpeg_quality": LD.JPEG_QUALITY,
                   "source": "verify_solve.py", **meta}, fh, indent=2)

    print(f"  saved: {out_dir}  ({len(buf)} frames, {n_moves} BLE moves)")
    if truth is None:
        print(f"         no moves.jsonl — without --ble there are no move "
              f"labels, so this\n         take is usable for the detector "
              f"but not for classifier training.")
    return out_dir


def record_phase(args, title, instructions, overlay=None, truth=None,
                 save_as: str | None = None, save_meta: dict | None = None,
                 max_seconds: float | None = None, cap_reason: str = ""):
    """
    Record one take. Returns (load_color, n_frames, fps, window, frame_times).

    `window` is the wall-clock span of the capture, which is how BLE moves
    get attributed to this take and not to the fiddling before it — the
    cube is connected across the whole sitting and reports every turn,
    including the ones made while reading the scramble off the screen.

    `frame_times` is the per-frame capture time, indexed the same way
    `load_color` is — so it is stamps[order], not stamps, once a slow capture
    has been resampled. analyse() uses it to carve classifier windows by time
    rather than by index arithmetic; see decode.window_from_anchor for what
    that is and is not worth.
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    for line in instructions:
        print(f"  {line}")

    status_fn = None
    if truth is not None:
        base = truth.count()
        status_fn = lambda: (f"BLE {truth.count() - base} moves"      # noqa: E731
                             if truth.connected else "BLE DISCONNECTED")

    buf, stamps = LD.capture(args.camera, overlay=overlay,
                             status_fn=status_fn, max_seconds=max_seconds,
                             cap_reason=cap_reason)
    if not buf or len(stamps) < 2:
        return None
    window = (float(stamps[0]), float(stamps[-1]))
    print()

    # Save before analysing: analysis is the part that can fail, and a take
    # that took a real solve to produce should not be lost to a bug in the
    # code that reads it.
    if save_as and args.save:
        save_take(Path(args.save) / save_as, buf, stamps, truth,
                  save_meta or {})

    fps = LD.report_capture_rate(stamps)
    order = np.arange(len(buf))
    if fps < 28 and not args.no_resample:
        order, fps = LD.resample_to_30fps(stamps)
        print(f"  resampling onto a 30fps timeline ({len(order)} frames)")
    import cv2
    return (lambda i: cv2.imdecode(buf[order[i]], cv2.IMREAD_COLOR),
            len(order), fps, window, stamps[order])


def run_live(args, tables):
    stack = (load_ctc_stack(args) if args.ctc else
             load_joint_stack(args) if args.joint else load_stack(args))
    detector, det_model, device, threshold, min_sep = stack
    truth = connect_truth(args)
    try:
        return _run_live(args, tables, stack, truth)
    finally:
        if truth is not None:
            truth.stop()


def _run_live(args, tables, stack, truth):
    detector, det_model, device, threshold, min_sep = stack
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Before anything is recorded, while adding a lamp still costs nothing.
    # This is a clock heuristic, not a measurement of the room: the
    # statistical out-of-distribution gates were built and measured and had
    # no power to separate a 56% take from a 90% one (see lighting_check.py).
    if not args.no_lighting_check:
        import lighting_check as LC
        note = LC.time_of_day_note()
        if note:
            print(f"\n{'='*70}\n{note}\n{'='*70}")
            if input("\n  Continue anyway? [Y/n] ").strip().lower() \
                    in ("n", "no"):
                sys.exit("Aborted — come back with more light.")
    if args.save:
        print(f"\n  Saving both takes under {args.save}/solve_{stamp}_*")

    if args.skip_scramble and not args.claim and args.seed is None:
        sys.exit("--skip-scramble asserts the cube is ALREADY in a known "
                 "state, so the\nscramble cannot be freshly random: pass "
                 "--claim \"R U R' ...\" with the word you\nactually "
                 "performed, or --seed N to regenerate the scramble a "
                 "previous run\nprinted.")
    if args.claim:
        scramble = args.claim.split()
        bad = [m for m in scramble if m not in RC.WCA12]
        if bad:
            sys.exit(f"--claim contains non-quarter-turn moves {bad}; the "
                     f"classifier's 12 classes are {' '.join(RC.WCA12)}")
    else:
        scramble = LD.generate_scramble(args.scramble, seed=args.seed)
    scramble_state = RC.seq_to_state(scramble)

    def analyse(src, label):
        load_color, n, fps, _window, ftimes = src
        print(f"\n  analysing {n} frames...")
        # `ftimes` is the real per-frame capture clock (resampled ordering
        # included). Every arm gets it: it is what each move's `time` field
        # is stamped from, and the coach's L1 metrics are differences of
        # those stamps. See decode.move_time for what nominal fps costs.
        if args.ctc:
            import joint_decode as JD
            res = JD.analyse_ctc_live(load_color, n, fps, detector,
                                      det_model, device,
                                      beam=args.ctc_beam, lm=args.lm,
                                      alpha=args.lm_alpha if args.lm else 0.0,
                                      beta=args.lm_beta if args.lm else 0.0,
                                      frame_times=ftimes)
        elif args.joint:
            import joint_decode as JD
            res = JD.analyse_joint_live(load_color, n, fps, detector,
                                        det_model, device, threshold,
                                        min_sep, frame_times=ftimes)
        else:
            res = LD.analyse(load_color, n, fps, detector, det_model, device,
                             threshold, min_sep, args.classifier,
                             frame_times=ftimes,
                             peak_threshold=args.candidate_threshold)
        if not res["moves"]:
            print(f"  No moves detected in the {label} — nothing to verify.")
            return None
        if res["class_names"] != RC.WCA12:
            sys.exit(f"classifier class order {res['class_names']} != "
                     f"{RC.WCA12} — refusing to decode")
        LD.print_sequence(res["moves"], fps)
        return res

    phase1 = None
    if not args.skip_scramble:
        # "Starting from a SOLVED cube" is an instruction everywhere else in
        # this file and an assumption everywhere downstream. With the cube
        # connected it is checkable, so check it — starting one move off
        # invalidates both phases and is invisible until the very end.
        if truth is not None:
            solved_now, wrong = truth.solved_report()
            if solved_now is False:
                print(f"\n  The cube reports it is NOT SOLVED right now "
                      f"({wrong} of 54 facelets off).")
                if wrong > 24:
                    print(f"  That is far more than any small misalignment: "
                          f"one quarter turn moves 6\n  facelets. If the cube "
                          f"in your hand LOOKS solved, believe your eyes — "
                          f"the\n  cube's own state tracking has drifted "
                          f"(a flat battery does this), and its\n  move log "
                          f"is equally untrustworthy. Charge it rather than "
                          f"proceeding.")
                else:
                    print(f"  Phase 1 measures a scramble applied to a solved "
                          f"cube, so this take would\n  be scored against the "
                          f"wrong start state.")
                if wrong > 24:
                    # Do not block on a reading we have just finished
                    # explaining is untrustworthy. One quarter turn moves 6
                    # facelets, so >24 is not a partly-finished solve; it is
                    # the cube's state tracking drifting, and aborting a
                    # good take on the strength of it is the wrong call.
                    # Warn, record, continue.
                    print(f"  Proceeding anyway — a reading this far off is "
                          f"the cube being wrong, not you.\n  The take is "
                          f"still valid; session_check.py will verify the "
                          f"move word afterwards.")
                elif input("  Solve it first, then press ENTER — or type "
                           "'go' to proceed anyway: ").strip().lower() != "go":
                    if truth.is_solved() is False:
                        sys.exit("Still not solved. Aborted.")
            elif solved_now is None:
                print(f"\n  (cube did not report its state; assuming it is "
                      f"solved as instructed)")

        print(f"\n{'='*70}")
        print(f"  PERFORM THIS SCRAMBLE — {len(scramble)} moves, "
              f"starting from a SOLVED cube")
        print(f"{'='*70}")
        for line in LD.format_scramble(scramble):
            print(f"    {line}")
        print(f"\n  Hold ONE orientation throughout. The classifier names "
              f"camera-relative\n  layers, so turning the whole cube "
              f"relabels every move after it.\n  Regrip without rotating."
              + ("" if not args.rotations else
                 "  (--rotations is on, so a rotation is recoverable — at "
                 "the cost of a wider search.)"))
        src = record_phase(args, "PHASE 1 of 2 — SCRAMBLE (measurement)",
                           ["SPACE to start recording, perform the scramble, "
                            "SPACE to stop."],
                           overlay=LD.format_scramble(scramble),
                           truth=truth,
                           save_as=f"solve_{stamp}_scramble",
                           save_meta={"phase": "scramble",
                                      "prescribed": " ".join(scramble)})
        if src is None:
            sys.exit("Nothing recorded.")
        res = analyse(src, "scramble")
        if res is None:
            sys.exit("Nothing to verify.")

        # The scramble that was PERFORMED, which is the one the cube is
        # actually in. Without BLE this is assumed equal to the prescribed
        # word, and that assumption is exactly what BLE is here to test.
        gt_word, mis_performed, from_cube = scramble, False, False
        if truth is not None:
            import ble_truth as BT
            performed = truth.words_between(*src[3])
            if not performed:
                print(f"\n  WARNING: the cube reported no moves during this "
                      f"take. Falling back to\n  the prescribed scramble as "
                      f"ground truth — check the connection.")
            else:
                cmp = BT.compare_words(scramble, performed)
                gt_word, mis_performed, from_cube = \
                    performed, not cmp["match"], True
                if mis_performed:
                    scramble_state = RC.seq_to_state(gt_word)

        LD.report_scramble(gt_word, res["moves"],
                           truth_label="cube-logged" if from_cube
                           else "prescribed")
        phase1 = verify_claim(
            "PHASE 1 CLAIM: solved cube -> the scramble that was performed"
            if mis_performed else
            "PHASE 1 CLAIM: solved cube -> the prescribed scramble",
            RC.SOLVED.copy(), scramble_state, res["moves"], threshold,
            args, tables, truth=gt_word)
        phase1["mis_performed"] = mis_performed
        phase1["truth_source"] = "ble" if from_cube else "prescribed"

        if (phase1.get("acc") or 0.0) < 0.9 or not phase1["res"]["solved"]:
            print(f"\n  CAUTION: the scramble did not read cleanly.")
            if truth is None:
                print(f"  If that is because it was performed differently "
                      f"from the printed one,\n  the cube is NOT in the state "
                      f"phase 2 is about to assert, and phase 2\n  will fail "
                      f"for a human reason. Re-run with --ble to rule that "
                      f"out.")
            else:
                print(f"  The cube confirmed which moves you performed, so "
                      f"this is the PIPELINE\n  misreading them, not a "
                      f"mis-performed scramble — phase 2's claimed start\n"
                      f"  state is correct regardless.")
            if input("\n  Continue to phase 2? [y/N] ").strip().lower() \
                    not in ("y", "yes"):
                return phase1, None, None
    else:
        print(f"\n  Assuming the cube is already scrambled by:")
        _print_word("claimed scramble", scramble)

    src = record_phase(args, "PHASE 2 of 2 — SOLVE (verification)",
                       ["Solve the cube. SPACE to start, SPACE when solved.",
                        "Same orientation rule as phase 1."],
                       truth=truth,
                       save_as=f"solve_{stamp}_solve",
                       save_meta={"phase": "solve",
                                  "claimed_start": " ".join(scramble)})
    if src is None:
        sys.exit("Nothing recorded.")
    res = analyse(src, "solve")
    if res is None:
        sys.exit("Nothing to verify.")

    # PHASE 3 — the post-timer verification scan. Not a third measurement:
    # it is the window the anticheat gate's two strongest arms live in.
    # Without it `observed_moves_after_stop` and `post_stop_continuity` are
    # both None, and adjudicate() then has nothing standing between it and
    # the two attacks that happen AFTER the clock stops (solve the cube once
    # the timer is off; make real moves without solving and swap a solved
    # cube in for the scan).
    scan_src, scan_res = None, None
    if not args.no_anticheat:
        scan_src = record_phase(
            args, "PHASE 3 — POST-TIMER SCAN (anticheat)",
            ["Present every face of the SOLVED cube to the camera.",
             "SPACE to start, SPACE when done. Do not turn the cube.",
             f"Hard cap {AG_POST_STOP_MAX_WINDOW_S:.0f}s — past that the "
             f"phantom allowance could hide a whole solve",
             "and the test provably has no power, so the gate abstains."],
            truth=truth, save_as=f"solve_{stamp}_scan",
            save_meta={"phase": "scan"},
            max_seconds=AG_POST_STOP_MAX_WINDOW_S,
            cap_reason="post-stop test loses power past this")
        if scan_src is None:
            print(f"\n  No scan recorded — the anticheat gate will ABSTAIN on "
                  f"the post-stop\n  window rather than pass it. That is the "
                  f"safe direction, not a free pass.")
        else:
            scan_res = analyse(scan_src, "post-timer scan")

    # Phase 2 is the real use case precisely BECAUSE it has no ground truth
    # — a solve is not a prescribed word and nobody remembers 60 moves
    # afterwards. The cube remembers. That does not weaken the claim (the
    # verdict is still decided from frames alone) but it is the only way to
    # see WHERE a failed verification went wrong.
    solve_word = None
    if truth is not None:
        solve_word = truth.words_between(*src[3]) or None
        if solve_word is None:
            print(f"\n  WARNING: the cube reported no moves during the solve.")
        else:
            print(f"\n  Ground truth for the solve: {len(solve_word)} moves "
                  f"from the cube.")
            LD.report_scramble(solve_word, res["moves"],
                               truth_label="cube-logged",
                               title="SOLVE ACCURACY")

        # An independent end-state check. The decode can only ever say "a
        # consistent story exists"; the cube can say whether the thing in
        # your hand is actually solved, and the two disagreeing is the most
        # informative failure this test can produce.
        ended_solved, wrong = truth.solved_report()
        if ended_solved is False:
            print(f"\n  The cube reports it is NOT SOLVED at the end of "
                  f"phase 2 ({wrong} of 54\n  facelets off). Phase 2 claims "
                  f"it reached the solved state, so a VERIFIED\n  verdict "
                  f"below would be wrong on the facts.")
            if wrong > 24:
                print(f"  But {wrong} facelets is too many to be a partly "
                      f"finished solve — if the cube\n  looks solved, this "
                      f"is the cube's state tracking drifting, not your "
                      f"solve.")
        elif ended_solved:
            print(f"  The cube confirms it is solved.")
        truth.warn_if_unhealthy()

    phase2 = verify_claim(
        "PHASE 2 CLAIM: the scrambled cube -> solved",
        scramble_state, RC.SOLVED.copy(), res["moves"], threshold,
        args, tables, truth=solve_word)
    phase2["ble_truth"] = solve_word is not None
    # No point pricing decoys against a claim that did not verify: there is
    # no baseline cost to compare them to.
    sweep = None
    if not args.no_sweep and phase2["res"]["solved"]:
        sweep = falsifiability_sweep(scramble_state, RC.SOLVED.copy(),
                                     phase2["cost_rows"], phase2["del_costs"],
                                     args, tables, seed=args.seed or 0,
                                     beam=phase2["res"]["beam_used"])

    # The anticheat gate, on the same frames. Run LAST because it is the
    # expensive part (a YOLO pass per window for the continuity guard, plus
    # the solved-at-stop check) and because a failure here must not cost the
    # reconstruction result that has already been printed.
    if not args.no_anticheat:
        try:
            phase2["anticheat"] = adjudicate_take(
                args, src, scan_src, res["moves"],
                # A scan that decoded NO moves is the good outcome, not a
                # missing measurement — analyse() returns None for an empty
                # decode, and confusing that with "not analysed" would turn
                # the cleanest possible scan into an abstention.
                (scan_res or {}).get("moves", []),
                detector=stack[0],
                lighting_ok=_lighting_ok(res))
            print_anticheat(phase2["anticheat"])
        except Exception as exc:                            # noqa: BLE001
            print(f"\n  anticheat gate failed to run: "
                  f"{type(exc).__name__}: {exc}")
            print(f"  The reconstruction verdict above is unaffected.")
    return phase1, phase2, sweep


def _session_src(d: Path, lo: int, hi: int):
    """A recorded session's frames [lo, hi) as a record_phase-shaped source.

    Exists so the anticheat path can be exercised WITHOUT a camera. Same
    5-tuple record_phase returns — (load_color, n, fps, window, frame_times)
    — so `adjudicate_take` cannot tell the difference, which is the point:
    a rehearsal that ran through a parallel code path would prove nothing
    about the live one.
    """
    import cv2

    recs = [json.loads(l) for l in open(d / "frames.jsonl") if l.strip()]
    recs = [r for r in recs if (d / "frames" / r["file"]).exists()]
    recs = recs[lo:hi]
    if len(recs) < 2:
        return None
    ts = np.array([r["ts"] for r in recs], dtype=np.float64)
    fps = (len(recs) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 30.0
    paths = [str(d / "frames" / r["file"]) for r in recs]
    return (lambda i: cv2.imread(paths[i]), len(recs), fps,
            (float(ts[0]), float(ts[-1])), ts)


def run_anticheat_session(args, _tables):
    """
    Replay a recorded session through the anticheat gate — no camera.

    The session is SPLIT into the two windows the live path records
    separately: everything up to a guard interval after the last true BLE
    onset is the timed solve, and the move-free tail after it stands in for
    the post-timer scan. That is the same substitution
    `anticheat_gate.py calibrate-poststop` makes, and it is honest for this
    purpose — the tail genuinely is "the cube being held in front of the
    camera with no moves in it".

    WHAT IT DOES AND DOES NOT ESTABLISH. It exercises the whole wiring:
    both continuity guards, solved-at-stop, the lighting probe, the count,
    and adjudicate() itself, on real frames. It does NOT measure the gate —
    `anticheat_gate.py score` does that over the corpus. This is the
    check you run before spending a live take, so that a broken import or a
    mis-shaped evidence field is found here rather than with a cube in your
    hand.
    """
    d = Path(args.anticheat_session)
    if not (d / "frames.jsonl").is_file():
        sys.exit(f"{d} has no frames.jsonl")

    npz = d / "detector_stream_color.npz"
    n_all = sum(1 for l in open(d / "frames.jsonl") if l.strip())
    if npz.is_file():
        z = np.load(npz, allow_pickle=True)
        onsets = z["onset_idx"].astype(int)
        fps = float(z["fps"])
        split = (int(onsets.max()) + int(round(args.guard_s * fps))
                 if onsets.size else n_all // 2)
    else:
        print(f"  no prepared stream — splitting at "
              f"{100 * (1 - args.tail_frac):.0f}% instead of at the last "
              f"true onset")
        split = int(n_all * (1 - args.tail_frac))
    split = int(np.clip(split, 8, n_all - 2))

    print(f"\n{'=' * 70}")
    print(f"  ANTICHEAT REHEARSAL — {d.name}")
    print(f"{'=' * 70}")
    print(f"  {n_all} frames; timed window [0,{split}), "
          f"post-stop window [{split},{n_all})")

    solve_src = _session_src(d, 0, split)
    scan_src = _session_src(d, split, n_all)
    if solve_src is None:
        sys.exit("Not enough frames in the timed window.")

    stack = (load_ctc_stack(args) if args.ctc else
             load_joint_stack(args) if args.joint else load_stack(args))
    detector, det_model, device, threshold, min_sep = stack

    def analyse_window(src, label):
        if src is None:
            return None
        load_color, n, fps, _w, ftimes = src
        print(f"\n  decoding the {label} ({n} frames)...")
        if args.ctc:
            import joint_decode as JD
            r = JD.analyse_ctc_live(load_color, n, fps, detector, det_model,
                                    device, beam=args.ctc_beam,
                                    frame_times=ftimes)
        elif args.joint:
            import joint_decode as JD
            r = JD.analyse_joint_live(load_color, n, fps, detector, det_model,
                                      device, threshold, min_sep,
                                      frame_times=ftimes)
        else:
            r = LD.analyse(load_color, n, fps, detector, det_model, device,
                           threshold, min_sep, args.classifier,
                           frame_times=ftimes)
        return r

    solve_res = analyse_window(solve_src, "timed window")
    scan_res = analyse_window(scan_src, "post-stop window")
    v = adjudicate_take(args, solve_src, scan_src,
                        (solve_res or {}).get("moves", []),
                        (scan_res or {}).get("moves", []),
                        detector=detector, lighting_ok=_lighting_ok(solve_res))
    print_anticheat(v)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # default=str, because solved_check.solved_at returns numpy scalars
        # (np.bool_, np.float32) that json refuses. Coercing them here rather
        # than in solved_check keeps that module free to stay numeric.
        Path(args.out).write_text(json.dumps(v, indent=2, default=str))
        print(f"\n  -> {args.out}")
    return v


def _lighting_ok(res):
    """True/False/None for the take's lighting, from the ANALYSED result.

    Takes `res` — what analyse() returned — rather than the raw capture, so
    the statistics are computed on `stream_frames`: the cropped 96x96 block
    the model itself was scored on. That is not a convenience, it is the
    only correct input; lighting_check's reference is aggregated over
    cropped streams, and feeding it raw camera frames makes every take read
    as out of distribution (see lighting_check.assess for the measurement).

    None means NOT ASSESSED and is deliberately harmless — adjudicate()
    abstains only on an explicit False, so neither a missing reference nor
    the default detector+classifier arm (which builds a GRAYSCALE stream and
    so has no colour block to assess) can push every take into review. Only
    `False` costs anything, and it costs an abstention rather than a
    rejection, because low light destroys the ONSET DETECTOR (measured: 40+
    points, against ~5 for the classifier) and the move count is exactly the
    detector's output.
    """
    from lighting_check import assess
    return assess((res or {}).get("stream_frames"))


# ---------------------------------------------------------------------------
# Rehearsal mode — a recorded session, BLE moves as ground truth
# ---------------------------------------------------------------------------

def run_session(args, tables):
    dirs = [Path(p) for pattern in args.session
            for p in (Path(".").glob(pattern) if "*" in pattern
                      else [Path(pattern)])]
    dirs = sorted(d for d in dirs if d.is_dir())
    if not dirs:
        sys.exit("No session directories matched --session.")

    results = []
    for d in dirs:
        gt = [m.get("wca_notation")
              for m in (json.loads(l) for l in open(d / "moves.jsonl")
                        if l.strip())]
        if not gt or any(g is None for g in gt):
            print(f"\n  {d.name}: no wca_notation — skipping")
            continue
        replay = RC._load_replay(d, args)
        if replay is None or not replay["moves"]:
            continue
        if replay["class_names"] != RC.WCA12:
            sys.exit(f"classifier class order {replay['class_names']} != "
                     f"{RC.WCA12} — refusing to decode")

        # The BLE move list defines the state the cube must have started in
        # for this video to be a solve — the stand-in for a scanned scramble.
        start = RC.start_from_gt(gt)
        threshold = replay["meta"].get("threshold", 0.5)
        is_unseen = RC.classifier_unseen(d.name, args.classifier)
        unseen = " [classifier-unseen]" if is_unseen \
            else " [classifier trained on this session]"
        out = verify_claim(f"{d.name}{unseen}: scrambled cube -> solved",
                           start, RC.SOLVED.copy(), replay["moves"],
                           threshold, args, tables, truth=gt)
        out["session"] = d.name
        out["unseen"] = is_unseen
        if out["res"]["solved"] and not args.no_sweep:
            out["sweep"] = falsifiability_sweep(
                start, RC.SOLVED.copy(), out["cost_rows"], out["del_costs"],
                args, tables, seed=0, beam=out["res"]["beam_used"])
        results.append(out)

    _session_summary(results)
    return results


def _session_summary(rows):
    if len(rows) < 2:
        return
    print(f"\n{'='*70}")
    for subset, label in ((rows, "ALL SESSIONS"),
                          ([r for r in rows if r["unseen"]],
                           "CLASSIFIER-UNSEEN ONLY (the honest number)")):
        if not subset:
            continue
        n = len(subset)
        ver = sum(r["res"]["solved"] for r in subset)
        exact = sum(r.get("exact", False) for r in subset)
        raw = np.mean([r["raw_acc"] for r in subset])
        # When verification fails the system falls back to the raw
        # sequence, so that is the accuracy the pipeline actually delivers.
        eff = np.mean([r["acc"] if r["res"]["solved"] else r["raw_acc"]
                       for r in subset])
        swept = [r for r in subset if r.get("sweep")]
        sound = [r for r in swept
                 if not any(s["verified"] and s["distance"] not in (0, None)
                            and s["cost"] < r["res"]["cost"] - 1e-6
                            for s in r["sweep"])]
        print(f"  {label}  ({n} sessions)")
        print(f"    verified:             {ver}/{n}")
        print(f"    exact reconstruction: {exact}/{n}")
        print(f"    per-move accuracy:    raw {raw*100:.1f}%  ->  system "
              f"{eff*100:.1f}%  ({(eff-raw)*100:+.1f} pts)")
        print(f"    true claim cheapest:  "
              + (f"{len(sound)}/{len(swept)} swept (falsifiability)"
                 if swept else "not measured (--no-sweep)"))
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def print_verdict(phase1, phase2, sweep, args):
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    if phase1:
        r = phase1
        dec = ("not verified" if r.get("acc") is None
               else f"{r['acc']*100:5.1f}% per move"
                    f"{'   (exact)' if r.get('exact') else ''}")
        src = ("cube-confirmed" if r.get("truth_source") == "ble"
               else "assumed performed correctly")
        print(f"  Measurement (scramble, {r['n_onsets']} onsets, {src}):")
        print(f"    raw pipeline    {(r.get('raw_acc') or 0.0)*100:5.1f}% "
              f"per move")
        print(f"    after decode    {dec}")
        if r.get("mis_performed"):
            print(f"    NOTE: the scramble was performed differently from "
                  f"the printed one.\n          Everything above is scored "
                  f"against what you actually did, and\n          phase 2 "
                  f"was verified against the resulting real state.")
    if phase2 is None:
        print(f"  Verification: not run.")
        return
    verified = phase2["res"]["solved"]
    print(f"\n  Verification (free solve, {phase2['n_onsets']} onsets): "
          f"{'VERIFIED SOLVED' if verified else 'NOT VERIFIED'}")

    # With BLE the free solve carries per-move numbers too, which is the
    # only place in this pipeline they can come from — and the only way to
    # tell a decode that failed from a read that was never good enough.
    if phase2.get("ble_truth"):
        dec = ("not verified" if phase2.get("acc") is None
               else f"{phase2['acc']*100:5.1f}% per move"
                    f"{'   (exact)' if phase2.get('exact') else ''}")
        print(f"    vs the cube's own move log:")
        print(f"      raw pipeline  {(phase2.get('raw_acc') or 0.0)*100:5.1f}%"
              f" per move")
        print(f"      after decode  {dec}")

    if verified and sweep:
        cheaper = [s for s in sweep if s["verified"] and s["distance"] != 0
                   and s["cost"] is not None
                   and s["cost"] < phase2["res"]["cost"] - 1e-6]
        sep = min((s["distance"] for s in sweep
                   if s["distance"] and not s["verified"]), default=None)
        if cheaper:
            print(f"    ...but a wrong claim was cheaper. Treat this as "
                  f"UNVERIFIED.")
        elif sep is None:
            print(f"    ...but every decoy was accepted too, so the verdict "
                  f"is uninformative.")
        else:
            print(f"    The true claim was the cheapest explanation, and "
                  f"claims >= {sep} quarter\n    turns away were rejected — "
                  f"that is the strength of this result.")

    # Where the remaining error lives, so the next move is not a guess.
    if phase1 and phase1.get("raw_acc") is not None:
        raw = phase1
        subs = raw.get("raw_errors", 0)
        if (raw.get("acc") or 0.0) < 0.98:
            # One model on --ctc/--joint, so "the classifier" is not a
            # component that exists to blame; the actionable half of the
            # sentence (training data from this room, not a wider beam) is
            # the same either way.
            where = ("the MODEL" if (args.ctc or args.joint)
                     else "the CLASSIFIER")
            print(f"\n  Remaining error is dominated by {where} "
                  f"({subs} raw errors on\n  a {raw['n_onsets']}-onset "
                  f"scramble). The decode repairs a handful; past ~6 mixed\n"
                  f"  errors it is out of envelope by construction, so the "
                  f"fix is training data\n  from THIS environment, not a "
                  f"wider --beam.")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="End-to-end verification: does the video show this cube "
                    "being solved?")
    p.add_argument("--session", nargs="+",
                   help="Rehearse offline on recorded session(s) using BLE "
                        "ground truth instead of the webcam (globs ok)")
    p.add_argument("--detector", type=str, default=LD.DETECTOR_PATH)
    p.add_argument("--classifier", type=str, default=LD.CLASSIFIER_PATH)
    p.add_argument("--joint", action="store_true",
                   help="Use Stage A's joint onset+class model "
                        "(MODEL_REWORK_PLAN.md) instead of the deployed "
                        "detector+classifier pair. Prototype, offline-"
                        "measured only so far — see MODEL_REWORK_PLAN.md "
                        "for what that does and does not cover.")
    p.add_argument("--joint-model", type=str, default="checkpoints/move_joint_seed0.pt",
                   help="Checkpoint to use with --joint")
    p.add_argument("--ctc", action="store_true",
                   help="Use the CTC arm (train_ctc.py): same trunk as "
                        "--joint, but move existence is decided inside a "
                        "prefix beam search over frames instead of by "
                        "peak-picking a threshold. Measured on the held-out "
                        "set, both seeds: MER 8.5%% -> 5.8%%, phantoms -64%%, "
                        "verified 1/8 -> 3/8.")
    p.add_argument("--ctc-model", type=str,
                   default="checkpoints/move_ctc_spd_s0.pt",
                   help="Checkpoint to use with --ctc. Defaults to the "
                        "SPEED-AUGMENTED model (2026-08-05) — see the "
                        "docstring; the anticheat gate's retention constants "
                        "are calibrated on this checkpoint, so changing it "
                        "here without re-running speed_sim.py desynchronises "
                        "the two.")
    p.add_argument("--ctc-beam", type=int, default=16,
                   help="Width of the CTC prefix beam search over frames. "
                        "Unrelated to --beam, which is the reconstruct.py "
                        "cube-state search.")
    p.add_argument("--lm", action="store_true", dest="lm_fusion",
                   help="--ctc only: fuse move_lm.py's n-gram move prior "
                        "into the prefix beam search. REGIME-DEPENDENT — "
                        "measured as a gain cross-day and a loss on the "
                        "same-day holdout, on both seeds. A live take is "
                        "cross-day, so it should help; that is a prediction, "
                        "not a measurement.")
    p.add_argument("--lm-order", type=int, default=4)
    p.add_argument("--lm-alpha", type=float, default=0.9,
                   help="LM weight (tune_lm_fusion.py chose this on the "
                        "unseen 07-29/30 dev sessions)")
    p.add_argument("--lm-beta", type=float, default=4.0,
                   help="Per-symbol insertion bonus; without it fusion "
                        "trades phantoms for misses instead of reducing "
                        "error")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--scramble", type=int, default=20,
                   help="Length of the prescribed scramble (default 20)")
    p.add_argument("--claim", type=str, default=None,
                   help="Use this exact scramble word instead of generating "
                        "one, e.g. \"R U R' F\"")
    p.add_argument("--skip-scramble", action="store_true",
                   help="Skip phase 1 — the cube is already scrambled by "
                        "--claim (or by the seeded scramble)")
    p.add_argument("--seed", type=int, default=None)
    # Smart cube as a ground-truth LABEL — never an input to the pipeline.
    p.add_argument("--ble", action="store_true",
                   help="Log ground truth from the smart cube alongside the "
                        "webcam: says which move was misread, and whether "
                        "the scramble was performed correctly")
    p.add_argument("--front", type=str, default=None,
                   help="Colour facing the camera (--ble), e.g. blue")
    p.add_argument("--top", type=str, default=None,
                   help="Colour facing up (--ble), e.g. yellow")
    p.add_argument("--address", type=str, default=None,
                   help="Cube BLE address; omit to auto-scan")
    p.add_argument("--no-echo", action="store_true",
                   help="Do not print BLE moves as they arrive")
    p.add_argument("--save", nargs="?", const="../training_data",
                   default=None, metavar="DIR",
                   help="Keep both takes as session directories (default "
                        "../training_data). With --ble they are training "
                        "data for the environment you just measured; "
                        "without it, detector data only")
    p.add_argument("--threshold", type=float, default=None,
                   help="Onset threshold (default: the checkpoint's, tuned "
                        "on held-out data)")
    p.add_argument("--min-sep", type=int, default=None, dest="min_sep")
    p.add_argument("--no-resample", action="store_true")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Re-run the replay in --session mode")
    p.add_argument("--no-sweep", action="store_true",
                   help="Skip the falsifiability decoys (they cost one "
                        "decode each)")
    p.add_argument("--no-lighting-check", action="store_true",
                   help="Skip the pre-take lighting reminder "
                        "(lighting_check.py)")
    p.add_argument("--anticheat-session", default=None, metavar="DIR",
                   help="Rehearse the ANTICHEAT gate on a recorded session, "
                        "no camera: the session is split into a timed window "
                        "and a move-free tail standing in for the post-timer "
                        "scan, and adjudicate() runs on the result. Use it to "
                        "check the wiring before spending a live take. Not a "
                        "measurement of the gate — anticheat_gate.py score "
                        "is.")
    p.add_argument("--guard-s", type=float, default=1.0,
                   help="--anticheat-session: seconds after the last true "
                        "onset before the post-stop window starts")
    p.add_argument("--tail-frac", type=float, default=0.15,
                   help="--anticheat-session: fraction of the session used "
                        "as the post-stop window when there is no prepared "
                        "stream to find the last onset in")
    p.add_argument("--out", default=None,
                   help="--anticheat-session: write the verdict here")
    p.add_argument("--no-anticheat", action="store_true",
                   help="Skip phase 3 (the post-timer scan) and the "
                        "anticheat_gate.adjudicate() verdict. The gate costs "
                        "a YOLO pass over each window for the continuity "
                        "guard, so this is the flag for a quick "
                        "reconstruction-only take — NOT a way to make a "
                        "take pass, since a missing scan window makes the "
                        "gate abstain rather than approve.")
    # Decoder knobs — same defaults and meanings as reconstruct.py.
    p.add_argument("--beam", type=int, default=RC.BEAM)
    p.add_argument("--retry-beam", type=int, default=4 * RC.BEAM)
    p.add_argument("--slices", action="store_true",
                   help="Let one onset decode as a middle-slice turn (the "
                        "smart cube reports a slice as two same-timestamp "
                        "face events, but the camera sees one motion). Off "
                        "by default = exact prior behaviour. Measured "
                        "+2/42 verified sessions with no falsifiability "
                        "cost — see GROUND_TRUTH_ARTIFACTS.md.")
    p.add_argument("--c-slice", type=float, default=RC.C_SLICE,
                   dest="c_slice",
                   help="Cost of a slice reading, on top of the onset's own "
                        "acceptance cost (default %(default)s)")
    p.add_argument("--slice-gate", type=float, default=RC.SLICE_GATE,
                   dest="slice_gate",
                   help="Minimum posterior mass on both halves of a pair "
                        "before an onset may be read as a slice "
                        "(default %(default)s)")
    p.add_argument("--del-cost", type=float, default=RC.C_DEL)
    p.add_argument("--ins-cost", type=float, default=RC.C_INS)
    p.add_argument("--rot-cost", type=float, default=RC.C_ROT)
    p.add_argument("--max-end-ins", type=int, default=RC.MAX_END_INS,
                   help="End-of-solve moves the camera never saw. Keep it "
                        "SMALL: it is also the radius inside which a false "
                        "claim is indistinguishable from a true one.")
    p.add_argument("--candidate-threshold", type=float, default=None,
                   help="D1 soft-onset lattice: peak-pick down to this "
                        "threshold and feed weak candidates to the "
                        "decoder near-free to delete (see reconstruct.py "
                        "--help). Default None: unchanged behaviour.")
    p.add_argument("--del-floor", type=float, default=RC.DEL_FLOOR)
    p.add_argument("--blend-inv", type=float, default=RC.BLEND_INV)
    p.add_argument("--blend-unif", type=float, default=RC.BLEND_UNIF)
    p.add_argument("--blend-adj", type=float, default=RC.BLEND_ADJ,
                   help="D2 confusion-calibrated prior (see "
                        "reconstruct.py --confusion to measure a value)")
    p.add_argument("--rel-weight", type=float, default=RC.REL_WEIGHT)
    p.add_argument("--rotations", action="store_true",
                   help="Allow whole-cube rotations as search state")
    p.add_argument("--bidir", action="store_true",
                   help="D3 bidirectional meet-in-the-middle decode "
                        "(PATH_TO_VERIFICATION.md §5) instead of the "
                        "single-pass decoder. Not compatible with "
                        "--rotations.")
    p.add_argument("--meet", type=int, default=None,
                   help="--bidir: onset index to meet at (default n//2)")
    p.add_argument("--meet-sweep", action="store_true",
                   help="--bidir: try meet points at n/3, n/2, 2n/3 and "
                        "keep the cheapest solved result")
    args = p.parse_args()

    if args.joint and args.ctc:
        sys.exit("--joint and --ctc are two different decoders for two "
                 "different checkpoints;\npick one. compare_onset_arms.py is "
                 "the tool for scoring them against each other.")
    if (args.joint or args.ctc) and args.session:
        arm = "--ctc" if args.ctc else "--joint"
        sys.exit(f"{arm} --session isn't wired — run_session() still "
                 f"replays through the deployed detector+classifier. Use "
                 f"verify_joint.py (peak-picked) or compare_onset_arms.py "
                 f"(both arms, incl. CTC)\nfor offline/recorded-session "
                 f"evaluation; {arm} here is for live capture only.")
    if args.lm_fusion and not args.ctc:
        sys.exit("--lm fuses a prior into the CTC prefix beam search, so it "
                 "only means anything\nwith --ctc. The peak-picking arms have "
                 "no beam over frames to fuse into.")
    args.lm = None                      # set by load_ctc_stack when --lm

    if args.anticheat_session and args.session:
        sys.exit("--anticheat-session and --session are two different "
                 "rehearsals;\npick one. --session replays the "
                 "RECONSTRUCTION, --anticheat-session the GATE.")

    if args.anticheat_session:
        # No pruning tables needed: the gate never decodes a cube state.
        run_anticheat_session(args, None)
        return

    tables = RC.build_tables()          # before any prompting, not mid-take
    if args.session:
        run_session(args, tables)
    else:
        print_verdict(*run_live(args, tables), args)


if __name__ == "__main__":
    main()
