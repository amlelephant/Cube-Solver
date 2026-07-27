"""
verify_solve.py

The whole pipeline, end to end, stated as the product claim:

    "This person was handed a cube in THIS state, and this video shows
     them solving it" — decided from webcam frames alone.

Everything upstream measures a component. This measures the claim:

    detector      when a move happened          (move_detector.pt)
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

import numpy as np  # noqa: E402

import live_detect as LD  # noqa: E402
import reconstruct as RC  # noqa: E402
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
    kw = dict(c_del=args.del_cost, c_ins=args.ins_cost, c_rot=args.rot_cost,
              max_end_ins=args.max_end_ins, rel_weight=args.rel_weight,
              del_costs=del_costs, rotations=args.rotations, tables=tables)
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
        with open(out_dir / "ble_meta.json", "w") as fh:
            json.dump({"end_state": truth.snapshot_state(),
                       "end_solved": solved, "end_facelets_wrong": wrong,
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
                 save_as: str | None = None, save_meta: dict | None = None):
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
                             status_fn=status_fn)
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
    stack = load_stack(args)
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
                if input("  Solve it first, then press ENTER — or type 'go' "
                         "to proceed anyway: ").strip().lower() != "go":
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
    return phase1, phase2, sweep


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
            print(f"\n  Remaining error is dominated by the CLASSIFIER "
                  f"({subs} raw errors on\n  a {raw['n_onsets']}-onset "
                  f"scramble). The decode repairs a handful; past ~6 mixed\n"
                  f"  errors it is out of envelope by construction, so the "
                  f"fix is classifier\n  training data from THIS "
                  f"environment, not a wider --beam.")


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
    # Decoder knobs — same defaults and meanings as reconstruct.py.
    p.add_argument("--beam", type=int, default=RC.BEAM)
    p.add_argument("--retry-beam", type=int, default=4 * RC.BEAM)
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

    tables = RC.build_tables()          # before any prompting, not mid-take
    if args.session:
        run_session(args, tables)
    else:
        print_verdict(*run_live(args, tables), args)


if __name__ == "__main__":
    main()
