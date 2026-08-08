"""
validate_and_promote.py — gate a candidate detector checkpoint against real
recorded sessions before it ever becomes the deployed detect_full_cube.pt.

Motivated by 2026-08-03: a hard-negative fine-tune looked "good" by its own
training-run val metrics (mAP50 0.808, precision 0.828) but, when actually
tested against the 69 real ble/training_data sessions via
batch_test_guard.py, had QUADRUPLED the false-DQ rate (10.1% -> 40.6%,
almost entirely from presence_gap — the detector losing the real cube far
more often). The training run's own small, noisy val split (200 images, 5
of them the new hard negatives) could not see that regression. This script
makes the real-session batch test a hard gate, not a manual afterthought:
swap in the candidate, run it, compare to a baseline batch result, and only
copy the candidate over the deployed weights if it's a strict improvement.

Promotion rule (deliberately conservative): total false-DQ count must not
increase, presence_gap count must not increase (that's what caught the
2026-08-03 regression), and two_cubes count must not increase either (a
retrain aimed at fixing two_cubes that makes it worse has failed at its one
job even if other reasons improve).

Run from inside cv/detection/ (bare model filenames — see CLAUDE.md):
    python validate_and_promote.py --candidate runs/detect/detect_hardneg2/weights/best.pt \\
        --baseline guard_batch_full_v2.json --root ../../ble/training_data
"""

import argparse
import hashlib
import json
import os
import shutil
import time

DEPLOYED = "detect_full_cube.pt"


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _summarize(results):
    n = len(results)
    dq = {k: v for k, v in results.items() if v["verdict"] == "dq"}
    reasons = {}
    for v in dq.values():
        for r in v["dq_reasons"]:
            reasons[r] = reasons.get(r, 0) + 1
    return {"n": n, "n_dq": len(dq), "reasons": reasons}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, help="path to candidate .pt")
    ap.add_argument("--baseline", required=True,
                     help="batch_test_guard.py --out JSON from the currently-deployed weights")
    ap.add_argument("--root", required=True, help="ble/training_data (or similar) session root")
    args = ap.parse_args()

    import batch_test_guard  # local import: needs to run after cwd/model setup

    with open(args.baseline) as f:
        baseline = _summarize(json.load(f))
    print(f"baseline: {baseline['n_dq']}/{baseline['n']} false-DQ, reasons={baseline['reasons']}")

    backup = f"{DEPLOYED}.pre_{time.strftime('%Y%m%d_%H%M%S')}.bak"
    shutil.copy2(DEPLOYED, backup)
    print(f"backed up deployed weights -> {backup}")
    shutil.copy2(args.candidate, DEPLOYED)
    print(f"swapped in candidate: {args.candidate}")

    sessions = sorted(
        d for d in os.listdir(args.root)
        if os.path.isfile(os.path.join(args.root, d, "frames.jsonl"))
        and os.path.isdir(os.path.join(args.root, d, "frames"))
    )
    results = {}
    for i, name in enumerate(sessions):
        rep = batch_test_guard.run_session(os.path.join(args.root, name))
        if rep is not None:
            results[name] = rep
        print(f"[{i + 1}/{len(sessions)}] {name}: "
              f"{'PASS' if rep and rep['verdict'] == 'pass' else '*** DQ ***'}")

    candidate_out = f"guard_batch_candidate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(candidate_out, "w") as f:
        json.dump(results, f, indent=2)

    cand = _summarize(results)
    print(f"\ncandidate: {cand['n_dq']}/{cand['n']} false-DQ, reasons={cand['reasons']}")
    print(f"full report -> {candidate_out}")

    base_r = baseline["reasons"]
    cand_r = cand["reasons"]
    ok = (cand["n_dq"] <= baseline["n_dq"]
          and cand_r.get("presence_gap", 0) <= base_r.get("presence_gap", 0)
          and cand_r.get("two_cubes", 0) <= base_r.get("two_cubes", 0))

    if ok:
        print("\nPROMOTED: candidate is not worse on any gated metric. "
              f"Deployed weights updated; previous weights backed up at {backup}.")
    else:
        shutil.copy2(backup, DEPLOYED)
        assert _md5(DEPLOYED) == _md5(backup)
        print(f"\nROLLED BACK: candidate regressed vs baseline "
              f"(total {cand['n_dq']} vs {baseline['n_dq']}, "
              f"presence_gap {cand_r.get('presence_gap', 0)} vs {base_r.get('presence_gap', 0)}, "
              f"two_cubes {cand_r.get('two_cubes', 0)} vs {base_r.get('two_cubes', 0)}). "
              "Deployed weights restored to the pre-candidate backup.")


if __name__ == "__main__":
    main()
