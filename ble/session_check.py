"""
session_check.py

Ground-truth endpoint consistency gate for recorded BLE sessions.

What it checks and why
----------------------
`OrientationTracker`'s face map is frozen at `calibrate()`. Its rotation
handling — `set_orientation_from_imu`, `notify_reoriented`,
`FaceMap.apply_whole_rotation`, `FaceMap.apply_face_rotation` — is defined
but never called anywhere in the repo, so a session that contains a
whole-cube rotation or a two-layer (wide) turn silently mislabels every
move after it. Nothing downstream notices: the classifier just trains on
wrong labels and the decoder just fails.

This is the cheap detector for that entire class of corruption. A
scramble/solve session PAIR has a known endpoint — the cube started solved
and ended solved — so applying the scramble's ground truth and then the
solve's ground truth to a solved cube must return it to solved. Any
untracked reorientation breaks that product, because the mislabelled moves
name the wrong faces.

Measured 2026-07-30: 16 of 16 pairs pass, so this corpus contains no wide
moves and no rotations. Which also means the check is currently a
regression gate rather than a bug hunt — run it on NEW sessions, where it
is the only thing standing between an untracked rotation and a silently
poisoned training set.

What this check is BLIND to, and why that mattered
--------------------------------------------------
An earlier version of this docstring claimed the passing checks
"incidentally validate the slice-move encoding". They validate half of it,
and the half they do not validate is where a real bug was hiding until
2026-08-03.

A middle-slice `M` is reported by the cube as `R` + `L'` because the core
rotates with the slice. That pair is a state-correct description of `M`
RELATIVE TO THE CENTRES, and these endpoint checks are genuine evidence for
that much. But `check_pair` applies the labels to a `CubieCube`, whose
model has no centres at all — it is centre-relative by construction. So it
cannot see that the slice also moved the centres, and therefore rotated the
cube's reporting frame away from the camera. Every label after a session's
first slice named the wrong face for the camera, on 7 of 76 sessions, while
this gate reported 16 of 16 pairs consistent. It was right and irrelevant
at the same time.

The camera-frame name now lives in `camera_notation` (see
`ble/slice_frame.py` and `backfill_camera_frame.py`); `wca_notation` stays
cube-frame, which is what this check needs and must keep getting. The slice
report below exists so the gate stops being silent about the axis it cannot
test.

Limits
------
Needs a scramble/solve PAIR. A lone solve session has no known start state,
so its endpoint is not checkable this way and it is reported as `skip` —
that is a gap in coverage, not a pass. The 23 pre-2026-07-23 sessions are
all in this category.

And it remains a CUBE-frame check. It will pass a session whose
`camera_notation` is missing or stale just as happily as one where it is
right, because it never reads that field. The `slices` column is the only
thing here that speaks to the camera frame.

Usage:
    python session_check.py --sessions training_data/solve_*/
    python session_check.py --sessions training_data/solve_20260729_221809_solve
"""

import argparse
import json
import sys
from pathlib import Path

# The vendored pure-Python solver supplies the cubie-level group model.
# Same sys.path bootstrap pattern the cv/ cross-topic scripts use.
_SOLVER_DIR = Path(__file__).resolve().parents[1] / "cv" / "solver"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

from twophase.cubes.cubiecube import CubieCube, MOVE_CUBE  # noqa: E402

# twophase face indexing (pieces.Color): U R F D L B
_TP_FACE = {"U": 0, "R": 1, "F": 2, "D": 3, "L": 4, "B": 5}

SCRAMBLE_SUFFIX = "_scramble"
SOLVE_SUFFIX = "_solve"


def gt_sequence(session_dir: Path) -> list[str]:
    """The session's WCA ground-truth move names, in order."""
    path = session_dir / "moves.jsonl"
    if not path.exists():
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        name = json.loads(line).get("wca_notation")
        if name:
            out.append(name)
    return out


def apply_sequence(cube: CubieCube, names: list[str]) -> list[str]:
    """
    Apply WCA move names to `cube` in place. Returns names it could not
    parse (which are skipped, and which make the result untrustworthy).
    """
    bad = []
    for name in names:
        face = name[0]
        if face not in _TP_FACE:
            bad.append(name)
            continue
        turns = 3 if name.endswith("'") else 2 if name.endswith("2") else 1
        for _ in range(turns):
            cube.multiply(MOVE_CUBE[_TP_FACE[face]])
    return bad


def is_solved(cube: CubieCube) -> bool:
    ref = CubieCube()
    return (cube.cp == ref.cp and cube.co == ref.co
            and cube.ep == ref.ep and cube.eo == ref.eo)


def misplaced(cube: CubieCube) -> tuple[int, int]:
    """(corner slots wrong, edge slots wrong) against a solved cube."""
    ref = CubieCube()
    c = sum(1 for a, b in zip(cube.cp, ref.cp) if a != b) + \
        sum(1 for a, b in zip(cube.co, ref.co) if a != b)
    e = sum(1 for a, b in zip(cube.ep, ref.ep) if a != b) + \
        sum(1 for a, b in zip(cube.eo, ref.eo) if a != b)
    return c, e


def closing_word(cube: CubieCube, max_depth: int = 3) -> list[str] | None:
    """
    A short word that would finish this cube, if one of at most
    `max_depth` face turns exists.

    This is the difference between the two causes a FAIL can have, and they
    want opposite responses. An untracked whole-cube rotation or wide turn
    relabels a long suffix; the residual is then a scrambled cube that no
    short word closes, and that session's labels are wrong from the
    reorientation onward. A residual a two- or three-move word closes is
    that many moves missing from the BLE log — dropped notifications, or a
    recording that stopped before the last turns were reported — which
    leaves every label that IS present correct.

    Brute force over the 18 face turns to depth 3 is ~6k states and
    instant, which is why this does not reach for the pattern databases in
    reconstruct.py: keeping this gate's dependencies at twophase alone is
    worth more than the tighter bound.

    Reported, never repaired. Appending the moves would be inventing ground
    truth, and WHERE in the log they belong is not knowable from the
    endpoint alone.
    """
    turns = [(f"{face}{suffix}", fi, n)
             for face, fi in _TP_FACE.items()
             for n, suffix in ((1, ""), (2, "2"), (3, "'"))]

    def dfs(state: CubieCube, depth: int, path: list[str]) -> list[str] | None:
        if is_solved(state):
            return list(path)
        if depth == 0:
            return None
        for name, fi, n in turns:
            # Two turns of the same face in a row are always expressible as
            # one, so they can never be on a shortest word.
            if path and path[-1][0] == name[0]:
                continue
            probe = CubieCube(cp=list(state.cp), co=list(state.co),
                              ep=list(state.ep), eo=list(state.eo))
            for _ in range(n):
                probe.multiply(MOVE_CUBE[fi])
            path.append(name)
            got = dfs(probe, depth - 1, path)
            path.pop()
            if got is not None:
                return got
        return None

    for d in range(1, max_depth + 1):
        got = dfs(cube, d, [])
        if got is not None:
            return got
    return None


def paired_dirs(session_dir: Path) -> tuple[Path, Path] | None:
    """
    (scramble_dir, solve_dir) for whichever half of a pair was passed, or
    None if this session has no partner.
    """
    name = session_dir.name
    if name.endswith(SCRAMBLE_SUFFIX):
        stem = name[: -len(SCRAMBLE_SUFFIX)]
    elif name.endswith(SOLVE_SUFFIX):
        stem = name[: -len(SOLVE_SUFFIX)]
    else:
        return None
    scramble = session_dir.parent / f"{stem}{SCRAMBLE_SUFFIX}"
    solve = session_dir.parent / f"{stem}{SOLVE_SUFFIX}"
    if scramble.is_dir() and solve.is_dir():
        return scramble, solve
    return None


def check_pair(scramble_dir: Path, solve_dir: Path) -> tuple[str, str]:
    """
    Returns (status, detail) where status is 'pass' or 'FAIL'.

    The cube starts solved, is scrambled, then solved — so the product of
    both ground-truth sequences must be the identity.
    """
    scramble, solve = gt_sequence(scramble_dir), gt_sequence(solve_dir)
    if not scramble or not solve:
        return "FAIL", "one half has no ground-truth moves"

    cube = CubieCube()
    bad = apply_sequence(cube, scramble) + apply_sequence(cube, solve)
    if bad:
        return "FAIL", f"{len(bad)} unparseable move name(s): {sorted(set(bad))[:5]}"
    if is_solved(cube):
        return "pass", f"{len(scramble)}+{len(solve)} moves"

    n = f"{len(scramble)}+{len(solve)} moves"
    word = closing_word(cube)
    if word is not None:
        return ("SHORT",
                f"{n}: {len(word)} move(s) short — `{' '.join(word)}` would "
                f"finish it, so the BLE log dropped {'a turn' if len(word)==1 else 'turns'} "
                f"(or the recording stopped early). Recoverable: nothing "
                f"here is RELABELLED, it is only incomplete.")
    c, e = misplaced(cube)
    return ("FAIL",
            f"{n} do NOT return to solved ({c} corner / {e} edge slots off, "
            f"and no word of <=3 turns closes it) — an untracked whole-cube "
            f"rotation or wide turn is the usual cause (OrientationTracker "
            f"cannot represent either)")


def slice_report(session_dir: Path) -> dict:
    """
    Slice count and camera-frame status for one session.

    The endpoint check above is blind to the camera frame (see the module
    docstring), so this reports the one thing that would tell you a session
    needs the camera-frame backfill: does it contain slices, and does it
    carry `camera_notation` for them.
    """
    path = session_dir / "moves.jsonl"
    if not path.exists():
        return {"moves": 0, "slices": 0, "state": "no moves.jsonl"}
    moves = [json.loads(l) for l in open(path) if l.strip()]
    if not moves:
        return {"moves": 0, "slices": 0, "state": "empty"}

    # Imported here rather than at module scope so this file keeps working
    # as a standalone gate even if ble/ is not on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from slice_frame import annotate

    probe = [dict(m) for m in moves]      # never mutate the real records
    summary = annotate(probe)
    have_cam = sum(1 for m in moves if m.get("camera_notation"))

    if summary["n_slices"] == 0:
        state = "no slices"
    elif have_cam < len(moves):
        state = "NEEDS BACKFILL"
    elif any(m.get("camera_notation") != p.get("camera_notation")
             for m, p in zip(moves, probe)):
        state = "STALE camera_notation"
    else:
        state = "camera frame ok"
    return {"moves": len(moves), "slices": summary["n_slices"],
            "relabelled": summary["n_relabelled"], "state": state}


def check_session(session_dir: Path) -> tuple[str, str]:
    """Endpoint check for one session, via its pair. 'skip' if unpaired."""
    pair = paired_dirs(session_dir)
    if pair is None:
        return "skip", "no paired scramble/solve — endpoint not checkable"
    return check_pair(*pair)


def run_check(session_dirs: list[Path]) -> int:
    """Check each distinct pair once. Returns a shell exit code."""
    seen: set[tuple[Path, Path]] = set()
    unpaired: list[Path] = []
    rows: list[tuple[str, str, str]] = []

    for d in sorted(session_dirs):
        pair = paired_dirs(d)
        if pair is None:
            unpaired.append(d)
            continue
        if pair in seen:
            continue
        seen.add(pair)
        status, detail = check_pair(*pair)
        stem = pair[0].name[: -len(SCRAMBLE_SUFFIX)]
        rows.append((stem, status, detail))

    print(f"\n{'session pair':<36} {'result':<6} detail")
    print(f"{'-'*36} {'-'*6} {'-'*44}")
    for stem, status, detail in rows:
        print(f"{stem:<36} {status:<6} {detail}")

    # -- the camera frame, which the endpoint check above cannot see
    slice_rows = []
    for d in sorted(session_dirs):
        r = slice_report(d)
        if r["slices"]:
            slice_rows.append((d.name, r))
    if slice_rows:
        print(f"\n{'session with slices':<40} {'slices':>7} {'relabelled':>11}"
              f"  camera frame")
        print(f"{'-'*40} {'-'*7} {'-'*11}  {'-'*18}")
        for name, r in slice_rows:
            print(f"{name:<40} {r['slices']:>7} {r['relabelled']:>11}  "
                  f"{r['state']}")
        stale = [n for n, r in slice_rows if r["state"] != "camera frame ok"]
        if stale:
            print(f"\n{len(stale)} session(s) need the camera-frame backfill "
                  f"— their labels name the wrong\nface after the first "
                  f"slice (ble/slice_frame.py):\n"
                  f"    python backfill_camera_frame.py --sessions "
                  f"training_data/solve_*/")

    failed = sum(1 for _, s, _ in rows if s == "FAIL")
    short = sum(1 for _, s, _ in rows if s == "SHORT")
    print()
    if rows:
        print(f"{len(rows) - failed - short} of {len(rows)} pair(s) "
              f"endpoint-consistent.")
    if short:
        print(f"{short} pair(s) are a few moves SHORT — dropped BLE "
              f"notifications, not a relabelling\nbug. Every label present "
              f"is still correct; the log is merely incomplete.")
    if unpaired:
        print(f"{len(unpaired)} session(s) unpaired and NOT checked — this is a "
              f"coverage gap, not a pass:")
        print("  " + ", ".join(d.name for d in sorted(unpaired)))
    if failed:
        print(f"\n{failed} pair(s) FAILED. Their labels are wrong from the "
              f"reorientation onward;\ndo not train on them or quote a number "
              f"from them until the cause is found.")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ground-truth endpoint consistency gate for BLE sessions")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Session folder(s) — supports globs")
    args = parser.parse_args()

    dirs = [Path(p) for pattern in args.sessions
            for p in (Path(".").glob(pattern) if "*" in pattern else [Path(pattern)])]
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        sys.exit("No session directories found.")
    sys.exit(run_check(dirs))
