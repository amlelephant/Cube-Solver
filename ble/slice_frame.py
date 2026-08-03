"""
slice_frame.py

The camera<-cube frame, and how a middle-slice turn rotates it.

The problem
-----------
The smart cube reports face turns RELATIVE TO ITS CORE, identifying each
face by its centre's colour. Turning a middle slice rotates the core, so
the two outer layers on that axis — which did not move in space at all —
appear to the cube to have turned backwards, and it emits a PAIR of
opposite-face quarter turns with opposite handedness (`R` + `L'` for an
`M`). `move_detector/GROUND_TRUTH_ARTIFACTS.md` measured that signature on
46 of 46 same-axis opposite-face pairs within 200ms, 31 of them sharing a
BLE timestamp exactly.

That pair describes the state change correctly in the CUBE's frame. What
it does not say is that four centres just moved. Since the cube's whole
coordinate system is defined by centre colour, from that moment the
colour->camera-position map is stale and every later move is reported in a
frame that has rotated relative to the camera. Do a slice twice — which
plenty of last-layer algorithms do — and the centres make a half turn: the
face physically on top now carries the old bottom centre's colour, the cube
calls a turn of it `D`, and the camera plainly sees `U`.

`OrientationTracker` freezes its face map at `calibrate()` and
`FaceMap.apply_whole_rotation` is never called, so nothing corrected for
this. Every label after a session's first slice was wrong.

The correction
--------------
Doing `M` (the slice between L and R, turning in the L direction) rotates
the core in the L direction, i.e. by `x'`, and the cube emits `(R, L')`.
So the core rotation is the INVERSE of the whole-cube rotation in the
UNPRIMED reported face's own direction:

    reported pair    physical    core rotation
      R , L'            M             x'
      L , R'            M'            x
      U , D'            E'            y'
      D , U'            E             y
      F , B'            S             z'
      B , F'            S'            z

Evidence, and its limits
------------------------
The vision model only ever sees the camera, so whichever labelling it
agrees with is the camera-relative truth. Per-move accuracy on the moves
after a session's first slice, held-out sessions only (38 moves pooled):

                    seed 0    seed 1
    today's labels   60.5%     42.1%
    corrected        71.1%     52.6%

Replicated on both seeds, same direction and magnitude. On sessions the
checkpoints trained on the correction makes agreement WORSE, which is the
expected signature of a model fitted to the corrupted labels rather than a
contradiction.

The MAGNITUDE is well supported; the SIGN is barely tested. Every
accumulated orientation in the current corpus sits in a small subgroup
dominated by half turns, which are self-inverse, so the rule above and its
inverse score identically on 11 of 12 session/seed cells. The one cell that
separates them favours this rule (65.0% vs 60.0%). Footage with an odd
number of same-axis slices would settle it; until then, treat the sign as
the better-supported of two options rather than as established.

What this module deliberately does not touch
--------------------------------------------
The slice pair's OWN two labels. The camera sees one motion and the 12-class
vocabulary has no symbol for it, so those two labels name two layers that
did not move in space. That is a vocabulary problem (`M`/`E`/`S` as real
classes) and a model change; this module fixes only the drift AFTER the
pair. See `move_detector/LABEL_FRAME_PLAN.md`.

No dependencies on purpose — not numpy, not twophase, not bleak. The live
recorder and the offline backfill must apply the SAME rule, and the live
path is the one that cannot be tested here, so the rule lives in one
importable place with nothing that could fail to import on the capture
machine.

    python slice_frame.py --selftest
"""

# U R F D L B, matching twophase's pieces.Color and reconstruct.TP_NAME.
FACES = "URFDLB"
FACE_IDX = {f: i for i, f in enumerate(FACES)}
FACE_OPP = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}

# The whole-cube rotation in each face's own turning direction.
FACE_ROT = {"U": "y", "D": "y'", "R": "x", "L": "x'", "F": "z", "B": "z'"}

# rho[r][camera_position] = the position whose face moves INTO it under r.
# After rotation r, sigma' = [sigma[rho[r][c]] for c in range(6)], the same
# convention as reconstruct._build_rotations (kept deliberately identical so
# the two cannot disagree about which way `x` turns).
_U, _R, _F, _D, _L, _B = range(6)


def _build_rho() -> dict:
    base = {}
    x = list(range(6))
    x[_F], x[_U], x[_B], x[_D] = _D, _F, _U, _B   # x: front comes from down
    y = list(range(6))
    y[_F], y[_R], y[_B], y[_L] = _R, _B, _L, _F   # y: front comes from right
    z = list(range(6))
    z[_U], z[_R], z[_D], z[_L] = _L, _U, _R, _D   # z: up comes from left
    for name, perm in (("x", x), ("y", y), ("z", z)):
        base[name] = perm
        inv = [0] * 6
        for i, p in enumerate(perm):
            inv[p] = i
        base[name + "'"] = inv
    return base


RHO = _build_rho()

IDENTITY = tuple(range(6))

# Two BLE move events this far apart or closer may be one physical motion.
# The cube's notification tick is 30ms and a slice's two halves usually
# share a timestamp exactly; 200ms is the window GROUND_TRUTH_ARTIFACTS.md
# measured the 46/46 signature over.
SLICE_MAX_GAP = 0.200


def invert_rotation(rot: str) -> str:
    return rot[:-1] if rot.endswith("'") else rot + "'"


def is_slice_pair(a_name: str, a_ts: float, b_name: str, b_ts: float,
                  max_gap: float = SLICE_MAX_GAP) -> bool:
    """
    Are these two consecutive BLE moves one physical slice turn?

    Three conditions, all measured rather than assumed (see the module
    docstring): same axis and opposite faces, opposite handedness, and
    close enough in time to be one motion. All three are required — an
    `R` and a genuine `L'` performed as two separate turns half a second
    apart is not a slice, and relabelling it as one would corrupt the very
    thing this exists to fix.
    """
    if not a_name or not b_name:
        return False
    fa, fb = a_name[0], b_name[0]
    if fa not in FACE_OPP or FACE_OPP[fa] != fb:
        return False
    if a_name.endswith("'") == b_name.endswith("'"):
        return False
    return abs(b_ts - a_ts) <= max_gap


def core_rotation(a_name: str, b_name: str) -> str:
    """
    The whole-cube rotation the core underwent, for a slice pair.

    Order-independent: it depends only on which half is unprimed.
    """
    unprimed = a_name if not a_name.endswith("'") else b_name
    return invert_rotation(FACE_ROT[unprimed[0]])


class CameraFrame:
    """
    The running camera<-cube orientation.

    `sigma[camera_position] = the cube face currently at that position`,
    the same representation reconstruct.py's `_SIGMAS` uses. Starts as the
    identity, because at `calibrate()` time the two frames agree by
    definition; a slice is the only thing here that moves it.
    """

    def __init__(self, sigma: tuple = IDENTITY):
        self.sigma = tuple(sigma)

    @property
    def is_identity(self) -> bool:
        return self.sigma == IDENTITY

    def reset(self) -> None:
        self.sigma = IDENTITY

    def rotate(self, rot: str) -> None:
        perm = RHO[rot]
        self.sigma = tuple(self.sigma[perm[c]] for c in range(6))

    def apply_slice(self, a_name: str, b_name: str) -> str:
        """Advance the frame for one slice pair; returns the rotation used."""
        rot = core_rotation(a_name, b_name)
        self.rotate(rot)
        return rot

    def camera_name(self, cube_name: str) -> str:
        """
        The camera-relative name of a move the cube reported in ITS frame.

        The cube says "the face whose centre is <colour> turned", which the
        orientation tracker has already resolved to a cube-frame letter.
        This asks where that face physically IS right now.
        """
        if not cube_name or cube_name[0] not in FACE_IDX:
            return cube_name
        cube_face = FACE_IDX[cube_name[0]]
        cam_pos = self.sigma.index(cube_face)
        return FACES[cam_pos] + ("'" if cube_name.endswith("'") else "")

    def describe(self) -> str:
        """Human-readable orientation, for recording alongside a move."""
        if self.is_identity:
            return "identity"
        return "".join(FACES[self.sigma[c]] for c in range(6))


def annotate(moves: list[dict], name_key: str = "wca_notation",
             ts_key: str = "timestamp",
             max_gap: float = SLICE_MAX_GAP) -> dict:
    """
    Add `camera_notation` and `orientation` to a list of move records.

    Mutates `moves` in place and returns a summary. `wca_notation` is left
    ALONE — it is the cube-frame name, several consumers depend on that
    (session_check.py applies it to a centre-relative CubieCube, and
    reconstruct.start_from_gt builds the start state from it), and silently
    redefining a field is how this class of bug happens in the first place.

    The slice pair's own two moves are annotated under the orientation in
    force BEFORE the rotation, and the rotation takes effect for the move
    after them. That is what the physics says: the centres finish moving
    when the slice finishes.
    """
    frame = CameraFrame()
    n_slices = 0
    rotations: list[str] = []
    i = 0
    while i < len(moves):
        m = moves[i]
        name = m.get(name_key)
        m["camera_notation"] = frame.camera_name(name) if name else name
        m["orientation"] = frame.describe()

        nxt = moves[i + 1] if i + 1 < len(moves) else None
        if nxt is not None and name and nxt.get(name_key) and \
                is_slice_pair(name, m.get(ts_key, 0.0),
                              nxt[name_key], nxt.get(ts_key, 0.0), max_gap):
            nxt["camera_notation"] = frame.camera_name(nxt[name_key])
            nxt["orientation"] = frame.describe()
            rotations.append(frame.apply_slice(name, nxt[name_key]))
            n_slices += 1
            i += 2
            continue
        i += 1

    n_changed = sum(1 for m in moves
                    if m.get("camera_notation") != m.get(name_key))
    return {"n_moves": len(moves), "n_slices": n_slices,
            "rotations": rotations, "n_relabelled": n_changed,
            "final_orientation": frame.describe(),
            "drifted": not frame.is_identity}


# ---------------------------------------------------------------------------

def _selftest() -> int:
    fails = []

    def check(name, cond):
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
        if not cond:
            fails.append(name)

    # -- permutation group laws
    for r in ("x", "y", "z"):
        f = CameraFrame()
        for _ in range(4):
            f.rotate(r)
        check(f"{r} four times is the identity", f.is_identity)
        f = CameraFrame()
        f.rotate(r)
        f.rotate(invert_rotation(r))
        check(f"{r} then {r}' is the identity", f.is_identity)

    # -- the documented geometry: after x, Front holds what was at Down
    f = CameraFrame()
    f.rotate("x")
    check("after x the front position holds the old down face",
          f.sigma[_F] == _D)
    f = CameraFrame()
    f.rotate("y")
    check("after y the front position holds the old right face",
          f.sigma[_F] == _R)

    # -- camera_name is the inverse relabelling, so a round trip is identity
    f = CameraFrame()
    f.rotate("y")
    inv = CameraFrame()
    inv.rotate("y'")
    ok = all(inv.camera_name(f.camera_name(n)) == n
             for n in ("U", "U'", "R", "R'", "F", "F'", "D", "L", "B"))
    check("camera_name under y then y' round-trips", ok)

    # -- the rotation table
    check("(R, L') gives x'", core_rotation("R", "L'") == "x'")
    check("(L', R) gives x' regardless of order",
          core_rotation("L'", "R") == "x'")
    check("(L, R') gives x", core_rotation("L", "R'") == "x")
    check("(U, D') gives y'", core_rotation("U", "D'") == "y'")
    check("(F, B') gives z'", core_rotation("F", "B'") == "z'")

    # -- WHICH slice moves which centres. Getting this backwards is the
    # easiest mistake here: only the M and S slices move the U/D centres.
    # M (the pair R + L') turns about x and carries U-F-D-B;
    # E (the pair U + D') turns about y and carries F-R-B-L, leaving U/D
    # exactly where they were.
    f = CameraFrame()
    f.apply_slice("U", "D'")
    f.apply_slice("U", "D'")
    check("two E slices (U + D') leave the up/down positions alone",
          f.camera_name("U") == "U" and f.camera_name("D") == "D")
    check("...but do swap front and back", f.camera_name("F") == "B")

    # -- the observed failure: M2, i.e. the pair (R, L') twice. The face
    # physically on top now carries the old bottom centre, so the cube
    # reports a turn of it as D and the camera correctly sees U.
    f = CameraFrame()
    f.apply_slice("R", "L'")
    f.apply_slice("R", "L'")
    check("after M2 the frame is a half turn, not identity", not f.is_identity)
    check("after M2 a cube-frame D reads as camera U",
          f.camera_name("D") == "U")
    check("...and a cube-frame U reads as camera D",
          f.camera_name("U") == "D")
    check("...while R and L are untouched (M turns about their axis)",
          f.camera_name("R") == "R" and f.camera_name("L'") == "L'")

    # -- pair detection rejects what it should
    check("R + L' 30ms apart is a slice",
          is_slice_pair("R", 0.0, "L'", 0.03))
    check("R + L' 500ms apart is NOT a slice",
          not is_slice_pair("R", 0.0, "L'", 0.5))
    check("R + L (same handedness) is NOT a slice",
          not is_slice_pair("R", 0.0, "L", 0.03))
    check("R + U' (different axis) is NOT a slice",
          not is_slice_pair("R", 0.0, "U'", 0.03))

    # -- annotate leaves a slice-free session completely untouched
    plain = [{"wca_notation": n, "timestamp": i * 0.5}
             for i, n in enumerate(["R", "U", "R'", "U'"])]
    s = annotate(plain)
    check("a slice-free session is not relabelled at all",
          s["n_relabelled"] == 0 and not s["drifted"]
          and all(m["camera_notation"] == m["wca_notation"] for m in plain))

    # -- and is idempotent
    before = [m["camera_notation"] for m in plain]
    annotate(plain)
    check("annotate is idempotent", before ==
          [m["camera_notation"] for m in plain])

    print(f"\n  {len(fails)} failure(s)" if fails else "\n  all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    import argparse
    import sys
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    p.print_help()
