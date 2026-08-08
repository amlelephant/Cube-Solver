"""
Import real recorded solves — and their coach analysis — onto one account.

Where `seed_demo` invents plausible players, this imports the actual
sessions in `ble/training_data/`: the times, move counts and every coach
metric are decoded from those recordings, and the wall-clock date on each
row is when the recording was actually made.

    python manage.py import_real_solves --report <coach_report.json> --user cuber

The report is `ble/move_detector/coach_report.py --out`'s output, i.e.

    cd ble/move_detector
    python coach_report.py --ctc checkpoints/move_ctc_spd_s0.pt \
        --sessions ../training_data/solve_*  --out results/<date>/report.json

WHAT THESE NUMBERS ARE, AND ARE NOT
-----------------------------------
`coach_report.py` decodes with a model that was TRAINED on most of these
sessions. The metric accuracies the payload carries (`accuracy_pct`) come
from `metric_robustness.py` on held-out solves and remain the honest
figures; the per-solve agreement you will see printed here is better than
that, because most of these are training sessions. Every imported row is
stamped `detail.held_out` so a later analysis can tell the two apart, and
`detail.trained_on` records the checkpoint that produced it.

Nothing imported here is evidence of anything. `verdict` is set to
VERIFIED because these are genuine recorded solves with a BLE cube
attached, not because the server verified them — no re-verification worker
exists yet (TODO §2B). Keep them off production.
"""

import json
from datetime import datetime, timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from core.models import Profile, Scramble, Solve
from core.scrambles import generate, sign
from core.views import SCRAMBLE_TTL

User = get_user_model()

#: Marks a row as ours to manage, so --reset cannot touch a real submission.
SOURCE = "import_real_solves"

#: Where the recordings live, relative to the repo root. Only read for the
#: capture timestamp, the true fps and the scramble that was actually set —
#: everything else comes out of the report.
DEFAULT_SESSIONS = Path(settings.BASE_DIR).parent / "ble" / "training_data"


def _session_facts(d: Path) -> dict:
    """Capture time, true fps and frame count for one session directory.

    fps is MEASURED from `frames.jsonl`'s per-frame timestamps rather than
    taken from the nominal 30 — webcam intervals jitter, and `derived_seconds`
    is defined as frames/fps, so a nominal figure would put that disagreement
    straight into the stored duration.
    """
    facts = {"captured_at": None, "fps": 0.0, "capture_frames": 0,
             "capture_seconds": None, "scramble": "", "ble_move_count": None}

    fl = d / "frames.jsonl"
    if fl.exists():
        ts = []
        for line in fl.read_text().splitlines():
            if line.strip():
                try:
                    ts.append(float(json.loads(line)["ts"]))
                except (ValueError, KeyError):
                    continue
        if len(ts) >= 2:
            span = ts[-1] - ts[0]
            facts["capture_frames"] = len(ts)
            facts["capture_seconds"] = round(span, 3)
            facts["fps"] = round((len(ts) - 1) / span, 3) if span > 0 else 0.0
            facts["captured_at"] = datetime.fromtimestamp(ts[0], dt_timezone.utc)

    cfg = d / "config.json"
    if cfg.exists():
        try:
            facts["scramble"] = json.loads(cfg.read_text()).get(
                "claimed_start", "") or ""
        except json.JSONDecodeError:
            pass

    meta = d / "ble_meta.json"
    if meta.exists():
        try:
            facts["ble_move_count"] = json.loads(meta.read_text()).get(
                "move_count")
        except json.JSONDecodeError:
            pass

    return facts


class Command(BaseCommand):
    help = "Import recorded solves + coach analysis onto one account."

    def add_arguments(self, parser):
        parser.add_argument("--report", required=True,
                            help="coach_report.py --out JSON.")
        parser.add_argument("--user", required=True,
                            help="Username to attach the solves to.")
        parser.add_argument("--sessions", default=str(DEFAULT_SESSIONS),
                            help="Root holding the solve_* session folders.")
        parser.add_argument("--reset", action="store_true",
                            help="Delete this account's previously imported "
                                 "solves first. Never touches submitted ones.")
        parser.add_argument("--force", action="store_true",
                            help="Bypass the DEBUG guard.")

    def handle(self, *args, **opts):
        if not settings.DEBUG and not opts["force"]:
            raise CommandError(
                "import_real_solves refuses to run with DJANGO_DEBUG off. "
                "These rows read as VERIFIED without a server ever having "
                "verified them. Pass --force if you genuinely mean it.")

        report_path = Path(opts["report"])
        if not report_path.exists():
            raise CommandError(f"No such report: {report_path}")
        report = json.loads(report_path.read_text())
        solves = report.get("solves", [])
        if not solves:
            raise CommandError(f"{report_path} contains no solves.")

        sessions_root = Path(opts["sessions"])
        try:
            user = User.objects.get(username=opts["user"])
        except User.DoesNotExist:
            raise CommandError(f"No such user: {opts['user']}")
        Profile.objects.get_or_create(user=user)

        if opts["reset"]:
            n = self._reset(user)
            self.stdout.write(f"Removed {n} previously imported solves.")

        existing = {s.detail.get("session")
                    for s in Solve.objects.filter(user=user)
                    if s.detail.get("source") == SOURCE}

        created = skipped = missing = 0
        for entry in solves:
            name = entry.get("session")
            if not name or not entry.get("decoded", {}).get("usable"):
                continue
            if name in existing:
                skipped += 1
                continue
            d = sessions_root / name
            if not d.is_dir():
                self.stdout.write(self.style.WARNING(
                    f"  {name}: session folder not found under "
                    f"{sessions_root} — skipped"))
                missing += 1
                continue
            self._make_solve(user, entry, d, report.get("model", ""))
            created += 1

        total = Solve.objects.filter(user=user).count()
        self.stdout.write(self.style.SUCCESS(
            f"\n{created} solves imported for {user.username} "
            f"({skipped} already present, {missing} without a session folder). "
            f"{total} solves on the account."))

    def _reset(self, user) -> int:
        rows = [s for s in Solve.objects.filter(user=user)
                if s.detail.get("source") == SOURCE]
        scramble_ids = [s.scramble_id for s in rows]
        # Solve PROTECTs its Scramble, so the solves have to go first.
        Solve.objects.filter(pk__in=[s.pk for s in rows]).delete()
        Scramble.objects.filter(pk__in=scramble_ids).delete()
        return len(rows)

    def _make_solve(self, user, entry: dict, d: Path, model: str):
        decoded = entry["decoded"]
        values = {k: m["value"] for k, m in decoded["metrics"].items()}
        facts = _session_facts(d)

        # The solve TIME is the coach's span: first onset to last. The
        # recording is longer at both ends — setup before, the verification
        # scan after — and neither belongs in a solve time.
        seconds = float(values.get("span_seconds") or 0.0)
        if seconds <= 0:
            return
        fps = facts["fps"] or 30.0
        # `derived_seconds` is defined as frame_count/fps, so frame_count is
        # the span's worth of frames rather than the whole recording's. The
        # real capture totals go in `detail`, where they are not mistaken for
        # the timed window.
        frame_count = max(1, round(seconds * fps))

        when = facts["captured_at"]
        moves = int(round(values.get("n_moves_qtm") or 0))

        scramble_text = facts["scramble"]
        scramble = Scramble(
            user=user,
            moves=scramble_text or generate(),
            expires_at=when + SCRAMBLE_TTL,
            consumed_at=when,
        )
        scramble.signature = sign(scramble.moves, scramble.nonce)

        with transaction.atomic():
            scramble.save()
            # auto_now_add ignores an assigned value, so backdate after insert.
            Scramble.objects.filter(pk=scramble.pk).update(issued_at=when)

            solve = Solve.objects.create(
                user=user, scramble=scramble,
                derived_seconds=round(seconds, 3),
                frame_count=frame_count,
                fps=fps,
                observed_moves=moves,
                observed_moves_after_stop=0,
                claimed_seconds=None,
                verdict=Solve.VERIFIED, result=Solve.SOLO,
                detail={
                    "source": SOURCE,
                    "session": entry["session"],
                    "regime": decoded.get("regime"),
                    "capture_hour": entry.get("hour"),
                    "trained_on": model,
                    # True only where the checkpoint never saw this session.
                    # Set from the report if it ever records it; until then
                    # the honest answer is "unknown", not "yes".
                    "held_out": entry.get("held_out"),
                    "capture_frames": facts["capture_frames"],
                    "capture_seconds": facts["capture_seconds"],
                    "capture_fps": facts["fps"],
                    "scramble_source": "recorded" if scramble_text
                                       else "generated",
                    "ble_move_count": facts["ble_move_count"],
                },
                analysis=decoded,
            )
            Solve.objects.filter(pk=solve.pk).update(received_at=when)
