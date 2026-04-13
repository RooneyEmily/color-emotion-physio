#!/usr/bin/env python3
"""
Batch timed-trial runs: participants × color spaces, with resume and light QC logging.

  python run_physio_grid.py --config physio_cohort.json [--dry-run] [--force]
  python run_physio_grid.py --config physio_cohort.json --qc-only   # physio_trial_qc only
  python run_physio_grid.py --config physio_cohort.json --qc-after   # after batch, run QC

Config: see physio_cohort.example.json. Paths use {base}, {k} from checkpoints_base
and checkpoint_run_index.

Optional keys: qc_after (bool), qc_glob (str, default *_fan.json). After QC, writes
output_dir/qc_report.json and qc_skip_candidates.txt (PIDs with
SESSION_VIDEO_SHORTER_THAN_RATINGS for merging into skip_participants).

preflight_session_coverage (bool, default true): before any process_videos job, scan each
participant's ratings vs session MP4 frame count (same logic as timed trials; no GPU).
preflight_on_coverage_short: "exit" (default) or "omit" — if ratings extend past the
last frame, abort the whole batch or drop those PIDs and continue.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _ensure_pythonpath(pythonpath: Any) -> None:
    if not pythonpath:
        return
    for p in str(pythonpath).split(os.pathsep):
        ap = os.path.abspath(p.strip())
        if ap and ap not in sys.path:
            sys.path.insert(0, ap)
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _run_preflight(
    data_root: Path,
    participant_ids: List[str],
    pythonpath: Any,
    out_dir: Path,
    dry_run: bool,
    on_fail: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (participant_ids_to_run, preflight_reports).
    Exits process if on_fail == exit and coverage_short present.
    """
    _ensure_pythonpath(pythonpath)
    from physio_session_preflight import scan_cohort

    reports, coverage_short, _missing = scan_cohort(data_root, participant_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Wrote {out_dir / 'preflight_report.json'}")

    if not coverage_short:
        return participant_ids, reports

    print(
        "PREFLIGHT: session MP4 shorter than subjective log (ratings end past last frame) for "
        + json.dumps(coverage_short),
        file=sys.stderr,
    )
    if on_fail == "omit":
        kept = [p for p in participant_ids if p not in set(coverage_short)]
        print(
            f"PREFLIGHT: omitting {len(coverage_short)} participant(s); continuing with {len(kept)}.",
            file=sys.stderr,
        )
        if not kept:
            print("PREFLIGHT: no participants left after omit.", file=sys.stderr)
            sys.exit(1)
        return kept, reports

    print(
        "PREFLIGHT: aborting. Fix data, add IDs to skip_participants, or set "
        "preflight_on_coverage_short to \"omit\" / run with --preflight-omit.",
        file=sys.stderr,
    )
    sys.exit(2)


def _expand_ckpt(pattern: str, base: str, k: int) -> str:
    return pattern.replace("{base}", base).replace("{k}", str(k))


def _run_physio_qc(
    out_dir: Path,
    qc_glob: str,
    dry_run: bool,
) -> List[str]:
    """
    Run physio_trial_qc.py on out_dir; write qc_report.json and qc_skip_candidates.txt.
    Returns sorted unique participant IDs flagged SESSION_VIDEO_SHORTER_THAN_RATINGS.
    """
    qc_script = Path(__file__).resolve().parent / "physio_trial_qc.py"
    if not qc_script.is_file():
        print(f"SKIP QC: missing {qc_script}", file=sys.stderr)
        return []
    json_out = out_dir / "qc_report.json"
    skip_out = out_dir / "qc_skip_candidates.txt"
    cmd = [
        sys.executable,
        str(qc_script),
        str(out_dir),
        "--glob",
        qc_glob,
        "--json-out",
        str(json_out),
    ]
    print("QC " + " ".join(cmd))
    if dry_run:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"QC failed rc={r.returncode}", file=sys.stderr)
        return []
    if not json_out.is_file():
        return []
    reports: List[Dict[str, Any]] = json.loads(json_out.read_text())
    bad: Set[str] = set()
    for rep in reports:
        if rep.get("error"):
            continue
        flags = rep.get("flags") or []
        if "SESSION_VIDEO_SHORTER_THAN_RATINGS" in flags:
            pid = rep.get("participant")
            if pid is not None:
                bad.add(str(pid))
    def _pid_key(x: str):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    ordered = sorted(bad, key=_pid_key)
    lines = [
        "# Participant IDs with SESSION_VIDEO_SHORTER_THAN_RATINGS (session MP4 shorter than subjective log).",
        "# Merge into skip_participants in your cohort JSON to avoid reprocessing.",
    ]
    lines.extend(ordered)
    skip_out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_out}")
    print(f"Wrote {skip_out} ({len(ordered)} PIDs)")
    if ordered:
        print(
            "QC: session shorter than ratings — consider skip_participants: "
            + json.dumps(ordered),
        )
    return ordered


def _load_participants(cfg: Dict[str, Any]) -> List[str]:
    skip: Set[str] = set(str(x) for x in (cfg.get("skip_participants") or []))
    raw: List[str] = []
    pf = cfg.get("participants_file")
    if pf:
        p = Path(pf)
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                raw.append(line)
    raw.extend(str(x) for x in (cfg.get("participants") or []))
    out: List[str] = []
    for pid in raw:
        if pid in skip:
            continue
        if pid not in out:
            out.append(pid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run process_videos_av.py grid (physio timed trials)")
    ap.add_argument("--config", type=str, required=True, help="JSON config path")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument("--force", action="store_true", help="Re-run even if output exists")
    ap.add_argument(
        "--qc-only",
        action="store_true",
        help="Only run physio_trial_qc on output_dir (no process_videos jobs).",
    )
    ap.add_argument(
        "--qc-after",
        action="store_true",
        help="After the batch, run physio_trial_qc and write qc_report.json + qc_skip_candidates.txt.",
    )
    ap.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip session coverage preflight (ratings vs MP4 length before GPU jobs).",
    )
    ap.add_argument(
        "--preflight-omit",
        action="store_true",
        help="On coverage mismatch, omit those participants and continue (overrides config).",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    base = str(cfg["checkpoints_base"])
    k = int(cfg["checkpoint_run_index"])
    ckpt_map = {
        space: _expand_ckpt(cfg["checkpoints"][space], base, k)
        for space in cfg["color_spaces"]
    }

    data_root = Path(cfg["data_root"])
    out_dir = Path(cfg["output_dir"])
    out_pat: str = cfg.get("output_filename", "{pid}_{space}_fan.json")
    script = Path(cfg["process_script"])
    pyp = cfg.get("pythonpath")
    participants = _load_participants(cfg)

    log_lines: List[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_lines.append(f"# physio grid log utc={ts} config={cfg_path}")

    env = os.environ.copy()
    if pyp:
        env["PYTHONPATH"] = pyp

    face = cfg.get("face_detector", "fan")
    subj = cfg.get("subjective_scale", "deap_1_9_to_neg1_1")
    trial_agg = str(cfg.get("trial_aggregation", "mean"))
    agg_ckpt = cfg.get("aggregator_checkpoint")
    agg_hidden = int(cfg.get("aggregator_hidden", 32))
    agg_layers = int(cfg.get("aggregator_layers", 1))
    qc_glob = str(cfg.get("qc_glob", "*_fan.json"))
    qc_after = bool(cfg.get("qc_after", False)) or args.qc_after

    if args.qc_only:
        _run_physio_qc(out_dir, qc_glob, args.dry_run)
        return

    do_preflight = bool(cfg.get("preflight_session_coverage", True)) and not args.no_preflight
    on_cov = str(cfg.get("preflight_on_coverage_short", "exit")).lower()
    if on_cov not in ("exit", "omit"):
        on_cov = "exit"
    if args.preflight_omit:
        on_cov = "omit"

    if do_preflight and participants:
        log_lines.append(
            f"PREFLIGHT session_coverage=1 on_fail={on_cov} n_participants={len(participants)}"
        )
        participants, preflight_reports = _run_preflight(
            data_root,
            participants,
            pyp,
            out_dir,
            args.dry_run,
            on_cov,
        )
        log_lines.append("PREFLIGHT_REPORT " + json.dumps(preflight_reports, default=str))

    for pid in participants:
        session = data_root / pid / "P1"
        if not session.is_dir():
            msg = f"SKIP missing session {session}"
            print(msg, file=sys.stderr)
            log_lines.append(msg)
            continue
        for space in cfg["color_spaces"]:
            ckpt = Path(ckpt_map[space])
            if not ckpt.is_file():
                msg = f"SKIP missing checkpoint {ckpt}"
                print(msg, file=sys.stderr)
                log_lines.append(msg)
                continue
            out_name = out_pat.replace("{pid}", pid).replace("{space}", space)
            out_path = out_dir / out_name
            if out_path.is_file() and not args.force:
                log_lines.append(f"SKIP exists {out_path}")
                continue
            cmd = [
                sys.executable,
                str(script),
                "--timed_trials",
                "--physio_session",
                str(session),
                "--checkpoint",
                str(ckpt),
                "--color_space",
                space,
                "--face_detector",
                face,
                "--subjective_scale",
                subj,
                "--trial-aggregation",
                trial_agg,
                "-o",
                str(out_path),
            ]
            if trial_agg == "gru":
                if not agg_ckpt:
                    print(
                        "Config trial_aggregation=gru requires aggregator_checkpoint",
                        file=sys.stderr,
                    )
                    raise SystemExit(3)
                cmd.extend(
                    [
                        "--aggregator-checkpoint",
                        str(Path(agg_ckpt).expanduser().resolve()),
                        "--aggregator-hidden",
                        str(agg_hidden),
                        "--aggregator-layers",
                        str(agg_layers),
                    ]
                )
            log_lines.append("RUN " + " ".join(cmd))
            print("RUN", " ".join(cmd))
            if args.dry_run:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(cmd, env=env)
            if r.returncode != 0:
                log_lines.append(f"FAIL rc={r.returncode} participant={pid} space={space}")
                print(f"FAIL participant={pid} space={space}", file=sys.stderr)
                continue
            log_lines.append(f"OK {out_path}")

    log_path = out_dir / "batch_run_log.txt"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n")
        print(f"Wrote {log_path}")

    if qc_after:
        _run_physio_qc(out_dir, qc_glob, args.dry_run)


if __name__ == "__main__":
    main()
