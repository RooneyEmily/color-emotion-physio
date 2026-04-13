#!/usr/bin/env python3
"""
QC checks on timed-trial JSON outputs:

1) Alignment: ratings CSV (start/end) duration vs frame-span duration at reported FPS.
2) Session coverage: ratings times mapped with the same clock as the pipeline must not
   extend past n_frames_session; flags end-clamping and degenerate one-frame spans when
   the CSV still implies a full trial (short recording vs long subjective log).
3) Face coverage: fraction of decoded frames with successful face in each trial.
4) Dropout pattern: correlation of trial index with bad spans / low face coverage.

Usage:
  python3 physio_trial_qc.py /path/to/emotion_batch_data --glob '*_rgb_fan.json'
  python3 physio_trial_qc.py /path/to/session.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_ratings_gui_time(s: str) -> datetime:
    s = s.strip()
    if " GMT" in s:
        s = s.split(" GMT")[0].strip()
    return datetime.strptime(s, "%a %b %d %Y %H:%M:%S")


def load_ratings_rows(ratings_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(ratings_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vn = (row.get("videoName") or "").strip()
            if not vn:
                continue
            try:
                t0 = parse_ratings_gui_time(row["startTime"])
                t1 = parse_ratings_gui_time(row["endTime"])
            except (KeyError, ValueError):
                continue
            rows.append({"videoName": vn, "start": t0, "end": t1})
    return rows


def spearman_rho(x: List[float], y: List[float]) -> float:
    """Minimal Spearman without scipy; returns nan if undefined."""
    n = len(x)
    if n < 3:
        return float("nan")

    def ranks(a: List[float]) -> List[float]:
        idx = sorted(range(n), key=lambda i: a[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[idx[j + 1]] == a[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx < 1e-12 or vy < 1e-12:
        return float("nan")
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    return cov / math.sqrt(vx * vy)


@dataclass
class TrialQC:
    trial_index: int
    videoName: str
    expected_duration_s: float
    frame_span_s: float
    align_delta_s: float
    face_ratio: float
    bad_span: bool
    valence_saturated_frac: float


def trial_face_ratio(trial: Dict[str, Any]) -> Tuple[float, int, int]:
    n = int(trial.get("n_frames") or 0)
    segs = trial.get("segments") or []
    with_face = 0
    for s in segs:
        with_face += int(s.get("n_frames_with_face") or 0)
    if n <= 0:
        return float("nan"), 0, 0
    return with_face / n, with_face, n


def valence_saturated_fraction(trial: Dict[str, Any]) -> float:
    segs = trial.get("segments") or []
    if not segs:
        return float("nan")
    n = 0
    sat = 0
    for s in segs:
        vm = s.get("valence_mean")
        if vm is None:
            continue
        n += 1
        if abs(float(vm) + 1.0) < 1e-6 or abs(float(vm) - 1.0) < 1e-6:
            sat += 1
    return sat / n if n else float("nan")


def _video_clock_zero_dt(data: Dict[str, Any]) -> Optional[datetime]:
    raw = data.get("video_clock_zero")
    if not raw or not isinstance(raw, str):
        return None
    try:
        # fromisoformat accepts "2025-04-15T14:18:34.827263"
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def analyze_json(path: Path, align_warn_s: float, face_warn: float) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    fps = float(data.get("fps_reported_session") or 25.0)
    n_frames_session = data.get("n_frames_session")
    t_video0 = _video_clock_zero_dt(data)
    ratings_path = Path(data.get("subjective_csv") or "")
    if not ratings_path.is_file():
        return {"file": str(path), "error": "missing_subjective_csv"}
    csv_rows = load_ratings_rows(ratings_path)
    trials = data.get("trials") or []
    out_trials: List[TrialQC] = []
    mismatches = 0
    n_end_clamped = 0
    n_degenerate_vs_csv = 0
    max_raw_f1 = float("-inf")
    last_end_sec_from_t0: Optional[float] = None
    if t_video0 is not None and isinstance(n_frames_session, int) and n_frames_session > 0:
        for cr in csv_rows:
            sec_e = (cr["end"] - t_video0).total_seconds()
            rf1 = int(sec_e * fps)
            max_raw_f1 = max(max_raw_f1, float(rf1))
            if rf1 > n_frames_session:
                n_end_clamped += 1
        if csv_rows:
            last_end_sec_from_t0 = (csv_rows[-1]["end"] - t_video0).total_seconds()

    for i, tr in enumerate(trials):
        vn = (tr.get("videoName") or "").strip()
        bad = tr.get("error") == "bad_or_empty_frame_span"
        fr = tr.get("frame_range")
        if i < len(csv_rows):
            cr = csv_rows[i]
            name_ok = cr["videoName"].strip() == vn or cr["videoName"] in vn or vn in cr["videoName"]
            if not name_ok:
                mismatches += 1
            exp_s = (cr["end"] - cr["start"]).total_seconds()
        else:
            exp_s = float("nan")
        if isinstance(fr, list) and len(fr) == 2 and not bad:
            f0, f1 = int(fr[0]), int(fr[1])
            span_s = (f1 - f0) / fps if fps > 0 else float("nan")
            dlt = abs(exp_s - span_s) if math.isfinite(exp_s) and math.isfinite(span_s) else float("nan")
        else:
            span_s = float("nan")
            dlt = float("nan")
            f0, f1 = -1, -1

        # Degenerate JSON span vs CSV: pipeline clamped to end of file (cf. process_videos_av).
        if (
            t_video0 is not None
            and isinstance(n_frames_session, int)
            and n_frames_session > 0
            and i < len(csv_rows)
        ):
            n_frames_effective = (f1 - f0) if f0 >= 0 and f1 >= 0 else 0
            if (
                math.isfinite(exp_s)
                and exp_s >= 5.0
                and n_frames_effective <= 1
                and not bad
            ):
                n_degenerate_vs_csv += 1

        fratio, _, _ = trial_face_ratio(tr)
        vsat = valence_saturated_fraction(tr)
        out_trials.append(
            TrialQC(
                trial_index=i,
                videoName=vn,
                expected_duration_s=exp_s,
                frame_span_s=span_s,
                align_delta_s=dlt,
                face_ratio=fratio,
                bad_span=bad,
                valence_saturated_frac=vsat,
            )
        )

    align_deltas = [t.align_delta_s for t in out_trials if math.isfinite(t.align_delta_s)]
    face_ratios = [t.face_ratio for t in out_trials if math.isfinite(t.face_ratio)]
    idxs = [float(t.trial_index) for t in out_trials]
    bads = [1.0 if t.bad_span else 0.0 for t in out_trials]
    low_face = [1.0 if (math.isfinite(t.face_ratio) and t.face_ratio < face_warn) else 0.0 for t in out_trials]

    max_align = max(align_deltas) if align_deltas else float("nan")
    mean_align = sum(align_deltas) / len(align_deltas) if align_deltas else float("nan")
    min_face = min(face_ratios) if face_ratios else float("nan")
    mean_face = sum(face_ratios) / len(face_ratios) if face_ratios else float("nan")

    rho_bad = spearman_rho(idxs, bads)
    rho_lowf = spearman_rho(idxs, low_face)

    n_align_warn = sum(1 for d in align_deltas if d > align_warn_s)
    n_face_warn = sum(1 for f in face_ratios if f < face_warn)

    session_duration_s: Optional[float] = None
    ratings_past_video = False
    if isinstance(n_frames_session, int) and n_frames_session > 0 and fps > 0:
        session_duration_s = n_frames_session / fps
        if last_end_sec_from_t0 is not None and last_end_sec_from_t0 > session_duration_s + 1e-3:
            ratings_past_video = True

    coverage_skipped = t_video0 is None or not isinstance(n_frames_session, int) or n_frames_session <= 0
    coverage_flags: List[str] = []
    if coverage_skipped:
        coverage_flags.append("SESSION_COVERAGE_SKIPPED_NO_CLOCK_OR_NFRAMES")
    else:
        if n_end_clamped > 0 or ratings_past_video:
            coverage_flags.append("SESSION_RATINGS_PAST_VIDEO_END")
        if n_degenerate_vs_csv > 0:
            coverage_flags.append("SESSION_DEGENERATE_SPANS_VS_CSV")

    short_video = (
        not coverage_skipped
        and (n_end_clamped > 0 or ratings_past_video or n_degenerate_vs_csv > 0)
    )

    return {
        "file": str(path),
        "participant": data.get("physio_participant_id"),
        "color_space": data.get("color_space"),
        "n_trials_json": len(trials),
        "n_ratings_rows": len(csv_rows),
        "video_name_order_mismatches": mismatches,
        "alignment_delta_s_mean": mean_align,
        "alignment_delta_s_max": max_align,
        "n_trials_align_delta_over_threshold": n_align_warn,
        "align_threshold_s": align_warn_s,
        "face_ratio_mean": mean_face,
        "face_ratio_min": min_face,
        "n_trials_face_ratio_below_threshold": n_face_warn,
        "face_warn_threshold": face_warn,
        "spearman_index_vs_bad_span": rho_bad,
        "spearman_index_vs_low_face": rho_lowf,
        "n_bad_spans": sum(bads),
        "n_frames_session": n_frames_session,
        "session_duration_s": session_duration_s,
        "n_trials_raw_end_past_n_frames": n_end_clamped,
        "max_raw_end_frame_index": max_raw_f1 if math.isfinite(max_raw_f1) else None,
        "last_rating_end_sec_from_video_zero": last_end_sec_from_t0,
        "n_trials_degenerate_span_vs_csv": n_degenerate_vs_csv,
        "session_coverage_flags": coverage_flags,
        "flags": [
            x
            for x in [
                "ALIGNMENT_CHECK_ORDER_NAMES" if mismatches > 0 else None,
                "ALIGNMENT_LARGE_DELTA" if (math.isfinite(max_align) and max_align > align_warn_s) else None,
                "LOW_FACE_COVERAGE" if (math.isfinite(min_face) and min_face < face_warn) else None,
                "DROPOUT_LATE_BAD_SPANS" if (math.isfinite(rho_bad) and rho_bad > 0.35) else None,
                "DROPOUT_LATE_LOW_FACE" if (math.isfinite(rho_lowf) and rho_lowf > 0.35) else None,
                "SESSION_COVERAGE_SKIPPED_NO_CLOCK_OR_NFRAMES" if coverage_skipped else None,
                "SESSION_VIDEO_SHORTER_THAN_RATINGS" if short_video else None,
            ]
            if x is not None
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Physio timed-trial QC: alignment, face coverage, dropout")
    ap.add_argument("inputs", nargs="+", help="JSON file(s) or directory")
    ap.add_argument("--glob", type=str, default="*_fan.json", help="Pattern when inputs include a directory")
    ap.add_argument("--align-warn-s", type=float, default=2.0, help="Flag trial if |CSV duration - frame span| > this")
    ap.add_argument("--face-warn", type=float, default=0.5, help="Flag trial if face_ratio < this")
    ap.add_argument("--json-out", type=str, default=None, help="Write full report JSON here")
    args = ap.parse_args()

    paths: List[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob(args.glob)))
        else:
            paths.append(p)

    reports: List[Dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            print(f"skip missing {p}", file=sys.stderr)
            continue
        reports.append(analyze_json(p, args.align_warn_s, args.face_warn))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2) + "\n")

    for r in reports:
        if "error" in r:
            print(r)
            continue
        cov = (
            f"  session: n_frames={r.get('n_frames_session')} duration_s={r.get('session_duration_s')} "
            f"raw_end_max={r.get('max_raw_end_frame_index')} "
            f"n_csv_ends_past_video={r.get('n_trials_raw_end_past_n_frames')} "
            f"n_degenerate_vs_csv={r.get('n_trials_degenerate_span_vs_csv')} "
            f"coverage={r.get('session_coverage_flags')}\n"
        )
        print(
            f"pid={r['participant']} space={r['color_space']} file={Path(r['file']).name}\n"
            f"  order_mismatches={r['video_name_order_mismatches']}  bad_spans={int(r['n_bad_spans'])}\n"
            f"{cov}"
            f"  align_delta_s: mean={r['alignment_delta_s_mean']:.3f} max={r['alignment_delta_s_max']:.3f} "
            f"(>{r['align_threshold_s']}s in {r['n_trials_align_delta_over_threshold']} trials)\n"
            f"  face_ratio: mean={r['face_ratio_mean']:.3f} min={r['face_ratio_min']:.3f} "
            f"(<{r['face_warn_threshold']} in {r['n_trials_face_ratio_below_threshold']} trials)\n"
            f"  Spearman(idx, bad_span)={r['spearman_index_vs_bad_span']:.3f}  "
            f"Spearman(idx, low_face)={r['spearman_index_vs_low_face']:.3f}\n"
            f"  flags={r['flags']}"
        )


if __name__ == "__main__":
    main()
