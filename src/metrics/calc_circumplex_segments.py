#!/usr/bin/env python3
"""
Compute circumplex angles and segment agreement for model vs self-report.

Angle convention follows the AV plane used in your PDF:
- x axis: valence
- y axis: arousal
- angle: atan2(arousal, valence) in degrees mapped to [0, 360)

Outputs per-trial rows with
- model and self-report AV coordinates
- model and self-report angles
- quadrant, octant, and 4-emotion region labels
- match flags for each segmentation scheme
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EMOTION_ANCHORS_DEG = {
    "surprise": 60.0,
    "anxiety": 110.0,
    "boredom": 240.0,
    "calmness": 290.0,
}


def _safe_float(v: Any) -> Optional[float]:
    try:
        z = float(v)
    except Exception:
        return None
    return z if math.isfinite(z) else None


def _model_to_sam(x: float) -> float:
    return 4.0 * x + 5.0


def _subjective_model_axis(trial: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    sv_s = _safe_float(trial.get("subjective_valence_scaled"))
    sa_s = _safe_float(trial.get("subjective_arousal_scaled"))
    sv_r = _safe_float(trial.get("subjective_valence"))
    sa_r = _safe_float(trial.get("subjective_arousal"))

    def one(scaled: Optional[float], raw: Optional[float]) -> Optional[float]:
        if scaled is not None:
            return scaled
        if raw is None:
            return None
        if abs(raw) > 1.25:
            return (raw - 5.0) / 4.0
        return raw

    return one(sv_s, sv_r), one(sa_s, sa_r)


def _subjective_sam_axis(trial: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    sv_raw = _safe_float(trial.get("subjective_valence"))
    sa_raw = _safe_float(trial.get("subjective_arousal"))
    sv_m, sa_m = _subjective_model_axis(trial)
    sv = sv_raw if sv_raw is not None and 1.0 <= sv_raw <= 9.0 else (_model_to_sam(sv_m) if sv_m is not None else None)
    sa = sa_raw if sa_raw is not None and 1.0 <= sa_raw <= 9.0 else (_model_to_sam(sa_m) if sa_m is not None else None)
    return sv, sa


def _angle_deg(arousal: float, valence: float) -> float:
    a = math.degrees(math.atan2(arousal, valence))
    if a < 0:
        a += 360.0
    return a


def _ang_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _quadrant(a: float, v: float) -> str:
    if v >= 0.0 and a >= 0.0:
        return "Q1"
    if v < 0.0 and a >= 0.0:
        return "Q2"
    if v < 0.0 and a < 0.0:
        return "Q3"
    return "Q4"


def _octant(angle_deg: float) -> str:
    idx = int(((angle_deg + 22.5) % 360.0) // 45.0)
    return f"O{idx + 1}"


def _anchor_region(angle_deg: float) -> str:
    return min(EMOTION_ANCHORS_DEG, key=lambda k: _ang_dist(angle_deg, EMOTION_ANCHORS_DEG[k]))


def _iter_inputs(inputs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        matches = sorted(Path(p) for p in glob.glob(s))
        if matches:
            out.extend(matches)
        else:
            p = Path(s)
            if p.exists():
                out.append(p)
    # de-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description="Calculate AV circumplex angles and segment matches.")
    ap.add_argument("--inputs", nargs="+", required=True, help="JSON paths or globs, e.g. results/*_xyz_fan.json")
    ap.add_argument("--out-csv", type=Path, required=True, help="Per-trial angle and segment table")
    ap.add_argument("--scale", choices=["model", "sam"], default="sam", help="Axis for angle calculation")
    args = ap.parse_args()

    files = _iter_inputs(args.inputs)
    if not files:
        raise SystemExit("No input JSON files found")

    rows: List[Dict[str, Any]] = []
    for p in files:
        data = json.loads(p.read_text())
        pid = str(data.get("physio_participant_id", p.stem.split("_")[0]))
        space = str(data.get("color_space", ""))
        trials = data.get("trials") or []
        for i, t in enumerate(trials):
            if t.get("error"):
                continue
            pv = _safe_float(t.get("video_mean_valence"))
            pa = _safe_float(t.get("video_mean_arousal"))
            if pv is None or pa is None:
                continue

            if args.scale == "model":
                sv, sa = _subjective_model_axis(t)
                mv, ma = pv, pa
            else:
                sv, sa = _subjective_sam_axis(t)
                mv, ma = _model_to_sam(pv), _model_to_sam(pa)
            if sv is None or sa is None:
                continue

            # For segment labeling, use centered coordinates in selected space.
            if args.scale == "sam":
                mv_c, ma_c = mv - 5.0, ma - 5.0
                sv_c, sa_c = sv - 5.0, sa - 5.0
            else:
                mv_c, ma_c = mv, ma
                sv_c, sa_c = sv, sa

            ang_m = _angle_deg(ma_c, mv_c)
            ang_s = _angle_deg(sa_c, sv_c)
            q_m, q_s = _quadrant(ma_c, mv_c), _quadrant(sa_c, sv_c)
            o_m, o_s = _octant(ang_m), _octant(ang_s)
            r_m, r_s = _anchor_region(ang_m), _anchor_region(ang_s)

            rows.append(
                {
                    "file": str(p),
                    "participant_id": pid,
                    "color_space": space,
                    "trial_index": i,
                    "video_name": str(t.get("videoName", "")),
                    "scale": args.scale,
                    "model_valence": mv,
                    "model_arousal": ma,
                    "self_valence": sv,
                    "self_arousal": sa,
                    "model_angle_deg": ang_m,
                    "self_angle_deg": ang_s,
                    "angle_abs_diff_deg": _ang_dist(ang_m, ang_s),
                    "model_quadrant": q_m,
                    "self_quadrant": q_s,
                    "quadrant_match": int(q_m == q_s),
                    "model_octant": o_m,
                    "self_octant": o_s,
                    "octant_match": int(o_m == o_s),
                    "model_region4": r_m,
                    "self_region4": r_s,
                    "region4_match": int(r_m == r_s),
                }
            )

    if not rows:
        raise SystemExit("No valid paired trials found after filtering")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    q = sum(r["quadrant_match"] for r in rows) / n
    o = sum(r["octant_match"] for r in rows) / n
    r4 = sum(r["region4_match"] for r in rows) / n
    mad = sum(float(r["angle_abs_diff_deg"]) for r in rows) / n
    print(f"Wrote {args.out_csv}")
    print(f"n_trials={n}")
    print(f"mean_abs_angle_diff_deg={mad:.2f}")
    print(f"quadrant_match_rate={q:.3f}")
    print(f"octant_match_rate={o:.3f}")
    print(f"region4_match_rate={r4:.3f}")
    print("region4 anchors deg: surprise=60 anxiety=110 boredom=240 calmness=290")


if __name__ == "__main__":
    main()
