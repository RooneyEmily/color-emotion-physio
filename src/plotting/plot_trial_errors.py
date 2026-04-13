#!/usr/bin/env python3
"""
Trial-level error analysis for timed-trial JSON outputs.

For each JSON (participant):
- Compute per-trial absolute error for valence/arousal between trial mean prediction and self-report.
- Plot per-subject trial error curves (blue=valence, orange=arousal).
- Plot per-video aggregate error across participants.
- Print overall MAE and RMSE for valence/arousal across all paired trials.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _pair_trial(tr: Dict[str, Any]) -> Optional[Tuple[float, float, float, float, str]]:
    pv = _safe_float(tr.get("video_mean_valence"))
    pa = _safe_float(tr.get("video_mean_arousal"))
    sv = _safe_float(tr.get("subjective_valence_scaled"))
    sa = _safe_float(tr.get("subjective_arousal_scaled"))
    if sv is None:
        sv = _safe_float(tr.get("subjective_valence"))
    if sa is None:
        sa = _safe_float(tr.get("subjective_arousal"))
    if None in (pv, pa, sv, sa):
        return None
    name = str(tr.get("videoName", "unknown"))
    return float(pv), float(pa), float(sv), float(sa), name


def _plot_subject_errors(
    pid: str,
    ev: np.ndarray,
    ea: np.ndarray,
    out_path: Path,
) -> None:
    x = np.arange(len(ev))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(x, ev, color="tab:blue", marker="o", linewidth=1.2, markersize=3, label="Valence abs error")
    ax.plot(x, ea, color="tab:orange", marker="o", linewidth=1.2, markersize=3, label="Arousal abs error")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"Participant {pid}: trial absolute error")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_video_aggregate(
    video_names: List[str],
    v_mae: np.ndarray,
    a_mae: np.ndarray,
    out_path: Path,
) -> None:
    x = np.arange(len(video_names))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, v_mae, color="tab:blue", marker="o", linewidth=1.2, markersize=3, label="Valence mean abs error")
    ax.plot(x, a_mae, color="tab:orange", marker="o", linewidth=1.2, markersize=3, label="Arousal mean abs error")
    ax.set_xlabel("Video index (sorted by name)")
    ax.set_ylabel("Mean absolute error across participants")
    ax.set_title("Per-video absolute error (averaged over participants)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot trial errors from timed-trial JSON files")
    ap.add_argument("--input-glob", type=str, required=True, help="Glob for participant JSONs, e.g. /path/*_lab_fan.json")
    ap.add_argument("--out-dir", type=Path, default=Path("/home/mabl-main/results/error_plots"))
    ap.add_argument("--tag", type=str, default="lab")
    args = ap.parse_args()

    paths = sorted(Path(x) for x in glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    # Collect pooled metrics and per-video errors
    pooled_pv: List[float] = []
    pooled_pa: List[float] = []
    pooled_sv: List[float] = []
    pooled_sa: List[float] = []
    by_video: Dict[str, List[Tuple[float, float]]] = {}

    for p in paths:
        data = json.loads(p.read_text())
        pid = str(data.get("physio_participant_id", p.stem.split("_")[0]))
        trials = data.get("trials") or []

        ev: List[float] = []
        ea: List[float] = []
        for tr in trials:
            tup = _pair_trial(tr)
            if tup is None:
                continue
            pv, pa, sv, sa, video = tup
            ev.append(abs(pv - sv))
            ea.append(abs(pa - sa))
            pooled_pv.append(pv)
            pooled_pa.append(pa)
            pooled_sv.append(sv)
            pooled_sa.append(sa)
            by_video.setdefault(video, []).append((abs(pv - sv), abs(pa - sa)))

        if ev:
            _plot_subject_errors(
                pid,
                np.asarray(ev, dtype=np.float64),
                np.asarray(ea, dtype=np.float64),
                args.out_dir / f"{args.tag}_subject_{pid}_trial_abs_error.png",
            )

    if not pooled_pv:
        raise SystemExit("No valid paired trial predictions found.")

    # Per-video means
    videos = sorted(by_video.keys())
    v_mae = np.asarray([float(np.mean([x[0] for x in by_video[v]])) for v in videos], dtype=np.float64)
    a_mae = np.asarray([float(np.mean([x[1] for x in by_video[v]])) for v in videos], dtype=np.float64)
    _plot_video_aggregate(videos, v_mae, a_mae, args.out_dir / f"{args.tag}_per_video_abs_error.png")

    # Overall summary
    pv = np.asarray(pooled_pv, dtype=np.float64)
    pa = np.asarray(pooled_pa, dtype=np.float64)
    sv = np.asarray(pooled_sv, dtype=np.float64)
    sa = np.asarray(pooled_sa, dtype=np.float64)
    mae_v = float(np.mean(np.abs(pv - sv)))
    mae_a = float(np.mean(np.abs(pa - sa)))
    rmse_v = float(np.sqrt(np.mean((pv - sv) ** 2)))
    rmse_a = float(np.sqrt(np.mean((pa - sa) ** 2)))

    summary = {
        "tag": args.tag,
        "n_files": len(paths),
        "n_paired_trials": int(len(pv)),
        "mae_valence": mae_v,
        "mae_arousal": mae_a,
        "rmse_valence": rmse_v,
        "rmse_arousal": rmse_a,
    }
    out_json = args.out_dir / f"{args.tag}_overall_error_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("Overall error summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Wrote {out_json}")
    print(f"Wrote subject and per-video plots in {args.out_dir}")


if __name__ == "__main__":
    main()
