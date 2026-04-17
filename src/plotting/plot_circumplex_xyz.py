#!/usr/bin/env python3
"""
Visualize circumplex distributions for XYZ model vs self-report.

Inputs can be either:
- the per-trial angle CSV from calc_circumplex_segments.py, or
- raw *_xyz_fan.json files via a glob (uses the same AV handling as existing code).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ANCHORS = {
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
    return z if np.isfinite(z) else None


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
    sv_r = _safe_float(trial.get("subjective_valence"))
    sa_r = _safe_float(trial.get("subjective_arousal"))
    sv_m, sa_m = _subjective_model_axis(trial)
    sv = sv_r if sv_r is not None and 1.0 <= sv_r <= 9.0 else (_model_to_sam(sv_m) if sv_m is not None else None)
    sa = sa_r if sa_r is not None and 1.0 <= sa_r <= 9.0 else (_model_to_sam(sa_m) if sa_m is not None else None)
    return sv, sa


def _load_from_json_glob(glob_pat: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for p in sorted(glob.glob(glob_pat)):
        data = json.loads(Path(p).read_text())
        if str(data.get("color_space", "")).lower().strip() != "xyz":
            continue
        for t in data.get("trials") or []:
            if t.get("error"):
                continue
            pv = _safe_float(t.get("video_mean_valence"))
            pa = _safe_float(t.get("video_mean_arousal"))
            sv, sa = _subjective_sam_axis(t)
            if pv is None or pa is None or sv is None or sa is None:
                continue
            rows.append(
                {
                    "model_x": _model_to_sam(pv) - 5.0,
                    "model_y": _model_to_sam(pa) - 5.0,
                    "self_x": sv - 5.0,
                    "self_y": sa - 5.0,
                }
            )
    return rows


def _load_from_csv(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r.get("color_space", "").strip().lower() != "xyz":
                continue
            mx = _safe_float(r.get("model_valence"))
            my = _safe_float(r.get("model_arousal"))
            sx = _safe_float(r.get("self_valence"))
            sy = _safe_float(r.get("self_arousal"))
            scale = (r.get("scale") or "").strip().lower()
            if mx is None or my is None or sx is None or sy is None:
                continue
            if scale == "sam":
                rows.append({"model_x": mx - 5.0, "model_y": my - 5.0, "self_x": sx - 5.0, "self_y": sy - 5.0})
            else:
                rows.append({"model_x": 4.0 * mx, "model_y": 4.0 * my, "self_x": 4.0 * sx, "self_y": 4.0 * sy})
    return rows


def _background(
    ax: plt.Axes,
    *,
    center_x: float,
    center_y: float,
    radius: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    xlabel: str,
    ylabel: str,
) -> None:
    circle = plt.Circle((center_x, center_y), radius, facecolor="#f7f7f7", edgecolor="black", linewidth=1.0, zorder=0)
    ax.add_patch(circle)
    ax.axhline(center_y, color="gray", linewidth=0.8)
    ax.axvline(center_x, color="gray", linewidth=0.8)
    for d in (45, 135, 225, 315):
        x = center_x + radius * math.cos(math.radians(d))
        y = center_y + radius * math.sin(math.radians(d))
        ax.plot([center_x, x], [center_y, y], linestyle="--", linewidth=0.7, color="#999999", alpha=0.7)
    for k, d in ANCHORS.items():
        x = center_x + (radius + 0.35) * math.cos(math.radians(d))
        y = center_y + (radius + 0.35) * math.sin(math.radians(d))
        ax.text(x, y, k, fontsize=8, ha="center", va="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot XYZ circumplex model vs self-report.")
    ap.add_argument("--angles-csv", type=Path, default=None, help="CSV from calc_circumplex_segments.py")
    ap.add_argument(
        "--json-glob",
        type=str,
        default="/home/mabl-main/results/emotion_batch_data/*_xyz_fan.json",
        help="Fallback raw JSON glob if --angles-csv not provided.",
    )
    ap.add_argument("--out", type=Path, default=Path("/home/mabl-main/results/xyz_circumplex_model_vs_self.png"))
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument(
        "--display-scale",
        choices=["sam", "centered"],
        default="sam",
        help="Display axis in SAM units (1..9) or centered units (-4..4). Angle math is identical.",
    )
    args = ap.parse_args()

    rows = _load_from_csv(args.angles_csv) if args.angles_csv is not None else _load_from_json_glob(args.json_glob)
    if not rows:
        raise SystemExit("No XYZ rows found for plotting.")

    mx_c = np.asarray([r["model_x"] for r in rows], dtype=float)
    my_c = np.asarray([r["model_y"] for r in rows], dtype=float)
    sx_c = np.asarray([r["self_x"] for r in rows], dtype=float)
    sy_c = np.asarray([r["self_y"] for r in rows], dtype=float)
    if args.display_scale == "sam":
        mx, my, sx, sy = mx_c + 5.0, my_c + 5.0, sx_c + 5.0, sy_c + 5.0
        bg = dict(
            center_x=5.0,
            center_y=5.0,
            radius=4.0,
            xlim=(0.8, 9.2),
            ylim=(0.8, 9.2),
            xlabel="Valence (SAM 1 to 9)",
            ylabel="Arousal (SAM 1 to 9)",
        )
    else:
        mx, my, sx, sy = mx_c, my_c, sx_c, sy_c
        bg = dict(
            center_x=0.0,
            center_y=0.0,
            radius=4.0,
            xlim=(-4.6, 4.6),
            ylim=(-4.6, 4.6),
            xlabel="Valence centered at SAM 5",
            ylabel="Arousal centered at SAM 5",
        )
    n = len(rows)

    q_match = np.mean(((mx_c >= 0) == (sx_c >= 0)) & ((my_c >= 0) == (sy_c >= 0)))
    mean_dist = np.mean(np.sqrt((mx - sx) ** 2 + (my - sy) ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), constrained_layout=True)

    _background(axes[0], **bg)
    axes[0].scatter(mx, my, s=16, alpha=0.35, color="tab:blue", edgecolors="none")
    axes[0].set_title("XYZ model points")

    _background(axes[1], **bg)
    axes[1].scatter(sx, sy, s=16, alpha=0.35, color="tab:orange", edgecolors="none")
    axes[1].set_title("Self-report points")

    _background(axes[2], **bg)
    axes[2].scatter(mx, my, s=14, alpha=0.22, color="tab:blue", edgecolors="none", label="Model")
    axes[2].scatter(sx, sy, s=14, alpha=0.22, color="tab:orange", edgecolors="none", label="Self-report")
    axes[2].legend(loc="upper right", frameon=True)
    axes[2].set_title("Overlay")

    fig.suptitle(
        f"XYZ circumplex comparison model vs self-report n={n} trials   mean AV distance={mean_dist:.2f}   quadrant match={q_match:.3f}",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
