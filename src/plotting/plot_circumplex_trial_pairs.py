#!/usr/bin/env python3
"""
Circumplex plot for one participant where each trial has:
- model point
- self-report point
- a connecting line between them
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def _angle_deg(yc: float, xc: float) -> float:
    a = math.degrees(math.atan2(yc, xc))
    return a + 360.0 if a < 0 else a


def _ang_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _background(ax: plt.Axes) -> None:
    c = plt.Circle((5.0, 5.0), 4.0, facecolor="#f7f7f7", edgecolor="black", linewidth=1.0, zorder=0)
    ax.add_patch(c)
    ax.axhline(5.0, color="gray", linewidth=0.8)
    ax.axvline(5.0, color="gray", linewidth=0.8)
    for d in (0, 60, 90, 110, 180, 240, 270, 290, 360):
        x = 5.0 + 4.0 * math.cos(math.radians(d))
        y = 5.0 + 4.0 * math.sin(math.radians(d))
        ax.plot([5.0, x], [5.0, y], linestyle="--", linewidth=0.7, color="#9a9a9a", alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.8, 9.2)
    ax.set_ylim(0.8, 9.2)
    ax.set_xlabel("Valence (SAM 1 to 9)")
    ax.set_ylabel("Arousal (SAM 1 to 9)")
    ax.grid(alpha=0.2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot trial-wise model/self circumplex pairs for one participant JSON.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=240)
    ap.add_argument("--annotate-trial-index", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    pid = data.get("physio_participant_id", "?")
    space = data.get("color_space", "?")
    trials = data.get("trials") or []

    rows: List[Tuple[int, float, float, float, float, float]] = []
    for i, t in enumerate(trials):
        if t.get("error"):
            continue
        pv = _safe_float(t.get("video_mean_valence"))
        pa = _safe_float(t.get("video_mean_arousal"))
        sv, sa = _subjective_sam_axis(t)
        if pv is None or pa is None or sv is None or sa is None:
            continue
        mv, ma = _model_to_sam(pv), _model_to_sam(pa)
        am = _angle_deg(ma - 5.0, mv - 5.0)
        asr = _angle_deg(sa - 5.0, sv - 5.0)
        rows.append((i, mv, ma, sv, sa, _ang_dist(am, asr)))

    if not rows:
        raise SystemExit("No valid paired trials found in input.")

    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    _background(ax)

    for i, mv, ma, sv, sa, _ in rows:
        ax.plot([mv, sv], [ma, sa], color="#888888", linewidth=1.0, alpha=0.65, zorder=1)
        ax.scatter([mv], [ma], color="tab:blue", s=34, zorder=3)
        ax.scatter([sv], [sa], color="tab:red", s=34, zorder=3)
        if args.annotate_trial_index:
            ax.text((mv + sv) / 2.0, (ma + sa) / 2.0, str(i), fontsize=7, color="#444444")

    # Legend proxies
    ax.plot([], [], color="#888888", linewidth=1.2, label="Trial pair link")
    ax.scatter([], [], color="tab:blue", s=36, label="Model")
    ax.scatter([], [], color="tab:red", s=36, label="Self-report")
    ax.legend(loc="upper right", frameon=True)

    mean_abs = float(np.mean([r[-1] for r in rows]))
    q_match = float(
        np.mean(
            [
                ((mv >= 5.0) == (sv >= 5.0)) and ((ma >= 5.0) == (sa >= 5.0))
                for _, mv, ma, sv, sa, _ in rows
            ]
        )
    )
    ax.set_title(f"PID {pid}  {space}  n={len(rows)}  mean angle diff={mean_abs:.1f} deg  quadrant match={q_match:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
