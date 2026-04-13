#!/usr/bin/env python3
"""
Plot segment-level valence/arousal trajectories for selected timed trials.

By default ``--trials all`` emits one figure per music-video trial (typically 22 per
participant). Output names include a sanitized ``videoName`` slug so files match the
stimulus label, not only the trial index.

Each figure shows:
- segment means over time (from JSON `segments`), same blue as the trial mean
- dashed blue horizontal line: trial-level model mean (`video_mean_*`)
- solid orange horizontal line: self-report (``--sam-style line``, default; use ``marker`` for end-of-trial star).
  Y-limits expand per panel so self-report is never clipped; raw SAM (1–9) is mapped to model
  space when scaled fields are absent.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
}


def _safe_float(v: Any) -> Optional[float]:
    try:
        z = float(v)
    except Exception:
        return None
    if np.isfinite(z):
        return z
    return None


def _model_to_sam(x: np.ndarray) -> np.ndarray:
    return 4.0 * x + 5.0


def _model_scalar_to_sam(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return 4.0 * float(x) + 5.0


def _subjective_model_axis(trial: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Valence/arousal in approximately [-1, 1] for overlay with model traces.
    Prefer *_scaled; else map raw SAM (1–9) via (x-5)/4 when |raw| > 1.25.
    """
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


def _subjective_sam_plot(trial: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Observer self-report on the SAM 1–9 axis for plotting alongside model→SAM traces.

    If ``subjective_*`` is already in [1, 9], use it as SAM. Otherwise use the same model-axis
    rule as segment means (``_subjective_model_axis``), then map m → SAM with ``4*m+5``.
    """
    sv_m, sa_m = _subjective_model_axis(trial)
    sv_r = _safe_float(trial.get("subjective_valence"))
    sa_r = _safe_float(trial.get("subjective_arousal"))

    def one_sam(model_m: Optional[float], raw: Optional[float]) -> Optional[float]:
        if raw is not None and 1.0 <= raw <= 9.0:
            return float(raw)
        return _model_scalar_to_sam(model_m)

    return one_sam(sv_m, sv_r), one_sam(sa_m, sa_r)


def _ylim_for_panel(
    seg: np.ndarray,
    mean: Optional[float],
    subj: Optional[float],
    default_lo: float,
    default_hi: float,
    pad_frac: float = 0.12,
) -> Tuple[float, float]:
    """Y-limits that include segment, mean, and self-report (never clip horizontal guides)."""
    vals: List[float] = []
    if seg.size:
        vals.extend([float(np.nanmin(seg)), float(np.nanmax(seg))])
    if mean is not None and np.isfinite(mean):
        vals.append(float(mean))
    if subj is not None and np.isfinite(subj):
        vals.append(float(subj))
    if not vals:
        return default_lo, default_hi
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    pad = max(0.06, pad_frac * span)
    return min(lo - pad, default_lo), max(hi + pad, default_hi)


def _trial_series(trial: Dict[str, Any], fps_default: float = 25.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    segs = trial.get("segments") or []
    fps = _safe_float(trial.get("fps_reported")) or fps_default
    if not segs:
        return np.array([]), np.array([]), np.array([])

    t_sec: List[float] = []
    vv: List[float] = []
    aa: List[float] = []
    f0_trial = float(segs[0].get("frame_start", 0))
    for s in segs:
        v = _safe_float(s.get("valence_mean"))
        a = _safe_float(s.get("arousal_mean"))
        if v is None or a is None:
            continue
        f0 = float(s.get("frame_start", 0))
        f1 = float(s.get("frame_end", f0))
        t_mid = ((f0 + f1) * 0.5 - f0_trial) / fps
        t_sec.append(t_mid)
        vv.append(v)
        aa.append(a)
    return np.asarray(t_sec), np.asarray(vv), np.asarray(aa)


def _slug_video_name(name: str, max_len: int = 80) -> str:
    """Filesystem-safe fragment from ratings ``videoName`` (e.g. ``Artist Song.mp4``)."""
    base = str(name).strip()
    for ext in (".mp4", ".MP4", ".mov", ".MOV", ".webm", ".WEBM"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    base = base.replace("\x00", "")
    base = re.sub(r'[<>:"/\\|?*]+', "_", base)
    base = re.sub(r"\s+", "_", base.strip())
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "video"
    if len(base) > max_len:
        base = base[:max_len].rstrip("_")
    return base


def _title_color_space(input_path: Path, data: Dict[str, Any]) -> str:
    """
    Color-space label for the figure suptitle.

    Prefer the canonical stem ``…_{space}_fan`` (e.g. ``15_xyz_fan.json`` → ``xyz``) so the
    title always matches the file you passed on the CLI. Fall back to JSON ``color_space``.
    If both exist and disagree, warn once (mis-tagged JSON or wrong ``--prefix``).
    """
    stem = input_path.stem
    from_fn: Optional[str] = None
    if stem.lower().endswith("_fan"):
        base = stem[: -len("_fan")]
        if "_" in base:
            from_fn = base.rsplit("_", 1)[-1].strip().lower() or None
    js = data.get("color_space")
    from_json = str(js).strip().lower() if js is not None and str(js).strip() != "" else None
    if from_fn and from_json and from_fn != from_json:
        print(
            f"Warning: input {input_path.name!r} implies color_space={from_fn!r} "
            f"but JSON has {from_json!r}; using filename for figure title.",
            file=sys.stderr,
        )
    if from_fn:
        return from_fn
    return from_json or "?"


def _plot_one(
    trial: Dict[str, Any],
    trial_idx: int,
    out_path: Path,
    title_prefix: str,
    sam_style: str,
    plot_scale: str,
    *,
    figure_style: str = "default",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 180,
) -> bool:
    t, v, a = _trial_series(trial)
    if len(t) == 0:
        return False

    pv = _safe_float(trial.get("video_mean_valence"))
    pa = _safe_float(trial.get("video_mean_arousal"))
    sv_m, sa_m = _subjective_model_axis(trial)

    # Visualization scale:
    # - model: native model units [-1,1]
    # - sam: convert model outputs to SAM-like units [1,9] for readability
    if plot_scale == "sam":
        v = _model_to_sam(v)
        a = _model_to_sam(a)
        pv = _model_scalar_to_sam(pv)
        pa = _model_scalar_to_sam(pa)
        sv_plot, sa_plot = _subjective_sam_plot(trial)
        ylab_v = "Valence (SAM 1–9)"
        ylab_a = "Arousal (SAM 1–9)"
        ylim_v = _ylim_for_panel(v, pv, sv_plot, 1.0, 9.0)
        ylim_a = _ylim_for_panel(a, pa, sa_plot, 1.0, 9.0)
    else:
        sv_plot = sv_m
        sa_plot = sa_m
        ylab_v = "Valence"
        ylab_a = "Arousal"
        ylim_v = _ylim_for_panel(v, pv, sv_plot, -1.05, 1.05)
        ylim_a = _ylim_for_panel(a, pa, sa_plot, -1.05, 1.05)

    video_name = str(trial.get("videoName", f"trial_{trial_idx}"))
    if figsize is None:
        figsize = (7.0, 4.25) if figure_style == "paper" else (9.0, 6.0)

    lbl_seg_v = "Valence (segment means)" if figure_style == "paper" else "segment valence"
    lbl_seg_a = "Arousal (segment means)" if figure_style == "paper" else "segment arousal"
    lbl_mean = "Trial mean (model)" if figure_style == "paper" else "trial mean pred"
    x_lab = "Time (s)" if figure_style == "paper" else "Time in trial (s)"
    suptitle_fs = 10 if figure_style == "paper" else 11
    sup = f"{title_prefix}\n{video_name}"

    rc = _PAPER_RC if figure_style == "paper" else {}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Valence panel
        axes[0].plot(
            t,
            v,
            marker="o",
            markersize=2.5 if figure_style == "default" else 2.0,
            linewidth=1.0,
            color="tab:blue",
            label=lbl_seg_v,
            zorder=2,
        )
        if pv is not None:
            axes[0].axhline(
                pv,
                linestyle="--",
                linewidth=1.2,
                color="tab:blue",
                label=f"{lbl_mean} ({pv:.2f})",
                zorder=4,
            )
        if sv_plot is not None and sam_style == "line":
            axes[0].axhline(
                sv_plot,
                linestyle="-",
                linewidth=1.6,
                color="tab:orange",
                label="Self-report (trial)",
                zorder=5,
            )
        elif sv_plot is not None and sam_style == "marker":
            axes[0].plot(
                [t[-1]],
                [sv_plot],
                marker="*",
                markersize=10,
                linestyle="None",
                color="tab:orange",
                label="Self-report (trial)",
                zorder=5,
            )
        axes[0].set_ylabel(ylab_v)
        axes[0].set_ylim(*ylim_v)
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="best")

        # Arousal panel
        axes[1].plot(
            t,
            a,
            marker="o",
            markersize=2.5 if figure_style == "default" else 2.0,
            linewidth=1.0,
            color="tab:blue",
            label=lbl_seg_a,
            zorder=2,
        )
        if pa is not None:
            axes[1].axhline(
                pa,
                linestyle="--",
                linewidth=1.2,
                color="tab:blue",
                label=f"{lbl_mean} ({pa:.2f})",
                zorder=4,
            )
        if sa_plot is not None and sam_style == "line":
            axes[1].axhline(
                sa_plot,
                linestyle="-",
                linewidth=1.6,
                color="tab:orange",
                label="Self-report (trial)",
                zorder=5,
            )
        elif sa_plot is not None and sam_style == "marker":
            axes[1].plot(
                [t[-1]],
                [sa_plot],
                marker="*",
                markersize=10,
                linestyle="None",
                color="tab:orange",
                label="Self-report (trial)",
                zorder=5,
            )
        axes[1].set_ylabel(ylab_a)
        axes[1].set_xlabel(x_lab)
        axes[1].set_ylim(*ylim_a)
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="best")

        fig.suptitle(sup, fontsize=suptitle_fs, y=1.02)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot segment-level trial snippets from timed-trial JSON")
    ap.add_argument("--input", type=Path, required=True, help="Timed-trial JSON file")
    ap.add_argument(
        "--trials",
        type=str,
        default="all",
        help="Comma-separated 0-based trial indices (e.g. 0,1,7) or 'all' (default: every music-video trial).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("snippet_plots"),
        help="Output directory for PNG plots",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="trial_snippet",
        help="Filename prefix (e.g. lab_15 or xyz_15)",
    )
    ap.add_argument(
        "--output-naming",
        type=str,
        default="video",
        choices=["video", "index"],
        help="video: trial02_Wilco_How_to_Fight_Loneliness.png; index: legacy trial02.png only.",
    )
    ap.add_argument(
        "--sam-style",
        type=str,
        default="line",
        choices=["marker", "line", "none"],
        help="Self-report: orange horizontal line (line, default), star at trial end (marker), or off (none).",
    )
    ap.add_argument(
        "--plot-scale",
        type=str,
        default="sam",
        choices=["model", "sam"],
        help="Visualization axis: sam (default) maps model outputs to SAM 1–9 (4*y+5); aligns self-report on same axis.",
    )
    ap.add_argument(
        "--figure-style",
        type=str,
        default="default",
        choices=["default", "paper"],
        help="paper: serif fonts, compact size, labels suitable for IEEE-style figures.",
    )
    ap.add_argument("--dpi", type=int, default=180, help="Raster resolution (PNG).")
    ap.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output figure format.",
    )
    ap.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches (optional; paper defaults to 7).",
    )
    ap.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Figure height in inches (optional; paper defaults to 4.25).",
    )
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    trials = data.get("trials") or []
    if not trials:
        raise SystemExit("No trials in input JSON.")
    if args.trials.strip().lower() == "all":
        idxs = list(range(len(trials)))
    else:
        idxs = [int(x.strip()) for x in args.trials.split(",") if x.strip()]
    cs_title = _title_color_space(args.input, data)
    title_prefix = f"PID {data.get('physio_participant_id', '?')} | {cs_title}"

    fs = args.figure_style
    fw = args.fig_width
    fh = args.fig_height
    if fw is not None or fh is not None:
        default_w, default_h = (7.0, 4.25) if fs == "paper" else (9.0, 6.0)
        figsize = (fw if fw is not None else default_w, fh if fh is not None else default_h)
    else:
        figsize = None

    for i in idxs:
        if i < 0 or i >= len(trials):
            continue
        trial = trials[i]
        vn = str(trial.get("videoName") or f"trial_{i}")
        slug = _slug_video_name(vn)
        if args.output_naming == "video":
            out = args.out_dir / f"{args.prefix}_trial{i:02d}_{slug}.{args.format}"
        else:
            out = args.out_dir / f"{args.prefix}_trial{i:02d}.{args.format}"
        ok = _plot_one(
            trial,
            i,
            out,
            title_prefix,
            args.sam_style,
            args.plot_scale,
            figure_style=fs,
            figsize=figsize,
            dpi=args.dpi,
        )
        if ok:
            print(f"Wrote {out}")
        else:
            print(f"Skip trial {i} ({vn}): no segment trace", file=sys.stderr)


if __name__ == "__main__":
    main()
