#!/usr/bin/env python3
"""
Decision-oriented circumplex evaluation plots.

Creates
1) 4x4 quadrant confusion matrix (rows=model, cols=self-report)
2) error distributions for angle difference and AV Euclidean distance
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)


Q_ORDER = ["Q1", "Q2", "Q3", "Q4"]
MAX_SAM_AV_DISTANCE = math.sqrt((9.0 - 1.0) ** 2 + (9.0 - 1.0) ** 2)  # sqrt(128)=11.3137


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _load_rows(path: Path, color_space: str, participant_id: str | None) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if (r.get("color_space") or "").strip().lower() != color_space.lower():
                continue
            if participant_id is not None and (r.get("participant_id") or "").strip() != participant_id:
                continue
            rows.append(r)
    return rows


def _confusion(rows: List[dict]) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=int)
    idx = {q: i for i, q in enumerate(Q_ORDER)}
    for r in rows:
        qm = r.get("model_quadrant", "")
        qs = r.get("self_quadrant", "")
        if qm in idx and qs in idx:
            mat[idx[qm], idx[qs]] += 1
    return mat


def _plot_confusion(mat: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(4), labels=Q_ORDER)
    ax.set_yticks(range(4), labels=Q_ORDER)
    ax.set_xlabel("Self-report quadrant")
    ax.set_ylabel("Model quadrant")
    ax.set_title(title, fontsize=20)
    for i in range(4):
        for j in range(4):
            v = int(mat[i, j])
            ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Trial count")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_errors(angle_diffs: np.ndarray, dists: np.ndarray, out_png: Path, title: str) -> None:
    nd = dists / MAX_SAM_AV_DISTANCE
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    angle_mean = float(np.mean(angle_diffs))
    angle_med = float(np.median(angle_diffs))
    dist_mean = float(np.mean(dists))
    dist_med = float(np.median(dists))
    nd_mean = float(np.mean(nd))
    nd_med = float(np.median(nd))

    axes[0].hist(angle_diffs, bins=22, color="tab:purple", alpha=0.8, edgecolor="white")
    axes[0].axvline(angle_mean, color="black", linestyle="--", linewidth=1.2, label=f"mean={angle_mean:.1f}")
    axes[0].axvline(angle_med, color="tab:red", linestyle="-", linewidth=1.8, label=f"median={angle_med:.1f}")
    axes[0].set_xlabel("Absolute angle difference (deg)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Directional mismatch", fontsize=14)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].hist(dists, bins=22, color="tab:green", alpha=0.8, edgecolor="white")
    axes[1].axvline(dist_mean, color="black", linestyle="--", linewidth=1.2, label=f"mean={dist_mean:.2f}")
    axes[1].axvline(dist_med, color="tab:red", linestyle="-", linewidth=1.8, label=f"median={dist_med:.2f}")
    axes[1].set_xlabel("AV Euclidean distance (SAM units)")
    axes[1].set_ylabel("")
    axes[1].set_title("Magnitude mismatch", fontsize=14)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].hist(nd, bins=22, color="tab:olive", alpha=0.8, edgecolor="white")
    axes[2].axvline(nd_mean, color="black", linestyle="--", linewidth=1.2, label=f"mean={nd_mean:.3f}")
    axes[2].axvline(nd_med, color="tab:red", linestyle="-", linewidth=1.8, label=f"median={nd_med:.3f}")
    axes[2].set_xlabel("Normalized AV distance (0 to 1)")
    axes[2].set_ylabel("")
    axes[2].set_title("Normalized magnitude mismatch", fontsize=14)
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    fig.suptitle(title, fontsize=20)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quadrant confusion and error distributions from circumplex angle CSV.")
    ap.add_argument("--angles-csv", type=Path, required=True)
    ap.add_argument("--color-space", type=str, default="xyz")
    ap.add_argument("--participant-id", type=str, default=None, help="Optional single participant filter")
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    rows = _load_rows(args.angles_csv, args.color_space, args.participant_id)
    if not rows:
        raise SystemExit("No rows matched filters")

    mat = _confusion(rows)
    n = int(np.sum(mat))
    acc = float(np.trace(mat) / n) if n > 0 else float("nan")
    # balanced accuracy over true classes (columns are self-report labels)
    col_sums = mat.sum(axis=0)
    recalls = [float(mat[i, i] / col_sums[i]) if col_sums[i] > 0 else float("nan") for i in range(4)]
    bal_acc = float(np.nanmean(recalls))
    # macro F1 (rows are predictions, columns are true labels)
    f1s = []
    for i in range(4):
        tp = float(mat[i, i])
        fp = float(np.sum(mat[i, :]) - tp)
        fn = float(np.sum(mat[:, i]) - tp)
        den = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / den) if den > 0 else float("nan"))
    macro_f1 = float(np.nanmean(f1s))
    # Cohen's kappa
    pe = float(np.sum(np.sum(mat, axis=1) * np.sum(mat, axis=0)) / (n * n)) if n > 0 else float("nan")
    kappa = float((acc - pe) / (1.0 - pe)) if n > 0 and (1.0 - pe) > 1e-12 else float("nan")

    ad = np.asarray([_safe_float(r.get("angle_abs_diff_deg", "")) for r in rows], dtype=float)
    ad = ad[np.isfinite(ad)]
    dists = []
    for r in rows:
        mv = _safe_float(r.get("model_valence", ""))
        ma = _safe_float(r.get("model_arousal", ""))
        sv = _safe_float(r.get("self_valence", ""))
        sa = _safe_float(r.get("self_arousal", ""))
        if not all(np.isfinite([mv, ma, sv, sa])):
            continue
        # CSV is generated with scale=sam in our workflow.
        dists.append(math.sqrt((mv - sv) ** 2 + (ma - sa) ** 2))
    d = np.asarray(dists, dtype=float)
    nd = d / MAX_SAM_AV_DISTANCE if len(d) else np.asarray([], dtype=float)
    # circular dispersion from signed wrapped angle differences
    signed_deg = []
    for r in rows:
        ma = _safe_float(r.get("model_angle_deg", ""))
        sa = _safe_float(r.get("self_angle_deg", ""))
        if not np.isfinite(ma) or not np.isfinite(sa):
            continue
        signed_deg.append(((ma - sa + 180.0) % 360.0) - 180.0)
    signed_rad = np.radians(np.asarray(signed_deg, dtype=float))
    if len(signed_rad):
        C = float(np.mean(np.cos(signed_rad)))
        S = float(np.mean(np.sin(signed_rad)))
        R = float(math.sqrt(C * C + S * S))
        circ_var = float(1.0 - R)
    else:
        R = float("nan")
        circ_var = float("nan")

    tag = f"{args.color_space}" + (f"_pid{args.participant_id}" if args.participant_id else "_all")
    unique_pids = sorted({(r.get("participant_id") or "").strip() for r in rows if (r.get("participant_id") or "").strip()})
    n_participants = len(unique_pids)
    t = f"{args.color_space.upper()} Model-Human Affect Misalignment (N = {n} Trials, {n_participants} Participants)"
    conf_png = args.out_prefix.with_name(args.out_prefix.name + f"_{tag}_quadrant_confusion.png")
    err_png = args.out_prefix.with_name(args.out_prefix.name + f"_{tag}_error_distributions.png")
    js = args.out_prefix.with_name(args.out_prefix.name + f"_{tag}_summary.json")

    _plot_confusion(mat, conf_png, t)
    _plot_errors(
        ad,
        d,
        err_png,
        f"{args.color_space.upper()} Model-Human Affect Misalignment (N = {n})",
    )

    summary: Dict[str, object] = {
        "color_space": args.color_space,
        "participant_id": args.participant_id,
        "n_trials": n,
        "n_participants": n_participants,
        "quadrant_confusion_rows_model_cols_self": mat.tolist(),
        "quadrant_accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "recall_by_quadrant_q1_q2_q3_q4": recalls,
        "prediction_distribution_q1_q2_q3_q4": (np.sum(mat, axis=1) / n).tolist() if n > 0 else None,
        "self_distribution_q1_q2_q3_q4": (np.sum(mat, axis=0) / n).tolist() if n > 0 else None,
        "mean_abs_angle_diff_deg": float(np.mean(ad)) if len(ad) else None,
        "median_abs_angle_diff_deg": float(np.median(ad)) if len(ad) else None,
        "iqr_abs_angle_diff_deg": float(np.percentile(ad, 75) - np.percentile(ad, 25)) if len(ad) else None,
        "circular_resultant_length": R,
        "circular_variance": circ_var,
        "mean_av_distance_sam": float(np.mean(d)) if len(d) else None,
        "median_av_distance_sam": float(np.median(d)) if len(d) else None,
        "iqr_av_distance_sam": float(np.percentile(d, 75) - np.percentile(d, 25)) if len(d) else None,
        "max_av_distance_sam": MAX_SAM_AV_DISTANCE,
        "mean_normalized_av_distance": float(np.mean(nd)) if len(nd) else None,
        "median_normalized_av_distance": float(np.median(nd)) if len(nd) else None,
        "iqr_normalized_av_distance": float(np.percentile(nd, 75) - np.percentile(nd, 25)) if len(nd) else None,
        "outputs": {"confusion_png": str(conf_png), "errors_png": str(err_png)},
    }
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {conf_png}")
    print(f"Wrote {err_png}")
    print(f"Wrote {js}")


if __name__ == "__main__":
    main()
