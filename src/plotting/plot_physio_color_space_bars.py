#!/usr/bin/env python3
"""
Bar charts: mean ± SE across participants for each color space (MABL-ME / timed-trial JSONs).

Style resembles the AFEW-VA paper figures (grouped bars, error bars = SE over participants).

  python3 plot_physio_color_space_bars.py \
    --data-dir /home/mabl-main/results/emotion_batch_data \
    --out /home/mabl-main/results/emotion_batch_data/mabl_me_color_space_ccc
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np

# Match paper / Table II ordering
SPACE_ORDER = ["rgb", "ycbcr", "xyz", "lab"]
SPACE_LABELS = ["RGB", "YCbCr", "XYZ", "LAB"]


def mean_se(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    if len(a) < 2:
        return m, 0.0
    se = float(np.std(a, ddof=1) / math.sqrt(len(a)))
    return m, se


def load_metrics(
    data_dir: Path,
) -> Tuple[DefaultDict[str, Dict[str, List[float]]], int]:
    """
    Returns (by_space[space_key][metric_name] -> list of per-participant values, n_files_used).
    metric names: ccc_v, ccc_a, pcc_v, pcc_a
    """
    by_space: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"ccc_v": [], "ccc_a": [], "pcc_v": [], "pcc_a": []}
    )
    n = 0
    for path in sorted(data_dir.glob("*_fan.json")):
        data = json.loads(path.read_text())
        cs = (data.get("color_space") or "").lower().strip()
        if cs not in SPACE_ORDER:
            continue
        comp = data.get("comparison") or {}
        if "note" in comp and "n_paired" not in comp:
            continue
        def pull(keys: List[str]) -> Any:
            for k in keys:
                if k in comp and comp[k] is not None:
                    return comp[k]
            return None

        for key, dst in [
            (["ccc_valence"], "ccc_v"),
            (["ccc_arousal"], "ccc_a"),
            (["pearson_valence", "pcc_valence"], "pcc_v"),
            (["pearson_arousal", "pcc_arousal"], "pcc_a"),
        ]:
            v = pull(key)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                by_space[cs][dst].append(float(v))
        n += 1
    return by_space, n


def plot_panel(
    ax,
    metric_key: str,
    ylabel: str,
    by_space: DefaultDict[str, Dict[str, List[float]]],
    title: str,
) -> None:
    means, ses = [], []
    for s in SPACE_ORDER:
        m, se = mean_se(by_space[s][metric_key])
        means.append(m)
        ses.append(se)
    x = np.arange(len(SPACE_ORDER))
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
    ax.bar(x, means, yerr=ses, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(SPACE_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.grid(axis="y", alpha=0.35, linestyle=":")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot color-space comparison for Physio JSON batch")
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory with *_fan.json")
    ap.add_argument("--out", type=Path, required=True, help="Output path without extension (adds .png/.pdf)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Install matplotlib: pip install matplotlib")

    by_space, n_files = load_metrics(args.data_dir)
    if n_files == 0:
        raise SystemExit(f"No *_fan.json found under {args.data_dir}")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    plot_panel(
        axes[0],
        "ccc_v",
        "CCC (valence)",
        by_space,
        "MABL-ME external validation — Valence",
    )
    plot_panel(
        axes[1],
        "ccc_a",
        "CCC (arousal)",
        by_space,
        "MABL-ME external validation — Arousal",
    )
    counts = [len(by_space[s]["ccc_v"]) for s in SPACE_ORDER]
    n_p = min(counts) if counts else 0
    fig.suptitle(
        f"Mean ± SE across participants (N={n_p} per space; {n_files} JSON files)\n"
        "Metrics computed trial-level within each participant; bars = mean CCC across participants, "
        "error bars = SE",
        fontsize=10,
    )

    out_base = args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    plt.close(fig)

    # Second figure: PCC
    fig2, axes2 = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    plot_panel(axes2[0], "pcc_v", "PCC (valence)", by_space, "MABL-ME — Pearson V")
    plot_panel(axes2[1], "pcc_a", "PCC (arousal)", by_space, "MABL-ME — Pearson A")
    fig2.suptitle(
        f"Mean ± SE across participants (PCC)\nN={n_p} per space",
        fontsize=10,
    )
    png2 = out_base.parent / (out_base.name + "_pcc")
    fig2.savefig(png2.with_suffix(".png"), dpi=args.dpi)
    fig2.savefig(png2.with_suffix(".pdf"))
    plt.close(fig2)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {png2.with_suffix('.png')}")
    print(f"Wrote {png2.with_suffix('.pdf')}")

    # Tiny summary table to stdout
    print("\n## Mean ± SE (participants)")
    for name, key in [("CCC_V", "ccc_v"), ("CCC_A", "ccc_a"), ("PCC_V", "pcc_v"), ("PCC_A", "pcc_a")]:
        parts = []
        for s, lab in zip(SPACE_ORDER, SPACE_LABELS):
            m, se = mean_se(by_space[s][key])
            parts.append(f"{lab}={m:.3f}±{se:.3f}")
        print(f"  {name}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
