#!/usr/bin/env python3
"""
Build Table 1 (train/val MSE at best mean val CCC checkpoint via best_checkpoint_metrics.json when present)
and Table 2 (test CCC/PCC/RMSE/SAGR
from comparison_report.json) in the same style as tables_afewva.tex.

  python export_afewva_methodology_tables.py \\
    --output-base comparison_output_methodology \\
    --report comparison_output_methodology/comparison_report.json \\
    --out comparison_output_methodology/tables_afewva_methodology.tex
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

TRAIN_RE = re.compile(
    r"Train Loss:\s*([\d.]+)\s*\(V:\s*([\d.]+),\s*A:\s*([\d.]+)\)"
)
VAL_RE = re.compile(
    r"Val Loss:\s*([\d.]+)\s*\(V:\s*([\d.]+),\s*A:\s*([\d.]+)\)"
)

# Descending mean overall test CCC (XYZ > RGB > LAB > YCbCr); same row order as Table I.
SPACE_ORDER = ["xyz", "rgb", "lab", "ycbcr"]
SPACE_LABEL = {"lab": "LAB", "ycbcr": "YCbCr", "xyz": "XYZ", "rgb": "RGB"}


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


def latest_log(run_dir: Path) -> Path | None:
    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        return None
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_final_epoch_losses(log_path: Path) -> Tuple[float, float, float, float, float, float] | None:
    text = log_path.read_text(errors="replace")
    trains = TRAIN_RE.findall(text)
    vals = VAL_RE.findall(text)
    if not trains or not vals:
        return None
    t = trains[-1]
    v = vals[-1]
    train_tot, train_v, train_a = map(float, t)
    val_tot, val_v, val_a = map(float, v)
    return train_tot, train_v, train_a, val_tot, val_v, val_a


def parse_best_checkpoint_metrics(run_dir: Path) -> Tuple[float, float, float, float, float, float] | None:
    """MSE at the checkpoint selected by highest mean val CCC (train_emonet*.py)."""
    p = run_dir / "best_checkpoint_metrics.json"
    if not p.is_file():
        return None
    j = json.loads(p.read_text())
    try:
        return (
            float(j["train_loss"]),
            float(j["train_v_loss"]),
            float(j["train_a_loss"]),
            float(j["val_loss"]),
            float(j["val_v_loss"]),
            float(j["val_a_loss"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def collect_table1(output_base: Path) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {
        s: {
            "train": [],
            "train_v": [],
            "train_a": [],
            "val": [],
            "val_v": [],
            "val_a": [],
        }
        for s in SPACE_ORDER
    }
    for space in SPACE_ORDER:
        for run_id in range(1, 6):
            rd = output_base / space / f"run_{run_id}"
            row = parse_best_checkpoint_metrics(rd)
            if row is None:
                lp = latest_log(rd)
                if lp is None:
                    continue
                row = parse_final_epoch_losses(lp)
                if row is None:
                    continue
            tt, tv, ta, vt, vv, va = row
            out[space]["train"].append(tt)
            out[space]["train_v"].append(tv)
            out[space]["train_a"].append(ta)
            out[space]["val"].append(vt)
            out[space]["val_v"].append(vv)
            out[space]["val_a"].append(va)
    return out


def cell(m: float, se: float, nd: int = 3) -> str:
    if not math.isfinite(m):
        return "---"
    if not math.isfinite(se):
        se = 0.0
    return f"${m:.{nd}f} \\pm {se:.{nd}f}$"


def cell_bold(m: float, se: float, nd: int = 3) -> str:
    if not math.isfinite(m):
        return "---"
    if not math.isfinite(se):
        se = 0.0
    return f"$\\mathbf{{{m:.{nd}f} \\pm {se:.{nd}f}}}$"


def emit_table1(t1: Dict[str, Dict[str, List[float]]]) -> str:
    rows = []
    for space in SPACE_ORDER:
        d = t1[space]
        rows.append(
            f"{SPACE_LABEL[space]:5} & "
            f"{cell(*mean_se(d['train']))} & {cell(*mean_se(d['train_v']))} & {cell(*mean_se(d['train_a']))} & "
            f"{cell(*mean_se(d['val']))} & {cell(*mean_se(d['val_v']))} & {cell(*mean_se(d['val_a']))} \\\\"
        )
    body = "\n".join(rows)
    return f"""% Table 1: Train/val MSE (mean $\\pm$ SE, $n=5$) at the checkpoint with highest mean validation CCC; 60/20/20; seeds 43--47. From best_checkpoint_metrics.json when present, else final log line.
\\begin{{table*}}[tp]
\\centering
\\footnotesize
\\caption{{Training and validation MSE across color spaces on AFEW-VA (mean $\\pm$ SE, $n=5$),
averaged over five runs with seeds 43--47, ordered by test CCC to match Table~\\ref{{tab:afewva}}. Values are at the selected checkpoint (best mean validation CCC), not necessarily the minimum validation loss epoch.}}
\\label{{tab:colorspace_results}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Color Space}} & \\textbf{{Train Loss}} & \\textbf{{Train MSE (V)}} & \\textbf{{Train MSE (A)}} & \\textbf{{Val. Loss}} & \\textbf{{Val. MSE (V)}} & \\textbf{{Val. MSE (A)}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def load_table2(report_path: Path) -> Dict[str, Dict[str, Tuple[float, float]]]:
    data = json.loads(report_path.read_text())
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for space in SPACE_ORDER:
        ms = data["results"][space]["mean_se"]
        out[space] = {
            "v_ccc": tuple(ms["valence_ccc"]),
            "v_pcc": tuple(ms["valence_pcc"]),
            "v_rmse": tuple(ms["valence_rmse"]),
            "v_sagr": tuple(ms["valence_sagr"]),
            "a_ccc": tuple(ms["arousal_ccc"]),
            "a_pcc": tuple(ms["arousal_pcc"]),
            "a_rmse": tuple(ms["arousal_rmse"]),
            "a_sagr": tuple(ms["arousal_sagr"]),
        }
    return out


def best_mask_table2(t2: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[str, Dict[str, bool]]:
    """Bold best among LAB/YCbCr/XYZ/RGB: maximize CCC/PCC/SAGR; minimize RMSE."""
    keys = [
        "v_ccc",
        "v_pcc",
        "v_rmse",
        "v_sagr",
        "a_ccc",
        "a_pcc",
        "a_rmse",
        "a_sagr",
    ]
    maximize = {"v_ccc", "v_pcc", "v_sagr", "a_ccc", "a_pcc", "a_sagr"}
    mask: Dict[str, Dict[str, bool]] = {s: {k: False for k in keys} for s in SPACE_ORDER}
    for k in keys:
        means = [(t2[s][k][0], s) for s in SPACE_ORDER]
        means = [(m, s) for m, s in means if math.isfinite(m)]
        if not means:
            continue
        if k in maximize:
            best_m = max(m for m, _ in means)
            winners = [s for m, s in means if abs(m - best_m) < 1e-6]
        else:
            best_m = min(m for m, _ in means)
            winners = [s for m, s in means if abs(m - best_m) < 1e-6]
        for s in winners:
            mask[s][k] = True
    return mask


def emit_table2(
    t2: Dict[str, Dict[str, Tuple[float, float]]],
    mask: Dict[str, Dict[str, bool]],
) -> str:
    rows = []
    for space in SPACE_ORDER:
        d = t2[space]
        msk = mask[space]
        cells = []
        order = ("v_ccc", "v_pcc", "v_rmse", "v_sagr", "a_ccc", "a_pcc", "a_rmse", "a_sagr")
        for k in order:
            mu, se = d[k]
            cells.append(cell_bold(mu, se) if msk[k] else cell(mu, se))
        # Two lines for readability (match tables_afewva.tex line break style)
        rows.append(
            f"{SPACE_LABEL[space]:5} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]}\n"
            f"      & {cells[4]} & {cells[5]} & {cells[6]} & {cells[7]} \\\\"
        )
    body = "\n".join(rows)
    return f"""% Table 2: Held-out test set (6013 frames, clip-level 20\\% test, seed 123). Mean $\\pm$ SE over $n=5$ runs. Best among fine-tuned spaces in \\textbf{{bold}}.
\\begin{{table*}}[tp]
\\centering
\\caption{{Results on AFEW-VA (mean $\\pm$ SE, $n=5$). EmoNet \\cite{{toisoul2021estimation}}
reflects full end-to-end training; ours use a frozen backbone. Rows follow Table~\\ref{{tab:colorspace_results}} (descending mean overall test CCC). Best fine-tuned result per metric in \\textbf{{bold}}. Valence metrics overlap across spaces; arousal CCC is lower for LAB than RGB/XYZ. Larger arousal PCC$-$CCC gaps than valence indicate weaker concordance calibration relative to correlation.}}
\\label{{tab:afewva}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccccc}}
\\hline
 & \\multicolumn{{4}}{{c}}{{Valence}} & \\multicolumn{{4}}{{c}}{{Arousal}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}}
Method & CCC & PCC & RMSE & SAGR & CCC & PCC & RMSE & SAGR \\\\
\\hline
EmoNet \\cite{{toisoul2021estimation}} & 0.69 & 0.70 & -- & -- & 0.66 & 0.67 & -- & -- \\\\
\\hline
{body}
\\hline
\\end{{tabular}}}}
\\end{{table*}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-base", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t1 = collect_table1(args.output_base)
    for s in SPACE_ORDER:
        n = len(t1[s]["train"])
        if n != 5:
            print(f"Warning: {s} has {n}/5 runs with parseable logs.")

    t2 = load_table2(args.report)
    mask = best_mask_table2(t2)

    tex = (
        "% Auto-generated by export_afewva_methodology_tables.py\n"
        "% \\usepackage{booktabs} for Table 1; same + \\cmidrule for Table 2.\n\n"
        + emit_table1(t1)
        + "\n\n"
        + emit_table2(t2, mask)
        + "\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
