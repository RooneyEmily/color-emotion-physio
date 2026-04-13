#!/usr/bin/env python3
"""
Per-participant and cohort-level agreement between trial-mean model predictions and SAM / subjective ratings.

Computes **CCC** (Lin), **PCC** (Pearson r), **RMSE**, and **SAGR** from timed-trial JSON
(``*_lab_fan.json``, ``*_xyz_fan.json``, …). Uses the same definitions as ``process_videos_av.py`` /
``mabl_permutation_baseline.py``.

- ``--axis model`` (default): predictions and ratings in model space (subjective scaled per JSON,
  same as the embedded ``comparison`` block).
- ``--axis sam``: map predictions to SAM via ``4 * m + 5``; ratings use raw SAM in ``[1, 9]`` when
  present, else ``4 * subjective_*_scaled + 5``. PCC and CCC match the model-space values (common
  affine); RMSE is reported in SAM units; SAGR uses agreement of signs relative to neutral **5**
  on the SAM axis (equivalent to model-space SAGR when ratings come from the DEAP linear map).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x, ddof=1) < 1e-12 or np.std(y, ddof=1) < 1e-12:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else float("nan")


def concordance_ccc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xm, ym = float(np.mean(x)), float(np.mean(y))
    xv, yv = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - xm) * (y - ym)))
    denom = xv + yv + (xm - ym) ** 2
    if denom < 1e-12:
        return float("nan")
    return 2.0 * cov / denom


def rmse_metric(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ground_truth - predictions) ** 2)))


def sagr_model(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Sign agreement in model space (AFEW-VA / pipeline convention)."""
    return float(np.mean(np.sign(ground_truth) == np.sign(predictions)))


def sagr_sam(ground_truth_sam: np.ndarray, predictions_sam: np.ndarray) -> float:
    """Sign agreement relative to SAM neutral 5 (matches model SAGR under DEAP linear map)."""
    return float(
        np.mean(np.sign(ground_truth_sam - 5.0) == np.sign(predictions_sam - 5.0))
    )


def _safe_float(v: Any) -> Optional[float]:
    try:
        z = float(v)
    except (TypeError, ValueError):
        return None
    return z if np.isfinite(z) else None


def trials_paired_arrays(
    trials: List[Dict[str, Any]],
    axis: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_v, pred_a, subj_v, subj_a) in the requested axis (``model`` or ``sam``)."""
    pv, pa, sv, sa = [], [], [], []
    for t in trials:
        if t.get("error"):
            continue
        vmi = t.get("video_mean_valence")
        ami = t.get("video_mean_arousal")
        if vmi is None or ami is None:
            continue
        if isinstance(vmi, float) and math.isnan(vmi):
            continue
        if isinstance(ami, float) and math.isnan(ami):
            continue
        pvf, paf = float(vmi), float(ami)

        if "subjective_valence_scaled" in t:
            sv_m = float(t["subjective_valence_scaled"])
            sa_m = float(t["subjective_arousal_scaled"])
        else:
            sv_m = float(t["subjective_valence"])
            sa_m = float(t["subjective_arousal"])

        if axis == "model":
            pv.append(pvf)
            pa.append(paf)
            sv.append(sv_m)
            sa.append(sa_m)
            continue

        # SAM axis: predictions
        pv.append(4.0 * pvf + 5.0)
        pa.append(4.0 * paf + 5.0)

        def one_sam(raw_key: str, scaled_m: float) -> float:
            raw = _safe_float(t.get(raw_key))
            if raw is not None and 1.0 <= raw <= 9.0:
                return raw
            return 4.0 * scaled_m + 5.0

        sv.append(one_sam("subjective_valence", sv_m))
        sa.append(one_sam("subjective_arousal", sa_m))

    return (
        np.asarray(pv, dtype=np.float64),
        np.asarray(pa, dtype=np.float64),
        np.asarray(sv, dtype=np.float64),
        np.asarray(sa, dtype=np.float64),
    )


def per_file_metrics(
    trials: List[Dict[str, Any]], axis: str
) -> Optional[Dict[str, float]]:
    pv, pa, sv, sa = trials_paired_arrays(trials, axis)
    if len(pv) < 2:
        return None
    if axis == "model":
        sagr_v = sagr_model(sv, pv)
        sagr_a = sagr_model(sa, pa)
    else:
        sagr_v = sagr_sam(sv, pv)
        sagr_a = sagr_sam(sa, pa)
    return {
        "n_paired": float(len(pv)),
        "ccc_v": concordance_ccc(pv, sv),
        "ccc_a": concordance_ccc(pa, sa),
        "pcc_v": pearson_r(pv, sv),
        "pcc_a": pearson_r(pa, sa),
        "rmse_v": rmse_metric(sv, pv),
        "rmse_a": rmse_metric(sa, pa),
        "sagr_v": sagr_v,
        "sagr_a": sagr_a,
    }


def mean_se(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    if len(a) < 2:
        return m, float("nan")
    se = float(np.std(a, ddof=1) / math.sqrt(len(a)))
    return m, se


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CCC, PCC, RMSE, SAGR: model vs subjective ratings from timed-trial JSON."
    )
    ap.add_argument(
        "--input-glob",
        type=str,
        required=True,
        help=r"Glob of timed-trial JSON, e.g. /home/.../emotion_batch_data/*_xyz_fan.json",
    )
    ap.add_argument(
        "--axis",
        type=str,
        choices=["model", "sam"],
        default="model",
        help="model: [-1,1] (matches JSON comparison). sam: SAM 1–9 (RMSE in SAM units).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write one row per input JSON with all metrics.",
    )
    ap.add_argument(
        "--show-json-comparison",
        action="store_true",
        help="Print embedded comparison.* fields when present (model axis only).",
    )
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob!r}")

    rows_out: List[Dict[str, Any]] = []
    keys = ["ccc_v", "ccc_a", "pcc_v", "pcc_a", "rmse_v", "rmse_a", "sagr_v", "sagr_a"]
    collected: Dict[str, List[float]] = {k: [] for k in keys}

    print(f"# axis={args.axis}  n_files={len(paths)}\n")

    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        pid = data.get("physio_participant_id", p.stem)
        cs = data.get("color_space", "")
        trials = data.get("trials") or []
        m = per_file_metrics(trials, args.axis)
        comp = data.get("comparison") or {}

        if m is None:
            print(f"{p.name}\tpid={pid}\tspace={cs}\tSKIP n_paired<2")
            continue

        row = {
            "file": str(p),
            "participant_id": str(pid),
            "color_space": str(cs),
            "axis": args.axis,
            "n_paired": int(m["n_paired"]),
            **{k: m[k] for k in keys},
        }
        rows_out.append(row)
        for k in keys:
            collected[k].append(float(m[k]))

        line = (
            f"{p.name}\tpid={pid}\tspace={cs}\tn={int(m['n_paired'])}\t"
            f"CCC_v={m['ccc_v']:.4f}\tCCC_a={m['ccc_a']:.4f}\t"
            f"PCC_v={m['pcc_v']:.4f}\tPCC_a={m['pcc_a']:.4f}\t"
            f"RMSE_v={m['rmse_v']:.4f}\tRMSE_a={m['rmse_a']:.4f}\t"
            f"SAGR_v={m['sagr_v']:.4f}\tSAGR_a={m['sagr_a']:.4f}"
        )
        print(line)

        if args.show_json_comparison and args.axis == "model":
            jv = comp.get("ccc_valence")
            ja = comp.get("ccc_arousal")
            rv = comp.get("pcc_valence", comp.get("pearson_valence"))
            ra = comp.get("pcc_arousal", comp.get("pearson_arousal"))
            ev = comp.get("rmse_valence")
            ea = comp.get("rmse_arousal")
            gv = comp.get("sagr_valence")
            ga = comp.get("sagr_arousal")
            if jv is not None:
                print(
                    f"    [JSON comparison] CCC_v={jv} CCC_a={ja} PCC_v={rv} PCC_a={ra} "
                    f"RMSE_v={ev} RMSE_a={ea} SAGR_v={gv} SAGR_a={ga}"
                )

    print("\n# Cohort mean ± SE (across participants with valid metrics)\n")
    for k in keys:
        mu, se = mean_se(collected[k])
        se_s = f"{se:.4f}" if math.isfinite(se) else "nan"
        print(f"  {k}: {mu:.4f} ± {se_s}  (n={len(collected[k])})")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "file",
            "participant_id",
            "color_space",
            "axis",
            "n_paired",
            *keys,
        ]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
