#!/usr/bin/env python3
"""One row per participant: trial counts + mean model vs mean subjective (scaled if present).

Optional merge of ``P1/Demographics/Participant*.csv`` (Physio-Code layout), e.g.::

  .../Data/<pid>/P1/Demographics/Participant1.csv
  .../Data/<pid>/P1/Demographics/Participant 15.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _accum(trials: List[Dict[str, Any]]) -> Dict[str, float]:
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
        if "subjective_valence_scaled" in t:
            sv.append(float(t["subjective_valence_scaled"]))
            sa.append(float(t["subjective_arousal_scaled"]))
        else:
            sv.append(float(t["subjective_valence"]))
            sa.append(float(t["subjective_arousal"]))
        pv.append(float(vmi))
        pa.append(float(ami))
    if not pv:
        return {}
    n = len(pv)
    return {
        "n_paired_trials": n,
        "mean_pred_valence": sum(pv) / n,
        "mean_pred_arousal": sum(pa) / n,
        "mean_subj_valence": sum(sv) / n,
        "mean_subj_arousal": sum(sa) / n,
    }


def _find_demographics_csv(data_root: Path, participant_id: str) -> Optional[Path]:
    """Return a single CSV under ``<root>/<pid>/P1/Demographics/`` if present."""
    demo_dir = data_root / participant_id / "P1" / "Demographics"
    if not demo_dir.is_dir():
        return None
    candidates = sorted(demo_dir.glob("Participant*.csv"))
    if not candidates:
        candidates = sorted(demo_dir.glob("*.csv"))
    if not candidates:
        return None
    pid = participant_id.strip()
    # Prefer filename that contains this participant id
    for c in candidates:
        stem = c.stem.replace(" ", "")
        if re.search(rf"(?:^|[^0-9]){re.escape(pid)}(?:[^0-9]|$)", stem):
            return c
    return candidates[0]


def _read_demographics_row(csv_path: Path) -> Dict[str, str]:
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            return {k: (v if v is not None else "") for k, v in row.items()}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path("/home/mabl-main/results/xyz_subject_summary.csv"))
    ap.add_argument(
        "--physio-data-root",
        type=Path,
        default=Path("/media/mabl-main/Data/Physio-code/Data"),
        help="Dataset root containing <pid>/P1/Demographics/*.csv",
    )
    ap.add_argument(
        "--no-demographics",
        action="store_true",
        help="Omit merge of P1/Demographics CSVs",
    )
    args = ap.parse_args()
    data_root: Optional[Path] = None if args.no_demographics else args.physio_data_root

    paths = sorted(Path(p) for p in glob.glob(args.input_glob))
    rows: List[Dict[str, Any]] = []

    for p in paths:
        data = json.loads(p.read_text())
        pid = str(data.get("physio_participant_id", p.stem.split("_")[0]))
        cs = str(data.get("color_space", ""))
        acc = _accum(data.get("trials") or [])
        if not acc:
            continue
        row: Dict[str, Any] = {
            "participant_id": pid,
            "color_space": cs,
            "demographics_csv": "",
            **acc,
        }
        if data_root is not None:
            dpath = _find_demographics_csv(data_root, pid)
            if dpath is not None:
                row["demographics_csv"] = str(dpath)
                demo = _read_demographics_row(dpath)
                for k, v in demo.items():
                    if k:
                        row[k] = v
        rows.append(row)

    rows.sort(key=lambda r: int(r["participant_id"]) if str(r["participant_id"]).isdigit() else r["participant_id"])

    core = [
        "participant_id",
        "color_space",
        "demographics_csv",
        "n_paired_trials",
        "mean_pred_valence",
        "mean_pred_arousal",
        "mean_subj_valence",
        "mean_subj_arousal",
    ]
    extra: set[str] = set()
    for r in rows:
        for k in r:
            if k not in core:
                extra.add(k)
    fieldnames = core + sorted(extra)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    n_demo = sum(1 for r in rows if r.get("demographics_csv"))
    print(f"Wrote {args.out} ({len(rows)} rows, demographics merged for {n_demo} participants)")


if __name__ == "__main__":
    main()
