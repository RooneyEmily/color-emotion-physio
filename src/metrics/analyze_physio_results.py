#!/usr/bin/env python3
"""
Aggregate timed-trial JSON outputs across participants and run label-shuffle sanity checks.

  # Fisher–Z meta-analysis (pass a directory or glob; shell glob is optional)
  python analyze_physio_results.py aggregate /path/to/emotion_batch --by-space
  python analyze_physio_results.py aggregate "/path/to/*_fan.json" --by-space

  # Shuffle subjective labels within each participant (null for correlation structure)
  python analyze_physio_results.py sanity /path/to/p15_lab_fan.json --n-shuffles 2000
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def expand_aggregate_paths(paths: List[Path]) -> List[Path]:
    """Resolve directories, glob patterns, and files (bash may pass * literally)."""
    expanded: List[Path] = []
    for raw in paths:
        s = str(raw)
        if "*" in s or "?" in s or "[" in s:
            expanded.extend(Path(p) for p in sorted(glob.glob(s)))
            continue
        p = raw
        if p.is_dir():
            found = sorted(p.glob("*_fan.json"))
            if not found:
                found = sorted(p.glob("*.json"))
            expanded.extend(found)
            continue
        if p.is_file():
            expanded.append(p)
            continue
        print(f"Warning: not a file, directory, or glob match: {p}", file=sys.stderr)
    seen: set[str] = set()
    uniq: List[Path] = []
    for x in expanded:
        try:
            key = str(x.resolve())
        except OSError:
            key = str(x)
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fisher_z(r: float) -> float:
    if not (-1.0 < r < 1.0):
        return float("nan")
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def inv_fisher_z(z: float) -> float:
    return math.tanh(z)


def meta_fisher_rs(rs: List[float], ns: List[int]) -> Tuple[float, float, float]:
    """Fixed-effect inverse-variance combine of Fisher z; return (z_bar, se_z, r_combined)."""
    zs_w: List[float] = []
    weights: List[float] = []
    for r, n in zip(rs, ns):
        if n < 4 or not (math.isfinite(r) and -1.0 < r < 1.0):
            continue
        z = fisher_z(r)
        if not math.isfinite(z):
            continue
        w = max(1, n - 3)
        zs_w.append(z * w)
        weights.append(float(w))
    if not weights:
        return float("nan"), float("nan"), float("nan")
    sw = sum(weights)
    z_bar = sum(zs_w) / sw
    se_z = math.sqrt(1.0 / sw)
    return z_bar, se_z, inv_fisher_z(z_bar)


def trials_to_arrays(
    trials: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pv, pa, sv, sa = [], [], [], []
    for t in trials:
        if t.get("error"):
            continue
        vmi = t.get("video_mean_valence")
        ami = t.get("video_mean_arousal")
        if vmi is None or ami is None:
            continue
        if isinstance(vmi, float) and isinstance(ami, float):
            if math.isnan(vmi) or math.isnan(ami):
                continue
        if "subjective_valence_scaled" in t:
            sv.append(float(t["subjective_valence_scaled"]))
            sa.append(float(t["subjective_arousal_scaled"]))
        else:
            sv.append(float(t["subjective_valence"]))
            sa.append(float(t["subjective_arousal"]))
        pv.append(float(vmi))
        pa.append(float(ami))
    return (
        np.array(pv, dtype=np.float64),
        np.array(pa, dtype=np.float64),
        np.array(sv, dtype=np.float64),
        np.array(sa, dtype=np.float64),
    )


def qc_one_record(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    trials = data.get("trials") or []
    n_csv = int(data.get("n_trials_csv") or 0)
    bad = sum(
        1
        for t in trials
        if t.get("error") == "bad_or_empty_frame_span"
    )
    comp = data.get("comparison") or {}
    n_pair = comp.get("n_paired")
    row = {
        "file": str(path),
        "participant": data.get("physio_participant_id"),
        "color_space": data.get("color_space"),
        "n_trials_csv": n_csv,
        "n_trials_json": len(trials),
        "bad_frame_spans": bad,
        "n_paired": n_pair,
        "pearson_v": comp.get("pearson_valence"),
        "pearson_a": comp.get("pearson_arousal"),
    }
    return row


def cmd_aggregate(paths: List[Path], by_space: bool) -> None:
    files = sorted(paths)
    groups: Dict[str, List[Path]] = {}
    for p in files:
        data = json.loads(p.read_text())
        sp = data.get("color_space") or "unknown"
        groups.setdefault(sp, []).append(p)

    if not by_space:
        groups = {"all": files}

    for space, spaths in sorted(groups.items()):
        rs_v: List[float] = []
        ns: List[int] = []
        rs_a: List[float] = []
        print(f"## color_space={space}  n_files={len(spaths)}")
        for p in sorted(spaths):
            data = json.loads(p.read_text())
            comp = data.get("comparison") or {}
            pid = data.get("physio_participant_id")
            nv = comp.get("pcc_valence", comp.get("pearson_valence"))
            na = comp.get("pcc_arousal", comp.get("pearson_arousal"))
            n = comp.get("n_paired")
            qc = qc_one_record(data, p)
            cv = comp.get("ccc_valence")
            ca = comp.get("ccc_arousal")
            rv = comp.get("rmse_valence")
            ra = comp.get("rmse_arousal")
            sv = comp.get("sagr_valence")
            sga = comp.get("sagr_arousal")
            so = comp.get("sagr_overall")
            print(
                f"  pid={pid} n_paired={n} bad_spans={qc['bad_frame_spans']}\n"
                f"    CCC_v={cv} CCC_a={ca}  PCC_v={nv} PCC_a={na}\n"
                f"    RMSE_v={rv} RMSE_a={ra}  SAGR_v={sv} SAGR_a={sga} SAGR={so}"
            )
            if isinstance(nv, (int, float)) and isinstance(n, int) and n >= 3:
                rs_v.append(float(nv))
                rs_a.append(float(na) if isinstance(na, (int, float)) else float("nan"))
                ns.append(int(n))
        if len(rs_v) >= 2 and len(ns) == len(rs_v):
            zv, sev, rcv = meta_fisher_rs(rs_v, ns)
            za, sea, rca = meta_fisher_rs(
                [x for x in rs_a if not math.isnan(x)],
                [ns[i] for i, x in enumerate(rs_a) if not math.isnan(x)],
            )
            print(
                f"  META Pearson V: combined_r≈{rcv:.4f} (Fisher z SE≈{sev:.4f}) over {len(rs_v)} participants"
            )
            xa = [x for x in rs_a if not math.isnan(x)]
            if len(xa) >= 2:
                print(
                    f"  META Pearson A: combined_r≈{rca:.4f} (Fisher z SE≈{sea:.4f})"
                )
        elif len(rs_v) == 1:
            print("  META: need ≥2 participants for Fisher meta; single-participant r is above.")
        print()


def cmd_sanity(path: Path, n_shuffles: int, seed: int) -> None:
    data = json.loads(path.read_text())
    trials = data.get("trials") or []
    pv, pa, sv, sa = trials_to_arrays(trials)
    if len(pv) < 3:
        print("Need ≥3 paired trials for shuffle test.", file=sys.stderr)
        raise SystemExit(1)
    obs_v = pearson_r(pv, sv)
    obs_a = pearson_r(pa, sa)
    rng = np.random.default_rng(seed)
    greater_v = greater_a = 0
    abs_obs_v, abs_obs_a = abs(obs_v), abs(obs_a)
    for _ in range(n_shuffles):
        sv_c = sv.copy()
        sa_c = sa.copy()
        rng.shuffle(sv_c)
        rng.shuffle(sa_c)
        rv = pearson_r(pv, sv_c)
        ra = pearson_r(pa, sa_c)
        if abs(rv) >= abs_obs_v:
            greater_v += 1
        if abs(ra) >= abs_obs_a:
            greater_a += 1
    p_two_v = (1 + greater_v) / (1 + n_shuffles)
    p_two_a = (1 + greater_a) / (1 + n_shuffles)
    print(f"file={path}")
    print(f"  n_trials={len(pv)}")
    print(f"  observed Pearson V={obs_v:.4f}  permutation p_two_sided≈{p_two_v:.4f} ({n_shuffles} shuffles)")
    print(f"  observed Pearson A={obs_a:.4f}  permutation p_two_sided≈{p_two_a:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate + sanity checks for timed-trial JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ag = sub.add_parser("aggregate", help="Per-file QC + Fisher–Z meta of Pearson r across files")
    ag.add_argument("json_files", nargs="+", type=Path)
    ag.add_argument(
        "--by-space",
        action="store_true",
        help="Group inputs by color_space field (recommended for multi-space batches)",
    )

    sn = sub.add_parser("sanity", help="Within-session label shuffle (destroys true pairing)")
    sn.add_argument("json_file", type=Path)
    sn.add_argument("--n-shuffles", type=int, default=2000)
    sn.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "aggregate":
        files = expand_aggregate_paths(args.json_files)
        if not files:
            print("No JSON inputs found.", file=sys.stderr)
            for p in args.json_files:
                if not p.exists():
                    print(f"  Missing path: {p}", file=sys.stderr)
                    continue
                if p.is_dir():
                    names = sorted(os.listdir(p))
                    j = [n for n in names if n.endswith(".json")]
                    print(
                        f"  Directory {p} has {len(j)} .json file(s)"
                        + (f": {j[:15]}..." if len(j) > 15 else f": {j}"),
                        file=sys.stderr,
                    )
                    if not j:
                        others = [n for n in names if not n.endswith(".json")][:8]
                        if others:
                            print(f"    (non-json: {others} ...)", file=sys.stderr)
                        print(
                            "    Generate with: python3 run_physio_grid.py --config physio_cohort.json",
                            file=sys.stderr,
                        )
                else:
                    print(f"  Not a file/dir: {p}", file=sys.stderr)
            print(
                "  Or point at results: aggregate /home/mabl-main/results \"*_lab_fan.json\"",
                file=sys.stderr,
            )
            raise SystemExit(1)
        cmd_aggregate(files, args.by_space)
    elif args.cmd == "sanity":
        cmd_sanity(args.json_file, args.n_shuffles, args.seed)


if __name__ == "__main__":
    main()
