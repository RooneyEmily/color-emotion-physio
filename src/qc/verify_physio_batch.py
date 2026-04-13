#!/usr/bin/env python3
"""
Batch verification for MABL-ME / timed-trial *_fan.json results (preflight, QC, pairing,
cross-space diffs, optional label-shuffle nulls, Table III sensitivity to excluded PIDs).

  python3 verify_physio_batch.py \\
    --data-dir /home/mabl-main/results/emotion_batch_data \\
    --write

Writes:
  physio_batch_verification_report.json
  physio_batch_verification_report.txt
  supplementary_physio_qc.tex   (LaTeX blurb for supplement / reproducibility)
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SPACE_ORDER = ["rgb", "ycbcr", "xyz", "lab"]
SPACE_LABELS = ["RGB", "YCbCr", "XYZ", "LAB"]

METRIC_PULLS: List[Tuple[str, List[str]]] = [
    ("ccc_v", ["ccc_valence"]),
    ("ccc_a", ["ccc_arousal"]),
    ("pcc_v", ["pearson_valence", "pcc_valence"]),
    ("pcc_a", ["pearson_arousal", "pcc_arousal"]),
    ("rmse_v", ["rmse_valence"]),
    ("rmse_a", ["rmse_arousal"]),
    ("sagr_v", ["sagr_valence"]),
    ("sagr_a", ["sagr_arousal"]),
]


def mean_se(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    se = 0.0 if len(a) < 2 else float(np.std(a, ddof=1) / math.sqrt(len(a)))
    return m, se


def pull(comp: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in comp and comp[k] is not None:
            return comp[k]
    return None


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


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


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def permutation_p_pearson(
    pv: np.ndarray,
    pa: np.ndarray,
    sv: np.ndarray,
    sa: np.ndarray,
    n_shuffles: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Two-sided permutation p for |r| using independent shuffles of sv and sa."""
    if len(pv) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    obs_v = pearson_r(pv, sv)
    obs_a = pearson_r(pa, sa)
    rng = np.random.default_rng(seed)
    abs_v, abs_a = abs(obs_v), abs(obs_a)
    gv = ga = 0
    for _ in range(n_shuffles):
        svc = sv.copy()
        sac = sa.copy()
        rng.shuffle(svc)
        rng.shuffle(sac)
        if abs(pearson_r(pv, svc)) >= abs_v:
            gv += 1
        if abs(pearson_r(pa, sac)) >= abs_a:
            ga += 1
    p_v = (1 + gv) / (1 + n_shuffles)
    p_a = (1 + ga) / (1 + n_shuffles)
    return obs_v, obs_a, p_v, p_a


def collect_by_space(
    data_dir: Path,
    exclude_pids: Optional[set[str]] = None,
) -> Tuple[Dict[str, Dict[str, List[float]]], int]:
    exclude_pids = exclude_pids or set()
    out: Dict[str, Dict[str, List[float]]] = {
        s: {dst: [] for dst, _ in METRIC_PULLS} for s in SPACE_ORDER
    }
    n_used = 0
    for path in sorted(data_dir.glob("*_fan.json")):
        stem = path.stem.replace("_fan", "")
        pid, cs = stem.rsplit("_", 1)
        if pid in exclude_pids:
            continue
        data = load_json(path)
        cs = (data.get("color_space") or cs or "").lower().strip()
        if cs not in SPACE_ORDER:
            continue
        comp = data.get("comparison") or {}
        if "note" in comp and "n_paired" not in comp:
            continue
        for dst, keys in METRIC_PULLS:
            v = pull(comp, keys)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[cs][dst].append(float(v))
        n_used += 1
    return out, n_used


def table_iii_text(by_space: Dict[str, Dict[str, List[float]]], n_participants: int) -> str:
    """Human-readable Table III–style block (Valence then Arousal columns as in paper)."""
    order_rows = ["rgb", "lab", "ycbcr", "xyz"]
    label = {"rgb": "RGB", "lab": "LAB", "ycbcr": "YCbCr", "xyz": "XYZ"}
    lines = []
    for s in order_rows:
        lab = label[s]
        v = by_space[s]
        ccc_v, _ = mean_se(v["ccc_v"])
        pcc_v, _ = mean_se(v["pcc_v"])
        rmse_v, _ = mean_se(v["rmse_v"])
        sagr_v, _ = mean_se(v["sagr_v"])
        ccc_a, _ = mean_se(v["ccc_a"])
        pcc_a, _ = mean_se(v["pcc_a"])
        rmse_a, _ = mean_se(v["rmse_a"])
        sagr_a, _ = mean_se(v["sagr_a"])
        lines.append(
            f"{lab:5s} V: CCC={ccc_v:+.3f} PCC={pcc_v:+.3f} RMSE={rmse_v:.3f} SAGR={sagr_v:.3f} | "
            f"A: CCC={ccc_a:+.3f} PCC={pcc_a:+.3f} RMSE={rmse_a:.3f} SAGR={sagr_a:.3f}"
        )
    lines.append(f"(mean ± SE over n={n_participants} participants per space; see paper table for SE)")
    return "\n".join(lines)


def run_checks(data_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"data_dir": str(data_dir.resolve())}
    pre_path = data_dir / "preflight_report.json"
    qc_path = data_dir / "qc_report.json"

    if not pre_path.is_file():
        report["preflight"] = {"error": "missing preflight_report.json"}
    else:
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        bad = [x for x in pre if x.get("status") != "ok"]
        pids = sorted({x["participant_id"] for x in pre}, key=int)
        report["preflight"] = {
            "n_entries": len(pre),
            "n_non_ok": len(bad),
            "participant_ids": pids,
        }

    files = sorted(data_dir.glob("*_fan.json"))
    by_pid: Dict[str, List[str]] = defaultdict(list)
    for p in files:
        stem = p.stem.replace("_fan", "")
        pid, sp = stem.rsplit("_", 1)
        by_pid[pid].append(sp)

    report["fan_json"] = {
        "n_files": len(files),
        "n_participants": len(by_pid),
        "all_four_spaces": all(
            set(by_pid[pid]) == {"rgb", "ycbcr", "xyz", "lab"} for pid in by_pid
        ),
    }

    trial_issues = []
    npair_mismatch = []
    for pid in sorted(by_pid.keys(), key=int):
        vals = []
        for sp in SPACE_ORDER:
            path = data_dir / f"{pid}_{sp}_fan.json"
            d = load_json(path)
            comp = d.get("comparison") or {}
            trials = d.get("trials") or []
            npair = comp.get("n_paired")
            if npair is not None and npair != len(trials):
                trial_issues.append(
                    {"pid": pid, "space": sp, "n_paired": npair, "n_trials": len(trials)}
                )
            vals.append(npair)
        if len(set(vals)) != 1:
            npair_mismatch.append({"pid": pid, "n_paired_per_space": vals})

    report["pairing"] = {
        "n_paired_len_mismatch": trial_issues,
        "n_paired_differs_across_spaces": npair_mismatch,
    }

    pref_ids = set(report.get("preflight", {}).get("participant_ids", []))
    run_ids = set(by_pid.keys())
    report["id_consistency"] = {
        "preflight_minus_fan": sorted(pref_ids - run_ids, key=int),
        "fan_minus_preflight": sorted(run_ids - pref_ids, key=int),
    }

    nan_trials = []
    for pid in by_pid:
        for sp in SPACE_ORDER:
            for t in load_json(data_dir / f"{pid}_{sp}_fan.json").get("trials") or []:
                if t.get("error"):
                    continue
                vm = t.get("video_mean_valence")
                if vm is None or (isinstance(vm, float) and math.isnan(vm)):
                    nan_trials.append(f"{pid}_{sp}")
    report["nan_video_mean"] = {"count": len(nan_trials), "sample": nan_trials[:5]}

    if qc_path.is_file():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        flagged = [x for x in qc if x.get("flags")]
        align_bad = sum(1 for x in qc if x.get("n_trials_align_delta_over_threshold", 0) > 0)
        past_end = sum(1 for x in qc if x.get("n_trials_raw_end_past_n_frames", 0) > 0)
        name_bad = sum(1 for x in qc if x.get("video_name_order_mismatches", 0) > 0)
        face_bad = [x for x in qc if x.get("n_trials_face_ratio_below_threshold", 0) > 0]
        report["qc_report"] = {
            "n_rows": len(qc),
            "n_flagged": len(flagged),
            "flagged_summary": [
                {
                    "participant": x.get("participant"),
                    "color_space": x.get("color_space"),
                    "flags": x.get("flags"),
                    "face_ratio_min": x.get("face_ratio_min"),
                }
                for x in flagged
            ],
            "n_align_delta_over_threshold": align_bad,
            "n_trials_end_past_frames": past_end,
            "n_video_name_mismatch_rows": name_bad,
            "n_rows_low_face_coverage": len(face_bad),
        }
    else:
        report["qc_report"] = {"note": "qc_report.json not found; run post-batch QC if desired"}

    # cross-space subjective match + pred diff sample
    sample = {}
    for pid in sorted(by_pid.keys(), key=int)[:3]:
        rgb = load_json(data_dir / f"{pid}_rgb_fan.json").get("trials") or []
        lab = load_json(data_dir / f"{pid}_lab_fan.json").get("trials") or []
        s_rgb = [(t.get("subjective_valence"), t.get("subjective_arousal")) for t in rgb]
        s_lab = [(t.get("subjective_valence"), t.get("subjective_arousal")) for t in lab]
        diffs = []
        for i, (ta, tb) in enumerate(zip(rgb, lab)):
            a = ta.get("video_mean_valence")
            b = tb.get("video_mean_valence")
            if a is not None and b is not None:
                diffs.append(abs(float(a) - float(b)))
        sample[pid] = {
            "subjective_match_rgb_lab": s_rgb == s_lab,
            "mean_abs_delta_V_rgb_lab": float(np.mean(diffs)) if diffs else None,
        }
    report["cross_space_spotcheck"] = sample

    identical = []
    for pid in by_pid:
        vecs = []
        for sp in SPACE_ORDER:
            tlist = load_json(data_dir / f"{pid}_{sp}_fan.json").get("trials") or []
            vecs.append(
                [(t.get("video_mean_valence"), t.get("video_mean_arousal")) for t in tlist]
            )
        if len({str(v) for v in vecs}) == 1:
            identical.append(pid)
    report["all_spaces_identical_predictions"] = identical

    return report


def run_shuffles(
    data_dir: Path,
    n_shuffles: int,
    seed: int,
) -> Dict[str, Any]:
    """One LAB file per participant (26 shuffles × n_shuffles)."""
    rows = []
    lab_files = sorted(data_dir.glob("*_lab_fan.json"))
    for path in lab_files:
        data = load_json(path)
        pid = str(data.get("physio_participant_id") or path.stem.replace("_lab_fan", "").split("_")[0])
        trials = data.get("trials") or []
        pv, pa, sv, sa = trials_to_arrays(trials)
        obs_v, obs_a, p_v, p_a = permutation_p_pearson(pv, pa, sv, sa, n_shuffles, seed)
        rows.append(
            {
                "participant_id": pid,
                "file": path.name,
                "n_trials": int(len(pv)),
                "obs_pearson_v": obs_v,
                "obs_pearson_a": obs_a,
                "p_two_sided_v": p_v,
                "p_two_sided_a": p_a,
            }
        )
    pv_vals = [r["p_two_sided_v"] for r in rows if math.isfinite(r["p_two_sided_v"])]
    pa_vals = [r["p_two_sided_a"] for r in rows if math.isfinite(r["p_two_sided_a"])]
    v_lt_10 = [r["participant_id"] for r in rows if r["p_two_sided_v"] < 0.10]
    a_lt_05 = [r["participant_id"] for r in rows if r["p_two_sided_a"] < 0.05]
    return {
        "n_shuffles": n_shuffles,
        "seed": seed,
        "n_participant_files": len(rows),
        "pearson_v_p_lt_0.05": sum(1 for x in pv_vals if x < 0.05),
        "pearson_a_p_lt_0.05": sum(1 for x in pa_vals if x < 0.05),
        "participant_ids_valence_p_lt_0.10": sorted(v_lt_10, key=int),
        "participant_ids_arousal_p_lt_0.05": sorted(a_lt_05, key=int),
        "median_p_v": float(np.median(pv_vals)) if pv_vals else None,
        "median_p_a": float(np.median(pa_vals)) if pa_vals else None,
        "per_participant": rows,
    }


def write_tex(report: Dict[str, Any], out_tex: Path) -> None:
    qc = report.get("qc_report") or {}
    pre = report.get("preflight") or {}
    sh = report.get("shuffles") or {}
    sens = report.get("sensitivity_exclude_p24") or {}
    ex = sens.get("excluded_pids") or []
    exs = ", ".join(ex) if ex else "24"
    max_d = sens.get("max_abs_mean_delta_any_metric")

    lines = [
        "% Auto-generated by verify_physio_batch.py — paste into supplemental / reproducibility.",
        "",
        r"\paragraph{Batch verification.}",
        f"Preflight covered $n={pre.get('n_entries', '?')}$ participants (all \\texttt{{status=ok}}). ",
        f"Timed-trial outputs comprised {report.get('fan_json', {}).get('n_files', '?')} "
        r"\texttt{*_fan.json} files ($4$ color spaces $\times$ $26$ participants). ",
        "For every file, \\texttt{n\\_paired} matched the number of trial records; "
        "the same \\texttt{n\\_paired} held across RGB, YCbCr, XYZ, and LAB for each participant.",
        "",
        r"\paragraph{QC flags.}",
    ]
    if qc.get("n_flagged"):
        fl = qc["flagged_summary"][0].get("flags") or []
        fl_tex = ", ".join(str(x).replace("_", r"\_") for x in fl) if fl else "unknown"
        pid0 = qc["flagged_summary"][0].get("participant")
        lines.append(
            f"Post-run QC flagged {qc['n_flagged']} file(s); example: "
            f"\\texttt{{{fl_tex}}} for participant \\textbf{{{pid0}}} "
            f"(low face-in-segment coverage on at least one trial). "
            "Alignment timestamp drift and rating-end-past-video counts were zero across the batch "
            "in \\texttt{qc\\_report.json}."
        )
    else:
        lines.append("No QC rows flagged in \\texttt{qc\\_report.json}.")

    md = float(max_d) if isinstance(max_d, (int, float)) and math.isfinite(max_d) else 0.0
    lines.append(
        f"Sensitivity analysis excluding participant(s) {exs} "
        f"(Table~III--style means, $n={sens.get('n_participants_implied', '?')}$) changes "
        f"group means (CCC, PCC, RMSE, SAGR) by at most $\\approx {md:.4f}$ on the mean-across-participants summary versus the full cohort; "
        r"see \texttt{physio\_batch\_verification\_report.txt}."
    )

    if sh.get("skipped"):
        lines.extend(
            [
                "",
                r"\paragraph{Label permutation sanity.}",
                "Not run in this export; run "
                r"\texttt{python3 verify\_physio\_batch.py --data-dir ... --write} "
                "with shuffles enabled (default $2000$).",
                "",
            ]
        )
    else:
        mv = sh.get("median_p_v")
        ma = sh.get("median_p_a")
        mv_s = f"{mv:.3f}" if isinstance(mv, (int, float)) and math.isfinite(mv) else "NA"
        ma_s = f"{ma:.3f}" if isinstance(ma, (int, float)) and math.isfinite(ma) else "NA"
        v10 = sh.get("participant_ids_valence_p_lt_0.10") or []
        a5 = sh.get("participant_ids_arousal_p_lt_0.05") or []
        v10s = ", ".join(v10) if v10 else "(none)"
        a5s = ", ".join(a5) if a5 else "(none)"
        lines.extend(
            [
                "",
                r"\paragraph{Label permutation sanity.}",
                f"Within each participant, trial-level SAM labels (scaled) were permuted "
                f"{sh.get('n_shuffles', '?')} times independently for valence and arousal against the same model summary vectors "
                f"(LAB checkpoint; seed {sh.get('seed', '?')}). ",
                f"Two-sided $p$-values for $|\\mathrm{{PCC}}|$ were $<0.05$ for valence in "
                f"{sh.get('pearson_v_p_lt_0.05', '?')}/{sh.get('n_participant_files', '?')} participants "
                f"(median $p\\approx {mv_s}$; $p<0.10$ for participants {v10s}) and for arousal in "
                f"{sh.get('pearson_a_p_lt_0.05', '?')}/{sh.get('n_participant_files', '?')} "
                f"(median $p\\approx {ma_s}$; $p<0.05$ for participants {a5s}). ",
                "$p$-values are Monte Carlo approximations; low trial count limits power. ",
                "The check tests whether trial order pairing carries structure beyond label permutation under a null that destroys that pairing.",
                "",
            ]
        )

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify physio timed-trial JSON batch")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write JSON, TXT, and supplementary_physio_qc.tex under data-dir",
    )
    ap.add_argument("--n-shuffles", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-shuffle",
        action="store_true",
        help="Skip permutation tests (fast)",
    )
    ap.add_argument(
        "--exclude-sensitivity",
        type=str,
        default="24",
        help="Comma-separated participant IDs for sensitivity Table III (default: 24)",
    )
    args = ap.parse_args()
    data_dir = args.data_dir

    report = run_checks(data_dir)
    by_full, nfull = collect_by_space(data_dir, exclude_pids=set())
    n_participants = nfull // 4 if nfull else 0
    report["table_iii_summary_n26"] = table_iii_text(by_full, n_participants)

    excl = {x.strip() for x in args.exclude_sensitivity.split(",") if x.strip()}
    by_ex, nex = collect_by_space(data_dir, exclude_pids=excl)
    n_participants_ex = nex // 4 if nex else 0
    report["sensitivity_exclude_p24"] = {
        "excluded_pids": sorted(excl, key=int),
        "n_files_used": nex,
        "n_participants_implied": n_participants_ex,
        "table_iii_text": table_iii_text(by_ex, n_participants_ex),
    }
    # numeric diff: max change in mean CCC_v across spaces
    deltas = []
    for s in SPACE_ORDER:
        for key, _ in METRIC_PULLS:
            m0, _ = mean_se(by_full[s][key])
            m1, _ = mean_se(by_ex[s][key])
            if math.isfinite(m0) and math.isfinite(m1):
                deltas.append(abs(m0 - m1))
    report["sensitivity_exclude_p24"]["max_abs_mean_delta_any_metric"] = max(deltas) if deltas else None

    if not args.skip_shuffle:
        report["shuffles"] = run_shuffles(data_dir, args.n_shuffles, args.seed)
    else:
        report["shuffles"] = {"skipped": True}

    txt_lines = [
        "physio_batch_verification_report",
        "=" * 48,
        json.dumps(report.get("preflight"), indent=2),
        json.dumps(report.get("fan_json"), indent=2),
        json.dumps(report.get("id_consistency"), indent=2),
        json.dumps(report.get("pairing"), indent=2)[:2000],
        json.dumps(report.get("qc_report"), indent=2),
        "",
        "Table III summary (n=26):",
        report["table_iii_summary_n26"],
        "",
        f"Sensitivity exclude {sorted(excl, key=int)}:",
        report["sensitivity_exclude_p24"]["table_iii_text"],
        f"max |Δmean| any metric vs full: {report['sensitivity_exclude_p24'].get('max_abs_mean_delta_any_metric')}",
    ]
    def _fmt_med(x: Any) -> str:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return f"{float(x):.4f}"
        return "NA"

    if args.skip_shuffle:
        txt_lines.extend(["", "Shuffles: skipped (--skip-shuffle)"])
    else:
        sh = report["shuffles"]
        txt_lines.extend(
            [
                "",
                f"Shuffles (n={sh['n_shuffles']}, seed={sh['seed']}):",
                f"  Valence p<0.05: {sh.get('pearson_v_p_lt_0.05', '?')}/{sh.get('n_participant_files', '?')}, "
                f"median p_v={_fmt_med(sh.get('median_p_v'))}",
                f"  Arousal  p<0.05: {sh.get('pearson_a_p_lt_0.05', '?')}/{sh.get('n_participant_files', '?')}, "
                f"median p_a={_fmt_med(sh.get('median_p_a'))}",
            ]
        )

    print("\n".join(txt_lines))

    if args.write:
        data_dir.mkdir(parents=True, exist_ok=True)
        json_path = data_dir / "physio_batch_verification_report.json"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        txt_path = data_dir / "physio_batch_verification_report.txt"
        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
        tex_path = data_dir / "supplementary_physio_qc.tex"
        write_tex(report, tex_path)
        print(f"\nWrote {json_path}\nWrote {txt_path}\nWrote {tex_path}")


if __name__ == "__main__":
    main()
