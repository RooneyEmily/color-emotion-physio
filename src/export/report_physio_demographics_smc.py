#!/usr/bin/env python3
"""
Summarize cohort demographics and analysis inclusion for IEEE SMC-style reporting.

Reads a subject-level CSV (e.g. from ``export_physio_subject_summary.py``). Detects
Age / Gender / Ethnicity columns by name when present. If those fields were not
collected in your Demographics CSVs, the report states that explicitly and still
reports available questionnaire columns.

Optionally scans timed-trial JSONs for trial-level errors and face coverage minima.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CORE_KEYS = {
    "participant_id",
    "color_space",
    "demographics_csv",
    "n_paired_trials",
    "mean_pred_valence",
    "mean_pred_arousal",
    "mean_subj_valence",
    "mean_subj_arousal",
}

# Form metadata (not clinical demographics); still merged from CSV.
SKIP_QUESTIONNAIRE_KEYS = {"timestamp", "participant #"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _classify_demo_column(name: str) -> str:
    n = _norm(name)
    if any(k in n for k in ("age", "how old", "years old", "yr old")):
        return "age"
    if any(k in n for k in ("gender", " sex", "^sex", "male or female")):
        return "gender"
    if any(k in n for k in ("ethnic", "race", "hispanic", "latino")):
        return "ethnicity"
    return "other"


def _parse_age(v: str) -> Optional[float]:
    v = (v or "").strip()
    if not v:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", v)
    if not m:
        return None
    try:
        x = float(m.group(1))
    except ValueError:
        return None
    if 5.0 <= x <= 120.0:
        return x
    return None


def _trial_face_ratio(trial: Dict[str, Any]) -> float:
    n = int(trial.get("n_frames") or 0)
    segs = trial.get("segments") or []
    with_face = sum(int(s.get("n_frames_with_face") or 0) for s in segs)
    if n <= 0:
        return float("nan")
    return with_face / n


def _scan_trial_jsons(glob_pat: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(Path(x) for x in glob.glob(glob_pat)):
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        pid = str(data.get("physio_participant_id", p.stem.split("_")[0]))
        trials = data.get("trials") or []
        err_types: Counter[str] = Counter()
        n_ok = 0
        ratios: List[float] = []
        for t in trials:
            if t.get("error"):
                err_types[str(t.get("error"))] += 1
                continue
            vm = t.get("video_mean_valence")
            if vm is None or (isinstance(vm, float) and math.isnan(vm)):
                continue
            n_ok += 1
            r = _trial_face_ratio(t)
            if math.isfinite(r):
                ratios.append(r)
        rows.append(
            {
                "participant_id": pid,
                "file": str(p),
                "n_trials_json": len(trials),
                "n_paired_predictions": n_ok,
                "trial_errors": dict(err_types),
                "face_ratio_min": min(ratios) if ratios else float("nan"),
                "face_ratio_mean_of_trial_means": sum(ratios) / len(ratios) if ratios else float("nan"),
            }
        )
    return rows


def _load_summary(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = [{k: (v if v is not None else "") for k, v in row.items()} for row in r]
    return fieldnames, rows


def _report_categorical(rows: Sequence[Dict[str, str]], col: str) -> List[str]:
    c = Counter((row.get(col) or "").strip() or "(blank)" for row in rows)
    out = [f"- **{col}**"]
    for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = 100.0 * v / len(rows) if rows else 0.0
        out.append(f"  - {k}: n = {v} ({pct:.1f}%)")
    return out


def build_report(
    summary_path: Path,
    trial_glob: Optional[str],
    expected_trials: int,
) -> str:
    fieldnames, rows = _load_summary(summary_path)
    if not rows:
        return f"No rows in {summary_path}.\n"

    demo_cols = [
        c
        for c in fieldnames
        if c and c not in CORE_KEYS and _norm(c) not in SKIP_QUESTIONNAIRE_KEYS
    ]
    by_kind: Dict[str, List[str]] = {"age": [], "gender": [], "ethnicity": [], "other": []}
    for c in demo_cols:
        by_kind[_classify_demo_column(c)].append(c)

    lines: List[str] = []
    lines.append(f"# Demographics and inclusion (SMC-style)\n")
    lines.append(f"_Source: `{summary_path}` ({len(rows)} rows)._")
    if trial_glob:
        lines.append(f"_Trial JSONs: `{trial_glob}`._\n")
    else:
        lines.append("")

    # --- N and cohort definition ---
    cs_vals = {str(r.get("color_space", "")).strip() for r in rows}
    lines.append("## Sample size and cohort")
    lines.append(
        f"- **Participants in this summary table:** N = **{len(rows)}** "
        f"(color_space values: {', '.join(sorted(cs_vals)) or '—'})."
    )
    missing_demo_path = sum(1 for r in rows if not (r.get("demographics_csv") or "").strip())
    if missing_demo_path:
        pids = [r["participant_id"] for r in rows if not (r.get("demographics_csv") or "").strip()]
        lines.append(
            f"- **Demographics CSV path missing** (no file merged from data root): "
            f"**{missing_demo_path}** participant(s): {', '.join(map(str, pids))}."
        )
    else:
        lines.append("- **Demographics CSV path:** present for all listed participants.")

    # Blank questionnaire row (all demo columns empty)
    if demo_cols:
        def all_demo_blank(r: Dict[str, str]) -> bool:
            return all(not (r.get(c) or "").strip() for c in demo_cols)

        blank_q = [r["participant_id"] for r in rows if all_demo_blank(r)]
        if blank_q:
            lines.append(
                f"- **Questionnaire fields empty** (merged row has no responses in known columns): "
                f"**{len(blank_q)}** — participant id(s): {', '.join(map(str, blank_q))}."
            )

    # Trial counts from summary
    nt = [int(float(r["n_paired_trials"])) for r in rows if (r.get("n_paired_trials") or "").strip().isdigit()]
    if not nt:
        nt = []
        for r in rows:
            try:
                nt.append(int(float(str(r.get("n_paired_trials", "")).strip())))
            except ValueError:
                pass
    if nt:
        lines.append(
            f"- **Timed trials with valid model means in summary** "
            f"(per participant): min = {min(nt)}, max = {max(nt)} "
            f"(expected design: **{expected_trials}** music videos per participant)."
        )
        short: List[str] = []
        for r in rows:
            raw = str(r.get("n_paired_trials", "")).strip()
            if not raw:
                continue
            try:
                npt = int(float(raw))
            except ValueError:
                continue
            if npt < expected_trials:
                short.append(str(r.get("participant_id", "?")))
        if short:
            lines.append(f"  - Participants below {expected_trials}: {', '.join(short)}.")

    # Optional JSON scan
    scan = _scan_trial_jsons(trial_glob) if trial_glob else []
    if scan:
        lines.append("\n## Trial pipeline / face coverage (from JSON glob)")
        lines.append(
            f"- **JSON files scanned:** {len(scan)} (pattern `{trial_glob}`)."
        )
        any_err = False
        for s in scan:
            if s["trial_errors"]:
                any_err = True
        if any_err:
            lines.append("- **Trial-level `error` flags (non-empty):**")
            for s in scan:
                if s["trial_errors"]:
                    lines.append(f"  - PID {s['participant_id']}: {s['trial_errors']}")
        else:
            lines.append("- **Trial-level errors:** none recorded in scanned files.")
        mins = [s["face_ratio_min"] for s in scan if math.isfinite(s["face_ratio_min"])]
        if mins:
            lines.append(
                f"- **Face-in-frame ratio** (min over trials, per participant; from segment `n_frames_with_face` / trial `n_frames`): "
                f"cohort minimum of per-participant minima = **{min(mins):.3f}**; "
                f"lowest participant PID = **{scan[mins.index(min(mins))]['participant_id']}**."
            )

    lines.append("\n## Age")
    age_cols = by_kind["age"]
    ages: List[float] = []
    for col in age_cols:
        for r in rows:
            a = _parse_age(r.get(col, ""))
            if a is not None:
                ages.append(a)
    if ages:
        lines.append(
            f"- Parsed from column(s): {', '.join(age_cols)} "
            f"(n = {len(ages)} non-missing)."
        )
        lines.append(
            f"- **Mean ± SD:** {statistics.mean(ages):.2f} ± {statistics.pstdev(ages):.2f} years "
            f"(range {min(ages):.0f}–{max(ages):.0f})."
        )
    else:
        lines.append(
            "- **Not available** in this CSV: no column matched as Age and no numeric age-like values found. "
            "Your current `P1/Demographics/Participant*.csv` forms appear to collect device/health items only "
            "(e.g. glasses, pacemaker), not date of birth / age."
        )

    lines.append("\n## Gender")
    gcols = by_kind["gender"]
    if gcols:
        for gc in gcols:
            lines.extend(_report_categorical(rows, gc))
    else:
        lines.append("- **Not available** in this CSV (no Gender/Sex column detected).")

    lines.append("\n## Ethnicity / race")
    ecols = by_kind["ethnicity"]
    if ecols:
        for ec in ecols:
            lines.extend(_report_categorical(rows, ec))
    else:
        lines.append(
            "- **Not available** in this CSV (no Ethnicity/Race column detected). "
            "If you add those items to the demographics form and re-run `export_physio_subject_summary.py`, "
            "they will appear here automatically."
        )

    lines.append("\n## Other questionnaire fields (merged)")
    others = by_kind["other"]
    if not others:
        lines.append("- _(None beyond core metrics.)_")
    else:
        for col in sorted(others):
            vals = [((r.get(col) or "").strip() or "(blank)") for r in rows]
            nuniq = len(set(vals))
            if nuniq <= 12:
                lines.extend(_report_categorical(rows, col))
            else:
                lines.append(f"- **{col}:** {nuniq} distinct values (omitted table; high cardinality).")

    lines.append("\n## Suggested SMC wording (edit to match your preregistration)")
    lines.append(
        f"We analyzed **N = {len(rows)}** participants "
        f"with complete timed-trial model outputs for **{expected_trials}** music videos each "
        f"(see `n_paired_trials` in the subject summary). "
    )
    if missing_demo_path or (demo_cols and any(not any((r.get(c) or "").strip() for c in demo_cols) for r in rows)):
        lines[-1] += (
            "Demographics file merge was incomplete or blank for at least one participant; "
            "we report only questionnaire items that were collected. "
        )
    if not ages and not gcols and not ecols:
        lines[-1] += (
            "**Age, gender, and ethnicity were not collected** in the archived demographics CSVs for this cohort; "
            "report supplementary health/vision items below where relevant. "
        )
    lines[-1] += (
        "Participants were not excluded from this summary table for low face detection by default; "
        "inspect optional face-ratio lines above if you apply a threshold."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="SMC-style demographics + inclusion report from subject summary CSV")
    ap.add_argument(
        "--subject-summary",
        type=Path,
        default=Path("/home/mabl-main/results/xyz_subject_summary.csv"),
        help="CSV from export_physio_subject_summary.py",
    )
    ap.add_argument(
        "--trial-json-glob",
        type=str,
        default="/home/mabl-main/results/emotion_batch_data/*_xyz_fan.json",
        help="Glob of timed-trial JSONs for error/face stats (set empty to skip).",
    )
    ap.add_argument(
        "--expected-trials",
        type=int,
        default=22,
        help="Design number of music-video trials per participant (for inclusion text).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write Markdown report to this path (default: print to stdout only).",
    )
    args = ap.parse_args()

    tg = (args.trial_json_glob or "").strip() or None
    text = build_report(args.subject_summary, tg, args.expected_trials)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
