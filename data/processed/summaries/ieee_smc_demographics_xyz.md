# Demographics and inclusion (SMC-style)

_Source: `/home/mabl-main/results/xyz_subject_summary.csv` (26 rows)._
_Trial JSONs: `/home/mabl-main/results/emotion_batch_data/*_xyz_fan.json`._

## Sample size and cohort
- **Participants in this summary table:** N = **26** (color_space values: xyz).
- **Demographics CSV path missing** (no file merged from data root): **1** participant(s): 14.
- **Questionnaire fields empty** (merged row has no responses in known columns): **1** — participant id(s): 14.
- **Timed trials with valid model means in summary** (per participant): min = 22, max = 22 (expected design: **22** music videos per participant).

## Trial pipeline / face coverage (from JSON glob)
- **JSON files scanned:** 26 (pattern `/home/mabl-main/results/emotion_batch_data/*_xyz_fan.json`).
- **Trial-level errors:** none recorded in scanned files.
- **Face-in-frame ratio** (min over trials, per participant; from segment `n_frames_with_face` / trial `n_frames`): cohort minimum of per-participant minima = **0.233**; lowest participant PID = **24**.

## Age
- **Not available** in this CSV: no column matched as Age and no numeric age-like values found. Your current `P1/Demographics/Participant*.csv` forms appear to collect device/health items only (e.g. glasses, pacemaker), not date of birth / age.

## Gender
- **Not available** in this CSV (no Gender/Sex column detected).

## Ethnicity / race
- **Not available** in this CSV (no Ethnicity/Race column detected). If you add those items to the demographics form and re-run `export_physio_subject_summary.py`, they will appear here automatically.

## Other questionnaire fields (merged)
- **Are you pregnant?**
  - No: n = 25 (96.2%)
  - (blank): n = 1 (3.8%)
- **Are you wearing makeup or sunscreen on your face?**
  - No: n = 20 (76.9%)
  - Yes: n = 5 (19.2%)
  - (blank): n = 1 (3.8%)
- **Do you have a pacemaker?**
  - No: n = 25 (96.2%)
  - (blank): n = 1 (3.8%)
- **Do you wear glasses?**
  - No: n = 16 (61.5%)
  - Yes: n = 9 (34.6%)
  - (blank): n = 1 (3.8%)
- **Is your vision ok without glasses?**
  - Yes: n = 20 (76.9%)
  - No: n = 5 (19.2%)
  - (blank): n = 1 (3.8%)

## Suggested SMC wording (edit to match your preregistration)
We analyzed **N = 26** participants with complete timed-trial model outputs for **22** music videos each (see `n_paired_trials` in the subject summary). Demographics file merge was incomplete or blank for at least one participant; we report only questionnaire items that were collected. **Age, gender, and ethnicity were not collected** in the archived demographics CSVs for this cohort; report supplementary health/vision items below where relevant. Participants were not excluded from this summary table for low face detection by default; inspect optional face-ratio lines above if you apply a threshold.
