# COLOR EMOTION PHYSIO REPO

This repository contains the final reproducibility package for color space emotion modeling on physio video trials.
The structure below lists the files included in the final paper release and what each group does at a high level.

## Core code

### src pipeline
`src/pipeline/process_videos_av.py`
Main trial processing script that runs model inference on video segments and writes per trial outputs with prediction and rating comparisons.

`src/pipeline/run_physio_grid.py`
Batch runner for cohort processing across participants and color spaces with run management and job orchestration.

### src metrics
`src/metrics/analyze_physio_results.py`
Aggregate analysis for participant json outputs with correlation style checks and group level summaries.

`src/metrics/calc_physio_pred_sam_agreement.py`
Agreement calculator for CCC PCC RMSE and SAGR between model predictions and SAM ratings.

`src/metrics/calc_circumplex_segments.py`
Computes trial level circumplex angles and segment assignments for model and self report including quadrant octant and region matching.

### src plotting
`src/plotting/plot_physio_color_space_bars.py`
Creates cross space summary figures for metric comparison across RGB YCbCr XYZ and LAB.

`src/plotting/plot_trial_snippets.py`
Creates trial level time series figures for valence and arousal with music video labels and SAM aligned scaling.

`src/plotting/plot_trial_errors.py`
Creates error focused plots used for trial quality and failure pattern inspection.

`src/plotting/plot_circumplex_xyz.py`
Creates XYZ circumplex distribution figures for model versus self report.

`src/plotting/plot_circumplex_trial_pairs.py`
Creates per participant circumplex pair plots with one line per trial between model and self report.

`src/plotting/plot_circumplex_evaluation.py`
Creates quadrant confusion matrices plus directional and magnitude error distributions including normalized distance.

### src export
`src/export/export_physio_color_space_overleaf.py`
Exports paper tables in latex format for direct overleaf import.

`src/export/export_physio_subject_summary.py`
Builds participant level summary tables from processed trial json files.

`src/export/report_physio_demographics_smc.py`
Builds manuscript ready demographics and inclusion text from summary data.

### src qc
`src/qc/physio_trial_qc.py`
Runs trial level quality checks including timing alignment and face coverage diagnostics.

`src/qc/verify_physio_batch.py`
Runs end to end verification checks on full processed batches before reporting.

## Processed dataset outputs

### data processed emotion_batch_data
Per subject per color space trial outputs

`data/processed/emotion_batch_data/*_rgb_fan.json`

`data/processed/emotion_batch_data/*_ycbcr_fan.json`

`data/processed/emotion_batch_data/*_xyz_fan.json`

`data/processed/emotion_batch_data/*_lab_fan.json`

### data processed summaries
`data/processed/summaries/xyz_subject_summary.csv`

`data/processed/summaries/physio_xyz_agreement_metrics.csv`

`data/processed/summaries/ieee_smc_demographics_xyz.md`

`data/processed/summaries/circumplex_angles_all_xyz.csv`

`data/processed/summaries/circumplex_eval_xyz_all_summary.json`

`data/processed/summaries/circumplex_eval_xyz_pid15_summary.json`

## Trained models final run only

### models
Each color space includes run_5 only

`models/rgb/run_5/`

`models/ycbcr/run_5/`

`models/xyz/run_5/`

`models/lab/run_5/`

Each run folder contains the following final artifacts when available

`best_model*.pth`

`args.json`

`split_manifest.json`

`logs/*`

`best_checkpoint_metrics.json`

## Reproducibility notes for paper github

Include exact commit id for this repository in the manuscript and in the release notes.

Include full environment lock file with python version torch version cuda version and all package versions.

Include commands used for training batch processing metric export and figure generation.

Include all processed json files used for tables and figures or include scripts plus checksum manifest if data cannot be redistributed.

Include all final run_5 checkpoint metadata files listed above for each color space.

Include one mapping file that links each paper figure and table to the script and input files used to create it.
