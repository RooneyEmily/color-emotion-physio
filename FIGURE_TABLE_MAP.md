# Figure and Table Map

This document maps paper figures and tables to the exact scripts, inputs, and outputs used in this repository.
It is intended as a reproducibility companion to `README.md`.

## Inputs used across most analyses

- Processed timed trial JSON files in `data/processed/emotion_batch_data/`
  - `*_rgb_fan.json`
  - `*_ycbcr_fan.json`
  - `*_xyz_fan.json`
  - `*_lab_fan.json`
- Subject level summaries in `data/processed/summaries/`

In the current local setup these map to

- `/home/mabl-main/results/emotion_batch_data/`
- `/home/mabl-main/results/`

## Table mapping

### Color space comparison table with CCC PCC RMSE SAGR

- Script
  - `src/export/export_physio_color_space_overleaf.py`
- Primary inputs
  - `data/processed/emotion_batch_data/*_fan.json`
- Command

```bash
python3 src/export/export_physio_color_space_overleaf.py \
  --data-dir data/processed/emotion_batch_data \
  --out outputs/fig_physio_color_spaces.tex
```

- Outputs
  - `outputs/fig_physio_color_spaces.tex` as standalone latex
  - `outputs/fig_physio_color_spaces_input.tex` as overleaf input snippet
- Notes
  - The script reads `comparison` metrics from each participant json
  - It computes mean plus standard error across participants by color space

### Participant level agreement table for XYZ

- Script
  - `src/metrics/calc_physio_pred_sam_agreement.py`
- Primary inputs
  - `data/processed/emotion_batch_data/*_xyz_fan.json`
- Command

```bash
python3 src/metrics/calc_physio_pred_sam_agreement.py \
  --input-glob 'data/processed/emotion_batch_data/*_xyz_fan.json' \
  --axis model \
  --out-csv data/processed/summaries/physio_xyz_agreement_metrics.csv
```

- Output
  - `data/processed/summaries/physio_xyz_agreement_metrics.csv`

### Demographics and inclusion text table

- Scripts
  - `src/export/export_physio_subject_summary.py`
  - `src/export/report_physio_demographics_smc.py`
- Commands

```bash
python3 src/export/export_physio_subject_summary.py \
  --input-glob 'data/processed/emotion_batch_data/*_xyz_fan.json' \
  --tag xyz \
  --out data/processed/summaries/xyz_subject_summary.csv

python3 src/export/report_physio_demographics_smc.py \
  --subject-summary data/processed/summaries/xyz_subject_summary.csv \
  --trial-json-glob 'data/processed/emotion_batch_data/*_xyz_fan.json' \
  --out data/processed/summaries/ieee_smc_demographics_xyz.md
```

- Outputs
  - `data/processed/summaries/xyz_subject_summary.csv`
  - `data/processed/summaries/ieee_smc_demographics_xyz.md`

## Figure mapping

### Cross space bar charts for CCC and PCC

- Script
  - `src/plotting/plot_physio_color_space_bars.py`
- Primary inputs
  - `data/processed/emotion_batch_data/*_fan.json`
- Command

```bash
python3 src/plotting/plot_physio_color_space_bars.py \
  --data-dir data/processed/emotion_batch_data \
  --out outputs/mabl_me_color_space_ccc
```

- Outputs
  - `outputs/mabl_me_color_space_ccc.png`
  - `outputs/mabl_me_color_space_ccc.pdf`
  - `outputs/mabl_me_color_space_ccc_pcc.png`
  - `outputs/mabl_me_color_space_ccc_pcc.pdf`

### Trial snippet trajectory figures

- Script
  - `src/plotting/plot_trial_snippets.py`
- Primary inputs
  - one participant json for one color space or full batch via loop
- Example command for one participant

```bash
python3 src/plotting/plot_trial_snippets.py \
  --input data/processed/emotion_batch_data/15_xyz_fan.json \
  --out-dir outputs/snippet_plots \
  --prefix xyz_15 \
  --figure-style paper \
  --plot-scale sam
```

- Output naming
  - `xyz_15_trial00_<video_slug>.png` through all available trials
- Notes
  - Current defaults generate all trials
  - Current defaults use SAM axis for visual alignment with self report
  - Titles use participant id plus color space plus video name

### Trial error and QC figures

- Scripts
  - `src/plotting/plot_trial_errors.py`
  - `src/qc/physio_trial_qc.py`
  - `src/qc/verify_physio_batch.py`
- Typical flow
  - run qc on processed batch
  - generate error summary artifacts
  - plot error focused visuals for supplementary material

## Metric definitions used in charts and tables

Primary source for metric formulas

- `src/pipeline/process_videos_av.py`

Definitions used

- PCC via Pearson correlation on paired trial means
- CCC via Lin concordance correlation coefficient
- RMSE on paired trial means
- SAGR as sign agreement rate

Supporting calculator with the same metric set

- `src/metrics/calc_physio_pred_sam_agreement.py`

## Model checkpoint mapping used for paper runs

Base directory

- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology`

Each run directory includes best checkpoint and run metadata such as `args.json` and `split_manifest.json`

### RGB

- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/rgb/run_1`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/rgb/run_2`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/rgb/run_3`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/rgb/run_4`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/rgb/run_5`
- Best checkpoint filename `best_model.pth`

### YCbCr

- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/ycbcr/run_1`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/ycbcr/run_2`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/ycbcr/run_3`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/ycbcr/run_4`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/ycbcr/run_5`
- Best checkpoint filename `best_model_ycbcr.pth`

### XYZ

- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/xyz/run_1`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/xyz/run_2`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/xyz/run_3`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/xyz/run_4`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/xyz/run_5`
- Best checkpoint filename `best_model_xyz.pth`

### LAB

- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/lab/run_1`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/lab/run_2`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/lab/run_3`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/lab/run_4`
- `/home/mabl-main/Documents/colorEmotionEstim/emonet_feature_extraction/comparison_output_methodology/lab/run_5`
- Best checkpoint filename `best_model.pth`

## Minimal paper release checklist

- Pin repository commit hash in manuscript and release tag
- Include environment lock file for exact package versions
- Include all commands used to produce final figures and tables
- Include checksums for released model checkpoints and key outputs
- Include this mapping file and keep filenames consistent with paper captions
