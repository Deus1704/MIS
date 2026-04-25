# Path C: FBP + Deep Learning Enhancement (Low-Dose Focus)

Path C is a focused branch for the objective:

- start from **classical FBP reconstruction**
- apply **image-domain deep learning enhancement** on top of FBP
- improve reconstruction quality while supporting **low-dose CT settings** (reduced radiation exposure)

This path reuses the validated training and evaluation engine from Path B, but narrows the default methods to post-FBP DL models:

- `RED-CNN`
- `U-Net`
- `AttentionUNet`

## Why This Path Exists

In low-dose CT, FBP remains clinically useful because it is fast and stable, but it amplifies noise and streak artifacts. Path C evaluates whether DL can reliably correct those artifacts *after* FBP, while preserving anatomy.

The pipeline logs all important quality evidence:

- training/validation trajectories per epoch
- per-slice test metrics
- method-level test summaries versus FBP baseline
- statistical significance versus FBP
- visual panels and metric comparison plots

## Directory Layout

```text
Path_C_FBP_DL/
├── ct_recon/                     # earlier evaluation core copied for local use
├── scripts/
│   └── collect_path_c_results.py # metric + visualization organizer
├── results/
│   └── <run_name>/               # each run contains raw + organized outputs
├── run_path_c_fbp_dl.sh          # main Path C runner
└── requirements.txt
```

## Run Path C

From `MIS_Project/`:

```bash
bash Path_C_FBP_DL/run_path_c_fbp_dl.sh
```

Optional environment overrides:

- `RUN_NAME` (default: UTC timestamp)
- `METHODS` (default: `"red_cnn unet attention_unet"`)
- `DOSE_FRACTION` (default: `0.25`)
- `EPOCHS` (default: `30`)
- `MAX_TRAIN`, `MAX_TEST` for quick experiments

## Stored Outputs Per Run

Inside `Path_C_FBP_DL/results/<run_name>/`:

- `training_curves.csv`, `inference_metrics.csv`, `method_summary.csv`
- `run_metadata.json`
- `ct_panel.png`, `training_curves.png`, `metrics_comparison.png`, `speed_quality_plot.png`, `radar_chart.png`, `freq_spectrum.png`
- `metrics/`
  - `train_val_accuracy_detailed.csv`
  - `train_val_best_epoch_summary.csv`
  - `test_accuracy_summary.csv`
- `visualizations/` (organized copy of key PNG outputs)
- `path_c_objective_report.md` (objective + evidence snapshot)

This gives a single location for all quantitative and visual evidence needed for reporting.
