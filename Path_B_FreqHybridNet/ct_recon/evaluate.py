from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import compute_masked_metrics, compute_metrics
from .stats import (
    bootstrap_ci,
    holm_bonferroni,
    paired_permutation_test,
    paired_wilcoxon,
    rank_biserial_effect_size,
)

REQUIRED_COLUMNS = ["patient_id", "slice_id", "target_path", "fbp_path", "dl_path"]


def load_array(path: str | Path) -> np.ndarray:
    """Load an array from .npy or .npz."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        data = np.load(p)
        keys = list(data.keys())
        if not keys:
            raise ValueError(f"NPZ file has no arrays: {p}")
        return data[keys[0]]
    raise ValueError(f"Unsupported array file format: {p}")


def _metric_record(
    patient_id: str,
    slice_id: str,
    method: str,
    region: str,
    metrics: Dict[str, float],
) -> Dict[str, float | str]:
    return {
        "patient_id": patient_id,
        "slice_id": slice_id,
        "method": method,
        "region": region,
        "ssim": metrics["ssim"],
        "psnr": metrics["psnr"],
        "rmse": metrics["rmse"],
    }


def evaluate_manifest(
    manifest_csv: str | Path,
    out_dir: str | Path,
    n_bootstrap: int = 2000,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Path]:
    """Run paired evaluation of FBP vs DL reconstructions."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    has_lesion_mask = "lesion_mask_path" in manifest.columns
    has_background_mask = "background_mask_path" in manifest.columns

    records: List[Dict[str, float | str]] = []

    for row in manifest.itertuples(index=False):
        patient_id = str(getattr(row, "patient_id"))
        slice_id = str(getattr(row, "slice_id"))

        target = load_array(getattr(row, "target_path"))
        fbp = load_array(getattr(row, "fbp_path"))
        dl = load_array(getattr(row, "dl_path"))

        records.append(_metric_record(patient_id, slice_id, "fbp", "global", compute_metrics(fbp, target)))
        records.append(_metric_record(patient_id, slice_id, "dl", "global", compute_metrics(dl, target)))

        if has_lesion_mask:
            lesion_mask = load_array(getattr(row, "lesion_mask_path")).astype(bool)
            records.append(
                _metric_record(
                    patient_id,
                    slice_id,
                    "fbp",
                    "lesion_roi",
                    compute_masked_metrics(fbp, target, lesion_mask),
                )
            )
            records.append(
                _metric_record(
                    patient_id,
                    slice_id,
                    "dl",
                    "lesion_roi",
                    compute_masked_metrics(dl, target, lesion_mask),
                )
            )

        if has_background_mask:
            background_mask = load_array(getattr(row, "background_mask_path")).astype(bool)
            records.append(
                _metric_record(
                    patient_id,
                    slice_id,
                    "fbp",
                    "background_roi",
                    compute_masked_metrics(fbp, target, background_mask),
                )
            )
            records.append(
                _metric_record(
                    patient_id,
                    slice_id,
                    "dl",
                    "background_roi",
                    compute_masked_metrics(dl, target, background_mask),
                )
            )

    slice_df = pd.DataFrame.from_records(records)
    patient_df = (
        slice_df.groupby(["patient_id", "method", "region"], as_index=False)[["ssim", "psnr", "rmse"]]
        .mean()
        .sort_values(["region", "patient_id", "method"])
    )

    summary_records: List[Dict[str, float | str]] = []
    metric_names = ["ssim", "psnr", "rmse"]

    for region in sorted(patient_df["region"].unique()):
        region_df = patient_df[patient_df["region"] == region]
        pivot = region_df.pivot(index="patient_id", columns="method", values=metric_names)

        if ("fbp" not in pivot.columns.get_level_values(1)) or ("dl" not in pivot.columns.get_level_values(1)):
            continue

        for metric in metric_names:
            x = pivot[(metric, "dl")].dropna()
            y = pivot[(metric, "fbp")].dropna()
            aligned = pd.concat([x, y], axis=1, join="inner").dropna()
            if aligned.empty:
                continue

            dl_vals = aligned.iloc[:, 0].to_numpy(dtype=np.float64)
            fbp_vals = aligned.iloc[:, 1].to_numpy(dtype=np.float64)
            diff = dl_vals - fbp_vals

            ci_low, ci_high = bootstrap_ci(diff, n_bootstrap=n_bootstrap, random_state=seed)
            wil = paired_wilcoxon(dl_vals, fbp_vals)
            perm = paired_permutation_test(dl_vals, fbp_vals, n_permutations=n_permutations, random_state=seed)
            eff = rank_biserial_effect_size(dl_vals, fbp_vals)

            summary_records.append(
                {
                    "region": region,
                    "metric": metric,
                    "n_patients": int(aligned.shape[0]),
                    "mean_dl": float(np.mean(dl_vals)),
                    "mean_fbp": float(np.mean(fbp_vals)),
                    "mean_diff_dl_minus_fbp": float(np.mean(diff)),
                    "ci95_low_diff": ci_low,
                    "ci95_high_diff": ci_high,
                    "wilcoxon_stat": wil["statistic"],
                    "wilcoxon_p": wil["p_value"],
                    "perm_p": perm["p_value"],
                    "effect_rank_biserial": eff,
                    "direction_note": "positive favors DL for ssim/psnr; negative favors DL for rmse",
                }
            )

    summary_df = pd.DataFrame.from_records(summary_records)
    if not summary_df.empty:
        corrected = holm_bonferroni(summary_df["wilcoxon_p"].to_numpy(dtype=np.float64))
        summary_df["wilcoxon_p_holm"] = corrected

    slice_csv = out_path / "slice_metrics.csv"
    patient_csv = out_path / "patient_metrics.csv"
    summary_csv = out_path / "summary_statistics.csv"
    summary_json = out_path / "summary_statistics.json"

    slice_df.to_csv(slice_csv, index=False)
    patient_df.to_csv(patient_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    summary_payload = {
        "n_slices_evaluated": int(slice_df["slice_id"].nunique()) if not slice_df.empty else 0,
        "n_patients_evaluated": int(patient_df["patient_id"].nunique()) if not patient_df.empty else 0,
        "regions": sorted(slice_df["region"].unique().tolist()) if not slice_df.empty else [],
        "metrics": metric_names,
        "results": summary_records,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "slice_metrics": slice_csv,
        "patient_metrics": patient_csv,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
    }
