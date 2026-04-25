from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


CSV_FILES = [
    "training_curves.csv",
    "inference_metrics.csv",
    "method_summary.csv",
]

PNG_FILES = [
    "ct_panel.png",
    "training_curves.png",
    "metrics_comparison.png",
    "speed_quality_plot.png",
    "radar_chart.png",
    "freq_spectrum.png",
]


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_train_val_summaries(training_df: pd.DataFrame, metrics_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if training_df.empty:
        return outputs

    detailed_path = metrics_dir / "train_val_accuracy_detailed.csv"
    training_df.sort_values(["model", "epoch"]).to_csv(detailed_path, index=False)
    outputs["train_val_accuracy_detailed"] = str(detailed_path)

    idx = training_df.groupby("model")["val_ssim"].idxmax()
    best_df = (
        training_df.loc[idx, ["model", "epoch", "train_loss", "val_loss", "val_ssim", "val_psnr", "val_rmse", "lr"]]
        .sort_values("val_ssim", ascending=False)
        .rename(columns={"epoch": "best_epoch"})
    )
    best_path = metrics_dir / "train_val_best_epoch_summary.csv"
    best_df.to_csv(best_path, index=False)
    outputs["train_val_best_epoch_summary"] = str(best_path)
    return outputs


def _build_test_summary(summary_df: pd.DataFrame, metrics_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if summary_df.empty:
        return outputs

    fbp_rows = summary_df[summary_df["method"] == "FBP"]
    if fbp_rows.empty:
        enhanced = summary_df.copy()
    else:
        fbp = fbp_rows.iloc[0]
        enhanced = summary_df.copy()
        enhanced["delta_ssim_vs_fbp"] = enhanced["mean_ssim"] - float(fbp["mean_ssim"])
        enhanced["delta_psnr_vs_fbp"] = enhanced["mean_psnr"] - float(fbp["mean_psnr"])
        enhanced["delta_rmse_vs_fbp"] = enhanced["mean_rmse"] - float(fbp["mean_rmse"])
        base_rmse = float(fbp["mean_rmse"])
        if abs(base_rmse) > 1e-12:
            enhanced["rmse_reduction_pct_vs_fbp"] = (
                (base_rmse - enhanced["mean_rmse"]) / base_rmse * 100.0
            )
        else:
            enhanced["rmse_reduction_pct_vs_fbp"] = 0.0

    enhanced = enhanced.sort_values("mean_ssim", ascending=False)
    test_summary_path = metrics_dir / "test_accuracy_summary.csv"
    enhanced.to_csv(test_summary_path, index=False)
    outputs["test_accuracy_summary"] = str(test_summary_path)
    return outputs


def _write_report(run_dir: Path, training_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    report_path = run_dir / "path_c_objective_report.md"
    lines: list[str] = []

    lines.append("# Path C Objective Report")
    lines.append("")
    lines.append("## Objective")
    lines.append(
        "Apply deep learning post-processing on top of Filtered Back Projection (FBP) "
        "to improve reconstruction quality under low-dose CT conditions."
    )
    lines.append("")
    lines.append("## Evidence Stored In This Run")
    lines.append("- Training/validation metrics per epoch (`training_curves.csv`).")
    lines.append("- Per-slice test metrics (`inference_metrics.csv`).")
    lines.append("- Method-level aggregate metrics and statistical tests (`method_summary.csv`).")
    lines.append("- Visual comparisons and diagnostic plots (`visualizations/`).")
    lines.append("")

    if not summary_df.empty:
        lines.append("## Test Summary (Mean Metrics)")
        keep_cols = ["method", "mean_ssim", "mean_psnr", "mean_rmse", "mean_inference_ms"]
        preview = summary_df[keep_cols].sort_values("mean_ssim", ascending=False)
        lines.append("")
        lines.append("```text")
        lines.append(preview.to_string(index=False))
        lines.append("```")
        lines.append("")

    if not training_df.empty:
        lines.append("## Training Coverage")
        lines.append(f"- Models trained: {', '.join(sorted(training_df['model'].unique()))}")
        lines.append(f"- Total epoch records: {len(training_df)}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize Path C CT reconstruction outputs.")
    parser.add_argument("--run-dir", required=True, help="Path to a Path C run directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = run_dir / "metrics"
    visuals_dir = run_dir / "visualizations"
    metadata_dir = run_dir / "metadata"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    copied_metrics: list[str] = []
    copied_visuals: list[str] = []

    for name in CSV_FILES:
        if _copy_if_exists(run_dir / name, metrics_dir / name):
            copied_metrics.append(name)

    for name in PNG_FILES:
        if _copy_if_exists(run_dir / name, visuals_dir / name):
            copied_visuals.append(name)

    _copy_if_exists(run_dir / "run_metadata.json", metadata_dir / "run_metadata.json")

    training_df = _safe_read_csv(run_dir / "training_curves.csv")
    summary_df = _safe_read_csv(run_dir / "method_summary.csv")

    derived_files: dict[str, str] = {}
    derived_files.update(_build_train_val_summaries(training_df, metrics_dir))
    derived_files.update(_build_test_summary(summary_df, metrics_dir))
    report_path = _write_report(run_dir, training_df, summary_df)

    manifest = {
        "run_dir": str(run_dir),
        "copied_metrics": copied_metrics,
        "copied_visuals": copied_visuals,
        "derived_files": derived_files,
        "report": str(report_path),
    }
    (run_dir / "path_c_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Organized Path C outputs in: {run_dir}")
    print(f"- Metrics folder: {metrics_dir}")
    print(f"- Visualizations folder: {visuals_dir}")
    print(f"- Report: {report_path}")


if __name__ == "__main__":
    main()
