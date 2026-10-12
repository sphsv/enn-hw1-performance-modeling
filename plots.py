from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equations import energy, latency, latency_components, memory


ROOT = Path(__file__).resolve().parent


def metric_by_batch_plot(
    frame,
    prediction,
    value_column,
    value_scale,
    value_label,
    title,
    model_label,
    output,
    show_oom=False,
):
    """Plot all 132 configurations as readable per-batch 2D panels."""
    batches = sorted(frame["batch"].unique())
    columns = 3
    rows = int(np.ceil(len(batches) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 3.35 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    size_curve = np.linspace(32, 512, 300)
    flat_axes = list(axes.flat)

    for axis, batch in zip(flat_axes, batches, strict=False):
        axis.plot(
            size_curve,
            prediction(size_curve, batch) / value_scale,
            color="#0072b2",
            linewidth=1.8,
            label=model_label,
        )
        batch_rows = frame[frame["batch"].eq(batch)]
        successful = batch_rows[
            batch_rows["status"].eq("ok") & batch_rows[value_column].notna()
        ]
        calibration = successful[~successful["is_validation"]]
        validation = successful[successful["is_validation"]]
        axis.scatter(
            calibration["image_size"],
            calibration[value_column] / value_scale,
            color="#4d4d4d",
            marker="o",
            s=27,
            label="measured: calibration",
            zorder=3,
        )
        axis.scatter(
            validation["image_size"],
            validation[value_column] / value_scale,
            color="#e69f00",
            marker="^",
            s=34,
            label="measured: held-out validation",
            zorder=3,
        )
        if show_oom:
            failed = batch_rows[batch_rows["status"].eq("OOM")]
            axis.scatter(
                failed["image_size"],
                prediction(failed["image_size"], failed["batch"]) / value_scale,
                color="#d55e00",
                marker="x",
                s=45,
                linewidths=1.5,
                label="OOM at analytical estimate",
                zorder=4,
            )
        axis.set_title(f"Batch B={batch}", fontsize=10)
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")

    for axis in flat_axes[len(batches) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Image size S [pixels]")
    for axis in axes[:, 0]:
        axis.set_ylabel(value_label)

    handles_by_label = {}
    for axis in flat_axes[: len(batches)]:
        handles, labels = axis.get_legend_handles_labels()
        handles_by_label.update(zip(labels, handles, strict=True))
    figure.legend(
        list(handles_by_label.values()),
        list(handles_by_label.keys()),
        loc="lower center",
        ncol=4 if show_oom else 3,
        frameon=False,
    )
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def prediction_scatter(frame, latency_theta, energy_theta, output):
    successful = frame[frame["status"].eq("ok")].copy()
    successful["latency_pred"] = latency(
        successful["image_size"], successful["batch"], latency_theta
    )
    successful["memory_pred"] = memory(
        successful["image_size"], successful["batch"]
    )
    successful["energy_pred"] = energy(
        successful["image_size"], successful["batch"], energy_theta
    )
    panels = (
        ("latency_s", "latency_pred", 1e-3, "Latency [ms]", "model"),
        (
            "peak_memory_bytes",
            "memory_pred",
            2**30,
            "Peak memory [GiB]",
            "analytical estimate",
        ),
        ("energy_j", "energy_pred", 1.0, "Energy [J]", "model"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, (actual, predicted, scale, label, prediction_name) in zip(
        axes, panels, strict=True
    ):
        valid = successful[actual].notna() & successful[predicted].notna()
        subset = successful[valid]
        for is_validation, marker, color, name in (
            (False, "o", "#777777", "calibration (used for fit)"),
            (True, "^", "#0072b2", "held-out validation"),
        ):
            split = subset[subset["is_validation"].eq(is_validation)]
            axis.scatter(
                split[actual] / scale,
                split[predicted] / scale,
                marker=marker,
                color=color,
                alpha=0.75,
                label=name,
            )
        values = np.concatenate(
            (subset[actual].to_numpy() / scale, subset[predicted].to_numpy() / scale)
        )
        low, high = np.nanmin(values), np.nanmax(values)
        axis.plot([low, high], [low, high], "k--", linewidth=1, label="ideal")
        validation = subset[subset["is_validation"]]
        if not validation.empty:
            denominator = np.maximum(
                np.abs(validation[actual].to_numpy()),
                np.finfo(np.float64).eps,
            )
            validation_mape = 100.0 * np.mean(
                np.abs(
                    validation[predicted].to_numpy()
                    - validation[actual].to_numpy()
                )
                / denominator
            )
            axis.set_title(f"Validation MAPE: {validation_mape:.1f}%")
        axis.set_xlabel(f"Measured {label}")
        axis.set_ylabel(f"{prediction_name.title()} {label}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].legend()
    figure.suptitle(
        "Calibration vs held-out validation (memory is analytical, not fitted)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def regime_plot(frame, theta, output):
    successful = frame[frame["status"].eq("ok")].copy()
    components = latency_components(
        successful["image_size"].to_numpy(),
        successful["batch"].to_numpy(),
        theta,
    )
    names = np.array(("launch", "memory", "compute"))
    values = np.column_stack([components[name] for name in names])
    successful["regime"] = names[np.argmax(values, axis=1)]
    successful["predicted"] = latency(
        successful["image_size"], successful["batch"], theta
    )
    successful["work"] = successful["batch"] * successful["image_size"] ** 2
    colors = {"launch": "#0072b2", "memory": "#e69f00", "compute": "#009e73"}

    figure, axis = plt.subplots(figsize=(8, 5))
    for name, color in colors.items():
        subset = successful[successful["regime"].eq(name)]
        axis.scatter(
            subset["work"],
            subset["latency_s"] * 1e3,
            color=color,
            alpha=0.7,
            label=f"measured: {name}",
        )
        axis.scatter(
            subset["work"],
            subset["predicted"] * 1e3,
            color=color,
            marker="x",
            label=f"predicted: {name}",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Work proxy $B S^2$ [pixel-images]")
    axis.set_ylabel("Forward latency [ms]")
    axis.grid(alpha=0.25, which="both")
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurements", type=Path, default=ROOT / "results" / "measurements.csv"
    )
    parser.add_argument("--theta", type=Path, default=ROOT / "results" / "theta.json")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "figures"
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.measurements)
    frame["is_validation"] = frame["is_validation"].astype(str).str.lower().eq("true")
    theta = json.loads(args.theta.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_by_batch_plot(
        frame,
        lambda size, batch: latency(size, batch, theta["latency"]),
        "latency_s",
        1e-3,
        "Latency [ms]",
        "Latency: measured points and calibrated model",
        "calibrated latency model",
        args.output_dir / "latency_by_batch.png",
    )
    metric_by_batch_plot(
        frame,
        memory,
        "peak_memory_bytes",
        2**30,
        "Peak allocated memory [GiB]",
        "Memory: measured peak and direct analytical calculation",
        "analytical memory (not fitted)",
        args.output_dir / "memory_by_batch.png",
        show_oom=True,
    )
    metric_by_batch_plot(
        frame,
        lambda size, batch: energy(size, batch, theta["energy"]),
        "energy_j",
        1.0,
        "Energy [J]",
        "Energy: measured points and calibrated model",
        "calibrated energy model",
        args.output_dir / "energy_by_batch.png",
    )
    prediction_scatter(
        frame,
        theta["latency"],
        theta["energy"],
        args.output_dir / "predicted_vs_measured.png",
    )
    regime_plot(frame, theta["latency"], args.output_dir / "latency_regimes.png")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
