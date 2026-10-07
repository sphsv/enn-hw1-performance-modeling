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


def surface_plot(frame, prediction, z_column, z_scale, z_label, output, oom=False):
    sizes = np.linspace(32, 512, 80)
    batches = np.geomspace(1, 256, 60)
    size_grid, batch_grid = np.meshgrid(sizes, batches)
    predicted = prediction(size_grid, batch_grid) / z_scale

    figure = plt.figure(figsize=(9, 6.5))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        size_grid,
        np.log2(batch_grid),
        predicted,
        cmap="viridis",
        alpha=0.55,
        linewidth=0,
    )
    successful = frame[frame["status"].eq("ok") & frame[z_column].notna()]
    axis.scatter(
        successful["image_size"],
        np.log2(successful["batch"]),
        successful[z_column] / z_scale,
        color="black",
        s=18,
        label="measured",
    )
    if oom:
        failed = frame[frame["status"].eq("OOM")]
        if not failed.empty:
            axis.scatter(
                failed["image_size"],
                np.log2(failed["batch"]),
                prediction(failed["image_size"], failed["batch"]) / z_scale,
                color="red",
                marker="x",
                s=45,
                label="OOM (at predicted memory)",
            )
    axis.set_xlabel("Image size S [pixels]")
    axis.set_ylabel("Batch size B")
    axis.set_zlabel(z_label)
    batch_ticks = np.arange(0, 9)
    axis.set_yticks(batch_ticks, [str(2**value) for value in batch_ticks])
    axis.legend()
    figure.subplots_adjust(left=0.02, right=0.96, bottom=0.06, top=0.98)
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
        ("latency_s", "latency_pred", 1e3, "Latency [ms]"),
        ("peak_memory_bytes", "memory_pred", 2**30, "Peak memory [GiB]"),
        ("energy_j", "energy_pred", 1.0, "Energy [J]"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, (actual, predicted, scale, label) in zip(axes, panels, strict=True):
        valid = successful[actual].notna() & successful[predicted].notna()
        subset = successful[valid]
        for is_validation, marker, name in (
            (False, "o", "calibration"),
            (True, "^", "validation"),
        ):
            split = subset[subset["is_validation"].eq(is_validation)]
            axis.scatter(
                split[actual] / scale,
                split[predicted] / scale,
                marker=marker,
                alpha=0.75,
                label=name,
            )
        values = np.concatenate(
            (subset[actual].to_numpy() / scale, subset[predicted].to_numpy() / scale)
        )
        low, high = np.nanmin(values), np.nanmax(values)
        axis.plot([low, high], [low, high], "k--", linewidth=1, label="ideal")
        axis.set_xlabel(f"Measured {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
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

    surface_plot(
        frame,
        lambda size, batch: latency(size, batch, theta["latency"]),
        "latency_s",
        1e-3,
        "Latency [ms]",
        args.output_dir / "latency_surface.png",
    )
    surface_plot(
        frame,
        memory,
        "peak_memory_bytes",
        2**30,
        "Peak allocated memory [GiB]",
        args.output_dir / "memory_surface.png",
        oom=True,
    )
    surface_plot(
        frame,
        lambda size, batch: energy(size, batch, theta["energy"]),
        "energy_j",
        1.0,
        "Energy [J]",
        args.output_dir / "energy_surface.png",
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
