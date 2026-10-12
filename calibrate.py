from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from equations import KERNEL_LAUNCHES, bytes_moved, energy, flops, latency, memory


ROOT = Path(__file__).resolve().parent


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - actual
    denominator = np.maximum(np.abs(actual), np.finfo(np.float64).eps)
    total = np.sum((actual - actual.mean()) ** 2)
    return {
        "count": int(actual.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mape_percent": float(100.0 * np.mean(np.abs(residual) / denominator)),
        "median_ape_percent": float(
            100.0 * np.median(np.abs(residual) / denominator)
        ),
        "r2": float(1.0 - np.sum(residual**2) / total) if total > 0 else None,
    }


def fit_latency(frame: pd.DataFrame):
    sizes = frame["image_size"].to_numpy(dtype=np.float64)
    batches = frame["batch"].to_numpy(dtype=np.float64)
    actual = frame["latency_s"].to_numpy(dtype=np.float64)
    operations = flops(sizes, batches)
    traffic = bytes_moved(sizes, batches)
    initial = np.array(
        [
            max(actual.min() / (2 * KERNEL_LAUNCHES), 1e-7),
            max(np.max(operations / actual), 1e9),
            max(np.max(traffic / actual), 1e9),
        ]
    )

    def unpack(log_parameters):
        values = np.exp(log_parameters)
        return {
            "launch_seconds_per_kernel": values[0],
            "effective_flops_per_second": values[1],
            "effective_bandwidth_bytes_per_second": values[2],
        }

    def residual(log_parameters):
        predicted = latency(sizes, batches, unpack(log_parameters))
        return np.log(predicted) - np.log(actual)

    fitted = least_squares(
        residual,
        np.log(initial),
        bounds=(
            np.log([1e-9, 1e8, 1e8]),
            np.log([1e-2, 1e16, 1e14]),
        ),
    )
    return {key: float(value) for key, value in unpack(fitted.x).items()}


def fit_energy(frame: pd.DataFrame):
    sizes = frame["image_size"].to_numpy(dtype=np.float64)
    batches = frame["batch"].to_numpy(dtype=np.float64)
    actual = frame["energy_j"].to_numpy(dtype=np.float64)
    features = np.column_stack(
        (
            np.full_like(actual, KERNEL_LAUNCHES),
            flops(sizes, batches),
            bytes_moved(sizes, batches),
        )
    )
    scales = np.maximum(np.max(features, axis=0), 1.0)
    scaled_coefficients, _ = nnls(features / scales, actual)
    coefficients = scaled_coefficients / scales
    return {
        "joules_per_kernel": float(coefficients[0]),
        "joules_per_flop": float(coefficients[1]),
        "joules_per_byte": float(coefficients[2]),
    }


def evaluate(frame, latency_theta, energy_theta):
    output = {}
    for split, subset in (
        ("calibration", frame[~frame["is_validation"]]),
        ("validation", frame[frame["is_validation"]]),
    ):
        if subset.empty:
            continue
        sizes = subset["image_size"].to_numpy()
        batches = subset["batch"].to_numpy()
        output[split] = {
            "latency_seconds": regression_metrics(
                subset["latency_s"], latency(sizes, batches, latency_theta)
            ),
            "memory_bytes": regression_metrics(
                subset["peak_memory_bytes"], memory(sizes, batches)
            ),
        }
        energy_subset = subset[np.isfinite(subset["energy_j"])]
        if not energy_subset.empty:
            output[split]["energy_joules"] = regression_metrics(
                energy_subset["energy_j"],
                energy(
                    energy_subset["image_size"].to_numpy(),
                    energy_subset["batch"].to_numpy(),
                    energy_theta,
                ),
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurements", type=Path, default=ROOT / "results" / "measurements.csv"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "theta.json")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "analysis_summary.json",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.measurements)
    raw["is_validation"] = raw["is_validation"].astype(str).str.lower().eq("true")
    successful = raw[raw["status"].eq("ok")].copy()
    calibration = successful[~successful["is_validation"]]
    validation = successful[successful["is_validation"]]
    if calibration.empty:
        raise SystemExit("No successful calibration rows")
    if validation.empty:
        raise SystemExit("No successful held-out validation rows")
    latency_theta = fit_latency(calibration)

    energy_calibration = calibration[np.isfinite(calibration["energy_j"])]
    if energy_calibration.empty:
        raise SystemExit("No energy measurements available for calibration")
    energy_theta = fit_energy(energy_calibration)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latency": latency_theta,
        "energy": energy_theta,
        "fit_rows": {
            "latency": int(len(calibration)),
            "energy": int(len(energy_calibration)),
        },
        "split": {
            "calibration": "is_validation=false; base S x base B only",
            "validation": "is_validation=true; any random S or random B; never used for fitting",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    summary = {
        "metrics": evaluate(successful, latency_theta, energy_theta),
        "rows": {
            "total": int(len(raw)),
            "successful": int(len(successful)),
            "oom": int(raw["status"].eq("OOM").sum()),
            "calibration": int((~raw["is_validation"]).sum()),
            "validation": int(raw["is_validation"].sum()),
            "successful_calibration": int(len(calibration)),
            "successful_validation": int(len(validation)),
        },
        "split_definition": payload["split"],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(payload | summary, indent=2))


if __name__ == "__main__":
    main()
