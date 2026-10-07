from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import random
import subprocess
import threading
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch

from equations import bytes_moved, flops, memory
from models import build_model


ROOT = Path(__file__).resolve().parent
BASE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CSV_FIELDS = (
    "image_size",
    "batch",
    "latency_s",
    "peak_memory_bytes",
    "energy_j",
    "status",
    "error",
    "is_validation",
    "measurement_order",
    "latency_repeats",
    "energy_repeats",
    "energy_method",
    "predicted_flops",
    "predicted_memory_bytes",
    "predicted_bytes_moved",
    "gpu_total_memory_bytes",
)


def measurement_grid(seed: int):
    rng = random.Random(seed)
    size_candidates = [size for size in range(32, 513, 16) if size not in BASE_SIZES]
    batch_candidates = [
        batch
        for batch in range(1, 257)
        if batch not in BASE_BATCHES and batch & (batch - 1)
    ]
    random_sizes = tuple(sorted(rng.sample(size_candidates, 4)))
    random_batches = tuple(sorted(rng.sample(batch_candidates, 3)))
    sizes = (*BASE_SIZES, *random_sizes)
    batches = (*BASE_BATCHES, *random_batches)
    configurations = list(product(sizes, batches))
    random.Random(seed + 1).shuffle(configurations)
    return configurations, random_sizes, random_batches


def median_latency(
    model,
    inputs,
    warmups: int,
    min_repeats: int,
    max_repeats: int,
    target_seconds: float,
):
    with torch.inference_mode():
        for _ in range(warmups):
            model(inputs)
        torch.cuda.synchronize()

        pilot_start = torch.cuda.Event(enable_timing=True)
        pilot_end = torch.cuda.Event(enable_timing=True)
        pilot_start.record()
        model(inputs)
        pilot_end.record()
        torch.cuda.synchronize()
        estimate = max(pilot_start.elapsed_time(pilot_end) / 1_000.0, 1e-6)
        repeats = max(
            min_repeats,
            min(max_repeats, math.ceil(target_seconds / estimate)),
        )

        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(repeats)
        ]
        for start, end in events:
            start.record()
            model(inputs)
            end.record()
        torch.cuda.synchronize()
    values = [start.elapsed_time(end) / 1_000.0 for start, end in events]
    return float(np.median(values)), repeats


def peak_memory(model, inputs) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model(inputs)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return int(peak)


class EnergyMeter:
    def __init__(self, device_index: int = 0, sample_interval: float = 0.02):
        import pynvml

        self.pynvml = pynvml
        self.sample_interval = sample_interval
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.has_energy_counter = True
        except pynvml.NVMLError:
            self.has_energy_counter = False

    def close(self) -> None:
        self.pynvml.nvmlShutdown()

    def measure(self, model, inputs, repeats: int) -> tuple[float, str]:
        if self.has_energy_counter:
            torch.cuda.synchronize()
            before = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            with torch.inference_mode():
                for _ in range(repeats):
                    model(inputs)
            torch.cuda.synchronize()
            after = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return (after - before) / 1_000.0 / repeats, "nvml_total_energy"

        samples: list[tuple[float, float]] = []
        stop = threading.Event()

        def sample_power() -> None:
            while not stop.is_set():
                power_w = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1_000.0
                samples.append((time.perf_counter(), power_w))
                stop.wait(self.sample_interval)

        sampler = threading.Thread(target=sample_power, daemon=True)
        sampler.start()
        time.sleep(self.sample_interval * 2)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                model(inputs)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        stop.set()
        sampler.join()
        active_power = [power for timestamp, power in samples if started <= timestamp <= finished]
        if not active_power:
            active_power = [power for _, power in samples]
        if not active_power:
            raise RuntimeError("NVML power sampler returned no samples")
        energy_j = float(np.mean(active_power)) * (finished - started) / repeats
        return energy_j, "nvml_power_sampling"


def environment_payload(seed, random_sizes, random_batches):
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "driver": driver,
        "seed": seed,
        "base_sizes": BASE_SIZES,
        "random_sizes": random_sizes,
        "base_batches": BASE_BATCHES,
        "random_batches": random_batches,
        "tf32": False,
        "cudnn_benchmark": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--min-repeats", type=int, default=7)
    parser.add_argument("--max-repeats", type=int, default=100)
    parser.add_argument("--latency-target-seconds", type=float, default=0.5)
    parser.add_argument("--energy-target-seconds", type=float, default=1.0)
    parser.add_argument("--max-energy-repeats", type=int, default=5_000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    output = args.output or ROOT / "results" / (
        "measurements_smoke.csv" if args.smoke else "measurements.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()

    configurations, random_sizes, random_batches = measurement_grid(args.seed)
    if args.smoke:
        configurations = [(32, 1), (64, 2), (128, 4)]
    validation_sizes, validation_batches = set(random_sizes), set(random_batches)
    completed = set()
    if output.exists():
        completed = {
            (int(row["image_size"]), int(row["batch"]))
            for row in csv.DictReader(output.open(newline=""))
        }

    model = build_model(args.seed).cuda().eval()
    gpu_total_memory = torch.cuda.get_device_properties(args.device).total_memory
    energy_meter = None if args.skip_energy else EnergyMeter(args.device)
    (ROOT / "results" / "environment.json").write_text(
        json.dumps(
            environment_payload(args.seed, random_sizes, random_batches),
            indent=2,
        )
        + "\n"
    )

    write_header = not output.exists() or output.stat().st_size == 0
    try:
        with output.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for order, (size, batch_size) in enumerate(configurations, start=1):
                if (size, batch_size) in completed:
                    continue
                row = {
                    "image_size": size,
                    "batch": batch_size,
                    "latency_s": "",
                    "peak_memory_bytes": "",
                    "energy_j": "",
                    "status": "ok",
                    "error": "",
                    "is_validation": size in validation_sizes or batch_size in validation_batches,
                    "measurement_order": order,
                    "latency_repeats": "",
                    "energy_repeats": "",
                    "energy_method": "disabled" if energy_meter is None else "",
                    "predicted_flops": float(flops(size, batch_size)),
                    "predicted_memory_bytes": float(memory(size, batch_size)),
                    "predicted_bytes_moved": float(bytes_moved(size, batch_size)),
                    "gpu_total_memory_bytes": gpu_total_memory,
                }
                inputs = None
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    inputs = torch.randn(
                        batch_size,
                        3,
                        size,
                        size,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    latency_s, latency_repeats = median_latency(
                        model,
                        inputs,
                        args.warmups,
                        args.min_repeats,
                        args.max_repeats,
                        args.latency_target_seconds,
                    )
                    row["latency_s"] = latency_s
                    row["latency_repeats"] = latency_repeats
                    row["peak_memory_bytes"] = peak_memory(model, inputs)
                    if energy_meter is not None:
                        energy_repeats = max(
                            1,
                            min(
                                args.max_energy_repeats,
                                math.ceil(args.energy_target_seconds / latency_s),
                            ),
                        )
                        energy_j, energy_method = energy_meter.measure(
                            model, inputs, energy_repeats
                        )
                        row["energy_j"] = energy_j
                        row["energy_repeats"] = energy_repeats
                        row["energy_method"] = energy_method
                except torch.cuda.OutOfMemoryError as exc:
                    row["status"] = "OOM"
                    row["error"] = str(exc).splitlines()[0][:300]
                finally:
                    del inputs
                    gc.collect()
                    torch.cuda.empty_cache()

                writer.writerow(row)
                handle.flush()
                print(
                    f"[{order:03d}/{len(configurations)}] S={size} B={batch_size} "
                    f"status={row['status']} latency={row['latency_s']}"
                )
    finally:
        if energy_meter is not None:
            energy_meter.close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
