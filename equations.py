from __future__ import annotations

from collections.abc import Mapping

import numpy as np


BYTES_PER_FP32 = 4.0
PARAMETER_COUNT = 1_040_324.0
KERNEL_LAUNCHES = 17


def _inputs(image_size, batch):
    size, batch_size = np.broadcast_arrays(
        np.asarray(image_size, dtype=np.float64),
        np.asarray(batch, dtype=np.float64),
    )
    if np.any(size <= 0) or np.any(batch_size <= 0):
        raise ValueError("image_size and batch must be positive")
    return size, batch_size


def _layer_costs(image_size, batch):
    """Return per-operator (FLOPs, ideal HBM bytes) arrays.

    Tensor traffic assumes that each input, output and weight tensor is read or
    written once per operator. Cache effects and cuDNN workspaces are excluded.
    """
    size, batch_size = _inputs(image_size, batch)
    work = batch_size * size**2

    input_elements = 3.0 * work
    conv1 = 8.0 * work
    pool = 2.0 * work
    conv2 = 4.0 * work
    conv3 = 2.0 * work
    conv4 = 4.0 * work
    conv5 = 1.0 * work
    conv6 = 2.0 * work
    gap = 512.0 * batch_size
    fc1 = 256.0 * batch_size
    fc2 = 100.0 * batch_size

    def traffic(*elements):
        return BYTES_PER_FP32 * sum(elements)

    return (
        (2.0 * conv1 * 3 * 7**2, traffic(input_elements, 4_704, conv1)),
        (conv1, traffic(conv1, conv1)),
        (8.0 * pool, traffic(conv1, pool)),
        (2.0 * conv2 * 32 * 5**2, traffic(pool, 51_200, conv2)),
        (conv2, traffic(conv2, conv2)),
        (2.0 * conv3 * 64 * 3**2, traffic(conv2, 73_728, conv3)),
        (conv3, traffic(conv3, conv3)),
        (2.0 * conv4 * 128, traffic(conv3, 32_768, conv4)),
        (conv4, traffic(conv4, conv4)),
        (2.0 * conv5 * 256 * 3**2, traffic(conv4, 589_824, conv5)),
        (conv5, traffic(conv5, conv5)),
        (2.0 * conv6 * 256, traffic(conv5, 131_072, conv6)),
        (conv6, traffic(conv6, conv6)),
        (conv6, traffic(conv6, gap)),
        (
            batch_size * (2 * 512 * 256 + 256),
            traffic(gap, 131_072 + 256, fc1),
        ),
        (fc1, traffic(fc1, fc1)),
        (
            batch_size * (2 * 256 * 100 + 100),
            traffic(fc1, 25_600 + 100, fc2),
        ),
    )


def flops(image_size, batch):
    """Forward-pass operations, using one MAC = two FLOPs."""
    size, batch_size = _inputs(image_size, batch)
    # Equivalent closed form: B * (17_751 * S**2 + 313_956).
    return batch_size * (17_751.0 * size**2 + 313_956.0)


def bytes_moved(image_size, batch):
    """Idealized FP32 traffic between GPU memory and operators."""
    size, batch_size = _inputs(image_size, batch)
    # Equivalent to summing the traffic term of every operator above.
    return BYTES_PER_FP32 * (
        91.0 * batch_size * size**2
        + 2_148.0 * batch_size
        + PARAMETER_COUNT
    )


def memory(image_size, batch):
    """Analytical peak allocated bytes for inference.

    The input owned by the caller remains live. The peak occurs at MaxPool:
    original input + convolution output + pooling output = 13 * B * S**2
    elements. cuDNN workspaces and allocator bookkeeping are intentionally not
    included; their effect is evaluated against measurements.
    """
    size, batch_size = _inputs(image_size, batch)
    return BYTES_PER_FP32 * (PARAMETER_COUNT + 13.0 * batch_size * size**2)


def latency(image_size, batch, theta: Mapping[str, float]):
    """Layer-wise roofline latency prediction in seconds."""
    launch = float(theta["launch_seconds_per_kernel"])
    compute_rate = float(theta["effective_flops_per_second"])
    bandwidth = float(theta["effective_bandwidth_bytes_per_second"])
    if min(launch, compute_rate, bandwidth) <= 0:
        raise ValueError("latency parameters must be positive")

    size, batch_size = _inputs(image_size, batch)
    result = np.zeros_like(size + batch_size, dtype=np.float64)
    for layer_flops, layer_bytes in _layer_costs(size, batch_size):
        result += launch + np.maximum(
            layer_flops / compute_rate,
            layer_bytes / bandwidth,
        )
    return result


def latency_components(image_size, batch, theta: Mapping[str, float]):
    """Return launch, memory-bound and compute-bound latency contributions."""
    size, batch_size = _inputs(image_size, batch)
    launch = np.full_like(
        size + batch_size,
        KERNEL_LAUNCHES * float(theta["launch_seconds_per_kernel"]),
    )
    memory_time = np.zeros_like(launch)
    compute_time = np.zeros_like(launch)
    for layer_flops, layer_bytes in _layer_costs(size, batch_size):
        compute = layer_flops / float(theta["effective_flops_per_second"])
        transfer = layer_bytes / float(theta["effective_bandwidth_bytes_per_second"])
        memory_time += np.where(transfer >= compute, transfer, 0.0)
        compute_time += np.where(compute > transfer, compute, 0.0)
    return {"launch": launch, "memory": memory_time, "compute": compute_time}


def energy(image_size, batch, theta_energy: Mapping[str, float]):
    """Whole-GPU energy prediction in joules."""
    return (
        KERNEL_LAUNCHES * float(theta_energy["joules_per_kernel"])
        + flops(image_size, batch) * float(theta_energy["joules_per_flop"])
        + bytes_moved(image_size, batch) * float(theta_energy["joules_per_byte"])
    )
