# Homework 1 — analytical performance model of a small CNN

Assignment: [Efficient Models course — Homework 1](https://github.com/On-Point-RND/Efficient-Models-course-ITMO-2026/blob/main/home_work_one.md).

This repository implements the CNN and four analytical functions required by
the assignment:

- `FLOPs(S, B)`;
- peak allocated `Memory(S, B)`;
- calibrated layer-wise roofline `Latency(S, B, θ)`;
- calibrated whole-GPU `Energy(S, B, θ_E)`.

The handwritten derivation must be added as `hw1_handwritten.pdf`.
`HANDWRITTEN_DERIVATION_GUIDE.md` contains the checked equations and
assumptions to rewrite by hand.

For a complete Russian-language explanation of the assignment, implementation,
current completion status and remaining steps, see
`HW1_FULL_WALKTHROUGH_RU.md`.

## Status

The code implementation and synthetic checks are complete. Final measurements
are intentionally not reported yet: the assignment requires a low-grade GPU
such as a T4/P100 or a local NVIDIA GPU with at most 16 GB. Results from the
A100 80 GB used only for implementation smoke tests would not meet that
requirement. After the low-grade GPU run, this section will contain the
measured errors, OOM boundary and one-page discussion.

## Model

The network follows the assigned layer table exactly: six convolutions with an
in-place ReLU after each, one MaxPool, global average pooling and a
512→256→100 classifier. Convolutions have no bias. The table does not list
BatchNorm, so the implementation does not add it. The model has 1,040,324 FP32
parameters.

## Closed-form equations

Using one MAC = two FLOPs:

```text
FLOPs(S,B) = B (17,751 S² + 313,956).
```

The analytical memory peak includes model parameters and live FP32 tensors.
It excludes hidden cuDNN workspaces and allocator bookkeeping:

```text
Memory(S,B) = 4 (1,040,324 + 13 B S²) bytes.
```

Memory is **not fitted or learned**. It is calculated directly from tensor
lifetimes and parameter count. The measured peak is collected only to check
this analytical calculation, quantify workspace/allocator effects and compare
the predicted boundary with actual OOM configurations.

Ideal tensor traffic assumes one read of each input/weight and one write of
each output per operator:

```text
BytesMoved(S,B) = 4 (91 B S² + 2,148 B + 1,040,324) bytes.
```

Latency is a layer-wise roofline sum over 17 conceptual operators:

```text
Latency(S,B;θ) =
Σ_l [t_launch + max(F_l(S,B)/P_eff, Q_l(S,B)/BW_eff)].
```

Whole-GPU energy is modelled with non-negative calibrated coefficients:

```text
Energy(S,B;θ_E) =
17 e_launch + e_flop FLOPs(S,B) + e_byte BytesMoved(S,B).
```

The functions in `equations.py` accept scalars or broadcast-compatible NumPy
arrays.

## Calibration and validation split

The base grid contains 7 image sizes and 9 batch sizes. With seed 7, the
additional random coordinates are:

- image sizes: 144, 272, 304, 448;
- batch sizes: 18, 24, 219.

All 132 Cartesian-product configurations are measured. The 63 base×base points
are used for calibration. Any configuration containing a random coordinate is
held out, giving 69 validation points. Only successful calibration rows are
passed to the latency and energy fit. Held-out validation rows are never used
to choose parameters; they are used only for the final error metrics and plots.

## Measurement protocol

- one CUDA GPU, FP32, `eval()`, `torch.inference_mode()`;
- `cudnn.benchmark=False`;
- TF32 disabled for cuDNN and matrix multiplication;
- random input tensors;
- latency is the median of adaptive repeated CUDA-event measurements;
- memory is `torch.cuda.max_memory_allocated()`;
- whole-GPU energy uses the NVML cumulative energy counter when available,
  otherwise concurrent NVML power sampling;
- OOM configurations are recorded and the process continues;
- rows are appended immediately, so an interrupted run resumes safely.

## Reproduction

### Google Colab (recommended for the final T4 run)

Open `HW1_COLAB_RUNNER.ipynb` in Colab, select a T4 GPU and run the cells from
top to bottom. The notebook is only an orchestration interface: it invokes the
same `measure.py`, `calibrate.py` and `plots.py` files used by the command-line
workflow. It stores the project in Google Drive by default, so an interrupted
132-point measurement can be resumed.

### Command line

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python measure.py
.venv/bin/python calibrate.py
.venv/bin/python plots.py
```

For a quick implementation check that is not part of the submitted results:

```bash
.venv/bin/python measure.py --smoke --skip-energy --overwrite
```

## Outputs

```text
results/
├── measurements.csv
├── environment.json
├── theta.json
├── analysis_summary.json
└── figures/
    ├── latency_by_batch.png
    ├── memory_by_batch.png
    ├── energy_by_batch.png
    ├── predicted_vs_measured.png
    └── latency_regimes.png
```

The three by-batch figures use readable 2D small multiples rather than a dense
3D surface. Every panel overlays the model/calculation with measured points:
circles are calibration points, triangles are genuinely held-out validation
points, and red crosses mark OOM configurations. The predicted-versus-measured
figure also reports validation MAPE and keeps calibration and validation
visually distinct.

## Discussion

This section will be completed from the low-grade GPU measurements. It will
cover:

1. validation error for latency, memory and energy;
2. the launch-, memory- and compute-bound regions;
3. the analytical versus observed OOM boundary;
4. deviations caused by cuDNN algorithm selection/workspace, caching, DVFS and
   power-counter resolution;
5. why one set of effective rates cannot reproduce every convolution shape.
