# Handwritten derivation guide

This file is a preparation aid, not a substitute for `hw1_handwritten.pdf`.
Rewrite the derivation legibly by hand, scan/photograph it, and combine the
pages into one PDF.

## 1. State the architecture and spatial sizes

For input `B × 3 × S × S`, where `S` is divisible by 16:

| Stage | Output shape |
|---|---|
| Conv 7×7, stride 2, 3→32 | `B × 32 × S/2 × S/2` |
| ReLU + MaxPool 3×3, stride 2 | `B × 32 × S/4 × S/4` |
| Conv 5×5, 32→64 + ReLU | `B × 64 × S/4 × S/4` |
| Conv 3×3, stride 2, 64→128 + ReLU | `B × 128 × S/8 × S/8` |
| Conv 1×1, 128→256 + ReLU | `B × 256 × S/8 × S/8` |
| Conv 3×3, stride 2, 256→256 + ReLU | `B × 256 × S/16 × S/16` |
| Conv 1×1, 256→512 + ReLU | `B × 512 × S/16 × S/16` |
| Global average pool | `B × 512` |
| Linear 512→256 + ReLU, Linear 256→100 | `B × 100` |

There is no BatchNorm because none is listed in the assigned layer table.
Convolutions have no bias; both linear layers have bias.

## 2. FLOPs

State the convention: one multiply-accumulate is two FLOPs. Count a ReLU
comparison as one operation, each MaxPool comparison as one operation, and
the global average of `n` values as `n-1` additions plus one division.

For a convolution:

```text
F_conv = 2 B H_out W_out C_out C_in K²
```

The six convolution terms are:

```text
2352 B S²
6400 B S²
2304 B S²
1024 B S²
4608 B S²
1024 B S²
```

They sum to `17712 B S²`. The six convolution ReLUs contribute
`21 B S²`, MaxPool contributes `16 B S²`, and global average pooling
contributes `2 B S²`.

The classifier contributes:

```text
2·512·256·B + 256B + 256B + 2·256·100·B + 100B
= 313956 B.
```

Final formula:

```text
FLOPs(S,B) = B (17751 S² + 313956).
```

## 3. Peak allocated memory

State the convention: FP32 uses four bytes; inference mode; in-place ReLUs;
the model parameters and input owned by the caller remain allocated.
The model has `1,040,324` parameters.

For a sequential network, an operator temporarily holds its input and output.
The largest live set occurs at MaxPool:

```text
original input:       3 B S²
Conv7×7 output:       8 B S²
MaxPool output:       2 B S²
total:               13 B S² elements.
```

Thus:

```text
Memory(S,B) = 4 (1,040,324 + 13 B S²) bytes.
```

Explicitly state that this equation excludes cuDNN workspace, allocator
rounding and unrelated GPU allocations. Their effect is evaluated from the
measured residual and OOM cases.

## 4. Bytes moved

Assume every operator reads each input and weight tensor once and writes each
output once. ReLU is in-place but still reads and writes its activation.
Caches and kernel fusion are ignored.

After summing all 17 operators:

```text
BytesMoved(S,B) = 4 (91 B S² + 2148 B + 1,040,324) bytes.
```

## 5. Latency

For operator `l`, let `F_l` be FLOPs and `Q_l` bytes moved. Use three
positive calibrated parameters:

- `t_launch`: launch overhead per operator;
- `P_eff`: effective FLOP/s;
- `BW_eff`: effective byte/s.

Layer-wise roofline formula:

```text
Latency(S,B; θ) =
Σ_l [t_launch + max(F_l(S,B)/P_eff, Q_l(S,B)/BW_eff)].
```

There are 17 conceptual operators. Explain that the launch term dominates
small workloads, memory time dominates low arithmetic-intensity layers, and
compute time dominates sufficiently large/high-intensity layers.

## 6. Energy

Energy is measured for the whole GPU without idle-power subtraction. Use:

```text
Energy(S,B; θ_E) =
17 e_launch
+ e_flop FLOPs(S,B)
+ e_byte BytesMoved(S,B).
```

The three non-negative coefficients are fitted on calibration points. State
that the model is phenomenological: DVFS, temperature and concurrent processes
are not explicitly represented.

## 7. Measurement assumptions

Write these explicitly:

- `model.eval()` and `torch.inference_mode()`;
- FP32, TF32 disabled;
- `torch.backends.cudnn.benchmark = False`;
- random inputs;
- CUDA-event median latency after warm-up;
- peak `torch.cuda.max_memory_allocated()`;
- whole-GPU NVML energy counter, or integrated NVML power if the counter is
  unsupported;
- base-grid points calibrate the parameters; every point with a random `S`
  or random `B` is validation data;
- OOM is recorded, not discarded.
