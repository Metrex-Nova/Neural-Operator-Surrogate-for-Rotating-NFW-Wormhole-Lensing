# Neural Operator Surrogate for Gravitational Lensing Simulation

A PyTorch project that learns the mapping

> **physical parameters (r₀, J) → lensing intensity profile I(b)**

using a **Fourier Neural Operator (FNO)**, replacing an expensive ray-tracing solver for a slowly rotating NFW wormhole.

---

## Project Structure

```
lensing_operator/
├── main.py                        ← unified entry point
├── requirements.txt
├── README.md
│
├── physics/
│   └── ray_tracing.py             ← parameterised physics solver (NFW wormhole)
│
├── data/
│   ├── generate_dataset.py        ← sweep (r0, J), call solver, save .npy
│   └── dataset_loader.py          ← PyTorch Dataset + DataLoader factory
│
├── models/
│   ├── fno.py                     ← 1-D Fourier Neural Operator
│   └── cnn_baseline.py            ← MLP + CNN baseline
│
├── training/
│   ├── train_fno.py               ← FNO training loop + checkpointing
│   └── train_baseline.py          ← MLP training loop + checkpointing
│
├── evaluation/
│   ├── metrics.py                 ← MSE, MAE, relative-L2
│   └── compare_models.py          ← test evaluation + speed benchmark + plots
│
├── data/                          ← generated .npy files (created at runtime)
├── checkpoints/                   ← saved model weights (created at runtime)
└── plots/                         ← output figures (created at runtime)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python main.py --mode generate --n_samples 2000
```

Samples `r₀ ∈ [0.3, 2.5] kpc` and `J ∈ [0.0, 1.5] kpc²` uniformly, solves the
full geodesic ODE for each pair, and saves:

| File | Shape | Contents |
|------|-------|----------|
| `data/params.npy` | `(N, 2)` | `[r0, J]` for each sample |
| `data/profiles.npy` | `(N, 300)` | normalised `I(b)` over 300-point b-grid |
| `data/b_grid.npy` | `(300,)` | impact-parameter grid [0.05, 13.0] kpc |
| `data/solver_time.npy` | `(1,)` | mean solver time per sample (seconds) |

### 3. Train the FNO

```bash
python main.py --mode train_fno --epochs 100 --modes 64 --width 64
```

Best checkpoint saved to `checkpoints/fno_best.pt`.

### 4. Train the MLP baseline

```bash
python main.py --mode train_baseline --epochs 100 --hidden 512
```

Best checkpoint saved to `checkpoints/baseline_best.pt`.

### 5. Evaluate and compare

```bash
python main.py --mode evaluate
```

Prints a metric table and speedup factor, then saves plots to `plots/`:

| Plot | Contents |
|------|----------|
| `sample_predictions.png` | True vs FNO vs MLP profiles (6 test samples) |
| `error_histograms.png` | Per-sample MSE distribution for both models |
| `training_curves.png` | Train / val loss curves (if results passed in-session) |
| `speedup_comparison.png` | Bar chart of inference speedup over physics solver |

---

## Model Architecture

### Fourier Neural Operator (FNO1d)

Input at each grid point: `[r₀_norm, J_norm, b_norm]` → shape `(B, N, 3)`

```
fc0  (3 → width)
  ↓
FNO blocks × n_layers
  each block: SpectralConv1d + pointwise Conv1d + InstanceNorm + GELU
  ↓
fc1  (width → 128) + GELU
fc2  (128 → 1)
  ↓
squeeze → (B, N)
```

Default: `modes=64`, `width=64`, `n_layers=4` ≈ **~300 K parameters**

### MLP Baseline

Input: `[r₀_norm, J_norm]` → shape `(B, 2)`

```
Linear(2 → 512) → GELU
[LayerNorm → Linear(512 → 512) → GELU] × 3
Linear(512 → 300)
```

---

## Physics

The solver integrates the null-geodesic equations for a slowly rotating NFW
wormhole with metric functions:

- **Redshift function** Φ(r): NFW gravitational potential
- **Shape function** b(r, r₀): wormhole throat at r₀
- **Frame-dragging** ω(r, J) = 2J / r³

The intensity integral follows standard accretion-disc emission formulas applied
along the photon path.  Normalisation is max-normalised per sample so the FNO
learns a unit-scale function.

---

## Key Parameters

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Throat radius | r₀ | 0.3 – 2.5 | kpc |
| Angular momentum | J | 0.0 – 1.5 | kpc² |
| NFW scale radius | Rₛ | 1.447 (fixed) | kpc |
| NFW density | ρₛ | 3.11×10⁻³ (fixed) | kpc⁻² |
| Observer distance | r_obs | 50 (fixed) | kpc |
