import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import make_loaders
from models.fno import FNO1d
from models.cnn_baseline import MLPBaseline
from evaluation.metrics import evaluate_model, mse

CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR      = "plots"


def load_fno(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt.get("config", {})
    model = FNO1d(modes=cfg.get("modes", 24), width=cfg.get("width", 32),
                  n_layers=cfg.get("n_layers", 4)).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def load_baseline(ckpt_path, device, n_grid=150):
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt.get("config", {})
    model = MLPBaseline(n_params=2, n_out=cfg.get("n_grid", n_grid),
                        hidden=cfg.get("hidden", 512),
                        n_layers=cfg.get("n_layers", 4)).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def _bench_model(model, loader, device, model_type, n_reps=3):
    times = []
    with torch.no_grad():
        for fno_input, params, _ in loader:
            inp = fno_input.to(device) if model_type == "fno" else params.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_reps):
                model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / (n_reps * inp.shape[0]))
            if len(times) >= 20:
                break
    return float(np.mean(times))


def _plot_sample_predictions(fno_model, mlp_model, dataset, b_grid, device, n_show=6):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    indices = np.random.choice(len(dataset), n_show, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    fig.suptitle("True vs Predicted Intensity Profiles", fontsize=14, fontweight="bold")

    for ax, idx in zip(axes.flatten(), indices):
        fno_inp, params, true_I = dataset[idx]
        true_I = true_I.numpy()
        with torch.no_grad():
            fno_pred = fno_model(fno_inp.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            mlp_pred = mlp_model(params.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        r0 = params[0].item() * (2.5 - 0.3) + 0.3
        J  = params[1].item() * 1.5

        ax.plot(b_grid, true_I,   "k-",  lw=2.0, label="Physics", zorder=3)
        ax.plot(b_grid, fno_pred, "r--", lw=1.8, label="FNO",     zorder=2)
        ax.plot(b_grid, mlp_pred, "b:",  lw=1.8, label="MLP",     zorder=1)
        ax.set_title(f"$r_0={r0:.2f}$ kpc,  $J={J:.2f}$ kpc²", fontsize=9)
        ax.set_xlabel("$b$ (kpc)", fontsize=8)
        ax.set_ylabel("$I(b)$",    fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, b_grid[-1])
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sample_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_error_histograms(fno_model, mlp_model, loader, device):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fno_mses, mlp_mses = [], []

    with torch.no_grad():
        for fno_inp, params, profile in loader:
            profile  = profile.to(device)
            fno_pred = fno_model(fno_inp.to(device))
            mlp_pred = mlp_model(params.to(device))
            for i in range(profile.shape[0]):
                t = profile[i].cpu().numpy()
                fno_mses.append(float(np.mean((fno_pred[i].cpu().numpy() - t) ** 2)))
                mlp_mses.append(float(np.mean((mlp_pred[i].cpu().numpy() - t) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.suptitle("Per-sample MSE on Test Set", fontsize=13, fontweight="bold")
    for ax, vals, name, color in zip(axes, [fno_mses, mlp_mses],
                                     ["FNO", "MLP Baseline"],
                                     ["tomato", "steelblue"]):
        ax.hist(vals, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(vals), color="k", ls="--", lw=1.5,
                   label=f"mean={np.mean(vals):.5f}")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("MSE", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "error_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_speedup_bar(fno_speedup, mlp_speedup):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    bars = ax.bar(["FNO", "MLP Baseline"], [fno_speedup, mlp_speedup],
                  color=["tomato", "steelblue"], edgecolor="white", width=0.4)
    for bar, val in zip(bars, [fno_speedup, mlp_speedup]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:,.0f}×", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup over physics solver", fontsize=11)
    ax.set_title("Inference Speedup vs Physics Solver", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "speedup_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_training_curves(fno_results, mlp_results):
    if fno_results is None and mlp_results is None:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_title("Training Loss Curves", fontsize=13, fontweight="bold")
    if fno_results:
        e = range(1, len(fno_results["train_losses"]) + 1)
        ax.semilogy(e, fno_results["train_losses"], "r-",  lw=1.5, label="FNO train")
        ax.semilogy(e, fno_results["val_losses"],   "r--", lw=1.5, label="FNO val")
    if mlp_results:
        e = range(1, len(mlp_results["train_losses"]) + 1)
        ax.semilogy(e, mlp_results["train_losses"], "b-",  lw=1.5, label="MLP train")
        ax.semilogy(e, mlp_results["val_losses"],   "b--", lw=1.5, label="MLP val")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE (log scale)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def compare_models(data_dir="data/", batch_size=64, device_str="auto",
                   fno_results=None, mlp_results=None, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if device_str == "auto" else torch.device(device_str)

    fno_ckpt = os.path.join(CHECKPOINT_DIR, "fno_best.pt")
    mlp_ckpt = os.path.join(CHECKPOINT_DIR, "baseline_best.pt")

    if not os.path.exists(fno_ckpt):
        print(f"[error] FNO checkpoint not found: {fno_ckpt}")
        return
    if not os.path.exists(mlp_ckpt):
        print(f"[error] MLP checkpoint not found: {mlp_ckpt}")
        return

    fno_model = load_fno(fno_ckpt, device)
    mlp_model = load_baseline(mlp_ckpt, device)

    _, _, test_loader = make_loaders(data_dir, batch_size=batch_size)

    if verbose:
        print("\n[1] Computing test-set metrics ...")
    fno_metrics = evaluate_model(fno_model, test_loader, device, model_type="fno")
    mlp_metrics = evaluate_model(mlp_model, test_loader, device, model_type="baseline")

    if verbose:
        w = 12
        print(f"\n{'Metric':<18} {'FNO':>{w}} {'MLP Baseline':>{w}}")
        print("-" * (18 + 2 * w + 2))
        for key in ("mse", "mae", "rel_l2"):
            print(f"  {key:<16} {fno_metrics[key]:>{w}.6f} {mlp_metrics[key]:>{w}.6f}")
        print(f"  {'n_test':<16} {fno_metrics['n_samples']:>{w}d}")

    if verbose:
        print(f"\n[2] Speed benchmark  (device={device}) ...")
    fno_t = _bench_model(fno_model, test_loader, device, "fno")
    mlp_t = _bench_model(mlp_model, test_loader, device, "baseline")

    solver_time = float(np.load(os.path.join(data_dir, "solver_time.npy"))[0])

    fno_speedup = solver_time / fno_t if fno_t > 0 else float("inf")
    mlp_speedup = solver_time / mlp_t if mlp_t > 0 else float("inf")

    if verbose:
        print(f"  Physics solver : {solver_time*1000:.2f} ms/sample")
        print(f"  FNO inference  : {fno_t*1000:.4f} ms/sample")
        print(f"  MLP inference  : {mlp_t*1000:.4f} ms/sample")
        print(f"\n  FNO speedup : {fno_speedup:,.0f}x")
        print(f"  MLP speedup : {mlp_speedup:,.0f}x")

    if verbose:
        print("\n[3] Generating plots ...")
    b_grid_np   = np.load(os.path.join(data_dir, "b_grid.npy"))
    test_subset = test_loader.dataset
    _plot_sample_predictions(fno_model, mlp_model, test_subset, b_grid_np, device)
    _plot_error_histograms(fno_model, mlp_model, test_loader, device)
    _plot_training_curves(fno_results, mlp_results)
    _plot_speedup_bar(fno_speedup, mlp_speedup)

    if verbose:
        print("\nEvaluation complete.  Plots saved to plots/")

    return dict(fno_metrics=fno_metrics, mlp_metrics=mlp_metrics,
                fno_speedup=fno_speedup, mlp_speedup=mlp_speedup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device",     type=str, default="auto")
    args = parser.parse_args()
    compare_models(data_dir=args.data_dir, batch_size=args.batch_size,
                   device_str=args.device)
