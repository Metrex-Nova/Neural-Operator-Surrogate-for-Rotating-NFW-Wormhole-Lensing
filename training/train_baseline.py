import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import make_loaders
from models.cnn_baseline import MLPBaseline

CHECKPOINT_DIR = "checkpoints"


def train_baseline(data_dir="data/", epochs=100, batch_size=32, lr=1e-3,
                   hidden=512, n_layers=4, weight_decay=1e-5,
                   device_str="auto", n_grid=150, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if device_str == "auto" else torch.device(device_str)
    if verbose:
        print(f"Device: {device}")

    train_loader, val_loader, _ = make_loaders(data_dir, batch_size=batch_size)

    model     = MLPBaseline(n_params=2, n_out=n_grid, hidden=hidden, n_layers=n_layers).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    if verbose:
        print(f"MLP parameters: {model.count_parameters():,}")
        print(f"\nTraining MLP Baseline  —  epochs={epochs}  lr={lr}  "
              f"hidden={hidden}  layers={n_layers}")
        print("-" * 70)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "baseline_best.pt")

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t0      = time.perf_counter()
        tr_loss = 0.0
        for _, params, profile in train_loader:
            params, profile = params.to(device), profile.to(device)
            optimizer.zero_grad()
            loss = criterion(model(params), profile)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * params.shape[0]
        tr_loss /= len(train_loader.dataset)
        train_losses.append(tr_loss)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for _, params, profile in val_loader:
                params, profile = params.to(device), profile.to(device)
                vl_loss += criterion(model(params), profile).item() * params.shape[0]
        vl_loss /= len(val_loader.dataset)
        val_losses.append(vl_loss)

        scheduler.step()

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(), "val_mse": vl_loss,
                "config": dict(hidden=hidden, n_layers=n_layers, n_grid=n_grid),
            }, ckpt_path)

        if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == 1):
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train_MSE={tr_loss:.6f}  val_MSE={vl_loss:.6f}  "
                  f"best={best_val:.6f}  t={time.perf_counter()-t0:.1f}s")

    if verbose:
        print("-" * 70)
        print(f"Best val MSE: {best_val:.6f}")
        print(f"Checkpoint  : {ckpt_path}")

    return dict(train_losses=train_losses, val_losses=val_losses,
                best_val_mse=best_val, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,   default="data/")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=int,   default=512)
    parser.add_argument("--n_layers",   type=int,   default=4)
    parser.add_argument("--device",     type=str,   default="auto")
    args = parser.parse_args()
    train_baseline(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, hidden=args.hidden, n_layers=args.n_layers,
                   device_str=args.device)
