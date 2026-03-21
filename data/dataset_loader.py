import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

R0_MIN, R0_MAX = 0.3, 2.5
J_MIN,  J_MAX  = 0.0, 1.5


class LensingDataset(Dataset):
    def __init__(self, data_dir="data/"):
        params_raw = np.load(os.path.join(data_dir, "params.npy"))
        profiles   = np.load(os.path.join(data_dir, "profiles.npy"))
        b_grid     = np.load(os.path.join(data_dir, "b_grid.npy"))

        param_min   = np.array([R0_MIN, J_MIN], dtype=np.float32)
        param_max   = np.array([R0_MAX, J_MAX], dtype=np.float32)
        params_norm = (params_raw - param_min) / (param_max - param_min)
        b_norm      = (b_grid - b_grid.min()) / (b_grid.max() - b_grid.min())

        self.params_norm = torch.from_numpy(params_norm)
        self.profiles    = torch.from_numpy(profiles)
        self.b_norm      = torch.from_numpy(b_norm)
        self.b_grid      = torch.from_numpy(b_grid)
        self.n_grid      = len(b_grid)

    def __len__(self):
        return len(self.params_norm)

    def __getitem__(self, idx):
        p = self.params_norm[idx]
        I = self.profiles[idx]

        N         = self.n_grid
        fno_input = torch.cat([
            p.unsqueeze(0).expand(N, -1),
            self.b_norm.unsqueeze(-1),
        ], dim=-1)

        return fno_input, p, I


def make_loaders(data_dir="data/", train_frac=0.80, val_frac=0.10,
                 batch_size=32, num_workers=0, seed=42):
    dataset = LensingDataset(data_dir)
    n       = len(dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"Dataset: {n} samples  →  train={n_train}  val={n_val}  test={n_test}")
    return train_loader, val_loader, test_loader
