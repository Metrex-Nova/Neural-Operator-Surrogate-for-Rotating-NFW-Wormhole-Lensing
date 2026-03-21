import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.ray_tracing import compute_intensity_profile

B_MIN  = 0.05
B_MAX  = 13.0
N_GRID = 150
B_GRID = np.linspace(B_MIN, B_MAX, N_GRID)

R0_MIN, R0_MAX = 0.3, 2.5
J_MIN,  J_MAX  = 0.0, 1.5


def _worker(args):
    i, r0, J, b_grid = args
    b_grid = np.asarray(b_grid, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        I, b_ph = compute_intensity_profile(r0, J, b_grid)
    except Exception:
        I, b_ph = np.zeros(len(b_grid), dtype=np.float32), 0.0
    return i, I.astype(np.float32), float(b_ph), time.perf_counter() - t0


def generate_dataset(n_samples=2000, seed=42, out_dir="data/", n_workers=4, verbose=True):
    rng    = np.random.default_rng(seed)
    r0_arr = rng.uniform(R0_MIN, R0_MAX, n_samples)
    J_arr  = rng.uniform(J_MIN,  J_MAX,  n_samples)
    params = np.stack([r0_arr, J_arr], axis=1).astype(np.float32)

    profiles = np.zeros((n_samples, N_GRID), dtype=np.float32)
    b_ph_arr = np.zeros(n_samples, dtype=np.float32)

    b_grid_list = B_GRID.tolist()  # plain list is picklable on Windows
    jobs  = [(i, float(r0_arr[i]), float(J_arr[i]), b_grid_list) for i in range(n_samples)]
    n_cpu = n_workers if n_workers > 0 else mp.cpu_count()

    if verbose:
        print(f"Generating {n_samples} samples  "
              f"(r0 ∈ [{R0_MIN},{R0_MAX}], J ∈ [{J_MIN},{J_MAX}])")
        print(f"b-grid: {N_GRID} pts, [{B_MIN}, {B_MAX}] kpc  |  workers: {n_cpu}")
        print("-" * 60)

    t_start = time.perf_counter()
    times   = []
    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  {postfix}"

    with mp.Pool(processes=n_cpu) as pool:
        with tqdm(total=n_samples, desc="Generating", unit="sample",
                  bar_format=bar_fmt, dynamic_ncols=True) as pbar:
            for result in pool.imap_unordered(_worker, jobs, chunksize=4):
                i, I, b_ph, elapsed = result
                profiles[i] = I
                b_ph_arr[i] = b_ph
                times.append(elapsed)
                pbar.set_postfix(r0=f"{r0_arr[i]:.2f}", J=f"{J_arr[i]:.2f}",
                                 b_ph=f"{b_ph:.2f}", avg_s=f"{np.mean(times):.2f}",
                                 refresh=False)
                pbar.update(1)

    avg_t = float(np.mean(times))
    if verbose:
        print("-" * 60)
        print(f"Done.  Avg solver time: {avg_t:.4f} s/sample  "
              f"| Wall time: {time.perf_counter()-t_start:.1f} s")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "params.npy"),      params)
    np.save(os.path.join(out_dir, "profiles.npy"),    profiles)
    np.save(os.path.join(out_dir, "b_grid.npy"),      B_GRID.astype(np.float32))
    np.save(os.path.join(out_dir, "b_ph.npy"),        b_ph_arr)
    np.save(os.path.join(out_dir, "solver_time.npy"), np.array([avg_t], dtype=np.float32))

    if verbose:
        print(f"Saved → '{out_dir}':  params{params.shape}  profiles{profiles.shape}")

    return params, profiles, B_GRID.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out_dir",   type=str, default="data/")
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()
    generate_dataset(args.n_samples, args.seed, args.out_dir, args.n_workers)
