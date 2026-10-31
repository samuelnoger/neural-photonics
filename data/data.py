import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from tqdm import tqdm

from train.arguments import get_args
from data.solver import HelmholtzSolver2D


def sample_random_design(shape, rng, args):
    """Samples a 2D design region density in [0, 1] using parameters from args."""
    nx_d, ny_d = shape
    probs = np.array([args.prob_smooth, args.prob_binary, args.prob_y_pert], dtype=np.float64)
    probs /= probs.sum()

    mode = rng.choice(["smooth", "binary", "y_perturbation"], p=probs)
    sigma = rng.uniform(args.blob_sigma_min, args.blob_sigma_max)

    raw = gaussian_filter(rng.standard_normal(shape), sigma=sigma)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

    if mode == "smooth":
        beta = rng.uniform(1.0, 4.0)
        rho = 1.0 / (1.0 + np.exp(-beta * (raw - rng.uniform(0.35, 0.65)) * 6.0))
    elif mode == "binary":
        thresh = rng.uniform(0.35, 0.65)
        rho = (raw > thresh).astype(np.float64)
        if rng.random() < 0.5:
            rho = gaussian_filter(rho, sigma=0.8)
    else:
        x = np.linspace(0, 1, nx_d)[:, None]
        y = np.arange(ny_d)[None, :] - (ny_d / 2.0)
        offset = float(args.port_offset) * x
        arm_width = rng.uniform(0.7 * args.wg_half, 1.2 * args.wg_half)
        arm1 = np.exp(-0.5 * ((y - offset) / arm_width) ** 2)
        arm2 = np.exp(-0.5 * ((y + offset) / arm_width) ** 2)
        skeleton = np.clip(arm1 + arm2, 0.0, 1.0)
        rho = np.clip(0.65 * skeleton + 0.35 * raw, 0.0, 1.0)
        if rng.random() < 0.5:
            rho = (rho > 0.5).astype(np.float64)

    if rng.random() < args.sym_prob:
        rho = 0.5 * (rho + rho[:, ::-1])

    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def _generate_single_sample(seed, args):
    """Worker function for parallel dataset generation."""
    rng = np.random.default_rng(seed)
    solver = HelmholtzSolver2D.from_args(args)
    rho = sample_random_design(solver.design_shape, rng, args)
    eps = solver.make_eps(rho)
    Ez = solver.solve(eps)

    eps_norm = ((eps - solver.eps_bg) / (solver.eps_si - solver.eps_bg)).astype(np.float32)
    field_2ch = np.stack([Ez.real.astype(np.float32), Ez.imag.astype(np.float32)], axis=0)
    return eps_norm[None, :, :], field_2ch


class PhotonicsDataset(Dataset):
    """PyTorch Dataset wrapping saved (eps_norm, Ez_2ch) tensors and solver config."""
    def __init__(self, path):
        data = torch.load(path, weights_only=False)
        self.eps = data["eps"]
        self.fields = data["fields"]
        self.field_std = data["field_std"]
        self.config = data.get("config", {})

    def __len__(self):
        return self.eps.shape[0]

    def __getitem__(self, idx):
        return self.eps[idx], self.fields[idx]


def generate_dataset(args):
    os.makedirs(args.save_dir, exist_ok=True)
    total = args.n_train + args.n_val
    print(f"Generating {total} 2D Helmholtz samples ({args.nx}x{args.ny}) across CPU cores (n_jobs={args.n_jobs})...")
    t0 = time.perf_counter()

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_generate_single_sample)(args.seed + i, args) for i in tqdm(range(total))
    )

    eps_all = np.stack([r[0] for r in results], axis=0)
    fields_all = np.stack([r[1] for r in results], axis=0)

    field_std = float(np.std(fields_all[:args.n_train]))
    fields_all = fields_all / field_std

    train_path = os.path.join(args.save_dir, "train.pt")
    val_path = os.path.join(args.save_dir, "val.pt")

    config_dict = vars(args)

    torch.save({
        "eps": torch.from_numpy(eps_all[:args.n_train]),
        "fields": torch.from_numpy(fields_all[:args.n_train]),
        "field_std": field_std,
        "config": config_dict,
    }, train_path)

    torch.save({
        "eps": torch.from_numpy(eps_all[args.n_train:]),
        "fields": torch.from_numpy(fields_all[args.n_train:]),
        "field_std": field_std,
        "config": config_dict,
    }, val_path)

    elapsed = time.perf_counter() - t0
    print(f"Saved {args.n_train} train and {args.n_val} val samples to '{args.save_dir}/' in {elapsed:.1f}s (field_std={field_std:.4f}).")


if __name__ == "__main__":
    args = get_args()
    generate_dataset(args)