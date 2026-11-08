import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, iterate_structure
from solver.solver import HelmholtzSolver2D

def metrics(solver, rho):
    Ez = solver.solve(solver.make_eps(rho))
    t1, t2 = solver.measure_port_powers(Ez)
    return t1, t2, t1 + t2, t1 / (t1 + t2)

def bias(rho_bin, n):                       # n>0 dilate (over-etch of oxide), n<0 erode
    if n == 0:
        return rho_bin
    st = iterate_structure(generate_binary_structure(2, 1), abs(n))
    p = abs(n) + 1
    padded = np.pad(rho_bin > 0.5, p, mode="edge")
    out = (binary_dilation if n > 0 else binary_erosion)(padded, structure=st)
    return out[p:-p, p:-p].astype(np.float32)

def robustness(fitter, rho_bin, target_ratio, biases=(-2, -1, 0, 1, 2), wls=(1.53, 1.55, 1.57), verbose=True):
    """Evaluates binary topology across wavelength and etch-bias perturbations and returns structured arrays."""
    biases = list(biases)
    wls = list(wls)
    tot_grid = np.zeros((len(wls), len(biases)), dtype=np.float32)
    ratio_grid = np.zeros((len(wls), len(biases)), dtype=np.float32)

    if verbose:
        print(f"target ratio {target_ratio:.2f}")

    for i, wl in enumerate(wls):
        cfg = dict(fitter.config)
        cfg["wl"] = wl
        solver = HelmholtzSolver2D.from_args(cfg)
        for j, n in enumerate(biases):
            t1, t2, tot, r = metrics(solver, bias(rho_bin, n))
            tot_grid[i, j] = tot
            ratio_grid[i, j] = r
            if verbose:
                print(f"wl={wl:.2f}  bias={n:+d}px  total={tot*100:5.1f}%  ratio={r:.3f}")

    return {
        "target_ratio": target_ratio,
        "biases": np.array(biases),
        "wls": np.array(wls),
        "total": tot_grid,     # shape: (len(wls), len(biases))
        "ratio": ratio_grid,   # shape: (len(wls), len(biases))
    }

def edge_shift_px(fitter, z, beta, delta, symmetric=False):
    with torch.no_grad():
        nom  = (fitter._project_density(z, beta, symmetric, 0.0)    > 0.5).float()
        plus = (fitter._project_density(z, beta, symmetric, +delta) > 0.5).float()
        minus= (fitter._project_density(z, beta, symmetric, -delta) > 0.5).float()
        # edge length ~ count of boundary pixels of the nominal mask
        edge = ((nom[1:, :] != nom[:-1, :]).sum() + (nom[:, 1:] != nom[:, :-1]).sum()).item()
        d_plus  = (plus.sum() - nom.sum()).item() / max(edge, 1)
        d_minus = (nom.sum() - minus.sum()).item() / max(edge, 1)
    return d_plus, d_minus