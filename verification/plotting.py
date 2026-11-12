"""Plotting helpers for solved Helmholtz fields."""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from solver.solver import HelmholtzSolver2D

def plot_result(result):
    """Visualizes the optimized silicon topology, FDFD wavefield intensity, and convergence for 1xN splitters."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Optimized Silicon Topology
    eps_grid = result["eps_norm"]
    axes[0].imshow(eps_grid.T, cmap="Greys", origin="lower")
    axes[0].set_title("Optimized Silicon Topology")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Exact FDFD Field Intensity |Ez|^2
    Ez = result["Ez_fit"]
    
    # Automatically detect powers across any number of ports (1x2, 1x3, 1xN)
    if "powers_fit" in result:
        powers = np.array(result["powers_fit"])
    else:
        powers = np.array([result.get("t1_fit", 0.0), result.get("t2_fit", 0.0)])
    
    total_power = powers.sum()

    intensity = np.abs(Ez) ** 2
    vmax = np.percentile(intensity, 99.5)
    axes[1].imshow(intensity.T, cmap="magma", vmin=0, vmax=vmax, origin="lower")
    axes[1].contour(eps_grid.T, levels=[0.5], colors=["cyan"], linewidths=[1.0], alpha=0.6)
    
    # Dynamically build title string for N ports
    ports_title_str = "  ".join(f"P{i+1}={p*100:.1f}%" for i, p in enumerate(powers))
    axes[1].set_title(f"FDFD |Ez|²\n{ports_title_str} (Tot={total_power*100:.1f}%)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Optimization Convergence Trajectory
    h = result["history"]
    if "powers" in h:
        powers_hist = np.array(h["powers"]) * 100  # shape: (steps, n_ports)
        for i in range(powers_hist.shape[1]):
            axes[2].plot(powers_hist[:, i], label=f"Port {i+1} (%)", lw=2)
        tot_hist = powers_hist.sum(axis=1)
    else:
        p1_hist = np.array(h.get("t1", h.get("t1_fit", [0]))) * 100
        p2_hist = np.array(h.get("t2", h.get("t2_fit", [0]))) * 100
        axes[2].plot(p1_hist, label="Port 1 (%)", lw=2)
        axes[2].plot(p2_hist, label="Port 2 (%)", lw=2)
        tot_hist = np.array(h.get("total", h.get("total_fit", p1_hist + p2_hist))) * 100

    axes[2].plot(tot_hist, "--", color="black", label="Total (%)")
    axes[2].set_xlabel("Optimization Step")
    axes[2].set_ylabel("Guided Port Power (%)")
    axes[2].set_title("Adjoint Inverse Design Convergence")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_robustness(rob_data, fitter=None, rho_bin=None):
    """
    Plots eroded/nominal/dilated masks alongside transmission & split-ratio stability across wavelengths and etch biases.
    """
    from verification.stability import bias as apply_bias

    biases = rob_data["biases"]
    wls = rob_data["wls"]
    tot_grid = rob_data["total"] * 100.0
    ratio_grid = rob_data["ratio"] * 100.0
    target_r = rob_data["target_ratio"] * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Total Guided Transmission vs. Etch Bias across Wavelengths
    for i, wl in enumerate(wls):
        axes[0].plot(biases, tot_grid[i], marker="o", lw=2, label=f"λ = {wl:.2f} μm")
    axes[0].axvline(0, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("Etch Bias (pixels, 1 px = 40 nm)")
    axes[0].set_ylabel("Total Transmission P1 + P2 (%)")
    axes[0].set_title("Throughput vs. Fabrication Etch Bias")
    axes[0].set_xticks(biases)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Split Ratio P1/(P1+P2) vs. Etch Bias across Wavelengths
    for i, wl in enumerate(wls):
        axes[1].plot(biases, ratio_grid[i], marker="s", lw=1.5, label=f"λ = {wl:.2f} μm")
    axes[1].axhline(target_r, color="black", ls="--", lw=1.2, label=f"Target ({target_r:.0f}%)")
    axes[1].set_xlabel("Etch Bias (pixels, 1 px = 40 nm)")
    axes[1].set_ylabel("Measured Split Ratio P1 / Total (%)")
    axes[1].set_title("Splitting Ratio Stability")
    axes[1].set_xticks(biases)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Boundary Contour Overlay for Eroded (-1px), Nominal (0px), Dilated (+1px)
    if rho_bin is not None:
        axes[2].imshow(rho_bin.T, cmap="Greys", origin="lower", alpha=0.25)
        for b_px, col, lbl in [(-1, "crimson", "-1 px (Eroded)"), (0, "black", "0 px (Nominal)"), (1, "dodgerblue", "+1 px (Dilated)")]:
            m = apply_bias(rho_bin, b_px)
            cs = axes[2].contour(m.T, levels=[0.5], colors=[col], linewidths=1.5)
            axes[2].plot([], [], color=col, lw=1.5, label=lbl)
        axes[2].set_title("Design Contours under ±1 px Etch Bias")
        axes[2].legend(loc="upper left", fontsize=8)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()

def plot_wdm_result(result, fitter):
    """Visualizes optimized WDM topology and its wavelength-selective field intensities."""
    wls = fitter.geometry.wls
    n_wls = len(wls)
    
    fig, axes = plt.subplots(1, n_wls + 1, figsize=(5 * (n_wls + 1), 4.5))
    
    # Optimized Silicon Topology
    eps_grid = result["eps_norm"]
    axes[0].imshow(eps_grid.T, cmap="Greys", origin="lower")
    axes[0].set_title("Optimized WDM Topology")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Field Intensity for each target wavelength
    # result["Ez_all"] and result["powers_all"] store the fields for each wavelength
    for idx, (wl, Ez_w, p_w) in enumerate(zip(wls, result["Ez_all"], result["powers_all"])):
        ax = axes[idx + 1]
        intensity = np.abs(Ez_w) ** 2
        vmax = np.percentile(intensity, 99.5)
        
        ax.imshow(intensity.T, cmap="magma", vmin=0, vmax=vmax, origin="lower")
        ax.contour(eps_grid.T, levels=[0.5], colors=["cyan"], linewidths=[1.0], alpha=0.6)
        
        ports_str = "  ".join(f"P{i+1}={p*100:.1f}%" for i, p in enumerate(p_w))
        ax.set_title(f"λ = {wl:.2f} μm\n{ports_str}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_spectrum(res,fitter, wl_min=1.20, wl_max=1.70, n_pts=30, in_db=False):
        """
        Sweeps wavelength across [wl_min, wl_max] on the final binary design `res["rho"]`
        and plots the continuous transmission spectrum for each output port.
        """
        wls_sweep = np.linspace(wl_min, wl_max, n_pts)
        port_traces = []
        for w in tqdm(wls_sweep, desc="Sweeping Spectrum"):
            s_w = HelmholtzSolver2D.from_args(fitter.config, geometry=fitter.geometry, wl_override=float(w))
            _, p_w = fitter.evaluate(res["rho"], solver=s_w)
            port_traces.append(p_w)

        port_traces = np.array(port_traces)  # shape: (n_pts, n_ports)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        colors = ["#2b5c8f", "#d95f02", "#7570b3"]

        for p_idx in range(port_traces.shape[1]):
            y_val = 10.0 * np.log10(np.clip(port_traces[:, p_idx], 1e-4, 1.0)) if in_db else port_traces[:, p_idx] * 100.0
            ax.plot(wls_sweep, y_val, "-o", markersize=4, lw=2, color=colors[p_idx % len(colors)], label=f"Port {p_idx + 1}")

        for w_design in fitter.wls:
            ax.axvline(w_design, color="gray", ls="--", alpha=0.6, label=f"Design λ = {w_design:.2f} µm")

        ax.set_xlabel("Wavelength λ (µm)")
        ax.set_ylabel("Transmission (dB)" if in_db else "Guided Transmission (%)")
        ax.set_title(f"Broadband Spectral Response [{fitter.device_type}]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()