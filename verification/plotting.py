"""Plotting helpers for solved Helmholtz fields."""
import numpy as np
import matplotlib.pyplot as plt


def plot_result(result):
    """Visualizes the optimized silicon topology, FDFD wavefield intensity, and convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Optimized Silicon Geometry
    eps_grid = result["eps_norm"]
    axes[0].imshow(eps_grid.T, cmap="Greys", origin="lower")
    axes[0].set_title("Optimized Silicon Topology")

    # Exact FDFD Field Intensity |Ez|^2
    Ez = result["Ez_fit"]
    t1 = result["t1_fit"]
    t2 = result["t2_fit"]

    intensity = np.abs(Ez) ** 2
    vmax = np.percentile(intensity, 99.5)
    axes[1].imshow(intensity.T, cmap="magma", vmin=0, vmax=vmax, origin="lower")
    axes[1].contour(eps_grid.T, levels=[0.5], colors=["cyan"], linewidths=[1.0], alpha=0.6)
    axes[1].set_title(f"FDFD |Ez|²\nP1={t1*100:.1f}%, P2={t2*100:.1f}% (Tot={(t1+t2)*100:.1f}%)")

    # Optimization Convergence Trajectory
    h = result["history"]
    p1_hist = np.array(h.get("t1", h.get("t1_fit"))) * 100
    p2_hist = np.array(h.get("t2", h.get("t2_fit"))) * 100
    tot_hist = np.array(h.get("total", h.get("total_fit"))) * 100

    axes[2].plot(p1_hist, label="Port 1 (%)", lw=2)
    axes[2].plot(p2_hist, label="Port 2 (%)", lw=2)
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