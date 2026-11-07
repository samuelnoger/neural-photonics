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