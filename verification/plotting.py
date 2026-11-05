"""Plotting helpers for solved Helmholtz fields."""
import numpy as np
import matplotlib.pyplot as plt


def plot_field(Ez, eps=None, solver=None, title="", axes=None):
    """
    Plot a solved field: Re(Ez) (the wave) and |Ez|^2 (the intensity).

    Ez     : (nx, ny) complex field (from solver.solve(), or a network prediction)
    eps    : optional (nx, ny) permittivity map; draws the silicon outline
    solver : optional HelmholtzSolver2D; adds micron axes, PML edges, the source
             and measurement lines, and the port powers in the title
    title  : optional label for the plot
    axes   : optional pair of matplotlib axes; if None, a new figure is created

    Returns the pair of axes. Call plt.show() yourself.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    nx, ny = Ez.shape
    dx = solver.dx if solver is not None else 1.0
    unit = "um" if solver is not None else "pixels"
    extent = [0, nx * dx, 0, ny * dx]                  # x horizontal, y vertical
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx

    re = Ez.real.T                                      # imshow expects (ny, nx)
    inten = (np.abs(Ez) ** 2).T
    vr = np.percentile(np.abs(re), 99.5)
    vi = np.percentile(inten, 99.5)
    panels = [(re, "RdBu_r", -vr, vr), (inten, "inferno", 0.0, vi)]

    for ax, (data, cmap, lo, hi) in zip(axes, panels):
        ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        if eps is not None and eps.max() > eps.min():
            ax.contour(xs, ys, eps.T, levels=[0.5 * (eps.min() + eps.max())],
                       colors="k", linewidths=0.7)
        if solver is not None:
            for xb in (solver.npml * dx, (nx - solver.npml) * dx):
                ax.axvline(xb, color="gray", ls=":", lw=0.8)
            for yb in (solver.npml * dx, (ny - solver.npml) * dx):
                ax.axhline(yb, color="gray", ls=":", lw=0.8)
            ax.axvline(solver.x_src * dx, color="lime", ls="--", lw=0.8)
            ax.axvline(solver.x_meas * dx, color="cyan", ls="--", lw=0.8)
        ax.set_xlabel(f"x ({unit})")

    head = f"{title}: " if title else ""
    first = f"{head}Re(Ez)"
    if solver is not None:
        p1, p2 = solver.measure_port_powers(Ez)
        first += f"\nP1={p1*100:.1f}%  P2={p2*100:.1f}%  total={(p1+p2)*100:.1f}%"
    axes[0].set_title(first)
    axes[1].set_title(f"{head}|Ez|$^2$")
    axes[0].set_ylabel(f"y ({unit})")
    return axes


def plot_result(result):
    """Visualizes the optimized geometry, convergence history, and Surrogate vs. FDFD fields."""
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes_grid.flatten()  # Flattens 2x2 grid into [ax0, ax1, ax2, ax3]

    # Optimized Silicon Geometry
    axes[0].imshow(result["eps_norm"].T, cmap="Greys", origin="lower")
    axes[0].set_title("Optimized Silicon Topology")

    # Optimization Trajectory
    h = result["history"]
    axes[1].plot(np.array(h["t1_surr"]) * 100, label="Port 1 (%)", lw=2)
    axes[1].plot(np.array(h["t2_surr"]) * 100, label="Port 2 (%)", lw=2)
    axes[1].plot(np.array(h["total_surr"]) * 100, "--", color="black", label="Total (%)")
    axes[1].set_xlabel("Optimization Step")
    axes[1].set_ylabel("Guided Port Power (%)")
    axes[1].set_title("Inverse Design Convergence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Surrogate Predicted Intensity
    int_surr = np.abs(result["Ez_surr"]) ** 2
    vmax = np.percentile(int_surr, 99.5)
    axes[2].imshow(int_surr.T, cmap="magma", vmin=0, vmax=vmax, origin="lower")
    axes[2].set_title(
        f"FNO Surrogate |Ez|²\nP1={result['t1_surr']*100:.1f}%, P2={result['t2_surr']*100:.1f}%"
    )
    eps_grid = result["eps_norm"]
    axes[2].contour(eps_grid.T, levels=[0.5], colors=["cyan"], linewidths=[1.0], alpha=0.5)

    # 3. True FDFD Verified Intensity
    if "Ez_true" in result:
        int_true = np.abs(result["Ez_true"]) ** 2
        vmax_t = np.percentile(int_true, 99.5)
        axes[3].imshow(int_true.T, cmap="magma", vmin=0, vmax=vmax_t, origin="lower")
        axes[3].set_title(
            f"FDFD Ground-Truth |Ez|²\nP1={result['t1_true']*100:.1f}%, P2={result['t2_true']*100:.1f}%"
        )
        axes[3].contour(eps_grid.T, levels=[0.5], colors=["cyan"], linewidths=[1.0], alpha=0.5)

    for ax in [axes[0], axes[2], axes[3]]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()