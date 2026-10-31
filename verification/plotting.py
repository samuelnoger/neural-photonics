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