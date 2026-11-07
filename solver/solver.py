import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib import pyplot as plt

from solver.arguments import get_args


class HelmholtzSolver2D:
    """
    2D Finite-Difference Frequency-Domain (FDFD) Scalar Helmholtz Solver (TM polarization).
    All physical and geometric parameters are configurable via train/arguments.py.
    """
    def __init__(
        self,
        nx=128,
        ny=128,
        dx=0.04,
        wl=1.55,
        npml=16,
        pml_sigma_max=4.0,
        eps_bg=2.08,
        eps_si=7.84,
        wg_half=5,
        port_offset=18,
        design_half_y=24,
        x_in_end=44,
        x_out_start=84,
        x_src=30,
        x_meas=100,
        src_amp=100.0,
        mode_waist_ratio=0.75,
    ):
        self.nx, self.ny = nx, ny
        self.dx = dx
        self.wl = wl
        self.k0 = 2 * np.pi / wl
        self.npml = npml
        self.pml_sigma_max = pml_sigma_max
        self.eps_bg = eps_bg
        self.eps_si = eps_si

        self.wg_half = wg_half
        self.port_offset = port_offset
        self.design_half_y = design_half_y
        self.x_in_end = x_in_end
        self.x_out_start = x_out_start
        self.x_src = x_src
        self.x_meas = x_meas
        self.src_amp = src_amp
        self.mode_waist_ratio = mode_waist_ratio

        self.L = self._build_laplacian_with_pml()
        self._setup_geometry_and_reference()

    @classmethod
    def from_args(cls, args):
        """Instantiates the solver directly from an argparse Namespace or config dict."""
        d = vars(args) if not isinstance(args, dict) else args
        return cls(
            nx=d["nx"],
            ny=d["ny"],
            dx=d["dx"],
            wl=d["wl"],
            npml=d["npml"],
            pml_sigma_max=d["pml_sigma_max"],
            eps_bg=d["eps_bg"],
            eps_si=d["eps_si"],
            wg_half=d["wg_half"],
            port_offset=d["port_offset"],
            design_half_y=d["design_half_y"],
            x_in_end=d["x_in_end"],
            x_out_start=d["x_out_start"],
            x_src=d["x_src"],
            x_meas=d["x_meas"],
            src_amp=d["src_amp"],
            mode_waist_ratio=d["mode_waist_ratio"],
        )

    def _build_laplacian_with_pml(self):
        sigma_max = self.pml_sigma_max / (self.k0 * self.dx * self.npml)

        def pml_stretch(n):
            s = np.ones(n, dtype=np.complex128)
            for i in range(self.npml):
                u = (self.npml - i) / self.npml
                sigma = sigma_max * (u ** 2)
                s[i] = 1.0 + 1j * sigma
                s[n - 1 - i] = 1.0 + 1j * sigma
            return s

        sx = pml_stretch(self.nx)
        sy = pml_stretch(self.ny)

        def d2_1d(n, s):
            inv_s = 1.0 / s
            s_half = 0.5 * (inv_s[:-1] + inv_s[1:])
            main = np.zeros(n, dtype=np.complex128)
            main[:-1] -= inv_s[:-1] * s_half
            main[1:]  -= inv_s[1:]  * s_half
            off_up = inv_s[:-1] * s_half
            off_dn = inv_s[1:]  * s_half
            return sp.diags([off_dn, main, off_up], [-1, 0, 1], shape=(n, n)) / (self.dx ** 2)

        Dxx = d2_1d(self.nx, sx)
        Dyy = d2_1d(self.ny, sy)
        return sp.kron(Dxx, sp.eye(self.ny)) + sp.kron(sp.eye(self.nx), Dyy)

    def _setup_geometry_and_reference(self):
        yc = self.ny // 2
        self.y1 = yc - self.port_offset
        self.y2 = yc + self.port_offset

        waist = self.wg_half * self.mode_waist_ratio
        y_grid = np.arange(self.ny)
        yc_f = (self.ny - 1) / 2

        g = lambda c: (lambda v: v / np.linalg.norm(v))(np.exp(-0.5 * ((y_grid - c) / waist) ** 2))

        self.mode_in, self.mode_p1, self.mode_p2 = g(yc_f), g(yc_f - self.port_offset), g(yc_f + self.port_offset)

        self.source = np.zeros((self.nx, self.ny), dtype=np.complex128)
        self.source[self.x_src, :] = self.src_amp * self.mode_in

        self.eps_template = np.full((self.nx, self.ny), self.eps_bg, dtype=np.float64)
        self.eps_template[:self.x_in_end, yc - self.wg_half : yc + self.wg_half] = self.eps_si
        self.eps_template[self.x_out_start:, self.y1 - self.wg_half : self.y1 + self.wg_half] = self.eps_si
        self.eps_template[self.x_out_start:, self.y2 - self.wg_half : self.y2 + self.wg_half] = self.eps_si

        self.design_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.design_mask[self.x_in_end:self.x_out_start, yc - self.design_half_y : yc + self.design_half_y] = True
        self.design_shape = (self.x_out_start - self.x_in_end, 2 * self.design_half_y)

        # Reference straight waveguide solve to normalize port transmission to [0, 1]
        eps_straight = np.full((self.nx, self.ny), self.eps_bg, dtype=np.float64)
        eps_straight[:, yc - self.wg_half : yc + self.wg_half] = self.eps_si
        Ez_ref = self.solve(eps_straight)
        self.ref_power = float(np.abs(np.vdot(self.mode_in, Ez_ref[self.x_meas, :])) ** 2)

    def make_eps(self, design_rho):
        eps = self.eps_template.copy()
        eps_patch = self.eps_bg + np.clip(design_rho, 0.0, 1.0) * (self.eps_si - self.eps_bg)
        eps[self.design_mask] = eps_patch.ravel()
        return eps

    def solve(self, eps_r):
        k2_eps = sp.diags((self.k0 ** 2) * eps_r.ravel(), 0)
        A = (self.L + k2_eps).tocsc()
        Ez_flat = spla.spsolve(A, self.source.ravel())
        return Ez_flat.reshape(self.nx, self.ny)

    def measure_port_powers(self, Ez):
        slice_out = Ez[self.x_meas, :]
        p1 = float(np.abs(np.vdot(self.mode_p1, slice_out)) ** 2) / self.ref_power
        p2 = float(np.abs(np.vdot(self.mode_p2, slice_out)) ** 2) / self.ref_power
        return p1, p2


if __name__ == "__main__":
    from verification.plotting import plot_field

    args = get_args()
    solver = HelmholtzSolver2D.from_args(args)

    t0 = time.perf_counter()
    eps_empty = solver.make_eps(np.zeros(solver.design_shape))
    Ez_empty = solver.solve(eps_empty)
    dt = time.perf_counter() - t0
    t1_empty, t2_empty = solver.measure_port_powers(Ez_empty)

    eps_full = solver.make_eps(np.ones(solver.design_shape))
    Ez_full = solver.solve(eps_full)
    t1_full, t2_full = solver.measure_port_powers(Ez_full)

    print(f"Single {args.nx}x{args.ny} solve time : {dt*1000:.1f} ms")
    print(f"Empty-gap baseline        : Port 1 = {t1_empty*100:.2f}%, Port 2 = {t2_empty*100:.2f}% (Total = {(t1_empty+t2_empty)*100:.2f}%)")
    print(f"Solid-Si block baseline   : Port 1 = {t1_full*100:.2f}%, Port 2 = {t2_full*100:.2f}% (Total = {(t1_full+t2_full)*100:.2f}%)")

    # Visualize the solved fields
    plot_field(Ez_empty, eps_empty, solver, title="Empty gap")
    plot_field(Ez_full, eps_full, solver, title="Solid Si block")
    plt.show()