import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from design.geometry import build_geometry


class HelmholtzSolver2D:
    """
    Universal 2D FDFD Scalar Helmholtz Solver (TM polarization) with:
      - Wavelength-dependent effective permittivity eps_si(wl)
      - Exact 1D transverse waveguide eigenmode solver for fundamental TM00 launch & port projection
    """
    def __init__(self, geometry, dx=0.02, wl=1.55, npml=32, pml_sigma_max=4.0, dispersive=True):
        self.geometry = geometry
        self.nx, self.ny = geometry.nx, geometry.ny
        self.dx = dx
        self.wl = wl
        self.k0 = 2 * np.pi / wl
        self.npml = npml
        self.pml_sigma_max = pml_sigma_max
        self.eps_bg = geometry.eps_bg

        # Wavelength-dependent effective permittivity for 220nm SOI slab (7.84 at 1.55 um, 8.35 at 1.31 um)
        if dispersive:
            self.eps_si = float(geometry.eps_si + 2.125 * (1.55 - wl))
        else:
            self.eps_si = float(geometry.eps_si)

        self.x_src = geometry.x_src
        self.x_meas = geometry.x_meas
        self.x_in_end = geometry.x_in_end
        self.x_out_start = geometry.x_out_start
        self.design_half_y = geometry.design_half_y
        self.design_shape = geometry.design_shape

        # 1. Solve exact 1D fundamental (00) waveguide eigenmodes at this wavelength
        self.mode_in, self.beta_in = self._solve_1d_eigenmode(self.geometry.yc)
        self.mode_ports = []
        for py in self.geometry.port_ys:
            m_port, _ = self._solve_1d_eigenmode(py)
            self.mode_ports.append(m_port)

        # Store stacked PyTorch port modes on geometry.device for fast vectorized loss evaluation
        modes_np = np.stack(self.mode_ports, axis=0).astype(np.float32)
        self.mode_ports_torch = torch.from_numpy(modes_np).to(self.geometry.device)

        # 2. Build input line source using the exact fundamental eigenmode
        self.source = np.zeros((self.nx, self.ny), dtype=np.complex128)
        self.source[self.x_src, :] = self.geometry.src_amp * self.mode_in

        # 3. Build 2D PML Laplacian
        self.L = self._build_laplacian_with_pml()

        # 4. Calibrate reference power at this wavelength using a straight waveguide with eps_si(wl)
        eps_straight_wl = np.full((self.nx, self.ny), self.eps_bg, dtype=np.float64)
        yc, wh = self.geometry.yc, self.geometry.wg_half
        eps_straight_wl[:, yc - wh : yc + wh] = self.eps_si
        Ez_ref = self.solve(eps_straight_wl)
        self.ref_power = float(np.abs(np.vdot(self.mode_in, Ez_ref[self.x_meas, :])) ** 2)

    def _solve_1d_eigenmode(self, center_y):
        """
        Solves the 1D transverse Helmholtz eigenvalue problem:
            (d^2/dy^2 + k0^2 * eps_1d(y)) * psi(y) = beta^2 * psi(y)
        for a single waveguide centered at `center_y`, returning the exact fundamental TM00 mode.
        """
        eps_1d = np.full(self.ny, self.eps_bg, dtype=np.float64)
        wh = self.geometry.wg_half
        eps_1d[center_y - wh : center_y + wh] = self.eps_si

        main_diag = -2.0 / (self.dx ** 2) + (self.k0 ** 2) * eps_1d
        off_diag = np.full(self.ny - 1, 1.0 / (self.dx ** 2), dtype=np.float64)

        # eigh_tridiagonal is O(ny) and returns real eigenvalues in ascending order
        eigvals, eigvecs = la.eigh_tridiagonal(main_diag, off_diag)

        # The fundamental (00) bound mode has the largest eigenvalue beta^2
        psi_0 = eigvecs[:, -1].copy()
        beta_0 = float(np.sqrt(max(eigvals[-1], 0.0)))

        # Fix sign convention so the mode peak at center_y is strictly positive
        if psi_0[center_y] < 0:
            psi_0 = -psi_0
        psi_0 /= np.linalg.norm(psi_0)
        return psi_0, beta_0

    @classmethod
    def from_args(cls, args, geometry=None, wl_override=None):
        d = vars(args) if not isinstance(args, dict) else args
        geom = geometry if geometry is not None else build_geometry(d)
        wl = wl_override if wl_override is not None else d["wl"]
        return cls(
            geometry=geom,
            dx=d["dx"],
            wl=wl,
            npml=d["npml"],
            pml_sigma_max=d["pml_sigma_max"],
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

    def make_eps(self, design_rho):
        """Builds the 2D permittivity grid at this solver's wavelength using self.eps_si."""
        rho_full = np.asarray(self.geometry.base_eps_norm.cpu().numpy()).squeeze()
        eps = self.eps_bg + rho_full * (self.eps_si - self.eps_bg)
        eps_patch = self.eps_bg + np.clip(design_rho, 0.0, 1.0) * (self.eps_si - self.eps_bg)
        eps[self.geometry.design_mask] = eps_patch.ravel()
        return eps

    def solve(self, eps_r):
        k2_eps = sp.diags((self.k0 ** 2) * eps_r.ravel(), 0)
        A = (self.L + k2_eps).tocsc()
        Ez_flat = spla.spsolve(A, self.source.ravel())
        return Ez_flat.reshape(self.nx, self.ny)

    def measure_port_powers(self, Ez):
        """Returns a list of normalized guided powers [P_1, ..., P_N] using exact wavelength eigenmodes."""
        slice_out = Ez[self.x_meas, :]
        return [
            float(np.abs(np.vdot(m, slice_out)) ** 2) / self.ref_power
            for m in self.mode_ports
        ]