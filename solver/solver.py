import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from design.geometry import build_geometry


class HelmholtzSolver2D:
    """
    Universal 2D FDFD Scalar Helmholtz Solver (TM polarization).
    Decoupled from specific device geometries via DeviceGeometry.
    """
    def __init__(self, geometry, dx=0.02, wl=1.55, npml=32, pml_sigma_max=4.0):
        self.geometry = geometry
        self.nx, self.ny = geometry.nx, geometry.ny
        self.dx = dx
        self.wl = wl
        self.k0 = 2 * np.pi / wl
        self.npml = npml
        self.pml_sigma_max = pml_sigma_max
        self.eps_bg = geometry.eps_bg
        self.eps_si = geometry.eps_si

        # Expose geometry attributes for backwards compatibility with adjoint.py & plotting.py
        self.source = geometry.source
        self.x_src = geometry.x_src
        self.x_meas = geometry.x_meas
        self.x_in_end = geometry.x_in_end
        self.x_out_start = geometry.x_out_start
        self.design_half_y = geometry.design_half_y
        self.design_shape = geometry.design_shape

        self.L = self._build_laplacian_with_pml()

        # Calibrate reference power at this wavelength using a straight waveguide
        Ez_ref = self.solve(self.geometry.eps_straight)
        self.ref_power = float(np.abs(np.vdot(self.geometry.mode_in, Ez_ref[self.x_meas, :])) ** 2)

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
        return self.geometry.make_eps(design_rho)

    def solve(self, eps_r):
        k2_eps = sp.diags((self.k0 ** 2) * eps_r.ravel(), 0)
        A = (self.L + k2_eps).tocsc()
        Ez_flat = spla.spsolve(A, self.source.ravel())
        return Ez_flat.reshape(self.nx, self.ny)

    def measure_port_powers(self, Ez):
        """Returns a list of normalized guided powers [P_1, ..., P_N] across all N output ports."""
        slice_out = Ez[self.x_meas, :]
        return [
            float(np.abs(np.vdot(m, slice_out)) ** 2) / self.ref_power
            for m in self.geometry.mode_ports
        ]