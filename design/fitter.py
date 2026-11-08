import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from solver.arguments import get_args
from solver.solver import HelmholtzSolver2D
from design.losses import SplitterLoss
from design.adjoint import solve_fdfd_differentiable


class PhotonicFitter:
    """
    End-to-end photonic inverse design fitter using Exact FDFD Adjoint Optimization.
    """
    def __init__(self, device="auto", smooth_sigma=3.0):
        self.device = self._resolve_device(device)
        self.config = vars(get_args([]))

        # Initialize ground-truth 2D Helmholtz FDFD solver
        self.solver = HelmholtzSolver2D.from_args(self.config)
        self.field_std = 1.0

        # Precompute normalized waveguide base template on device
        eps_empty = self.solver.make_eps(np.zeros(self.solver.design_shape, dtype=np.float32))
        eps_norm_empty = (eps_empty - self.solver.eps_bg) / (self.solver.eps_si - self.solver.eps_bg)
        self.base_eps_norm = torch.from_numpy(eps_norm_empty).float().to(self.device).unsqueeze(0).unsqueeze(0)

        # Design region slice coordinates
        self.x0, self.x1 = self.solver.x_in_end, self.solver.x_out_start
        yc = self.solver.ny // 2
        self.y0, self.y1 = yc - self.solver.design_half_y, yc + self.solver.design_half_y

        # Precompute circular Gaussian smoothing kernel for manufacturable feature sizes
        k_rad = int(np.ceil(2.5 * smooth_sigma))
        coords = torch.arange(-k_rad, k_rad + 1, device=self.device, dtype=torch.float32)
        kx, ky = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-0.5 * (kx ** 2 + ky ** 2) / (smooth_sigma ** 2))
        kernel = kernel / kernel.sum()
        self.smooth_kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.k_pad = k_rad

        nx_d, ny_d = self.solver.design_shape
        c  = self.solver.design_half_y            # local axis index
        wg = self.solver.wg_half
        po = self.solver.port_offset
        yy = torch.arange(ny_d, device=self.device)
        band = lambda lo, hi: ((yy >= lo) & (yy < hi)).float()

        left  = band(c - wg, c + wg)                                              # input waveguide
        right = band(c - po - wg, c - po + wg) + band(c + po - wg, c + po + wg)   # two output waveguides

        n_pin = self.k_pad + 2                    # >= filter radius, so the outer pinned columns stay clean
        self.pin = torch.zeros(nx_d, ny_d, device=self.device)
        pin_z    = torch.zeros(nx_d, ny_d, device=self.device)
        self.pin[:n_pin, :]  = 1;  pin_z[:n_pin, :]  = (2 * left  - 1) * 2.0      # +2 inside silicon, -2 outside
        self.pin[-n_pin:, :] = 1;  pin_z[-n_pin:, :] = (2 * right - 1) * 2.0
        self.pin_z = pin_z

        self.criterion = SplitterLoss(self.solver, self.field_std, self.device)

    @staticmethod
    def _resolve_device(device_arg):
        if device_arg != "auto":
            return torch.device(device_arg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def evaluate(self, rho_np):
        Ez = self.solver.solve(self.solver.make_eps(rho_np))
        t1, t2 = self.solver.measure_port_powers(Ez)
        return Ez, t1, t2

    def _project_density(self, z, beta, symmetric=False, shift=0.0):
        if symmetric:
            z = 0.5 * (z + torch.flip(z, dims=[-1]))
        z = self.pin * self.pin_z + (1 - self.pin) * z          # pinned columns are fixed
        z_4d = F.pad(z.unsqueeze(0).unsqueeze(0), (self.k_pad,) * 4, mode="replicate")
        z_smooth = F.conv2d(z_4d, self.smooth_kernel).squeeze()
        return torch.sigmoid(beta * (z_smooth + shift * (1 - self.pin))) 

    def _assemble_eps_norm(self, rho):
        """Inserts the (40, 48) differentiable rho patch into the (1, 1, 128, 128) chip grid."""
        eps = self.base_eps_norm.clone()
        eps[0, 0, self.x0 : self.x1, self.y0 : self.y1] = rho
        return eps

    def _init_latent(self, init_mode="y_branch", fill_val=0.5):
        nx_d, ny_d = self.solver.design_shape
        x = torch.linspace(0, 1, nx_d, device=self.device)[:, None]
        y = torch.arange(ny_d, device=self.device, dtype=torch.float32)[None, :] - ((ny_d - 1) / 2.0)

        if init_mode == "y_branch":
            offset = float(self.solver.port_offset) * x
            w = float(self.solver.wg_half)
            skel = torch.exp(-0.5 * ((y - offset) / w) ** 2) + torch.exp(-0.5 * ((y + offset) / w) ** 2)
            rho0 = torch.clamp(skel, 0.05, 0.95)
        elif init_mode == "taper":
            half_width = float(self.solver.wg_half) + x * float(self.solver.port_offset + self.solver.wg_half)
            rho0 = torch.sigmoid((half_width - torch.abs(y)) / 2.0)
            rho0 = torch.clamp(rho0, 0.05, 0.95)
        elif init_mode == "empty":
            rho0 = torch.full((nx_d, ny_d), 0.10, device=self.device)
        elif init_mode == "filled":
            rho0 = torch.full((nx_d, ny_d), 0.90, device=self.device)
        else:
            val = float(np.clip(fill_val, 0.05, 0.95))
            rho0 = torch.full((nx_d, ny_d), val, device=self.device)

        return torch.logit(rho0)

    def fit(
        self,
        target_ratio=0.5,
        total_target=0.98,
        steps=200,
        lr=0.05,
        init_mode="y_branch",
        symmetric=None,
        beta_start=1.5,
        beta_end=12.0,
        robust_delta=0.0,     # latent-space threshold shift; 0 disables robust optimization
        robust_start=0.5,
    ):
        """
        Optimizes the 40x48 design region using exact FDFD adjoint gradients.
        """
        if symmetric is None:
            symmetric = abs(target_ratio - 0.5) < 1e-3

        z = self._init_latent(init_mode).requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)

        history = {"loss": [], "t1": [], "t2": [], "total": [], "t1_surr": [], "t2_surr": [], "total_surr": []}
        pbar = tqdm(
            range(1, steps + 1),
            desc=f"Fitting [FDFD Adjoint] ({target_ratio*100:.0f}/{(1-target_ratio)*100:.0f})",
        )

        t0 = time.perf_counter()
        for step in pbar:
            frac = min((step - 1) / max(0.75 * steps, 1), 1.0)
            beta = beta_start + (beta_end - beta_start) * (frac ** 1.5)
            bin_weight = 0.01 + 0.08 * (frac ** 2)

            optimizer.zero_grad()

            use_robust = robust_delta > 0 and step > robust_start * steps
            shifts = (0.0, robust_delta, -robust_delta) if use_robust else (0.0,)

            loss = 0.0
            for shift in shifts:
                rho_s = self._project_density(z, beta=beta, symmetric=symmetric, shift=shift)
                field = solve_fdfd_differentiable(self._assemble_eps_norm(rho_s), self.solver, self.field_std)
                l, t1_s, t2_s = self.criterion(field, rho_s, target_ratio=target_ratio,
                                               total_target=total_target, bin_weight=bin_weight)
                loss = loss + l / len(shifts)
                if shift == 0.0:
                    t1, t2 = t1_s, t2_s

            loss.backward()
            optimizer.step()
            scheduler.step()

            t1_v, t2_v = t1.item(), t2.item()
            history["loss"].append(loss.item())
            history["t1"].append(t1_v)
            history["t2"].append(t2_v)
            history["total"].append(t1_v + t2_v)
            # Keep *_surr aliases so existing plotting.py works without any edits
            history["t1_surr"].append(t1_v)
            history["t2_surr"].append(t2_v)
            history["total_surr"].append(t1_v + t2_v)

            pbar.set_postfix(
                T1=f"{t1_v*100:.1f}%",
                T2=f"{t2_v*100:.1f}%",
                Tot=f"{(t1_v+t2_v)*100:.1f}%",
                beta=f"{beta:.1f}",
            )

        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            rho_gray = self._project_density(z, beta=beta_end, symmetric=symmetric)
            rho_gray_np = rho_gray.cpu().numpy()
            rho_bin_np = (rho_gray_np > 0.5).astype(np.float32)

        Ez_bin, t1_b, t2_b = self.evaluate(rho_bin_np)
        print(f" [FDFD Adjoint Fit in {elapsed:.1f}s]\n")

        return {
            # headline = binary design (what you would fabricate)
            "rho": rho_bin_np,
            "eps_norm": self._assemble_eps_norm(torch.from_numpy(rho_bin_np).to(self.device)).squeeze().cpu().numpy(),
            "Ez_fit": Ez_bin,   
            "t1_fit": t1_b,
            "t2_fit": t2_b,

            "z": z.detach().cpu(),

            "history": history,
            "target_ratio": target_ratio,
            "elapsed_sec": elapsed,
        }