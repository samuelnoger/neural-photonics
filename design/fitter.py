import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from solver.arguments import get_args
from solver.solver import HelmholtzSolver2D
from design.geometry import build_geometry
from design.losses import SplitterLoss
from design.adjoint import solve_fdfd_differentiable


class PhotonicFitter:
    """
    Universal photonic inverse design fitter supporting:
      - 1x2 Splitters (device_type="splitter_1x2")
      - 1x3 Splitters (device_type="splitter_1x3")
      - 2-Wavelength WDM Demultiplexers (device_type="wdm_1x2")
    """
    def __init__(self, device_type="splitter_1x2", device="auto", smooth_sigma=3.0, **overrides):
        self.device = self._resolve_device(device)
        self.config = vars(get_args(device_type=device_type, custom_args=[]))
        self.config.update(overrides)
        self.device_type = self.config["device_type"]

        # Build geometry and per-wavelength FDFD solvers + loss criteria
        self.geometry = build_geometry(self.config)
        self.field_std = 1.0
        self.wls = self.geometry.wls if self.device_type == "wdm_1x2" else [self.config["wl"]]

        self.solvers = [HelmholtzSolver2D.from_args(self.config, geometry=self.geometry, wl_override=w) for w in self.wls]
        self.solver = self.solvers[0]  # Primary solver reference
        self.criteria = [SplitterLoss(s, self.field_std, self.device) for s in self.solvers]

        # Precompute normalized waveguide base template on device
        eps_empty = self.geometry.make_eps(np.zeros(self.geometry.design_shape, dtype=np.float32))
        eps_norm_empty = (eps_empty - self.geometry.eps_bg) / (self.geometry.eps_si - self.geometry.eps_bg)
        self.base_eps_norm = torch.from_numpy(eps_norm_empty).float().to(self.device).unsqueeze(0).unsqueeze(0)

        self.x0, self.x1 = self.geometry.x0, self.geometry.x1
        self.y0, self.y1 = self.geometry.y0, self.geometry.y1

        # Precompute circular Gaussian smoothing kernel
        k_rad = int(np.ceil(2.5 * smooth_sigma))
        coords = torch.arange(-k_rad, k_rad + 1, device=self.device, dtype=torch.float32)
        kx, ky = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-0.5 * (kx ** 2 + ky ** 2) / (smooth_sigma ** 2))
        self.smooth_kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)
        self.k_pad = k_rad

        # Universal port pinning from geometry
        self.pin, self.pin_z = self.geometry.get_pin_masks(n_pin=self.k_pad + 2, device=self.device)

    @staticmethod
    def _resolve_device(device_arg):
        if device_arg != "auto":
            return torch.device(device_arg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def evaluate(self, rho_np, solver=None):
        s = solver if solver is not None else self.solver
        Ez = s.solve(s.make_eps(rho_np))
        powers = s.measure_port_powers(Ez)
        return Ez, powers

    def _project_density(self, z, beta, symmetric=False, shift=0.0):
        if symmetric:
            z = 0.5 * (z + torch.flip(z, dims=[-1]))
        z = self.pin * self.pin_z + (1.0 - self.pin) * z
        z_4d = F.pad(z.unsqueeze(0).unsqueeze(0), (self.k_pad,) * 4, mode="replicate")
        z_smooth = F.conv2d(z_4d, self.smooth_kernel).squeeze()
        return torch.sigmoid(beta * (z_smooth + shift * (1.0 - self.pin)))

    def _assemble_eps_norm(self, rho):
        eps = self.base_eps_norm.clone()
        eps[0, 0, self.x0 : self.x1, self.y0 : self.y1] = rho
        return eps

    def _resolve_targets(self, target_ratio, target_ratios):
        """Resolves per-wavelength target ratio vectors for 1x2, 1x3, or WDM."""
        if self.device_type == "wdm_1x2":
            # Wavelength 0 -> Port 1 ([1, 0]), Wavelength 1 -> Port 2 ([0, 1])
            return [[1.0, 0.0], [0.0, 1.0]]
        if target_ratios is not None:
            r = np.asarray(target_ratios, dtype=np.float32)
            return [(r / r.sum()).tolist()]
        if self.geometry.n_ports == 2:
            tr = 0.5 if target_ratio is None else float(target_ratio)
            return [[tr, 1.0 - tr]]
        # Default equal split for 1xN (e.g. [1/3, 1/3, 1/3] for 1x3)
        n = self.geometry.n_ports
        return [[1.0 / n] * n]

    def fit(
        self,
        target_ratio=None,
        target_ratios=None,
        total_target=0.96,
        steps=200,
        lr=0.05,
        init_mode="branched",
        symmetric=None,
        beta_start=1.5,
        beta_end=12.0,
        robust_delta=0.0,
        robust_start=0.5,
    ):
        wl_targets = self._resolve_targets(target_ratio, target_ratios)
        if symmetric is None:
            # Enforce y-symmetry automatically if targets are symmetric and not WDM
            t0 = wl_targets[0]
            symmetric = (self.device_type != "wdm_1x2") and np.allclose(t0, t0[::-1], atol=1e-3)

        z = self.geometry.init_latent(init_mode=init_mode, device=self.device).requires_grad_(True)
        
        optimizer = torch.optim.Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)

        history = {"loss": [], "powers": [], "total": [], "t1": [], "t2": [], "t1_surr": [], "t2_surr": [], "total_surr": []}
        pbar = tqdm(range(1, steps + 1), desc=f"Fitting [{self.device_type}]")

        t0 = time.perf_counter()
        for step in pbar:
            frac = min((step - 1) / max(0.75 * steps, 1), 1.0)
            beta = beta_start + (beta_end - beta_start) * (frac ** 1.5)
            bin_weight = 0.01 + 0.08 * (frac ** 2)

            optimizer.zero_grad()
            use_robust = robust_delta > 0 and step > robust_start * steps
            shifts = (0.0, robust_delta, -robust_delta) if use_robust else (0.0,)

            loss = 0.0
            nom_powers = []
            for shift in shifts:
                rho_s = self._project_density(z, beta=beta, symmetric=symmetric, shift=shift)
                eps_in = self._assemble_eps_norm(rho_s)
                for s_idx, (solver, criterion) in enumerate(zip(self.solvers, self.criteria)):
                    field = solve_fdfd_differentiable(eps_in, solver, self.field_std)
                    l, p_vec = criterion(
                        field, rho_s, target_ratios=wl_targets[s_idx],
                        total_target=total_target, bin_weight=bin_weight
                    )
                    loss = loss + l / (len(shifts) * len(self.solvers))
                    if shift == 0.0:
                        nom_powers.append(p_vec.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            scheduler.step()

            p0 = nom_powers[0]  # Primary wavelength port powers
            tot_v = float(np.sum(p0))
            history["loss"].append(loss.item())
            history["powers"].append(p0.tolist())
            history["total"].append(tot_v)
            history["t1"].append(float(p0[0]))
            history["t2"].append(float(p0[1]))
            history["t1_surr"].append(float(p0[0]))
            history["t2_surr"].append(float(p0[1]))
            history["total_surr"].append(tot_v)

            ports_str = "/".join(f"{p*100:.1f}%" for p in p0)
            pbar.set_postfix(Ports=ports_str, Tot=f"{tot_v*100:.1f}%", beta=f"{beta:.1f}")

        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            rho_gray = self._project_density(z, beta=beta_end, symmetric=symmetric)
            rho_bin_np = (rho_gray.cpu().numpy() > 0.5).astype(np.float32)

        Ez_list, powers_list = [], []
        for s in self.solvers:
            Ez_b, p_b = self.evaluate(rho_bin_np, solver=s)
            Ez_list.append(Ez_b)
            powers_list.append(p_b)

        p_main = powers_list[0]
        print(f" [{self.device_type} Fit in {elapsed:.1f}s] Ports: {[round(x*100, 1) for x in p_main]}%")

        return {
            "device_type": self.device_type,
            "rho": rho_bin_np,
            "eps_norm": self._assemble_eps_norm(torch.from_numpy(rho_bin_np).to(self.device)).squeeze().cpu().numpy(),
            "Ez_fit": Ez_list[0],
            "Ez_all": Ez_list,
            "powers_fit": p_main,
            "powers_all": powers_list,
            "t1_fit": p_main[0],
            "t2_fit": p_main[1],
            "z": z.detach().cpu(),
            "history": history,
            "target_ratios": wl_targets[0],
            "target_ratio": wl_targets[0][0],
            "elapsed_sec": elapsed,
        }