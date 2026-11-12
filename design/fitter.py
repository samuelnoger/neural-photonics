import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from solver.arguments import get_args
from solver.solver import HelmholtzSolver2D
from design.geometry import build_geometry
from design.losses import splitter_loss, wdm_loss
from design.adjoint import solve_fdfd_differentiable


class PhotonicFitter:
    """
    Universal photonic inverse design fitter supporting:
      - 1x2 Splitters (device_type="splitter_1x2")
      - 1x3 Splitters (device_type="splitter_1x3")
      - 2-Wavelength WDM Demultiplexers (device_type="wdm_1x2")
    """
    def __init__(self, device_type="splitter_1x2", device="cpu", smooth_sigma=4.0, **overrides):
        self.device = torch.device("cpu") if device in ("auto", "cpu") else torch.device(device)
        self.config = vars(get_args(device_type=device_type, custom_args=[]))
        self.config.update(overrides)
        self.device_type = self.config["device_type"]

        # Build geometry (with tensors on self.device) and per-wavelength FDFD solvers
        self.geometry = build_geometry(self.config, device=self.device)
        self.wls = self.geometry.wls if self.device_type == "wdm_1x2" else [self.config["wl"]]

        self.solvers = [HelmholtzSolver2D.from_args(self.config, geometry=self.geometry, wl_override=w) for w in self.wls]
        self.solver = self.solvers[0]  # Primary solver reference
        self.loss_fn = wdm_loss if self.device_type == "wdm_1x2" else splitter_loss

        self.x0, self.x1 = self.geometry.x0, self.geometry.x1
        self.y0, self.y1 = self.geometry.y0, self.geometry.y1

        # Precompute circular Gaussian smoothing kernel on self.device
        k_rad = int(np.ceil(2.5 * smooth_sigma))
        coords = torch.arange(-k_rad, k_rad + 1, device=self.device, dtype=torch.float32)
        kx, ky = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-0.5 * (kx ** 2 + ky ** 2) / (smooth_sigma ** 2))
        self.smooth_kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)
        self.k_pad = k_rad

        # Universal port & corner pinning on self.device
        self.pin, self.pin_z = self.geometry.get_pin_masks(n_pin=self.k_pad + 2, flare_margin=500)

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
        eps = self.geometry.base_eps_norm.clone()
        eps[0, 0, self.x0 : self.x1, self.y0 : self.y1] = rho
        return eps

    def _resolve_targets(self, target_ratio, target_ratios):
        """Returns target ratio tensor of shape (n_wls, n_ports) on `self.device`."""
        if self.device_type == "wdm_1x2":
            t_list = [[1.0, 0.0], [0.0, 1.0]]
        elif target_ratios is not None:
            r = np.asarray(target_ratios, dtype=np.float32)
            t_list = [(r / r.sum()).tolist()]
        elif self.geometry.n_ports == 2:
            tr = 0.5 if target_ratio is None else float(target_ratio)
            t_list = [[tr, 1.0 - tr]]
        else:
            n = self.geometry.n_ports
            t_list = [[1.0 / n] * n]
        return torch.tensor(t_list, dtype=torch.float32, device=self.device)

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
        beta_end=16.0,
        robust_delta=0.0,
        robust_start=0.5,
    ):
        wl_targets = self._resolve_targets(target_ratio, target_ratios)
        t0_np = wl_targets[0].cpu().numpy()
        if symmetric is None:
            symmetric = (self.device_type != "wdm_1x2") and np.allclose(t0_np, t0_np[::-1], atol=1e-3)

        z = self.geometry.init_latent(init_mode=init_mode).requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)

        history = {"loss": [], "powers": [], "powers_all": [], "total": [], "t1": [], "t2": []}
        pbar = tqdm(range(1, steps + 1), desc=f"Fitting [{self.device_type}]")

        wl_mean = float(np.mean(self.wls))
        t0 = time.perf_counter()
        for step in pbar:
            frac = min((step - 1) / max(0.75 * steps, 1), 1.0)
            beta = beta_start + (beta_end - beta_start) * (frac ** 1.5)
            bin_weight = 0.01 + 0.10 * (frac ** 2)

            optimizer.zero_grad()
            use_robust = robust_delta > 0 and step > robust_start * steps
            shifts = (0.0, robust_delta, -robust_delta) if use_robust else (0.0,)

            loss = 0.0
            nom_powers = []
            for shift in shifts:
                rho_s = self._project_density(z, beta=beta, symmetric=symmetric, shift=shift)
                eps_in = self._assemble_eps_norm(rho_s)
                pass_powers = []
                for s_idx, solver in enumerate(self.solvers):
                    field = solve_fdfd_differentiable(eps_in, solver)
                    l, p_vec = self.loss_fn(
                        field,
                        rho_s,
                        solver,
                        targets=wl_targets[s_idx],
                        total_target=total_target,
                        bin_weight=bin_weight,
                    )
                    # Compensate for k0^2 ~ 1/wl^2 adjoint scaling in multi-wavelength WDM
                    wl_weight = (solver.wl / wl_mean) ** 2 if len(self.solvers) > 1 else 1.0
                    loss = loss + (wl_weight * l) / (len(shifts) * len(self.solvers))
                    pass_powers.append((p_vec * wl_targets[s_idx]).sum())
                    if shift == 0.0:
                        nom_powers.append(p_vec.detach().cpu().numpy())

                # Explicitly penalize pass-port asymmetry between WDM channels
                if len(pass_powers) > 1:
                    loss = loss + 1.5 * ((pass_powers[0] - pass_powers[1]) ** 2) / len(shifts)

            loss.backward()
            optimizer.step()
            scheduler.step()

            p0 = nom_powers[0]
            history["loss"].append(loss.item())
            history["powers"].append(p0.tolist())
            history["powers_all"].append([p.tolist() for p in nom_powers])
            history["t1"].append(float(p0[0]))
            history["t2"].append(float(p0[1]))

            if len(nom_powers) > 1:
                tot_v = 0.5 * float(np.sum(nom_powers[0]) + np.sum(nom_powers[1]))
                ports_str = f"λ1:{nom_powers[0][0]*100:.1f}%|λ2:{nom_powers[1][1]*100:.1f}%"
            else:
                tot_v = float(np.sum(p0))
                ports_str = "/".join(f"{p*100:.1f}%" for p in p0)

            history["total"].append(tot_v)
            pbar.set_postfix(Ports=ports_str, Tot=f"{tot_v*100:.1f}%", beta=f"{beta:.1f}")

        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            rho_gray = self._project_density(z, beta=beta_end, symmetric=symmetric)
            rho_bin_t = (rho_gray > 0.5).float()
            rho_bin_np = rho_bin_t.cpu().numpy()
            eps_norm_np = self._assemble_eps_norm(rho_bin_t).squeeze().cpu().numpy()

        Ez_list, powers_list = [], []
        for s in self.solvers:
            Ez_b, p_b = self.evaluate(rho_bin_np, solver=s)
            Ez_list.append(Ez_b)
            powers_list.append(p_b)

        p_main = powers_list[0]
        if len(powers_list) > 1:
            msg = " | ".join(
                f"λ={w:.2f}um: {[round(x*100, 1) for x in p]}%"
                for w, p in zip(self.wls, powers_list)
            )
            print(f" [{self.device_type} Fit in {elapsed:.1f}s] {msg}")
        else:
            print(f" [{self.device_type} Fit in {elapsed:.1f}s] Ports: {[round(x*100, 1) for x in p_main]}%")

        return {
            "device_type": self.device_type,
            "rho": rho_bin_np,
            "eps_norm": eps_norm_np,
            "Ez_fit": Ez_list[0],
            "Ez_all": Ez_list,
            "powers_fit": p_main,
            "powers_all": powers_list,
            "t1_fit": p_main[0],
            "t2_fit": p_main[1],
            "z": z.detach().cpu(),
            "history": history,
            "target_ratios": t0_np.tolist(),
            "target_ratio": float(t0_np[0]),
            "elapsed_sec": elapsed,
        }