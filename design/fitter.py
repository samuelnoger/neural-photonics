import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from data.solver import HelmholtzSolver2D
from model.model import build_model
from design.losses import SplitterLoss


class PhotonicFitter:
    """
    End-to-end inverse design fitter using a frozen neural surrogate (FNO/U-Net).
    """
    def __init__(self, ckpt_path="checkpoints/best_fno.pt", device="auto"):
        self.device = self._resolve_device(device)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        self.field_std = ckpt["field_std"]

        # Initialize ground-truth FDFD solver for template & final verification
        self.solver = HelmholtzSolver2D.from_args(self.config)

        # Load and freeze neural surrogate
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Precompute normalized waveguide base template on device
        eps_empty = self.solver.make_eps(np.zeros(self.solver.design_shape, dtype=np.float32))
        eps_norm_empty = (eps_empty - self.solver.eps_bg) / (self.solver.eps_si - self.solver.eps_bg)
        self.base_eps_norm = torch.from_numpy(eps_norm_empty).float().to(self.device).unsqueeze(0).unsqueeze(0)

        # Design region slice coordinates
        self.x0, self.x1 = self.solver.x_in_end, self.solver.x_out_start
        yc = self.solver.ny // 2
        self.y0, self.y1 = yc - self.solver.design_half_y, yc + self.solver.design_half_y

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

    def _project_density(self, z, beta, symmetric=False):
        if symmetric:
            z = 0.5 * (z + torch.flip(z, dims=[-1]))
        # Two passes of 3x3 average pooling = 5x5 triangular tent filter
        z_smooth = z.unsqueeze(0).unsqueeze(0)
        z_smooth = F.avg_pool2d(z_smooth, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        z_smooth = F.avg_pool2d(z_smooth, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        return torch.sigmoid(beta * z_smooth.squeeze())

    def _assemble_eps_norm(self, rho):
        """Inserts the (40, 48) differentiable rho patch into the (1, 1, 128, 128) chip grid."""
        eps = self.base_eps_norm.clone()
        eps[0, 0, self.x0 : self.x1, self.y0 : self.y1] = rho
        return eps

    def _init_latent(self, init_mode="y_branch", fill_val=0.5):
        """
        Initializes the unconstrained latent tensor z for the (nx_d, ny_d) design region.
        Supported init_mode:
          - "y_branch" : Smooth Gaussian Y-junction skeleton
          - "taper"    : Wide linear horn/funnel connecting input to both outputs
          - "mmi"      : Rectangular central slab of silicon (Multi-Mode Interference block)
          - "empty"    : Low-density oxide background (rho ≈ 0.10)
          - "filled"   : High-density silicon block (rho ≈ 0.90)
          - "gray"     : Uniform 50% gray mist (rho = fill_val, default 0.50)
          - "random"   : Smooth random Gaussian noise around 0.50
        """
        nx_d, ny_d = self.solver.design_shape
        x = torch.linspace(0, 1, nx_d, device=self.device)[:, None]
        y = torch.arange(ny_d, device=self.device, dtype=torch.float32)[None, :] - ((ny_d - 1) / 2.0)

        if init_mode == "y_branch":
            offset = float(self.solver.port_offset) * x
            w = float(self.solver.wg_half)
            skel = torch.exp(-0.5 * ((y - offset) / w) ** 2) + torch.exp(-0.5 * ((y + offset) / w) ** 2)
            rho0 = torch.clamp(skel, 0.05, 0.95)

        elif init_mode == "taper":
            # Smoothly widens from input waveguide half-width to outer edge of output ports
            half_width = float(self.solver.wg_half) + x * float(self.solver.port_offset + self.solver.wg_half)
            rho0 = torch.sigmoid((half_width - torch.abs(y)) / 2.0)
            rho0 = torch.clamp(rho0, 0.05, 0.95)

        elif init_mode == "mmi":
            # Rectangular silicon slab covering the port span across all (nx_d, ny_d) pixels
            slab_half_y = float(self.solver.port_offset + self.solver.wg_half)
            rho0 = torch.sigmoid((slab_half_y - torch.abs(y)) / 1.5).expand(nx_d, ny_d)
            rho0 = torch.clamp(rho0, 0.05, 0.95)

        elif init_mode == "empty":
            rho0 = torch.full((nx_d, ny_d), 0.10, device=self.device)

        elif init_mode == "filled":
            rho0 = torch.full((nx_d, ny_d), 0.90, device=self.device)

        elif init_mode == "random":
            rho0 = torch.clamp(0.5 + 0.15 * torch.randn((nx_d, ny_d), device=self.device), 0.05, 0.95)

        else:  # "gray" or custom fill_val
            val = float(np.clip(fill_val, 0.05, 0.95))
            rho0 = torch.full((nx_d, ny_d), val, device=self.device)

        return torch.logit(rho0)

    def _predict_equivariant(self, eps_in):
        # f_sym(x) = 0.5 * (f(x) + flip(f(flip(x))))
        pred1 = self.model(eps_in)
        pred2 = torch.flip(self.model(torch.flip(eps_in, dims=[-1])), dims=[-1])
        return 0.5 * (pred1 + pred2)

    def fit(
        self,
        target_ratio=0.5,
        total_target=0.92,
        steps=150,
        lr=0.08,
        init_mode="y_branch",
        symmetric=None,
        beta_start=1.5,
        beta_end=6.0,
        verify_fdfd=True,
    ):
        """
        Optimizes the 40x48 design region to achieve `target_ratio` power splitting.
        If symmetric is None, automatically enforces symmetry when target_ratio == 0.5.
        """
        if symmetric is None:
            symmetric = abs(target_ratio - 0.5) < 1e-3

        z = self._init_latent(init_mode).requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)

        history = {"loss": [], "t1_surr": [], "t2_surr": [], "total_surr": []}
        pbar = tqdm(range(1, steps + 1), desc=f"Fitting Splitter ({target_ratio*100:.0f}/{(1-target_ratio)*100:.0f})")

        for step in pbar:

            # Reach full binarization at 75% of total steps, then let the optimizer fine-tune at fixed beta
            frac = min((step - 1) / max(0.75 * steps, 1), 1.0)
            beta = beta_start + (beta_end - beta_start) * (frac ** 1.5)
            bin_weight = 0.01 + 0.08 * (frac ** 2)

            optimizer.zero_grad()
            rho = self._project_density(z, beta=beta, symmetric=symmetric)
            eps_in = self._assemble_eps_norm(rho)

            pred_field = self._predict_equivariant(eps_in)
            loss, t1, t2 = self.criterion(
                pred_field,
                rho,
                target_ratio=target_ratio,
                total_target=total_target,
                bin_weight=bin_weight,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            t1_v, t2_v = t1.item(), t2.item()
            history["loss"].append(loss.item())
            history["t1_surr"].append(t1_v)
            history["t2_surr"].append(t2_v)
            history["total_surr"].append(t1_v + t2_v)

            pbar.set_postfix(
                T1=f"{t1_v*100:.1f}%",
                T2=f"{t2_v*100:.1f}%",
                Tot=f"{(t1_v+t2_v)*100:.1f}%",
                beta=f"{beta:.1f}",
            )

        # Finalize design & run ground-truth FDFD check
        with torch.no_grad():
            rho_final = self._project_density(z, beta=beta_end, symmetric=symmetric)
            eps_final_in = self._assemble_eps_norm(rho_final)
            pred_final = self._predict_equivariant(eps_final_in).squeeze(0).cpu().numpy()
            Ez_surr = (pred_final[0] + 1j * pred_final[1]) * self.field_std
            rho_np = rho_final.cpu().numpy()

        result = {
            "rho": rho_np,
            "eps_norm": eps_final_in.squeeze().cpu().numpy(),
            "Ez_surr": Ez_surr,
            "t1_surr": history["t1_surr"][-1],
            "t2_surr": history["t2_surr"][-1],
            "history": history,
            "target_ratio": target_ratio,
        }

        if verify_fdfd:
            eps_phys = self.solver.make_eps(rho_np)
            Ez_true = self.solver.solve(eps_phys)
            t1_true, t2_true = self.solver.measure_port_powers(Ez_true)
            result.update({
                "Ez_true": Ez_true,
                "t1_true": t1_true,
                "t2_true": t2_true,
            })
            print(
                f" Surrogate Prediction : Port 1 = {result['t1_surr']*100:.1f}%, Port 2 = {result['t2_surr']*100:.1f}% "
                f"(Total = {(result['t1_surr']+result['t2_surr'])*100:.1f}%)\n"
                f" FDFD Ground Truth    : Port 1 = {t1_true*100:.1f}%, Port 2 = {t2_true*100:.1f}% "
                f"(Total = {(t1_true+t2_true)*100:.1f}%)"
            )

        return result