import numpy as np
import torch


class DeviceGeometry:
    """
    Universal 2D photonic device geometry base class.
    Defines waveguides, design region, port mode profiles, port pinning, and initial latent shapes.
    """
    def __init__(self, config, port_offsets_y):
        d = vars(config) if not isinstance(config, dict) else config
        self.config = d
        self.nx, self.ny = d["nx"], d["ny"]
        self.eps_bg, self.eps_si = d["eps_bg"], d["eps_si"]
        self.wg_half = d["wg_half"]
        self.design_half_y = d["design_half_y"]
        self.x_in_end = d["x_in_end"]
        self.x_out_start = d["x_out_start"]
        self.x_src = d["x_src"]
        self.x_meas = d["x_meas"]
        self.src_amp = d["src_amp"]
        self.mode_waist_ratio = d["mode_waist_ratio"]
        self.wls = d.get("wls", [d.get("wl", 1.55)])

        yc = self.ny // 2
        yc_f = (self.ny - 1) / 2.0
        self.yc = yc
        self.port_offsets_y = list(port_offsets_y)  # e.g. [-36, +36] or [-52, 0, +52]
        self.port_ys = [yc + off for off in self.port_offsets_y]
        self.n_ports = len(self.port_ys)

        # Design box slice coordinates & shape
        self.x0, self.x1 = self.x_in_end, self.x_out_start
        self.y0, self.y1 = yc - self.design_half_y, yc + self.design_half_y
        self.design_shape = (self.x1 - self.x0, self.y1 - self.y0)
        self.design_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.design_mask[self.x0 : self.x1, self.y0 : self.y1] = True

        # Transverse Gaussian mode profiles for input and all N output ports
        waist = self.wg_half * self.mode_waist_ratio
        y_grid = np.arange(self.ny)
        g = lambda c: (lambda v: v / np.linalg.norm(v))(np.exp(-0.5 * ((y_grid - c) / waist) ** 2))

        self.mode_in = g(yc_f)
        self.mode_ports = [g(yc_f + off) for off in self.port_offsets_y]

        # Input line source
        self.source = np.zeros((self.nx, self.ny), dtype=np.complex128)
        self.source[self.x_src, :] = self.src_amp * self.mode_in

        # Base permittivity template (input waveguide + N output waveguides)
        self.eps_template = np.full((self.nx, self.ny), self.eps_bg, dtype=np.float64)
        self.eps_template[: self.x0, yc - self.wg_half : yc + self.wg_half] = self.eps_si
        for py in self.port_ys:
            self.eps_template[self.x1 :, py - self.wg_half : py + self.wg_half] = self.eps_si

        # Reference straight waveguide for normalizing port power to [0, 1]
        self.eps_straight = np.full((self.nx, self.ny), self.eps_bg, dtype=np.float64)
        self.eps_straight[:, yc - self.wg_half : yc + self.wg_half] = self.eps_si

    def make_eps(self, design_rho):
        eps = self.eps_template.copy()
        eps_patch = self.eps_bg + np.clip(design_rho, 0.0, 1.0) * (self.eps_si - self.eps_bg)
        eps[self.design_mask] = eps_patch.ravel()
        return eps

    def get_pin_masks(self, n_pin, device, flare_margin=16):
        """
        Pins left/right port interfaces to match the waveguides AND pins the dead
        top-left / bottom-left outer corners to SiO2 so detached islands cannot form.
        """
        nx_d, ny_d = self.design_shape
        thresh = 0.5 * (self.eps_bg + self.eps_si)
        left_si = (self.eps_template[self.x0 - 1, self.y0 : self.y1] > thresh).astype(np.float32)
        right_si = (self.eps_template[self.x1, self.y0 : self.y1] > thresh).astype(np.float32)

        left_t = torch.from_numpy(left_si).to(device)
        right_t = torch.from_numpy(right_si).to(device)

        pin = torch.zeros(nx_d, ny_d, device=device)
        pin_z = torch.zeros(nx_d, ny_d, device=device)

        # 1. Pin left and right waveguide port columns
        pin[:n_pin, :] = 1.0
        pin_z[:n_pin, :] = (2.0 * left_t - 1.0) * 2.0
        pin[-n_pin:, :] = 1.0
        pin_z[-n_pin:, :] = (2.0 * right_t - 1.0) * 2.0

        # 2. Pin dead top-left / bottom-left corners outside a flaring envelope to oxide (-2.0)
        x_norm = torch.linspace(0.0, 1.0, nx_d, device=device)[:, None]
        y_abs = torch.abs(torch.arange(ny_d, device=device, dtype=torch.float32)[None, :] - (ny_d - 1) / 2.0)
        max_port_extent = max(abs(off) for off in self.port_offsets_y) + self.wg_half
        allowed_half_y = (self.wg_half + flare_margin) + x_norm * (max_port_extent + flare_margin - self.wg_half)

        dead_corner = (y_abs > allowed_half_y) & (pin == 0.0)
        pin[dead_corner] = 1.0
        pin_z[dead_corner] = -2.0  # Fixed SiO2 cladding

        return pin, pin_z

    def init_latent(self, init_mode="branched", fill_val=0.5, device="cpu"):
        """Initializes latent design tensor z for 1x2, 1x3, or general 1xN devices."""
        nx_d, ny_d = self.design_shape
        x = torch.linspace(0, 1, nx_d, device=device)[:, None]
        y = torch.arange(ny_d, device=device, dtype=torch.float32)[None, :] - ((ny_d - 1) / 2.0)

        if init_mode in ("y_branch", "branched"):
            w = float(self.wg_half)
            skel = torch.zeros((nx_d, ny_d), device=device)
            for off in self.port_offsets_y:
                skel = skel + torch.exp(-0.5 * ((y - float(off) * x) / w) ** 2)
            rho0 = torch.clamp(skel, 0.05, 0.95)
        elif init_mode == "taper":
            max_off = max(abs(off) for off in self.port_offsets_y)
            half_width = float(self.wg_half) + x * float(max_off + self.wg_half)
            rho0 = torch.clamp(torch.sigmoid((half_width - torch.abs(y)) / 2.0), 0.05, 0.95)
        elif init_mode == "empty":
            rho0 = torch.full((nx_d, ny_d), 0.10, device=device)
        elif init_mode == "filled":
            rho0 = torch.full((nx_d, ny_d), 0.90, device=device)
        else:
            rho0 = torch.full((nx_d, ny_d), float(np.clip(fill_val, 0.05, 0.95)), device=device)

        return torch.logit(rho0)


class Splitter1x2Geometry(DeviceGeometry):
    def __init__(self, config):
        d = vars(config) if not isinstance(config, dict) else config
        po = d["port_offset"]
        super().__init__(d, port_offsets_y=[-po, +po])


class Splitter1x3Geometry(DeviceGeometry):
    def __init__(self, config):
        d = vars(config) if not isinstance(config, dict) else config
        po = d["port_offset"]
        super().__init__(d, port_offsets_y=[-po, 0, +po])


class WDM1x2Geometry(Splitter1x2Geometry):
    """1x2 Wavelength Demultiplexer: routes wls[0] -> Port 1 and wls[1] -> Port 2."""
    pass


def build_geometry(config):
    d = vars(config) if not isinstance(config, dict) else config
    dev_type = d.get("device_type", "splitter_1x2")
    if dev_type == "splitter_1x2":
        return Splitter1x2Geometry(d)
    elif dev_type == "splitter_1x3":
        return Splitter1x3Geometry(d)
    elif dev_type == "wdm_1x2":
        return WDM1x2Geometry(d)
    raise ValueError(f"Unknown device_type: '{dev_type}'")