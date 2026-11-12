import argparse

# Clean, decoupled device presets: only specify the box length, height, port offset, and wavelengths!
# x_out_start, x_meas, nx, and ny are automatically computed to preserve exact PML/source/measurement margins.
DEVICE_PRESETS = {
    "splitter_1x2": {
        "design_len_x": 120,   # 2.40 um long design region
        "design_half_y": 48,   # 1.92 um tall design region (96 px)
        "port_offset": 36,     # Ports at yc ± 36 (0.72 um offset)
        "wls": [1.55],
    },
    "splitter_1x3": {
        "design_len_x": 150,   # 3.00 um long design region
        "design_half_y": 72,   # 2.88 um tall design region (144 px)
        "port_offset": 50,     # Ports at yc - 50, yc, yc + 50 (1.00 um spacing)
        "wls": [1.55],
    },
    "wdm_1x2": {
        "design_len_x": 180,   # 3.60 um long design region (needed for 1.31/1.55 um beat length)
        "design_half_y": 56,   # 2.24 um tall design region (112 px)
        "port_offset": 36,     # Ports at yc ± 36 (0.72 um offset)
        "wls": [1.31, 1.55],
    },
}


def get_args(device_type="splitter_1x2", custom_args=None):
    parser = argparse.ArgumentParser(
        description="2D Photonic Inverse Design via Differentiable FDFD Adjoint Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--device-type",
        type=str,
        default=device_type,
        choices=list(DEVICE_PRESETS.keys()),
        help="Photonic device geometry preset",
    )

    # Physics & Waveguide Geometry
    phys = parser.add_argument_group("Physics & Waveguide Geometry")
    phys.add_argument("--dx", type=float, default=0.02, help="Grid spacing in microns (0.02 = 20 nm)")
    phys.add_argument("--wl", type=float, default=1.55, help="Single laser wavelength in microns")
    phys.add_argument("--wls", type=float, nargs="+", default=[1.55], help="Wavelengths list (used by WDM)")
    phys.add_argument("--npml", type=int, default=32, help="PML absorbing boundary thickness in pixels")
    phys.add_argument("--pml-sigma-max", type=float, default=8.0, help="Dimensionless PML damping prefactor")
    phys.add_argument("--eps-bg", type=float, default=2.08, help="Background cladding permittivity (SiO2)")
    phys.add_argument("--eps-si", type=float, default=7.84, help="Silicon core effective permittivity at 1.55 um")

    phys.add_argument("--wg-half", type=int, default=10, help="Waveguide half-width in pixels (10 px = 200 nm)")
    phys.add_argument("--port-offset", type=int, default=36, help="Y-offset between adjacent output ports in pixels")
    phys.add_argument("--design-len-x", type=int, default=120, help="Length of central design region along x in pixels")
    phys.add_argument("--design-half-y", type=int, default=48, help="Half-height of central design region along y in pixels")
    phys.add_argument("--x-src", type=int, default=60, help="X-index of input line source")
    phys.add_argument("--x-in-end", type=int, default=88, help="X-index where input enters design region")

    # Optional explicit overrides (if None, auto-computed from design_len_x and design_half_y!)
    phys.add_argument("--x-out-start", type=int, default=None, help="X-index where outputs leave design region (auto if None)")
    phys.add_argument("--x-meas", type=int, default=None, help="X-index of output measurement slice (auto if None)")
    phys.add_argument("--nx", type=int, default=None, help="Total grid points along x (auto if None)")
    phys.add_argument("--ny", type=int, default=None, help="Total grid points along y (auto if None)")

    phys.add_argument("--src-amp", type=float, default=100.0, help="Amplitude scaling of line source")

    # Inverse Design Optimization
    inv = parser.add_argument_group("Inverse Design")
    inv.add_argument("--target-ratios", type=float, nargs="+", default=None, help="Target split ratios per port")
    inv.add_argument("--total-target", type=float, default=0.96, help="Target total guided transmission")
    inv.add_argument("--init-mode", type=str, default="branched", help="Initial latent geometry mode")
    inv.add_argument("--design-steps", type=int, default=200, help="Number of adjoint gradient steps")
    inv.add_argument("--design-lr", type=float, default=0.05, help="Learning rate for geometry optimization")
    inv.add_argument("--beta-start", type=float, default=1.5, help="Initial sigmoid projection steepness")
    inv.add_argument("--beta-end", type=float, default=16.0, help="Final sigmoid projection steepness")
    inv.add_argument("--smooth-sigma", type=float, default=4.0, help="Gaussian spatial filter sigma in pixels")
    inv.add_argument("--device", type=str, default="cpu", help="Compute device")
    inv.add_argument("--results-dir", type=str, default="results", help="Directory for output plots")

    # 1. Apply preset defaults for the chosen device_type
    if device_type in DEVICE_PRESETS:
        parser.set_defaults(**DEVICE_PRESETS[device_type])

    args = parser.parse_args(custom_args)
    if custom_args is not None and args.device_type != device_type:
        parser.set_defaults(**DEVICE_PRESETS[args.device_type])
        args = parser.parse_args(custom_args)

    # 2. Automatically derive coupled x/y grid coordinates so margins around PML & measurement slice are always safe!
    if args.x_out_start is None:
        args.x_out_start = args.x_in_end + args.design_len_x          # e.g. 88 + 120 = 208
    if args.x_meas is None:
        args.x_meas = args.x_out_start + 32                           # 32 px (0.64 um) after design box (e.g. 240)
    if args.nx is None:
        args.nx = args.x_meas + 24 + args.npml                        # 24 px margin + 32 px PML after x_meas (e.g. 296)
    if args.ny is None:
        args.ny = 2 * (args.design_half_y + 48 + args.npml)           # 48 px margin + 32 px PML above/below box (e.g. 256)

    return args