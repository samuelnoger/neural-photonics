import argparse


def get_args(custom_args=None):
    parser = argparse.ArgumentParser(
        description="2D Photonic Inverse Design via Differentiable FDFD Adjoint Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physics & Waveguide Geometry
    phys = parser.add_argument_group("Physics & Waveguide Geometry")
    phys.add_argument("--nx", type=int, default=296, help="Grid points along x (propagation direction)")
    phys.add_argument("--ny", type=int, default=256, help="Grid points along y (transverse direction)")
    phys.add_argument("--dx", type=float, default=0.02, help="Grid spacing in microns (0.02 = 20 nm)")
    phys.add_argument("--wl", type=float, default=1.55, help="Laser wavelength in microns")
    phys.add_argument("--npml", type=int, default=32, help="PML absorbing boundary thickness in pixels")
    phys.add_argument("--pml-sigma-max", type=float, default=4.0, help="Dimensionless PML damping prefactor")
    phys.add_argument("--eps-bg", type=float, default=2.08, help="Background cladding relative permittivity (SiO2)")
    phys.add_argument("--eps-si", type=float, default=7.84, help="Silicon core effective permittivity")

    phys.add_argument("--wg-half", type=int, default=10, help="Waveguide half-width in pixels")
    phys.add_argument("--port-offset", type=int, default=36, help="Y-offset of the two output ports from center")
    phys.add_argument("--design-half-y", type=int, default=48, help="Half-height of the central design region")
    phys.add_argument("--x-in-end", type=int, default=88, help="X-index where input waveguide enters design region")
    phys.add_argument("--x-out-start", type=int, default=208, help="X-index where output waveguides leave design region")
    phys.add_argument("--x-src", type=int, default=60, help="X-index of the input laser line source")
    phys.add_argument("--x-meas", type=int, default=240, help="X-index of the output port measurement slice")
    phys.add_argument("--src-amp", type=float, default=100.0, help="Amplitude scaling of the line source")
    phys.add_argument("--mode-waist-ratio", type=float, default=0.75, help="Gaussian mode waist relative to wg_half")

    # Inverse Design Optimization (design/fitter.py)
    inv = parser.add_argument_group("Inverse Design")
    inv.add_argument("--target-ratio", type=float, default=0.5, help="Target power split ratio Port1 / (Port1 + Port2)")
    inv.add_argument("--total-target", type=float, default=0.92, help="Target total guided transmission")
    inv.add_argument("--init-mode", type=str, default="y_branch", help="Initial latent geometry mode")
    inv.add_argument("--design-steps", type=int, default=200, help="Number of adjoint gradient steps")
    inv.add_argument("--design-lr", type=float, default=0.05, help="Learning rate for geometry optimization")
    inv.add_argument("--beta-start", type=float, default=1.5, help="Initial sigmoid projection steepness")
    inv.add_argument("--beta-end", type=float, default=10.0, help="Final sigmoid projection steepness")
    inv.add_argument("--smooth-sigma", type=float, default=1.5, help="Gaussian spatial filter sigma in pixels")
    inv.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cuda', 'mps', or 'cpu'")
    inv.add_argument("--results-dir", type=str, default="results", help="Directory for output plots and metrics")

    return parser.parse_args(custom_args)