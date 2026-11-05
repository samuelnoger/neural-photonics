import argparse


def get_args(custom_args=None):
    parser = argparse.ArgumentParser(
        description="2D Neural Photonics: FDFD Solver, Surrogate Training, and Inverse Design",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physics & Waveguide Geometry (data/solver.py)
    phys = parser.add_argument_group("Physics & Waveguide Geometry")
    phys.add_argument("--nx", type=int, default=128, help="Grid points along x (propagation direction)")
    phys.add_argument("--ny", type=int, default=128, help="Grid points along y (transverse direction)")
    phys.add_argument("--dx", type=float, default=0.04, help="Grid spacing in microns (0.04 = 40 nm)")
    phys.add_argument("--wl", type=float, default=1.55, help="Laser wavelength in microns")
    phys.add_argument("--npml", type=int, default=16, help="PML absorbing boundary thickness in pixels")
    phys.add_argument("--pml-sigma-max", type=float, default=4.0, help="Dimensionless PML damping prefactor")
    phys.add_argument("--eps-bg", type=float, default=2.08, help="Background clade relative permittivity (SiO2 ~ 1.44^2)")
    phys.add_argument("--eps-si", type=float, default=7.84, help="Core slab effective permittivity (SOI 220nm ~ 2.8^2)")

    phys.add_argument("--wg-half", type=int, default=5, help="Waveguide half-width in pixels (5 px = 10 px total = 400 nm)")
    phys.add_argument("--port-offset", type=int, default=18, help="Y-offset of the two output ports from center in pixels")
    phys.add_argument("--design-half-y", type=int, default=24, help="Half-height of the central design region in pixels")
    phys.add_argument("--x-in-end", type=int, default=44, help="X-index where input waveguide enters the design region")
    phys.add_argument("--x-out-start", type=int, default=84, help="X-index where output waveguides leave the design region")
    phys.add_argument("--x-src", type=int, default=30, help="X-index of the input laser line source")
    phys.add_argument("--x-meas", type=int, default=100, help="X-index of the output port power measurement slice")
    phys.add_argument("--src-amp", type=float, default=100.0, help="Amplitude scaling of the line source")
    phys.add_argument("--mode-waist-ratio", type=float, default=0.75, help="Gaussian mode waist relative to wg_half")

    # Random Geometry & Dataset Generation (data/data.py)
    data = parser.add_argument_group("Dataset Generation")
    data.add_argument("--save-dir", type=str, default="datasets", help="Directory to save/load train.pt and val.pt")
    data.add_argument("--n-train", type=int, default=40000, help="Number of training simulations to generate")
    data.add_argument("--n-val", type=int, default=2000, help="Number of validation simulations to generate")
    data.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU cores for parallel solving (-1 = all cores)")
    data.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")

    data.add_argument("--prob-smooth", type=float, default=0.3, help="Fraction of smooth grayscale random designs")
    data.add_argument("--prob-binary", type=float, default=0.3, help="Fraction of binarized island designs")
    data.add_argument("--prob-y-pert", type=float, default=0.4, help="Fraction of perturbed Y-splitter skeleton designs")
    data.add_argument("--blob-sigma-min", type=float, default=1.5, help="Min Gaussian smoothing sigma for random blobs")
    data.add_argument("--blob-sigma-max", type=float, default=4.0, help="Max Gaussian smoothing sigma for random blobs")
    data.add_argument("--sym-prob", type=float, default=0.3, help="Probability of enforcing y-symmetry on a sample")

    # Neural Surrogate Model & Training (model/model.py, train/train.py)
    train = parser.add_argument_group("Model & Training")
    train.add_argument("--arch", type=str, default="fno", choices=["unet", "fno"], help="Surrogate architecture")
    train.add_argument("--base-ch", type=int, default=32, help="Base channel width for U-Net or FNO")
    train.add_argument("--fno-modes", type=int, default=16, help="Number of Fourier modes to keep per dimension in FNO")
    train.add_argument("--fno-layers", type=int, default=4, help="Number of Fourier operator layers in FNO")
    train.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    train.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    train.add_argument("--lr", type=float, default=1e-3, help="Initial AdamW learning rate")
    train.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    train.add_argument("--grad-loss-weight", type=float, default=0.2, help="Weight of spatial gradient loss dE/dx, dE/dy")
    train.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    train.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cuda', 'mps', or 'cpu'")

    # Inverse Design Optimization
    inv = parser.add_argument_group("Inverse Design")
    inv.add_argument("--target-ratio", type=float, default=0.5, help="Target power split ratio Port1 / (Port1 + Port2)")
    inv.add_argument("--design-steps", type=int, default=200, help="Number of gradient-descent steps on geometry")
    inv.add_argument("--design-lr", type=float, default=0.08, help="Learning rate for geometry optimization")
    inv.add_argument("--results-dir", type=str, default="results", help="Directory for output plots and metrics")

    return parser.parse_args(custom_args)