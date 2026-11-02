import torch
import torch.nn as nn
import torch.nn.functional as F


def append_coords(x):
    """
    Appends normalized (x, y) coordinate channels in [-1, 1] to input tensor x.
    Input:  (B, C, H, W)
    Output: (B, C + 2, H, W)
    """
    B, _, H, W = x.shape
    device, dtype = x.device, x.dtype
    gx = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    gy = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, gx, gy], dim=1)


import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

# 2D Fourier Neural Operator (FNO)
class SpectralConv2d(nn.Module):
    """2D Fourier layer optimized for Apple Silicon (MPS) and CUDA."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1_re = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights1_im = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights2_re = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights2_im = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    @staticmethod
    def compl_mul2d(x_re, x_im, w_re, w_im):
        # Permute (B, I, X, Y) -> (X, Y, B, I) and (I, O, X, Y) -> (X, Y, I, O)
        xr = x_re.permute(2, 3, 0, 1)
        xi = x_im.permute(2, 3, 0, 1)
        wr = w_re.permute(2, 3, 0, 1)
        wi = w_im.permute(2, 3, 0, 1)

        out_re = (xr @ wr - xi @ wi).permute(2, 3, 0, 1)
        out_im = (xr @ wi + xi @ wr).permute(2, 3, 0, 1)
        return torch.complex(out_re, out_im)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        x1 = x_ft[:, :, :m1, :m2]
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x1.real, x1.imag, self.weights1_re[:, :, :m1, :m2], self.weights1_im[:, :, :m1, :m2]
        )

        x2 = x_ft[:, :, -m1:, :m2]
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x2.real, x2.imag, self.weights2_re[:, :, :m1, :m2], self.weights2_im[:, :, :m1, :m2]
        )

        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    """2D Fourier Neural Operator mapping (B, 1, nx, ny) -> (B, 2, nx, ny)."""
    def __init__(self, in_ch=1, out_ch=2, width=48, modes=20, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch + 2, width, kernel_size=1)

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)
        ])
        self.pointwise_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.InstanceNorm2d(width, affine=True) for _ in range(n_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_ch, kernel_size=1),
        )

    def forward(self, x):
        x = append_coords(x)
        x = self.lift(x)
        for spec, pw, norm in zip(self.spectral_layers, self.pointwise_layers, self.norms):
            x = F.gelu(norm(spec(x) + pw(x)))
        return self.proj(x)


# 2D U-Net Architecture
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class UNet2d(nn.Module):
    """4-level Residual U-Net mapping (B, 1, nx, ny) -> (B, 2, nx, ny)."""
    def __init__(self, in_ch=1, out_ch=2, base_ch=48):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.enc1 = ConvBlock(in_ch + 2, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x):
        x = append_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_model(args):
    """Factory function to instantiate FNO or U-Net from args."""
    d = vars(args) if not isinstance(args, dict) else args
    arch = d["arch"].lower()
    if arch == "fno":
        return FNO2d(
            in_ch=1,
            out_ch=2,
            width=d["base_ch"],
            modes=d["fno_modes"],
            n_layers=d["fno_layers"],
        )
    elif arch == "unet":
        return UNet2d(
            in_ch=1,
            out_ch=2,
            base_ch=d["base_ch"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")