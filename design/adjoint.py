import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


class _HelmholtzAdjointFn(torch.autograd.Function):
    """
    Exact differentiable 2D FDFD Helmholtz solver using the Adjoint Method.
    Reuses the forward sparse LU factorization in the backward pass.
    """
    @staticmethod
    def forward(ctx, eps_norm, solver, field_std):
        # eps_norm shape: (1, 1, nx, ny) -> convert to physical permittivity eps_r
        eps_norm_np = eps_norm.detach().cpu().numpy().squeeze()
        eps_r = solver.eps_bg + eps_norm_np * (solver.eps_si - solver.eps_bg)

        # Build complex Helmholtz operator A = L + k0^2 * diag(eps_r) and factorize ONCE
        k2_eps = sp.diags((solver.k0 ** 2) * eps_r.ravel(), 0)
        A = (solver.L + k2_eps).tocsc()
        lu = spla.splu(A)

        # Forward solve: A * Ez = b
        Ez_flat = lu.solve(solver.source.ravel())
        Ez = Ez_flat.reshape(solver.nx, solver.ny)

        # Save LU factorization and forward wavefield for the backward adjoint pass
        ctx.lu = lu
        ctx.Ez = Ez
        ctx.solver = solver
        ctx.field_std = field_std
        ctx.device = eps_norm.device

        # Return normalized 2-channel tensor (1, 2, nx, ny) to match SplitterLoss
        out_np = np.stack([Ez.real, Ez.imag], axis=0) / field_std
        return torch.from_numpy(out_np.astype(np.float32)).unsqueeze(0).to(eps_norm.device)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output shape: (1, 2, nx, ny) -> upstream dL/d(Ez_norm)
        grad_np = grad_output.detach().cpu().numpy().squeeze(0) / ctx.field_std
        g = (grad_np[0] + 1j * grad_np[1]).ravel()

        # Adjoint solve: A^T * lambda_adj = conj(g)  (reuses LU factors -> ~2 ms!)
        lam_flat = ctx.lu.solve(np.conj(g), trans="T")
        lam = lam_flat.reshape(ctx.solver.nx, ctx.solver.ny)

        # Exact physics gradient: dL/d(eps_norm) = -k0^2 * (eps_si - eps_bg) * Re(lam * Ez)
        scale = (ctx.solver.k0 ** 2) * (ctx.solver.eps_si - ctx.solver.eps_bg)
        grad_eps_norm = -scale * np.real(lam * ctx.Ez)

        grad_tensor = (
            torch.from_numpy(grad_eps_norm.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(ctx.device)
        )
        return grad_tensor, None, None


def solve_fdfd_differentiable(eps_norm, solver, field_std=1.0):
    return _HelmholtzAdjointFn.apply(eps_norm, solver, field_std)