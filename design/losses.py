import torch
import torch.nn.functional as F


class SplitterLoss(torch.nn.Module):
    def __init__(self, solver, field_std, device):
        super().__init__()
        self.x_meas = solver.x_meas
        self.field_std = float(field_std)
        self.p_ref = float(solver.ref_power)

        # Directly reuse the normalized mode profiles from HelmholtzSolver2D
        self.m1 = torch.from_numpy(solver.mode_p1).float().to(device)
        self.m2 = torch.from_numpy(solver.mode_p2).float().to(device)

    def measure_ports_torch(self, pred_field):
        """
        Differentiable equivalent of solver.measure_port_powers(Ez).
        pred_field: (1, 2, nx, ny) standardized real/imag field from surrogate.
        Returns (T1, T2) as differentiable scalar tensors.
        """
        # Un-standardize back to physical Ez units at the measurement slice x_meas
        ez_re = pred_field[0, 0, self.x_meas, :] * self.field_std
        ez_im = pred_field[0, 1, self.x_meas, :] * self.field_std

        # Complex overlap integral |(Ez, m_k)|^2 = (Re(Ez).m_k)^2 + (Im(Ez).m_k)^2
        c1_re = torch.dot(ez_re, self.m1)
        c1_im = torch.dot(ez_im, self.m1)
        p1 = (c1_re ** 2 + c1_im ** 2) / self.p_ref

        c2_re = torch.dot(ez_re, self.m2)
        c2_im = torch.dot(ez_im, self.m2)
        p2 = (c2_re ** 2 + c2_im ** 2) / self.p_ref

        return p1, p2

    def forward(self, pred_field, rho, target_ratio=0.5, total_target=0.85, bin_weight=0.05):
        t1, t2 = self.measure_ports_torch(pred_field)

        t1_goal = total_target * target_ratio
        t2_goal = total_target * (1.0 - target_ratio)

        # 1. Port transmission matching loss + maximize total throughput
        ratio_loss = (t1 - t1_goal) ** 2 + (t2 - t2_goal) ** 2
        throughput_loss = F.relu(total_target - (t1 + t2)) ** 2

        # 2. Binarization penalty: 4 * rho * (1 - rho) is 1 at rho=0.5 and 0 at rho={0, 1}
        bin_loss = (4.0 * rho * (1.0 - rho)).mean()

        loss = ratio_loss + 0.5 * throughput_loss + bin_weight * bin_loss
        return loss, t1, t2