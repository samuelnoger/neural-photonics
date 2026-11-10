import torch
import torch.nn.functional as F


class SplitterLoss(torch.nn.Module):
    def __init__(self, solver, field_std, device):
        super().__init__()
        self.x_meas = solver.x_meas
        self.field_std = float(field_std)
        self.p_ref = float(solver.ref_power)
        self.modes = [torch.from_numpy(m).float().to(device) for m in solver.geometry.mode_ports]

    def measure_ports_torch(self, pred_field):
        ez_re = pred_field[0, 0, self.x_meas, :] * self.field_std
        ez_im = pred_field[0, 1, self.x_meas, :] * self.field_std
        powers = []
        for m in self.modes:
            c_re = torch.dot(ez_re, m)
            c_im = torch.dot(ez_im, m)
            powers.append((c_re ** 2 + c_im ** 2) / self.p_ref)
        return torch.stack(powers)  # shape: (n_ports,)

    def forward(self, pred_field, rho, target_ratios, total_target=0.96, bin_weight=0.05, tv_weight=0.015):
        powers = self.measure_ports_torch(pred_field)
        targets = torch.as_tensor(target_ratios, dtype=powers.dtype, device=powers.device)
        targets = targets / targets.sum()

        goals = total_target * targets
        ratio_loss = ((powers - goals) ** 2).sum()
        throughput_loss = F.relu(total_target - powers.sum()) ** 2
        bin_loss = (4.0 * rho * (1.0 - rho)).mean()

        # Total Variation (perimeter) penalty + tiny L1 void-clearing penalty
        tv_x = torch.abs(rho[1:, :] - rho[:-1, :]).mean()
        tv_y = torch.abs(rho[:, 1:] - rho[:, :-1]).mean()
        reg_loss = tv_weight * (tv_x + tv_y) + 0.005 * rho.mean()

        loss = ratio_loss + 0.5 * throughput_loss + bin_weight * bin_loss + reg_loss
        return loss, powers