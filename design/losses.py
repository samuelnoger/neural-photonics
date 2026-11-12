import torch
import torch.nn.functional as F


def measure_ports_torch(pred_field, solver):
    """
    Vectorized mode-overlap measurement across all N ports at once.
    Both `pred_field` and `solver.mode_ports_torch` are on `self.device`.
    """
    x_m = solver.x_meas
    modes = solver.mode_ports_torch  # shape: (n_ports, ny)

    ez_re = pred_field[0, 0, x_m, :]  # shape: (ny,)
    ez_im = pred_field[0, 1, x_m, :]  # shape: (ny,)

    c_re = torch.mv(modes, ez_re)
    c_im = torch.mv(modes, ez_im)
    return (c_re ** 2 + c_im ** 2) / float(solver.ref_power)  # shape: (n_ports,)


def reg_loss(rho, bin_weight=0.05, tv_weight=0.015):
    """Binarization + Total Variation (perimeter) + L1 void-clearing regularization."""
    bin_loss = (4.0 * rho * (1.0 - rho)).mean()
    tv_x = torch.abs(rho[1:, :] - rho[:-1, :]).mean()
    tv_y = torch.abs(rho[:, 1:] - rho[:, :-1]).mean()
    return bin_weight * bin_loss + tv_weight * (tv_x + tv_y) + 0.005 * rho.mean()


def splitter_loss(pred_field, rho, solver, targets, total_target=0.96, bin_weight=0.05, tv_weight=0.015):
    """Loss function for 1x2 and 1x3 power splitters."""
    powers = measure_ports_torch(pred_field, solver)
    goals = total_target * targets

    ratio_loss = ((powers - goals) ** 2).sum()
    throughput_loss = F.relu(total_target - powers.sum()) ** 2
    loss = ratio_loss + 0.5 * throughput_loss + reg_loss(rho, bin_weight, tv_weight)
    return loss, powers


def wdm_loss(pred_field, rho, solver, targets, total_target=0.96, bin_weight=0.05, tv_weight=0.015, crosstalk_weight=3.0):
    """
    Loss function for WDM demultiplexers:
    Maximizes pass-port transmission and applies quadratic + linear/log extinction
    on the isolation port so crosstalk continues to be pushed down below 1-2%.
    """
    powers = measure_ports_torch(pred_field, solver)
    goals = total_target * targets

    pass_mask = (targets > 0.5).float()
    isolation_mask = 1.0 - pass_mask

    pass_loss = (pass_mask * (powers - goals) ** 2).sum()
    p_xt = (isolation_mask * powers).sum()
    crosstalk_loss = crosstalk_weight * (p_xt ** 2 + 0.15 * p_xt) + 0.015 * torch.log(p_xt + 1e-3)
    throughput_loss = F.relu(total_target - (pass_mask * powers).sum()) ** 2

    loss = pass_loss + crosstalk_loss + 0.5 * throughput_loss + reg_loss(rho, bin_weight, tv_weight)
    return loss, powers