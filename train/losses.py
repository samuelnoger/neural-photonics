import torch
import torch.nn.functional as F

def field_loss(pred, target, grad_weight=0.2, port_weight=0.5, x_meas=100):
    # 1. Direct Relative L2 loss per sample (matches evaluation metric!)
    diff_norm = torch.linalg.vector_norm(pred - target, dim=(1, 2, 3))
    true_norm = torch.linalg.vector_norm(target, dim=(1, 2, 3)) + 1e-8
    rel_l2 = (diff_norm / true_norm).mean()

    # Port measurement slice supervision (x_meas - 1 : x_meas + 2)
    port_diff = torch.linalg.vector_norm(
        pred[:, :, x_meas - 1 : x_meas + 2, :] - target[:, :, x_meas - 1 : x_meas + 2, :],
        dim=(1, 2, 3),
    )
    port_true = torch.linalg.vector_norm(
        target[:, :, x_meas - 1 : x_meas + 2, :], dim=(1, 2, 3)
    ) + 1e-8
    port_loss = (port_diff / port_true).mean()

    total = rel_l2 + port_weight * port_loss
    if grad_weight <= 0.0:
        return total

    dx_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_targ = target[:, :, 1:, :] - target[:, :, :-1, :]
    dy_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_targ = target[:, :, :, 1:] - target[:, :, :, :-1]

    grad_l2 = (
        torch.linalg.vector_norm(dx_pred - dx_targ, dim=(1, 2, 3)) / true_norm
        + torch.linalg.vector_norm(dy_pred - dy_targ, dim=(1, 2, 3)) / true_norm
    ).mean()

    return total + grad_weight * grad_l2


@torch.no_grad()
def evaluate(model, loader, device, grad_weight):
    model.eval()
    total_loss = 0.0
    total_rel_l2 = 0.0
    n_samples = 0

    for eps, fields in loader:
        eps, fields = eps.to(device), fields.to(device)
        pred = model(eps)

        loss = field_loss(pred, fields, grad_weight)
        B = eps.size(0)
        total_loss += loss.item() * B

        # Relative L2 error per sample: ||pred - true||_2 / ||true||_2
        diff_norm = torch.linalg.vector_norm(pred - fields, dim=(1, 2, 3))
        true_norm = torch.linalg.vector_norm(fields, dim=(1, 2, 3)) + 1e-8
        total_rel_l2 += (diff_norm / true_norm).sum().item()
        n_samples += B

    return total_loss / n_samples, (total_rel_l2 / n_samples) * 100.0