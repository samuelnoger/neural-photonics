import torch
import torch.nn.functional as F

def field_loss(pred, target, grad_weight=0.2):
    """MSE loss on field values + finite-difference spatial gradients (dE/dx, dE/dy)."""
    mse = F.mse_loss(pred, target)
    if grad_weight <= 0.0:
        return mse

    dx_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_targ = target[:, :, 1:, :] - target[:, :, :-1, :]
    dy_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_targ = target[:, :, :, 1:] - target[:, :, :, :-1]

    grad_mse = F.mse_loss(dx_pred, dx_targ) + F.mse_loss(dy_pred, dy_targ)
    return mse + grad_weight * grad_mse


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