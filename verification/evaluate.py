import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from train.arguments import get_args
from train.train import resolve_device
from data.data import PhotonicsDataset
from data.solver import HelmholtzSolver2D
from model.model import build_model


@torch.no_grad()
def verify_and_plot(args):
    device = resolve_device(args.device)
    ckpt_path = os.path.join(args.ckpt_dir, f"best_{args.arch}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", vars(args))
    history = ckpt["history"]
    field_std = ckpt["field_std"]

    # Load validation dataset and solver
    val_path = os.path.join(args.save_dir, "val.pt")
    val_ds = PhotonicsDataset(val_path)
    solver = HelmholtzSolver2D.from_args(saved_cfg)

    # Rebuild and load trained model
    model = build_model(saved_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    os.makedirs(args.results_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Training vs. Validation Loss Curve (Overfitting Diagnostic)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE + Grad)")
    ax1.set_title("Training vs. Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_rel_l2"], color="tab:orange", lw=2)
    best_ep = int(np.argmin(history["val_rel_l2"])) + 1
    best_val = min(history["val_rel_l2"])
    ax2.scatter([best_ep], [best_val], color="red", zorder=5, label=f"Best: {best_val:.1f}% (Ep {best_ep})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Relative L2 Error (%)")
    ax2.set_title("Validation Relative L2 Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(args.results_dir, f"{args.arch}_training_curves.png")
    plt.savefig(curve_path, dpi=180)
    plt.close()
    print(f"Saved training curves to '{curve_path}' (Best Val Rel L2 = {best_val:.2f}% at Epoch {best_ep}).")

    # -------------------------------------------------------------------------
    # Visual Field Comparison on 3 Unseen Validation Samples
    # -------------------------------------------------------------------------
    sample_indices = [0, 1, 2]
    fig, axes = plt.subplots(len(sample_indices), 5, figsize=(16, 3.2 * len(sample_indices)))

    for row, idx in enumerate(sample_indices):
        eps_t, true_f = val_ds[idx]
        pred_f = model(eps_t.unsqueeze(0).to(device)).squeeze(0).cpu()

        # Un-standardize back to physical electric field units
        Ez_true = (true_f[0].numpy() + 1j * true_f[1].numpy()) * field_std
        Ez_pred = (pred_f[0].numpy() + 1j * pred_f[1].numpy()) * field_std

        rel_l2 = np.linalg.norm(Ez_pred - Ez_true) / (np.linalg.norm(Ez_true) + 1e-8) * 100.0
        t1_true, t2_true = solver.measure_port_powers(Ez_true)
        t1_pred, t2_pred = solver.measure_port_powers(Ez_pred)

        vmax_re = np.percentile(np.abs(Ez_true.real), 99)
        vmax_int = np.percentile(np.abs(Ez_true) ** 2, 99)

        # Col 0: Silicon Geometry
        axes[row, 0].imshow(eps_t[0].numpy().T, cmap="Greys", origin="lower")
        axes[row, 0].set_title(f"Val #{idx}: Silicon Geometry")

        # Col 1: True Re(Ez)
        axes[row, 1].imshow(Ez_true.real.T, cmap="RdBu_r", vmin=-vmax_re, vmax=vmax_re, origin="lower")
        axes[row, 1].set_title(f"True Re(Ez)\nP1={t1_true*100:.1f}%, P2={t2_true*100:.1f}%")

        # Col 2: Predicted Re(Ez)
        axes[row, 2].imshow(Ez_pred.real.T, cmap="RdBu_r", vmin=-vmax_re, vmax=vmax_re, origin="lower")
        axes[row, 2].set_title(f"Pred Re(Ez) (Err: {rel_l2:.1f}%)\nP1={t1_pred*100:.1f}%, P2={t2_pred*100:.1f}%")

        # Col 3: True Intensity |Ez|^2
        axes[row, 3].imshow((np.abs(Ez_true) ** 2).T, cmap="magma", vmin=0, vmax=vmax_int, origin="lower")
        axes[row, 3].set_title("True Intensity |Ez|²")

        # Col 4: Predicted Intensity |Ez|^2
        axes[row, 4].imshow((np.abs(Ez_pred) ** 2).T, cmap="magma", vmin=0, vmax=vmax_int, origin="lower")
        axes[row, 4].set_title("Pred Intensity |Ez|²")

        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    comp_path = os.path.join(args.results_dir, f"{args.arch}_val_comparison.png")
    plt.savefig(comp_path, dpi=180)
    plt.close()
    print(f"Saved validation wave comparison plot to '{comp_path}'.")


if __name__ == "__main__":
    args = get_args()
    verify_and_plot(args)