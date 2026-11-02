import os
import time
import torch
from torch.utils.data import DataLoader

from train.arguments import get_args
from data.data import PhotonicsDataset, generate_dataset
from model.model import build_model
from train.losses import field_loss, evaluate
from tqdm import tqdm


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train(args):
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_path = os.path.join(args.save_dir, "train.pt")
    val_path = os.path.join(args.save_dir, "val.pt")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print(f"Dataset not found in '{args.save_dir}/'. Generating now...")
        generate_dataset(args)

    train_ds = PhotonicsDataset(train_path)
    val_ds = PhotonicsDataset(val_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {args.arch.upper()} ({n_params:,} params) on {device} | Train: {len(train_ds)}, Val: {len(val_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.ckpt_dir, f"best_{args.arch}.pt")
    best_rel_l2 = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_rel_l2": []}

    epochs = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch", ncols=120)

    for epoch in epochs:
        model.train()
        running_loss = torch.zeros(1, device=device)
        n_seen = 0

        for eps, fields in train_loader:
            eps, fields = eps.to(device), fields.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(eps)
            loss = field_loss(pred, fields, args.grad_loss_weight)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach() * eps.size(0)
            n_seen += eps.size(0)

        scheduler.step()
        train_loss = (running_loss / n_seen).item()
        val_loss, val_rel_l2 = evaluate(model, val_loader, device, args.grad_loss_weight)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rel_l2"].append(val_rel_l2)

        if val_rel_l2 < best_rel_l2:
            best_rel_l2 = val_rel_l2
            torch.save({
                "model_state": model.state_dict(),
                "field_std": train_ds.field_std,
                "config": vars(args),
                "best_val_rel_l2": best_rel_l2,
                "history": history,
            }, best_ckpt_path)

        # Update the right side of the progress bar with formatted metrics
        epochs.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            rel_l2=f"{val_rel_l2:.1f}%",
            best=f"{best_rel_l2:.1f}%",
        )

    print(f"\nTraining complete! Best Val Relative L2 Error: {best_rel_l2:.2f}% saved to '{best_ckpt_path}'.")


if __name__ == "__main__":
    args = get_args()
    train(args)