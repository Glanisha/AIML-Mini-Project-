"""R4: fine-tune CNNs with 5-fold grouped cross-validation on the development set.

Outputs per architecture (ml/artifacts/<arch>/):
  fold<k>.pt          best-validation-loss checkpoint of fold k
  oof.npz             out-of-fold logits for every dev image (used for calibration/thresholds)
  test_logits.npy     (folds, n_test, C) logits, for the architecture comparison table only
  history.json        per-epoch train/val curves
The test set is never used for any training or selection decision.
"""

import argparse
import copy
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .common import ARTIFACTS, N_FOLDS, SEED, save_json, seed_everything
from .data import TrainDataset, eval_tensor, load_manifest
from .models import CamNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(x[i : i + batch].to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def train_fold(arch, train_df, val_x, val_y, epochs, lr, batch_size, seed):
    seed_everything(seed)
    model = CamNet(arch).to(DEVICE)
    loader = DataLoader(
        TrainDataset(train_df.path.tolist(), train_df.y.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    params = [
        {"params": model.features.parameters(), "lr": lr},
        {"params": model.fc.parameters(), "lr": lr * 10},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 10], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    best = {"loss": float("inf")}
    history = []
    val_y_t = torch.as_tensor(val_y)
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(yb)
            n += len(yb)

        logits = predict_logits(model, val_x)
        val_loss = float(loss_fn(torch.from_numpy(logits), val_y_t))
        probs = torch.softmax(torch.from_numpy(logits), 1)[:, 1].numpy()
        val_auc = float(roc_auc_score(val_y, probs))
        val_acc = float(((probs >= 0.5) == val_y).mean())
        history.append({"epoch": epoch + 1, "train_loss": running / n, "val_loss": val_loss, "val_auc": val_auc, "val_acc": val_acc})
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": epoch + 1, "state": copy.deepcopy(model.state_dict()), "logits": logits}
    return best, history


def run(arch: str, epochs: int = 30, lr: float = 2e-4, batch_size: int = 24) -> dict:
    df = load_manifest()
    dev = df[df.split == "dev"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    out_dir = ARTIFACTS / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    test_x = eval_tensor(test.path.tolist())
    oof = np.zeros((len(dev), 2), dtype=np.float32)
    test_logits, histories, summary = [], {}, []
    for k in range(N_FOLDS):
        t0 = time.time()
        tr, va = dev[dev.fold != k], dev[dev.fold == k]
        val_x = eval_tensor(va.path.tolist())
        best, hist = train_fold(arch, tr, val_x, va.y.to_numpy(), epochs, lr, batch_size, SEED + k)
        torch.save({"arch": arch, "state_dict": best["state"], "epoch": best["epoch"]}, out_dir / f"fold{k}.pt")
        oof[va.index] = best["logits"]

        model = CamNet(arch, pretrained=False).to(DEVICE)
        model.load_state_dict(best["state"])
        test_logits.append(predict_logits(model, test_x))
        histories[f"fold{k}"] = hist
        h = hist[best["epoch"] - 1]
        summary.append({"fold": k, "best_epoch": best["epoch"], "val_loss": h["val_loss"], "val_auc": h["val_auc"], "seconds": round(time.time() - t0, 1)})
        print(f"[{arch}] fold {k}: best epoch {best['epoch']} val_loss {h['val_loss']:.3f} val_auc {h['val_auc']:.3f} ({time.time() - t0:.0f}s)")

    np.savez(out_dir / "oof.npz", logits=oof, y=dev.y.to_numpy(), path=dev.path.to_numpy(), fold=dev.fold.to_numpy())
    np.save(out_dir / "test_logits.npy", np.stack(test_logits))
    oof_auc = float(roc_auc_score(dev.y, torch.softmax(torch.from_numpy(oof), 1)[:, 1].numpy()))
    result = {"arch": arch, "epochs": epochs, "lr": lr, "batch_size": batch_size, "folds": summary, "oof_auc": oof_auc}
    save_json({"summary": result, "history": histories}, out_dir / "history.json")
    print(f"[{arch}] out-of-fold AUC {oof_auc:.3f}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    run(a.arch, a.epochs)
