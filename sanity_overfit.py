"""
Sanity check: can the WiFi DeepConvLSTM overfit a tiny subset?

If a model with no dropout cannot reach near-100% train accuracy on ~200 samples
in a few hundred epochs, the data pipeline (labels, shapes, normalization) is
the problem, not the architecture or hyperparameters.

Usage:
    python sanity_overfit.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import WiFiDataset
from src.models.deepconvlstm import DeepConvLSTM


N_SAMPLES = 200
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    full = WiFiDataset(split="train")
    labels = np.asarray(full.y)
    classes = np.unique(labels)
    per_class = max(1, N_SAMPLES // len(classes))
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([
        rng.choice(np.where(labels == c)[0], size=per_class, replace=False)
        for c in classes
    ])
    rng.shuffle(idx)
    subset = Subset(full, idx.tolist())
    print(f"Subset: {len(subset)} samples across {len(classes)} classes "
          f"(~{per_class}/class)")

    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = DeepConvLSTM(in_channels=224, num_classes=27, dropout=0.0).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        avg_loss = total_loss / total
        acc = correct / total
        best_acc = max(best_acc, acc)
        if epoch <= 10 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | acc={acc:.4f}")
        if acc >= 0.99:
            print(f"Reached >=99% train accuracy at epoch {epoch}. Stopping.")
            break

    print(f"\nBest train accuracy: {best_acc:.4f}")
    if best_acc >= 0.95:
        print("PASS — model can overfit. Pipeline is fine; issue is architecture/regularization.")
    else:
        print("FAIL — model cannot overfit a tiny subset. Investigate labels, shapes, normalization.")


if __name__ == "__main__":
    main()
