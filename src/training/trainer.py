import csv
import inspect
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Generic supervised trainer with optional fusion-side hooks.

    Two optional behaviours, off by default for backwards compatibility:

    - moddrop_p (Track 2A): during training, with probability moddrop_p, a
      single modality (uniformly chosen) is zeroed in the input batch. Only
      applies when the batch carries two inputs (fusion models). Standard
      ModDrop / OPM-style regularisation: forces the network not to depend
      exclusively on one modality at training time. Has no effect when
      moddrop_p <= 0 or for single-modality batches.

    - track_gate (Track 2B): after each epoch, runs a no-grad forward pass on
      the validation loader with `return_gate=True` and logs the gate's
      mean / std / sigmoid-entropy / min / max to gate_trajectory.csv. Skipped
      if the model's forward does not accept a return_gate kwarg.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        exp_dir: Path,
        early_stop_patience: int = 10,
        moddrop_p: float = 0.0,
        track_gate: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.exp_dir = Path(exp_dir)
        self.patience = early_stop_patience
        self.moddrop_p = float(moddrop_p)

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.exp_dir / "training_log.csv"
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        self._track_gate = bool(track_gate) and self._model_supports_gate()
        self._gate_log_path = self.exp_dir / "gate_trajectory.csv"

    # ------------------------------------------------------------------
    # train / eval loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int):
        with open(self._log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        if self._track_gate:
            with open(self._gate_log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "gate_mean", "gate_std", "gate_entropy",
                     "gate_min", "gate_max", "n_samples"]
                )

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss, val_acc = self._eval_epoch()
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | {elapsed:.1f}s"
            )

            with open(self._log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc])

            if self._track_gate:
                self._log_gate_stats(epoch)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                torch.save(self.model.state_dict(), self.exp_dir / "best_model.pt")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            loss = self._forward(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _eval_epoch(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss, preds, labels = self._forward(batch, return_preds=True)
                total_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(self.val_loader), correct / total

    def _forward(self, batch, return_preds: bool = False):
        if len(batch) == 3:
            x1, x2, y = [t.to(self.device) for t in batch]
            x1, x2 = self._maybe_moddrop(x1, x2)
            logits = self.model(x1, x2)
        else:
            x, y = [t.to(self.device) for t in batch]
            logits = self.model(x)
        loss = self.criterion(logits, y)
        if return_preds:
            return loss, logits.argmax(dim=1), y
        return loss

    # ------------------------------------------------------------------
    # modality dropout (Track 2A)
    # ------------------------------------------------------------------

    def _maybe_moddrop(self, x1: torch.Tensor, x2: torch.Tensor):
        """With prob moddrop_p, zero one modality (uniformly chosen)."""
        if not self.model.training or self.moddrop_p <= 0.0:
            return x1, x2
        if torch.rand(1).item() < self.moddrop_p:
            if torch.rand(1).item() < 0.5:
                x1 = torch.zeros_like(x1)
            else:
                x2 = torch.zeros_like(x2)
        return x1, x2

    # ------------------------------------------------------------------
    # gate trajectory logging (Track 2B)
    # ------------------------------------------------------------------

    def _model_supports_gate(self) -> bool:
        try:
            sig = inspect.signature(self.model.forward)
        except (TypeError, ValueError):
            return False
        return "return_gate" in sig.parameters

    def _log_gate_stats(self, epoch: int):
        """Run val pass with return_gate=True, log gate stats for this epoch."""
        self.model.eval()
        sums = torch.zeros(5, dtype=torch.float64)   # sum, sum_sq, sum_H, min_acc, max_acc
        n_total = 0
        gate_min = float("inf")
        gate_max = float("-inf")
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) != 3:
                    return  # not a fusion batch — bail
                x1, x2, _ = [t.to(self.device) for t in batch]
                _, g = self.model(x1, x2, return_gate=True)
                # g shape: (B, T, gate_dim) for raw GMU, (B, gate_dim) for stats GMU
                g_flat = g.reshape(-1).double()
                # Sigmoid entropy: -p log p - (1-p) log (1-p), elementwise
                eps = 1e-12
                p = g_flat.clamp(eps, 1.0 - eps)
                H = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
                sums[0] += g_flat.sum().item()
                sums[1] += (g_flat * g_flat).sum().item()
                sums[2] += H.sum().item()
                gate_min = min(gate_min, float(g_flat.min().item()))
                gate_max = max(gate_max, float(g_flat.max().item()))
                n_total += g_flat.numel()
        if n_total == 0:
            return
        mean = sums[0].item() / n_total
        var = sums[1].item() / n_total - mean * mean
        std = max(var, 0.0) ** 0.5
        ent = sums[2].item() / n_total
        with open(self._gate_log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{mean:.6f}", f"{std:.6f}", f"{ent:.6f}",
                 f"{gate_min:.6f}", f"{gate_max:.6f}", n_total]
            )
