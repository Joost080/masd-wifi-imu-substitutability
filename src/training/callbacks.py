class EarlyStopping:
    """Stateless patience counter — used standalone or by Trainer."""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self):
        self.best = float("inf")
        self.counter = 0
