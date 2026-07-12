import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def weighted_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)


def per_class_f1(y_true, y_pred, num_classes: int = 27) -> np.ndarray:
    return f1_score(
        y_true, y_pred,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def get_confusion_matrix(y_true, y_pred, num_classes: int = 27) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def print_report(y_true, y_pred) -> None:
    print(classification_report(y_true, y_pred, zero_division=0))
