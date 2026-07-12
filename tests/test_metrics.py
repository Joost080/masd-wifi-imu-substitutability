import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import weighted_accuracy, per_class_f1, get_confusion_matrix, macro_f1


def test_perfect_accuracy():
    y = np.arange(27)
    assert weighted_accuracy(y, y) == 1.0


def test_zero_accuracy():
    y_true = np.zeros(27, dtype=int)
    y_pred = np.ones(27, dtype=int)
    assert weighted_accuracy(y_true, y_pred) == 0.0


def test_per_class_f1_shape():
    y_true = np.random.randint(0, 27, 200)
    y_pred = np.random.randint(0, 27, 200)
    f1 = per_class_f1(y_true, y_pred)
    assert f1.shape == (27,)
    assert (f1 >= 0).all() and (f1 <= 1).all()


def test_confusion_matrix_shape():
    y_true = np.random.randint(0, 27, 200)
    y_pred = np.random.randint(0, 27, 200)
    cm = get_confusion_matrix(y_true, y_pred)
    assert cm.shape == (27, 27)


def test_confusion_matrix_diagonal_on_perfect_preds():
    y = np.repeat(np.arange(27), 10)
    cm = get_confusion_matrix(y, y)
    assert (cm == np.diag(np.diag(cm))).all()
