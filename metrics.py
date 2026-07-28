# metrics.py
import numpy as np

def compute_classification_accuracy(Atrue, predicted_labels, eps=1e-8):
    """Return per-fluorophore accuracy from the spectral-angle class labels."""
    _, _, R = Atrue.shape
    predicted = np.asarray(predicted_labels).reshape(-1)
    T_true = Atrue.reshape(-1, R)
    acc_vals = []

    for r in range(R):
        mask_true = T_true[:, r] > eps
        if not np.any(mask_true):
            acc_vals.append(np.nan)
            continue
        acc_vals.append(float(np.mean(predicted[mask_true] == r)))

    return acc_vals
