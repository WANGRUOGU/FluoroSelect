# metrics.py
import numpy as np


def compute_true_class_rmse(Atrue, Ahat, eps=1e-8):
    """Per-dye RMSE on pixels truly containing that dye, excluding background."""
    if Atrue.shape != Ahat.shape:
        raise ValueError("True and estimated abundance arrays must have the same shape.")
    result = []
    for r in range(Atrue.shape[-1]):
        mask = Atrue[..., r] > eps
        result.append(float(np.sqrt(np.mean((Ahat[..., r][mask] - Atrue[..., r][mask]) ** 2)))
                      if np.any(mask) else np.nan)
    return result


def summarize_performance(Atrue, Ahat, predicted):
    """Paper endpoints, computed within one image replicate."""
    accuracy = np.asarray(compute_classification_accuracy(Atrue, predicted))
    rmse = np.asarray(compute_true_class_rmse(Atrue, Ahat))
    counts = np.sum(Atrue > 1e-8, axis=tuple(range(Atrue.ndim - 1)))
    present = counts > 0
    return dict(accuracy=accuracy, rmse=rmse,
                macro_accuracy=float(np.mean(accuracy[present])),
                worst_accuracy=float(np.min(accuracy[present])),
                true_class_rmse=float(np.sqrt(np.average(rmse[present] ** 2, weights=counts[present]))),
                worst_class_rmse=float(np.max(rmse[present])))

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
