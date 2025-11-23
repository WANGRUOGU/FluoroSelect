# metrics.py
import numpy as np

def compute_prop_and_accuracy(Atrue, Ahat, eps=1e-8):
    """
    Compute per-fluorophore:
      - proportion: mean_r( A_r / sum_k A_k ) over pixels where the true fluor is present
      - accuracy:  fraction of those pixels where this fluor has the largest estimated abundance

    Atrue, Ahat: (H, W, R)
    Returns:
      prop_vals: list length R
      acc_vals:  list length R
    """
    H, W, R = Atrue.shape
    T_true = Atrue.reshape(-1, R)
    T_hat = Ahat.reshape(-1, R)

    prop_vals = []
    acc_vals = []

    for r in range(R):
        # Pixels where this fluor has non-zero true abundance
        mask_true = T_true[:, r] > eps
        if not np.any(mask_true):
            prop_vals.append(np.nan)
            acc_vals.append(np.nan)
            continue

        Ah = T_hat[mask_true, :]  # (N_r, R)
        Ar = Ah[:, r]
        sums = Ah.sum(axis=1)

        # Avoid division by zero: keep only pixels where sum_k A_k > 0
        valid = sums > eps
        if not np.any(valid):
            prop_vals.append(np.nan)
            acc_vals.append(np.nan)
            continue

        # proportion = mean( A_r / sum_k A_k )
        prop_vals.append(float(np.mean(Ar[valid] / sums[valid])))

        # accuracy = mean( A_r is the argmax over k )
        winners = np.argmax(Ah[valid, :], axis=1)
        acc_vals.append(float(np.mean(winners == r)))

    return prop_vals, acc_vals
