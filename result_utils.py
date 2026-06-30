# result_utils.py
import numpy as np
import pandas as pd
import streamlit as st

from data_helpers import fluor_from_label
from utils import similarity_matrix


def select_worst_group(
    E_norm,
    labels,
    idx_groups,
    required_count,
    use_pool,
    similarity_metric="Cosine similarity",
):
    """Simple contrast selector: intentionally picks high-similarity candidates."""
    if E_norm.size == 0:
        return []

    S = similarity_matrix(E_norm, metric=similarity_metric)
    max_sim = np.max(S, axis=1)

    if use_pool:
        if required_count is None:
            required_count = E_norm.shape[1]
        order = np.argsort(-max_sim)
        return [int(i) for i in order[: int(required_count)]]

    worst = []
    used_dyes = set()

    for idxs in idx_groups:
        if not idxs:
            continue

        best_j = None
        best_score = -1.0

        for j in idxs:
            dye = fluor_from_label(labels[j])
            if dye in used_dyes:
                continue

            score = max_sim[j]
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is None:
            best_j = max(idxs, key=lambda j: max_sim[j])

        worst.append(int(best_j))
        used_dyes.add(fluor_from_label(labels[best_j]))

    return worst


def select_worst_group_constrained(
    E_norm,
    labels,
    idx_groups,
    required_count,
    use_pool,
    fixed_fluorophores=None,
    allowed_fluorophores=None,
    similarity_metric="Cosine similarity",
):
    """Worst-group heuristic that mirrors pool constraints where relevant."""
    if not use_pool:
        return select_worst_group(
            E_norm,
            labels,
            idx_groups,
            required_count,
            use_pool,
            similarity_metric=similarity_metric,
        )

    fixed_set = set(fixed_fluorophores or [])
    allowed_set = set(allowed_fluorophores or [])

    fixed_idx = [j for j, lab in enumerate(labels) if fluor_from_label(lab) in fixed_set]
    allowed_idx = [j for j, lab in enumerate(labels) if fluor_from_label(lab) in allowed_set]
    allowed_idx = [j for j in allowed_idx if j not in fixed_idx]

    n_total = int(required_count or 0)
    n_add = max(0, n_total - len(fixed_idx))

    if n_add == 0 or E_norm.size == 0:
        return fixed_idx[:n_total]

    S = similarity_matrix(E_norm, metric=similarity_metric)
    max_sim = np.max(S, axis=1)
    order = sorted(allowed_idx, key=lambda j: -max_sim[j])

    return fixed_idx + order[:n_add]


def render_metrics_table(names, rmse_vals, prop_vals, acc_vals):
    """Render a compact 3 x N metrics table."""
    rows = []

    for measurement, values in [
        ("RMSE", rmse_vals),
        ("Proportion", prop_vals),
        ("Accuracy", acc_vals),
    ]:
        row = {"Measurement": measurement}
        for name, value in zip(names, values):
            try:
                row[name] = f"{float(value):.4f}"
            except Exception:
                row[name] = ""
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def build_result_context(
    run_id,
    mode,
    source_mode,
    laser_strategy,
    laser_list,
    spec_res_mode,
    use_pool,
    fixed_probe_pairs,
    fixed_fluorophores,
    allowed_fluorophores,
    selected_labels,
    tops,
    names,
    rmse_vals,
    prop_vals,
    acc_vals,
    pair_formatter,
    similarity_metric="Cosine similarity",
):
    return {
        "run_id": run_id,
        "mode": mode,
        "selection_source": source_mode,
        "laser_strategy": laser_strategy,
        "lasers": laser_list,
        "spectral_resolution": spec_res_mode,
        "similarity_metric": similarity_metric,
        "use_pool": use_pool,
        "fixed_probe_fluorophore_pairs": fixed_probe_pairs,
        "fixed_fluorophores": fixed_fluorophores,
        "allowed_fluorophores": allowed_fluorophores,
        "selected_labels": selected_labels,
        "selected_fluorophores": [fluor_from_label(x) for x in selected_labels],
        "top_pairwise_similarities": [
            {
                "similarity": float(val),
                "score": float(val),
                "label_1": a,
                "label_2": b,
                "fluorophore_pair": pair_formatter(a, b),
            }
            for val, a, b in tops
        ],
        "metrics": {
            "names": names,
            "rmse": [float(x) for x in rmse_vals],
            "proportion": [float(x) for x in prop_vals],
            "accuracy": [float(x) for x in acc_vals],
        },
    }
