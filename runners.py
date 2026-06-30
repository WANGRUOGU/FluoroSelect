# runners.py
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ai_ui import render_ai_result_panel
from config import DETECTION_CHANNELS
from data_helpers import (
    apply_mbs_zeroing,
    cached_build_effective_with_lasers,
    cached_interpolate_E_on_channels,
    fluor_from_label,
    sorted_order_by_peak,
)
from metrics import compute_prop_and_accuracy
from result_utils import (
    build_result_context,
    render_metrics_table,
    select_worst_group_constrained,
)
from sim_core import (
    argmax_labelmap,
    colorize_composite,
    simulate_rods_and_unmix,
    to_uint8_gray,
)
from ui_helpers import (
    ensure_colors,
    html_two_row_table,
    metric_header,
    pair_only_fluor,
    prettify_name,
    rgb01_to_plotly,
    show_bw_grid,
)
from utils import (
    build_emission_only_matrix,
    derive_powers_separate,
    derive_powers_simultaneous,
    similarity_matrix,
    solve_lexicographic_k,
    top_k_pairwise,
)


def _constraint_indices(
    labels,
    use_pool,
    fixed_fluorophores,
    allowed_fluorophores,
    fixed_probe_pairs,
):
    """
    Convert user constraints into column indices for the optimizer.
    """
    fixed_indices = []
    allowed_indices = None

    if use_pool:
        fixed_set = set(fixed_fluorophores or [])
        allowed_set = set(allowed_fluorophores or [])
        allowed_indices = []

        for j, label in enumerate(labels):
            fluor = fluor_from_label(label)
            if fluor in fixed_set:
                fixed_indices.append(j)
            if fluor in allowed_set:
                allowed_indices.append(j)

        return fixed_indices, allowed_indices

    fixed_pair_set = set(fixed_probe_pairs or [])

    for j, label in enumerate(labels):
        if label in fixed_pair_set:
            fixed_indices.append(j)

    return fixed_indices, None


def _render_selection_tables(use_pool, labels, sel_idx, worst_idx, predicted=False):
    """Render selected and worst-comparison tables."""
    if predicted:
        title = (
            "Selected fluorophores (with lasers, best)"
            if use_pool
            else "Selected probe–fluorophore pairs (with lasers, best)"
        )
    else:
        title = (
            "Selected fluorophores (best)"
            if use_pool
            else "Selected probe–fluorophore pairs (best)"
        )

    st.subheader(title)

    if use_pool:
        fluors = [fluor_from_label(labels[j]) for j in sel_idx]

        html_two_row_table(
            "Slot",
            "Fluorophore",
            [f"Slot {i + 1}" for i in range(len(fluors))],
            fluors,
        )

        worst_fluors = [fluor_from_label(labels[j]) for j in worst_idx]

        if worst_fluors:
            st.markdown("**Worst fluorophores (same count)**")
            html_two_row_table(
                "Slot",
                "Fluorophore",
                [f"Slot {i + 1}" for i in range(len(worst_fluors))],
                worst_fluors,
            )

    else:
        sel_pairs = [labels[j] for j in sel_idx]

        html_two_row_table(
            "Probe",
            "Fluorophore",
            [s.split(" – ", 1)[0] for s in sel_pairs],
            [s.split(" – ", 1)[1] for s in sel_pairs],
        )

        worst_pairs = [labels[j] for j in worst_idx]

        if worst_pairs:
            st.markdown("**Worst probe–fluorophore pairs (same count)**")
            html_two_row_table(
                "Probe",
                "Fluorophore",
                [s.split(" – ", 1)[0] for s in worst_pairs],
                [s.split(" – ", 1)[1] for s in worst_pairs],
            )


def _render_pairwise_table(
    E_norm,
    labels,
    k_show,
    similarity_metric="Cosine similarity",
):
    """Render top pairwise similarity/confusability scores."""
    S = similarity_matrix(E_norm, metric=similarity_metric)
    tops = top_k_pairwise(S, labels, k=k_show)

    st.subheader(f"Top pairwise scores ({similarity_metric})")

    html_two_row_table(
        "Pair",
        "Score",
        [pair_only_fluor(a, b) for _, a, b in tops],
        [val for val, _, _ in tops],
        color_second_row=True,
        color_thresh=0.9,
        fmt2=True,
    )

    return tops


def _render_spectra(x_axis, spectra, labels, colors, y_title, normalize_by=None):
    """Render spectra viewer."""
    st.subheader("Spectra viewer")

    fig = go.Figure()

    for t, label in enumerate(labels):
        y = spectra[:, t]

        if normalize_by is None:
            y = y / (np.max(y) + 1e-12)
        else:
            y = y / (normalize_by + 1e-12)

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y,
                mode="lines",
                name=label,
                line=dict(color=rgb01_to_plotly(colors[t]), width=2),
            )
        )

    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_title,
        yaxis=dict(range=[0, 1.05]),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_simulation_and_metrics(E_chan, colors, names):
    """
    Run synthetic rod simulation, render unmixing images, and compute metrics.

    Current noise model: the clean image is scaled to peak intensity 255, then
    Poisson shot noise is sampled.
    """
    Atrue, Ahat = simulate_rods_and_unmix(E_chan, rods_per=3)

    col_l, col_r = st.columns(2)

    true_rgb = (colorize_composite(Atrue, colors) * 255).astype(np.uint8)
    labelmap_rgb = argmax_labelmap(Ahat, colors)

    with col_l:
        st.image(true_rgb, use_container_width=True, clamp=True)
        st.caption("True")

    with col_r:
        st.image(labelmap_rgb, use_container_width=True, clamp=True)
        st.caption("Unmixing results")

    unmix_bw = [to_uint8_gray(Ahat[:, :, r]) for r in range(Ahat.shape[2])]

    st.divider()

    show_bw_grid(
        "Per-fluorophore (Unmixing, grayscale)",
        unmix_bw,
        names,
        cols_per_row=6,
    )

    prop_vals, acc_vals = compute_prop_and_accuracy(Atrue, Ahat)

    rmse_vals = [
        float(np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r]) ** 2)))
        for r in range(len(names))
    ]

    metric_header(
        "Per-fluorophore metrics",
        "RMSE: Root-mean-square error of the estimated abundance map for each fluorophore.\n"
        "Proportion: For each fluorophore, we look at pixels where its true abundance is nonzero "
        "and compute A_r / sum_k A_k, then average these ratios.\n"
        "Accuracy: For each fluorophore, among pixels where its true abundance is nonzero, "
        "accuracy is the fraction of pixels where this fluorophore has the largest estimated abundance.",
    )

    render_metrics_table(names, rmse_vals, prop_vals, acc_vals)

    st.caption(
        "Simulation note: synthetic images are generated under Poisson shot noise "
        "after scaling the clean image to peak intensity 255."
    )

    return rmse_vals, prop_vals, acc_vals


def _make_small_groups(use_pool, selected_labels):
    """Build a small groups dict for the final selected labels."""
    if use_pool:
        return {"Pool": [fluor_from_label(s) for s in selected_labels]}

    small_groups = {}

    for s in selected_labels:
        probe, fluor = s.split(" – ", 1)
        small_groups.setdefault(probe, []).append(fluor)

    return small_groups


def run_fluoroselect(
    *,
    wl,
    dye_db,
    groups,
    config,
    constraints,
    app_context,
):
    """Main public runner called by app.py."""
    mode = config["mode"]
    laser_strategy = config["laser_strategy"]
    laser_list = config["laser_list"]
    spec_res_mode = config["spec_res_mode"]
    source_mode = config["source_mode"]
    k_show = config["k_show"]
    similarity_metric = config.get("similarity_metric", "Cosine similarity")

    use_pool = constraints["use_pool"]
    required_count = constraints["required_count"] if use_pool else None

    fixed_fluorophores = constraints["fixed_fluorophores"]
    allowed_fluorophores = constraints["allowed_fluorophores"]
    fixed_probe_pairs = constraints["fixed_probe_pairs"]

    if mode == "Emission spectra":
        _run_emission_mode(
            wl=wl,
            dye_db=dye_db,
            groups=groups,
            mode=mode,
            source_mode=source_mode,
            laser_strategy=laser_strategy,
            laser_list=laser_list,
            spec_res_mode=spec_res_mode,
            k_show=k_show,
            similarity_metric=similarity_metric,
            use_pool=use_pool,
            required_count=required_count,
            fixed_fluorophores=fixed_fluorophores,
            allowed_fluorophores=allowed_fluorophores,
            fixed_probe_pairs=fixed_probe_pairs,
            app_context=app_context,
        )
    else:
        _run_predicted_mode(
            wl=wl,
            dye_db=dye_db,
            groups=groups,
            mode=mode,
            source_mode=source_mode,
            laser_strategy=laser_strategy,
            laser_list=laser_list,
            spec_res_mode=spec_res_mode,
            k_show=k_show,
            similarity_metric=similarity_metric,
            use_pool=use_pool,
            required_count=required_count,
            fixed_fluorophores=fixed_fluorophores,
            allowed_fluorophores=allowed_fluorophores,
            fixed_probe_pairs=fixed_probe_pairs,
            app_context=app_context,
        )


def _run_emission_mode(
    *,
    wl,
    dye_db,
    groups,
    mode,
    source_mode,
    laser_strategy,
    laser_list,
    spec_res_mode,
    k_show,
    similarity_metric,
    use_pool,
    required_count,
    fixed_fluorophores,
    allowed_fluorophores,
    fixed_probe_pairs,
    app_context,
):
    E_norm, labels, idx_groups = build_emission_only_matrix(wl, dye_db, groups)

    if E_norm.shape[1] == 0:
        st.error("No spectra.")
        st.stop()

    fixed_indices, allowed_indices = _constraint_indices(
        labels,
        use_pool,
        fixed_fluorophores,
        allowed_fluorophores,
        fixed_probe_pairs,
    )

    try:
        sel_idx, _ = solve_lexicographic_k(
            E_norm,
            idx_groups,
            labels,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
            fixed_indices=fixed_indices,
            allowed_indices=allowed_indices,
            similarity_metric=similarity_metric,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    labels_sel_tmp = [labels[j] for j in sel_idx]
    order, _ = sorted_order_by_peak(labels_sel_tmp, wl, dye_db)
    sel_idx = [sel_idx[i] for i in order]

    worst_idx = select_worst_group_constrained(
        E_norm,
        labels,
        idx_groups,
        required_count,
        use_pool,
        fixed_fluorophores=fixed_fluorophores,
        allowed_fluorophores=allowed_fluorophores,
        similarity_metric=similarity_metric,
    )

    labels_worst_tmp = [labels[j] for j in worst_idx]
    order_w, _ = sorted_order_by_peak(labels_worst_tmp, wl, dye_db)
    worst_idx = [worst_idx[i] for i in order_w]

    colors = ensure_colors(len(sel_idx))

    _render_selection_tables(
        use_pool=use_pool,
        labels=labels,
        sel_idx=sel_idx,
        worst_idx=worst_idx,
        predicted=False,
    )

    selected_labels = [labels[j] for j in sel_idx]

    tops = _render_pairwise_table(
        E_norm[:, sel_idx],
        selected_labels,
        k_show,
        similarity_metric=similarity_metric,
    )

    _render_spectra(
        x_axis=wl,
        spectra=E_norm[:, sel_idx],
        labels=selected_labels,
        colors=colors,
        y_title="Normalized intensity",
        normalize_by=None,
    )

    E_chan = cached_interpolate_E_on_channels(
        wl,
        E_norm[:, sel_idx],
        DETECTION_CHANNELS,
    )

    names = [prettify_name(label) for label in selected_labels]

    rmse_vals, prop_vals, acc_vals = _render_simulation_and_metrics(
        E_chan,
        colors,
        names,
    )

    result_context = build_result_context(
        run_id="emission",
        mode=mode,
        source_mode=source_mode,
        laser_strategy=laser_strategy,
        laser_list=laser_list,
        spec_res_mode=spec_res_mode,
        similarity_metric=similarity_metric,
        use_pool=use_pool,
        fixed_probe_pairs=fixed_probe_pairs,
        fixed_fluorophores=fixed_fluorophores,
        allowed_fluorophores=allowed_fluorophores,
        selected_labels=selected_labels,
        tops=tops,
        names=names,
        rmse_vals=rmse_vals,
        prop_vals=prop_vals,
        acc_vals=acc_vals,
        pair_formatter=pair_only_fluor,
    )

    render_ai_result_panel(result_context, app_context)


def _run_predicted_mode(
    *,
    wl,
    dye_db,
    groups,
    mode,
    source_mode,
    laser_strategy,
    laser_list,
    spec_res_mode,
    k_show,
    similarity_metric,
    use_pool,
    required_count,
    fixed_fluorophores,
    allowed_fluorophores,
    fixed_probe_pairs,
    app_context,
):
    if not laser_list:
        st.error("Please specify laser wavelengths.")
        st.stop()

    # Round A: provisional selection on emission-only spectra.
    E0, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)

    fixed_indices0, allowed_indices0 = _constraint_indices(
        labels0,
        use_pool,
        fixed_fluorophores,
        allowed_fluorophores,
        fixed_probe_pairs,
    )

    try:
        sel0, _ = solve_lexicographic_k(
            E0,
            idx0,
            labels0,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
            fixed_indices=fixed_indices0,
            allowed_indices=allowed_indices0,
            similarity_metric=similarity_metric,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    provisional_labels = [labels0[j] for j in sel0]

    # Estimate laser powers on provisional set.
    if laser_strategy == "Simultaneous":
        powers_A, _ = derive_powers_simultaneous(
            wl,
            dye_db,
            provisional_labels,
            laser_list,
        )
    else:
        powers_A, _ = derive_powers_separate(
            wl,
            dye_db,
            provisional_labels,
            laser_list,
        )

    # Build all candidate effective spectra.
    E_raw_all, E_norm_all, labels_all, idx_all = cached_build_effective_with_lasers(
        wl,
        dye_db,
        groups,
        laser_list,
        laser_strategy,
        powers_A,
    )

    # Choose selection-resolution matrix.
    if spec_res_mode == "9.8 nm" and laser_strategy == "Simultaneous":
        E_raw_all_98 = cached_interpolate_E_on_channels(
            wl,
            E_raw_all,
            DETECTION_CHANNELS,
        )

        E_raw_all_98 = apply_mbs_zeroing(
            E_raw_all_98,
            laser_strategy,
            spec_res_mode,
            laser_list,
        )

        E_norm_for_select = E_raw_all_98 / (
            np.linalg.norm(E_raw_all_98, axis=0, keepdims=True) + 1e-12
        )
    else:
        E_norm_for_select = E_norm_all

    fixed_indices_all, allowed_indices_all = _constraint_indices(
        labels_all,
        use_pool,
        fixed_fluorophores,
        allowed_fluorophores,
        fixed_probe_pairs,
    )

    try:
        sel_idx, _ = solve_lexicographic_k(
            E_norm_for_select,
            idx_all,
            labels_all,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
            fixed_indices=fixed_indices_all,
            allowed_indices=allowed_indices_all,
            similarity_metric=similarity_metric,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    final_labels = [labels_all[j] for j in sel_idx]

    worst_idx = select_worst_group_constrained(
        E_norm_for_select,
        labels_all,
        idx_all,
        required_count,
        use_pool,
        fixed_fluorophores=fixed_fluorophores,
        allowed_fluorophores=allowed_fluorophores,
        similarity_metric=similarity_metric,
    )

    worst_labels = [labels_all[j] for j in worst_idx]

    # Recalibrate powers on final set.
    if laser_strategy == "Simultaneous":
        powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
    else:
        powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

    small_groups = _make_small_groups(use_pool, final_labels)

    E_raw_sel_1nm, E_norm_sel_1nm, labels_sel, _ = cached_build_effective_with_lasers(
        wl,
        dye_db,
        small_groups,
        laser_list,
        laser_strategy,
        powers,
    )

    # Final display/simulation resolution.
    if spec_res_mode == "9.8 nm" and laser_strategy == "Simultaneous":
        E_raw_sel = cached_interpolate_E_on_channels(
            wl,
            E_raw_sel_1nm,
            DETECTION_CHANNELS,
        )

        E_raw_sel = apply_mbs_zeroing(
            E_raw_sel,
            laser_strategy,
            spec_res_mode,
            laser_list,
        )

        E_norm_sel = E_raw_sel / (
            np.linalg.norm(E_raw_sel, axis=0, keepdims=True) + 1e-12
        )
        x_axis = DETECTION_CHANNELS
    else:
        E_raw_sel = E_raw_sel_1nm
        E_norm_sel = E_norm_sel_1nm
        x_axis = wl

    # Sort selected labels by emission peak.
    if labels_sel:
        order, labels_sel = sorted_order_by_peak(labels_sel, wl, dye_db)
        E_raw_sel = E_raw_sel[:, order]
        E_norm_sel = E_norm_sel[:, order]

    # Sort worst labels for display only.
    if worst_labels:
        _, worst_labels = sorted_order_by_peak(worst_labels, wl, dye_db)

    colors = ensure_colors(len(labels_sel))

    if use_pool:
        st.subheader("Selected fluorophores (with lasers, best)")
        fluors = [fluor_from_label(s) for s in labels_sel]

        html_two_row_table(
            "Slot",
            "Fluorophore",
            [f"Slot {i + 1}" for i in range(len(fluors))],
            fluors,
        )

        if worst_labels:
            st.markdown("**Worst fluorophores (same count)**")
            worst_fluors = [fluor_from_label(s) for s in worst_labels]
            html_two_row_table(
                "Slot",
                "Fluorophore",
                [f"Slot {i + 1}" for i in range(len(worst_fluors))],
                worst_fluors,
            )

    else:
        st.subheader("Selected probe–fluorophore pairs (with lasers, best)")

        html_two_row_table(
            "Probe",
            "Fluorophore",
            [s.split(" – ", 1)[0] for s in labels_sel],
            [s.split(" – ", 1)[1] for s in labels_sel],
        )

        if worst_labels:
            st.markdown("**Worst probe–fluorophore pairs (same count)**")
            html_two_row_table(
                "Probe",
                "Fluorophore",
                [s.split(" – ", 1)[0] for s in worst_labels],
                [s.split(" – ", 1)[1] for s in worst_labels],
            )

    tops = _render_pairwise_table(
        E_norm_sel,
        labels_sel,
        k_show,
        similarity_metric=similarity_metric,
    )

    _render_spectra(
        x_axis=x_axis,
        spectra=E_raw_sel,
        labels=labels_sel,
        colors=colors,
        y_title="Normalized intensity (relative to B)",
        normalize_by=B,
    )

    E_chan = E_raw_sel / (B + 1e-12)
    names = [prettify_name(s) for s in labels_sel]

    rmse_vals, prop_vals, acc_vals = _render_simulation_and_metrics(
        E_chan,
        colors,
        names,
    )

    result_context = build_result_context(
        run_id="predicted",
        mode=mode,
        source_mode=source_mode,
        laser_strategy=laser_strategy,
        laser_list=laser_list,
        spec_res_mode=spec_res_mode,
        similarity_metric=similarity_metric,
        use_pool=use_pool,
        fixed_probe_pairs=fixed_probe_pairs,
        fixed_fluorophores=fixed_fluorophores,
        allowed_fluorophores=allowed_fluorophores,
        selected_labels=labels_sel,
        tops=tops,
        names=names,
        rmse_vals=rmse_vals,
        prop_vals=prop_vals,
        acc_vals=acc_vals,
        pair_formatter=pair_only_fluor,
    )

    render_ai_result_panel(result_context, app_context)
