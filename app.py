# app.py
import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import DYES_YAML, PROBE_MAP_YAML, READOUT_POOL_YAML, DETECTION_CHANNELS
from utils import (
    load_dyes_yaml,
    load_probe_fluor_map,
    build_emission_only_matrix,
    build_effective_with_lasers,
    derive_powers_simultaneous,
    derive_powers_separate,
    solve_lexicographic_k,
    cosine_similarity_matrix,
    top_k_pairwise,
)
from sim_core import (
    simulate_rods_and_unmix,
    colorize_composite,
    argmax_labelmap,
    to_uint8_gray,
)
from ui_helpers import (
    ensure_colors,
    rgb01_to_plotly,
    pair_only_fluor,
    html_two_row_table,
    show_bw_grid,
    metric_header,
    prettify_name,
)
from metrics import compute_prop_and_accuracy

st.set_page_config(page_title="Fluorophore Selection", layout="wide")
st.title("Fluorophore Selection for Multiplexed Imaging")
# -------------------- Data --------------------
wl, dye_db = load_dyes_yaml(DYES_YAML)
probe_map = load_probe_fluor_map(PROBE_MAP_YAML)


def _load_readout_pool(path):
    try:
        import yaml
        import os

        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = data.get("fluorophores", []) or []
        pool = sorted({s.strip() for s in items if isinstance(s, str) and s.strip()})
        return [f for f in pool if f in dye_db]
    except Exception:
        return []


readout_pool = _load_readout_pool(READOUT_POOL_YAML)


def _get_inventory_from_probe_map():
    """Union of all fluorophores appearing in probe_fluor_map.yaml that exist in dyes.yaml."""
    inv = set()
    for _, vals in probe_map.items():
        if not isinstance(vals, (list, tuple)):
            continue
        for f in vals:
            if isinstance(f, str):
                fs = f.strip()
                if fs and fs in dye_db:
                    inv.add(fs)
    return sorted(inv)


inventory_pool = _get_inventory_from_probe_map()


def _get_eub338_pool():
    """Candidates under the EUB 338 probe key (various spellings), filtered to dyes.yaml presence."""
    targets = {"eub338", "eub 338", "eub-338"}

    def norm(s):
        return "".join(s.lower().split())

    for k in probe_map.keys():
        if norm(k) in targets:
            cands = [f for f in probe_map.get(k, []) if f in dye_db]
            return sorted({c.strip() for c in cands})
    # relaxed fallback
    import re

    def norm2(s):
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    for k in probe_map.keys():
        if norm2(k) == "eub338":
            cands = [f for f in probe_map.get(k, []) if f in dye_db]
            return sorted({c.strip() for c in cands})
    return []


def peak_wavelength_for_label(label: str) -> float:
    """
    Return the emission peak wavelength (in nm) for a given label.
    Label may be 'Probe – Fluor' or just 'Fluor'.
    If not found or invalid, return +inf so it is sorted to the end.
    """
    name = label.split(" – ", 1)[1] if " – " in label else label
    rec = dye_db.get(name)
    if rec is None:
        return float("inf")
    em = np.asarray(rec.get("emission", []), dtype=float)
    if em.size != len(wl) or np.max(em) <= 0:
        return float("inf")
    jmax = int(np.argmax(em))
    return float(wl[jmax])


def sorted_order_by_peak(labels_list):
    """
    Given a list of labels, return:
      - order: np.array of indices that sorts them by emission peak wavelength
      - sorted_labels: labels_list reordered by that order
    """
    peaks = [peak_wavelength_for_label(lbl) for lbl in labels_list]
    order = np.argsort(peaks)
    sorted_labels = [labels_list[i] for i in order]
    return order, sorted_labels


@st.cache_data(show_spinner=False)
def cached_build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers):
    groups_key = json.dumps({k: sorted(v) for k, v in sorted(groups.items())}, ensure_ascii=False)
    _ = (
        tuple(sorted(laser_list)),
        laser_strategy,
        tuple(np.asarray(powers, float)) if powers is not None else None,
        groups_key,
    )
    return build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers)


@st.cache_data(show_spinner=False)
def cached_interpolate_E_on_channels(wl, spectra_cols, chan_centers_nm):
    spectra_cols = np.asarray(spectra_cols, dtype=float)
    if spectra_cols.ndim == 1:
        spectra_cols = spectra_cols[:, None]
    W, N = spectra_cols.shape
    E = np.zeros((len(chan_centers_nm), N), dtype=float)
    for j in range(N):
        y = spectra_cols[:, j]
        E[:, j] = np.interp(chan_centers_nm, wl, y, left=float(y[0]), right=float(y[-1]))
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)


def apply_mbs_zeroing(E_raw_on_det, laser_strategy, spec_res_mode, laser_list):
    """
    In Simultaneous + Valm-lab 33-channel mode, if lasers are [405,488,561,639],
    zero out MBS-blocked detection channels:
      414, 486, 557, 566, 628, 637, 646 nm.
    """
    if spec_res_mode != "33 detection channels (Valm lab)":
        return E_raw_on_det
    if laser_strategy != "Simultaneous":
        return E_raw_on_det
    try:
        if sorted(int(l) for l in laser_list) != [405, 488, 561, 639]:
            return E_raw_on_det
    except Exception:
        return E_raw_on_det

    blocked = np.array([414, 486, 557, 566, 628, 637, 646], dtype=float)
    mask = np.isin(DETECTION_CHANNELS, blocked)
    if E_raw_on_det.ndim == 2:
        E_raw_on_det[mask, :] = 0.0
    return E_raw_on_det


# -------------------- Sidebar --------------------
st.sidebar.header("Configuration")

# 1. 选择模式
mode = st.sidebar.radio(
    "Mode",
    options=("Emission spectra", "Predicted spectra"),
    help=(
        "Emission: emission-only, peak-normalized.\n"
        "Predicted: effective spectra with lasers (excitation · QY · EC)."
    ),
    key="mode_radio",
)

laser_list = []
laser_strategy = None
spec_res_mode = "1 nm (general)"

# 2. 在 Predicted 模式下，先选 laser 使用方式 + 光谱分辨率
if mode == "Predicted spectra":
    laser_strategy = st.sidebar.radio(
        "Laser usage",
        ("Simultaneous", "Separate"),
        key="laser_strategy_radio",
    )

    # ✅ Spectral resolution 提到 Selection source 之前
    if laser_strategy == "Simultaneous":
        spec_res_mode = st.sidebar.radio(
            "Spectral resolution",
            ("1 nm (general)", "33 detection channels (Valm lab)"),
            key="spec_res_radio",
        )
    else:
        spec_res_mode = "1 nm (general)"

# 3. 再选 Selection source
source_mode = st.sidebar.radio(
    "Selection source",
    ("By probes", "From readout pool", "All fluorophores", "EUB338 only"),
    key="source_radio",
)

# 4. 其他 sidebar 控件
k_show = st.sidebar.slider("Show top-K similarities", 5, 50, 10, 1, key="k_show_slider")

# 5. 在 Predicted 模式下继续设置 laser 波长
if mode == "Predicted spectra":
    n = st.sidebar.number_input("Number of lasers", 1, 8, 4, 1, key="num_lasers_input")
    cols_l = st.sidebar.columns(2)
    defaults = [405, 488, 561, 639]
    for i in range(n):
        lam = cols_l[i % 2].number_input(
            f"Laser {i+1} (nm)",
            int(wl.min()),
            int(max(700, wl.max())),
            defaults[i] if i < len(defaults) else int(wl.min()),
            1,
            key=f"laser_{i+1}",
        )
        laser_list.append(int(lam))


# -------------------- Source selection -> groups --------------------
use_pool = False
if source_mode == "From readout pool":
    pool = readout_pool[:]
    if not pool:
        st.info("Readout pool not found (data/readout_fluorophores.yaml).")
        st.stop()
    max_n = len(pool)
    N_pick = st.number_input(
        "How many fluorophores", 1, max_n, min(4, max_n), 1, key="n_pick_pool"
    )
    groups = {"Pool": pool}
    use_pool = True

elif source_mode == "All fluorophores":
    pool = inventory_pool[:]
    if not pool:
        st.error("No fluorophores found in probe_fluor_map.yaml that also exist in dyes.yaml.")
        st.stop()
    max_n = len(pool)
    N_pick = st.number_input(
        "How many fluorophores", 1, max_n, min(4, max_n), 1, key="n_pick_inv"
    )
    groups = {"Pool": pool}
    use_pool = True

elif source_mode == "EUB338 only":
    pool = _get_eub338_pool()
    if not pool:
        st.error("No candidates found for EUB 338 in probe_fluor_map.yaml.")
        st.stop()
    max_n = len(pool)
    N_pick = st.number_input(
        "How many fluorophores", 1, max_n, min(4, max_n), 1, key="n_pick_eub338"
    )
    groups = {"Pool": pool}
    use_pool = True

else:  # "By probes"
    all_probes = sorted(probe_map.keys())
    picked = st.multiselect("Probes", options=all_probes, key="picked_probes")
    if not picked:
        st.info("Select at least one probe to proceed.")
        st.stop()
    groups = {}
    for p in picked:
        cands = [f for f in probe_map.get(p, []) if f in dye_db]
        if cands:
            groups[p] = cands
    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml.")
        st.stop()
    N_pick = None


def run(groups, mode, laser_strategy, laser_list, spec_res_mode):
    required_count = N_pick if use_pool else None

    # ---------- EMISSION MODE ----------
    if mode == "Emission spectra":
        E_norm, labels, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra.")
            st.stop()

        sel_idx, _ = solve_lexicographic_k(
            E_norm,
            idx_groups,
            labels,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
        )

        # sort selection by emission peak wavelength
        labels_sel_tmp = [labels[j] for j in sel_idx]
        order, _ = sorted_order_by_peak(labels_sel_tmp)
        sel_idx = [sel_idx[i] for i in order]

        colors = ensure_colors(len(sel_idx))

        # Selected fluorophores table
        if use_pool:
            fluors = [labels[j].split(" – ", 1)[1] for j in sel_idx]
            st.subheader("Selected fluorophores")
            html_two_row_table(
                "Slot",
                "Fluorophore",
                [f"Slot {i+1}" for i in range(len(fluors))],
                fluors,
            )
        else:
            sel_pairs = [labels[j] for j in sel_idx]
            st.subheader("Selected fluorophores")
            html_two_row_table(
                "Probe",
                "Fluorophore",
                [s.split(" – ", 1)[0] for s in sel_pairs],
                [s.split(" – ", 1)[1] for s in sel_pairs],
            )

        # Pairwise similarity
        S = cosine_similarity_matrix(E_norm[:, sel_idx])
        tops = top_k_pairwise(S, [labels[j] for j in sel_idx], k=k_show)
        st.subheader("Top pairwise similarities")
        html_two_row_table(
            "Pair",
            "Similarity",
            [pair_only_fluor(a, b) for _, a, b in tops],
            [val for val, _, _ in tops],
            color_second_row=True,
            color_thresh=0.9,
            fmt2=True,
        )

        # Spectra viewer (emission-only, wavelength grid)
        st.subheader("Spectra viewer")
        fig = go.Figure()
        for t, j in enumerate(sel_idx):
            y = E_norm[:, j]
            y = y / (np.max(y) + 1e-12)
            fig.add_trace(
                go.Scatter(
                    x=wl,
                    y=y,
                    mode="lines",
                    name=labels[j],
                    line=dict(color=rgb01_to_plotly(colors[t]), width=2),
                )
            )
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Normalized intensity",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Simulation ----------
        chan = DETECTION_CHANNELS
        E_chan = cached_interpolate_E_on_channels(wl, E_norm[:, sel_idx], chan)
        Atrue, Ahat = simulate_rods_and_unmix(E_chan, rods_per=3)

        colL, colR = st.columns(2)
        true_rgb = (colorize_composite(Atrue, colors) * 255).astype(np.uint8)
        labelmap_rgb = argmax_labelmap(Ahat, colors)
        with colL:
            st.image(true_rgb, use_container_width=True, clamp=True)
            st.caption("True")
        with colR:
            st.image(labelmap_rgb, use_container_width=True, clamp=True)
            st.caption("Unmixing results")

        names = [prettify_name(labels[j]) for j in sel_idx]
        unmix_bw = [to_uint8_gray(Ahat[:, :, r]) for r in range(Ahat.shape[2])]

        st.divider()
        show_bw_grid("Per-fluorophore (Unmixing, grayscale)", unmix_bw, names, cols_per_row=6)

        # proportion & accuracy
        prop_vals, acc_vals = compute_prop_and_accuracy(Atrue, Ahat)
        prop_show = [v if np.isfinite(v) else "" for v in prop_vals]
        acc_show = [v if np.isfinite(v) else "" for v in acc_vals]

        metric_header(
            "Per-fluorophore proportion",
            "For each fluorophore, we look at pixels where its true abundance is nonzero "
            "and compute A_r / sum_k A_k, then average these ratios (ignoring pixels where the sum is zero).",
        )
        html_two_row_table(
            "Fluorophore",
            "Proportion",
            names,
            prop_show,
            fmt2=True,
        )

        metric_header(
            "Per-fluorophore accuracy",
            "For each fluorophore, among pixels where its true abundance is nonzero, "
            "accuracy is the fraction of pixels where this fluorophore has the largest estimated abundance.",
        )
        html_two_row_table(
            "Fluorophore",
            "Accuracy",
            names,
            acc_show,
            fmt2=True,
        )

        # RMSE
        rmse_vals = []
        for r in range(len(names)):
            rmse_vals.append(
                np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r]) ** 2))
            )
        st.subheader("Per-fluorophore RMSE")
        html_two_row_table(
            "Fluorophore",
            "RMSE",
            names,
            rmse_vals,
            fmt2=True,
        )

        return

    # ---------- PREDICTED MODE ----------
    else:
        if not laser_list:
            st.error("Please specify laser wavelengths.")
            st.stop()

        # Round A: provisional selection on emission-only
        E0, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
        sel0, _ = solve_lexicographic_k(
            E0,
            idx0,
            labels0,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
        )
        A_labels = [labels0[j] for j in sel0]

        # (1) powers on provisional set
        if laser_strategy == "Simultaneous":
            powers_A, _ = derive_powers_simultaneous(
                wl, dye_db, A_labels, laser_list
            )
        else:
            powers_A, _ = derive_powers_separate(
                wl, dye_db, A_labels, laser_list
            )

        # First build (all candidates, with lasers) at 1 nm grid
        E_raw_all, E_norm_all, labels_all, idx_all = cached_build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers_A
        )

        # For selection + similarity: choose resolution
        if (
            spec_res_mode == "33 detection channels (Valm lab)"
            and laser_strategy == "Simultaneous"
        ):
            # Compress 1 nm spectra to 33 detection channels
            E_raw_all_33 = cached_interpolate_E_on_channels(
                wl, E_raw_all, DETECTION_CHANNELS
            )
            E_raw_all_33 = apply_mbs_zeroing(
                E_raw_all_33, laser_strategy, spec_res_mode, laser_list
            )
            denom_all = np.linalg.norm(E_raw_all_33, axis=0, keepdims=True) + 1e-12
            E_norm_for_select = E_raw_all_33 / denom_all
        else:
            E_norm_for_select = E_norm_all

        # Lexicographic selection
        sel_idx, _ = solve_lexicographic_k(
            E_norm_for_select,
            idx_all,
            labels_all,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
        )
        final_labels = [labels_all[j] for j in sel_idx]

        # (2) recalibrate on final set
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(
                wl, dye_db, final_labels, laser_list
            )
        else:
            powers, B = derive_powers_separate(
                wl, dye_db, final_labels, laser_list
            )

        # Build final spectra at 1 nm grid (only selected set)
        if use_pool:
            small_groups = {"Pool": [s.split(" – ", 1)[1] for s in final_labels]}
        else:
            small_groups = {}
            for s in final_labels:
                p, f = s.split(" – ", 1)
                small_groups.setdefault(p, []).append(f)

        E_raw_sel_1nm, E_norm_sel_1nm, labels_sel, _ = cached_build_effective_with_lasers(
            wl, dye_db, small_groups, laser_list, laser_strategy, powers
        )

        # For display / simulation: choose final resolution
        if (
            spec_res_mode == "33 detection channels (Valm lab)"
            and laser_strategy == "Simultaneous"
        ):
            # Compress to 33 detection channels
            E_raw_sel = cached_interpolate_E_on_channels(
                wl, E_raw_sel_1nm, DETECTION_CHANNELS
            )
            E_raw_sel = apply_mbs_zeroing(
                E_raw_sel, laser_strategy, spec_res_mode, laser_list
            )
            denom_sel = np.linalg.norm(E_raw_sel, axis=0, keepdims=True) + 1e-12
            E_norm_sel = E_raw_sel / denom_sel
            x_axis = DETECTION_CHANNELS
        else:
            E_raw_sel = E_raw_sel_1nm
            E_norm_sel = E_norm_sel_1nm
            x_axis = wl

        # sort by emission peak wavelength
        if labels_sel:
            order, labels_sel = sorted_order_by_peak(labels_sel)
            E_raw_sel = E_raw_sel[:, order]
            E_norm_sel = E_norm_sel[:, order]

        colors = ensure_colors(len(labels_sel))

        # Selected fluorophores table
        st.subheader("Selected fluorophores (with lasers)")
        fluors = [s.split(" – ", 1)[1] for s in labels_sel]
        html_two_row_table(
            "Slot",
            "Fluorophore",
            [f"Slot {i+1}" for i in range(len(fluors))],
            fluors,
        )

        # Pairwise similarity
        S = cosine_similarity_matrix(E_norm_sel)
        tops = top_k_pairwise(S, labels_sel, k=k_show)
        st.subheader("Top pairwise similarities")
        html_two_row_table(
            "Pair",
            "Similarity",
            [pair_only_fluor(a, b) for _, a, b in tops],
            [val for val, _, _ in tops],
            color_second_row=True,
            color_thresh=0.9,
            fmt2=True,
        )

        # Spectra viewer
        st.subheader("Spectra viewer")
        fig = go.Figure()
        for t in range(len(labels_sel)):
            y = E_raw_sel[:, t] / (B + 1e-12)
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y,
                    mode="lines",
                    name=labels_sel[t],
                    line=dict(color=rgb01_to_plotly(colors[t]), width=2),
                )
            )
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Normalized intensity (relative to B)",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Simulation ----------
        if (
            spec_res_mode == "33 detection channels (Valm lab)"
            and laser_strategy == "Simultaneous"
        ):
            chan = DETECTION_CHANNELS
            E_chan = E_raw_sel / (B + 1e-12)
        else:
            chan = wl
            E_chan = E_raw_sel / (B + 1e-12)

        Atrue, Ahat = simulate_rods_and_unmix(E_chan, rods_per=3)

        colL, colR = st.columns(2)
        true_rgb = (colorize_composite(Atrue, colors) * 255).astype(np.uint8)
        labelmap_rgb = argmax_labelmap(Ahat, colors)
        with colL:
            st.image(true_rgb, use_container_width=True, clamp=True)
            st.caption("True")
        with colR:
            st.image(labelmap_rgb, use_container_width=True, clamp=True)
            st.caption("Unmixing results")

        names = [prettify_name(s) for s in labels_sel]
        unmix_bw = [to_uint8_gray(Ahat[:, :, r]) for r in range(Ahat.shape[2])]

        st.divider()
        show_bw_grid("Per-fluorophore (Unmixing, grayscale)", unmix_bw, names, cols_per_row=6)

        # proportion & accuracy
        prop_vals, acc_vals = compute_prop_and_accuracy(Atrue, Ahat)
        prop_show = [v if np.isfinite(v) else "" for v in prop_vals]
        acc_show = [v if np.isfinite(v) else "" for v in acc_vals]

        metric_header(
            "Per-fluorophore proportion",
            "For each fluorophore, we look at pixels where its true abundance is nonzero "
            "and compute A_r / sum_k A_k, then average these ratios (ignoring pixels where the sum is zero).",
        )
        html_two_row_table(
            "Fluorophore",
            "Proportion",
            names,
            prop_show,
            fmt2=True,
        )

        metric_header(
            "Per-fluorophore accuracy",
            "For each fluorophore, among pixels where its true abundance is nonzero, "
            "accuracy is the fraction of pixels where this fluorophore has the largest estimated abundance.",
        )
        html_two_row_table(
            "Fluorophore",
            "Accuracy",
            names,
            acc_show,
            fmt2=True,
        )

        # RMSE
        rmse_vals = []
        for r in range(len(names)):
            rmse_vals.append(
                np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r]) ** 2))
            )
        st.subheader("Per-fluorophore RMSE")
        html_two_row_table(
            "Fluorophore",
            "RMSE",
            names,
            rmse_vals,
            fmt2=True,
        )

        return


# -------------------- Execute --------------------

run(groups, mode, laser_strategy, laser_list, spec_res_mode)
