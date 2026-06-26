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

from ai_helper import (
    parse_user_request,
    explain_result,
    suggest_improvements,
    answer_light_question,
    generate_methods_text,
)

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
    """Candidates under the EUB 338 probe key, filtered to dyes.yaml presence."""
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


def fluor_from_label(label: str) -> str:
    """Extract fluorophore name from 'Probe – Fluor' or return label itself."""
    return label.split(" – ", 1)[1] if " – " in label else label


def select_worst_group(E_norm, labels, idx_groups, required_count, use_pool):
    """
    Heuristic 'worst group' selector.

    Pool mode:
        Pick required_count columns with largest max similarity.
    By-probes mode:
        For each probe group, pick the candidate with largest max similarity,
        enforcing global fluorophore uniqueness as much as possible.
    """
    if E_norm.size == 0:
        return []

    S = cosine_similarity_matrix(E_norm)
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
):
    """
    Worst-group heuristic that mirrors the manual constraints in pool mode.
    This is only a contrast list, not the optimizer.
    """
    if not use_pool:
        return select_worst_group(E_norm, labels, idx_groups, required_count, use_pool)

    fixed_set = set(fixed_fluorophores or [])
    allowed_set = set(allowed_fluorophores or [])

    fixed_idx = [j for j, lab in enumerate(labels) if fluor_from_label(lab) in fixed_set]
    allowed_idx = [j for j, lab in enumerate(labels) if fluor_from_label(lab) in allowed_set]
    allowed_idx = [j for j in allowed_idx if j not in fixed_idx]

    n_total = int(required_count or 0)
    n_add = max(0, n_total - len(fixed_idx))

    if n_add == 0:
        return fixed_idx[:n_total]

    if E_norm.size == 0:
        return fixed_idx[:n_total]

    S = cosine_similarity_matrix(E_norm)
    max_sim = np.max(S, axis=1)
    order = sorted(allowed_idx, key=lambda j: -max_sim[j])
    return fixed_idx + order[:n_add]


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
    zero out MBS-blocked detection channels: 414, 486, 557, 566, 628, 637, 646 nm.
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


def render_metrics_table(names, rmse_vals, prop_vals, acc_vals):
    """
    Render a 3 x N metrics table:
    - rows: RMSE, Proportion, Accuracy
    - columns: fluorophore names
    """

    def esc(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def fmt(v):
        try:
            v = float(v)
        except Exception:
            return esc(v)
        if not np.isfinite(v):
            return ""
        return f"{v:.4f}"

    headers = ["Measurement"] + [esc(n) for n in names]
    rows = [
        ["RMSE"] + [fmt(v) for v in rmse_vals],
        ["Proportion"] + [fmt(v) for v in prop_vals],
        ["Accuracy"] + [fmt(v) for v in acc_vals],
    ]

    thead = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = []
        for j, cell in enumerate(r):
            align = "left" if j == 0 else "right"
            tds.append(f'<td style="text-align:{align}; padding:4px 8px;">{cell}</td>')
        trs.append(f"<tr>{''.join(tds)}</tr>")

    html = f"""
    <table style="border-collapse:collapse; font-size:0.9rem;">
        <thead><tr>{thead}</tr></thead>
        <tbody>{''.join(trs)}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)

# -------------------- AI input assistant --------------------
def build_ai_app_context():
    all_probes = sorted(probe_map.keys())

    probe_to_fluors = {}
    pair_options = []

    for p in all_probes:
        cands = sorted([f for f in probe_map.get(p, []) if f in dye_db])
        probe_to_fluors[p] = cands
        for f in cands:
            pair_options.append(f"{p} – {f}")

    return {
        "available_modes": ["Emission spectra", "Predicted spectra"],
        "available_laser_strategies": ["Simultaneous", "Separate"],
        "available_spectral_resolutions": [
            "1 nm (general)",
            "33 detection channels (Valm lab)",
        ],
        "available_selection_sources": [
            "By probes",
            "From readout pool",
            "All fluorophores",
            "EUB338 only",
        ],
        "available_lasers_common": [405, 445, 488, 514, 561, 594, 633, 639],
        "probes": all_probes,
        "probe_to_fluorophores": probe_to_fluors,
        "probe_fluorophore_pairs": pair_options,
        "readout_pool": readout_pool,
        "inventory_pool": inventory_pool,
        "eub338_pool": _get_eub338_pool(),
    }


def apply_ai_plan_to_session_state(plan, app_context):
    """
    Apply parsed AI plan to Streamlit widget keys.
    Values are filtered against available options before assignment.
    """

    def valid(x, options):
        return x if x in options else None

    # Mode
    mode_val = valid(plan.get("mode"), app_context["available_modes"])
    if mode_val:
        st.session_state["mode_radio"] = mode_val

    # Source
    source_val = valid(plan.get("selection_source"), app_context["available_selection_sources"])
    if source_val:
        st.session_state["source_radio"] = source_val

    # Laser strategy and lasers
    strategy_val = valid(plan.get("laser_strategy"), app_context["available_laser_strategies"])
    if strategy_val:
        st.session_state["laser_strategy_radio"] = strategy_val

    spec_val = valid(plan.get("spectral_resolution"), app_context["available_spectral_resolutions"])
    if spec_val:
        st.session_state["spec_res_radio"] = spec_val

    lasers = plan.get("lasers") or []
    lasers = [int(x) for x in lasers if isinstance(x, (int, float, str)) and str(x).isdigit()]
    if lasers:
        st.session_state["num_lasers_input"] = len(lasers)
        for i, lam in enumerate(lasers, start=1):
            st.session_state[f"laser_{i}"] = int(lam)

    # By probes
    fixed_pairs = []
    for item in plan.get("fixed_probe_fluorophore_pairs") or []:
        p = item.get("probe")
        f = item.get("fluorophore")
        pair = f"{p} – {f}"
        if pair in app_context["probe_fluorophore_pairs"]:
            fixed_pairs.append(pair)

    if fixed_pairs:
        st.session_state["fixed_probe_pairs"] = fixed_pairs
        st.session_state["source_radio"] = "By probes"

    additional_probes = [
        p for p in (plan.get("additional_probes") or [])
        if p in app_context["probes"]
    ]
    if additional_probes:
        st.session_state["picked_additional_probes"] = additional_probes
        st.session_state["source_radio"] = "By probes"

    # Pool modes
    source_after = st.session_state.get("source_radio", source_val or "By probes")

    if source_after == "From readout pool":
        suffix = "pool"
        available_fluors = app_context["readout_pool"]
    elif source_after == "EUB338 only":
        suffix = "eub338"
        available_fluors = app_context["eub338_pool"]
    else:
        suffix = "inventory"
        available_fluors = app_context["inventory_pool"]

    fixed_fluors = [
        f for f in (plan.get("fixed_fluorophores") or [])
        if f in available_fluors
    ]

    candidate_fluors_raw = plan.get("candidate_fluorophores") or []
    if candidate_fluors_raw:
        candidate_fluors = [f for f in candidate_fluors_raw if f in available_fluors and f not in fixed_fluors]
    else:
        candidate_fluors = [f for f in available_fluors if f not in fixed_fluors]

    if source_after in ["From readout pool", "All fluorophores", "EUB338 only"]:
        st.session_state[f"fixed_fluorophores_{suffix}"] = fixed_fluors
        st.session_state[f"allowed_fluorophores_{suffix}"] = candidate_fluors

        n_add = plan.get("n_additional_fluorophores")
        if isinstance(n_add, int):
            n_add = max(0, min(n_add, len(candidate_fluors)))
            st.session_state[f"n_additional_{suffix}"] = n_add


ai_app_context = build_ai_app_context()

with st.sidebar.expander("AI input assistant", expanded=False):
    user_ai_request = st.text_area(
        "Describe what you want to select",
        placeholder=(
            "Example: Fix EUB338 with AF488, then choose Streptococcus probes. "
            "Use predicted spectra with 488, 561, and 639 nm lasers."
        ),
        key="ai_user_request",
    )

    if st.button("Parse and apply", key="ai_parse_apply"):
        if not user_ai_request.strip():
            st.warning("Please enter a request first.")
        else:
            with st.spinner("Parsing request with Gemini..."):
                try:
                    plan = parse_user_request(user_ai_request, ai_app_context)
                    apply_ai_plan_to_session_state(plan, ai_app_context)
                    st.session_state["last_ai_plan"] = plan
                    st.success("AI plan applied to the sidebar controls.")
                    if plan.get("warnings"):
                        st.warning("\n".join(plan["warnings"]))
                    st.rerun()
                except Exception as e:
                    st.error(f"AI parsing failed: {e}")

    if "last_ai_plan" in st.session_state:
        st.caption("Last parsed AI plan")
        st.json(st.session_state["last_ai_plan"])

# -------------------- Sidebar --------------------
st.sidebar.header("Configuration")

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

if mode == "Predicted spectra":
    laser_strategy = st.sidebar.radio(
        "Laser usage",
        ("Simultaneous", "Separate"),
        key="laser_strategy_radio",
    )

    if laser_strategy == "Simultaneous":
        spec_res_mode = st.sidebar.radio(
            "Spectral resolution",
            ("1 nm (general)", "33 detection channels (Valm lab)"),
            key="spec_res_radio",
        )
    else:
        spec_res_mode = "1 nm (general)"

source_mode = st.sidebar.radio(
    "Selection source",
    ("By probes", "From readout pool", "All fluorophores", "EUB338 only"),
    key="source_radio",
)

k_show = st.sidebar.slider("Show top-K similarities", 5, 50, 10, 1, key="k_show_slider")

if mode == "Predicted spectra":
    n = st.sidebar.number_input("Number of lasers", 1, 8, 4, 1, key="num_lasers_input")
    cols_l = st.sidebar.columns(2)
    defaults = [405, 488, 561, 639]
    for i in range(n):
        lam = cols_l[i % 2].number_input(
            f"Laser {i + 1} (nm)",
            int(wl.min()),
            int(max(700, wl.max())),
            defaults[i] if i < len(defaults) else int(wl.min()),
            1,
            key=f"laser_{i + 1}",
        )
        laser_list.append(int(lam))


# -------------------- Source selection -> groups --------------------
use_pool = False
pool = []

fixed_fluorophores = []
allowed_fluorophores = []
fixed_probe_pairs = []
required_count = None

if source_mode == "From readout pool":
    pool = readout_pool[:]
    if not pool:
        st.info("Readout pool not found (data/readout_fluorophores.yaml).")
        st.stop()

    use_pool = True
    all_available_fluors = sorted(set(pool))

    st.sidebar.subheader("Fluorophore constraints")

    fixed_fluorophores = st.sidebar.multiselect(
        "Fixed fluorophores",
        options=all_available_fluors,
        default=[],
        help=(
            "These fluorophores will be forced into the final selected panel. "
            "The optimizer will choose the remaining fluorophores around them."
        ),
        key="fixed_fluorophores_pool",
    )

    candidate_options = [f for f in all_available_fluors if f not in fixed_fluorophores]

    allowed_fluorophores = st.sidebar.multiselect(
        "Candidate fluorophores for additional selection",
        options=candidate_options,
        default=candidate_options,
        help=(
            "The optimizer can only choose additional fluorophores from this list. "
            "Fixed fluorophores are always included."
        ),
        key="allowed_fluorophores_pool",
    )

    label_n = (
        "How many additional fluorophores to choose"
        if fixed_fluorophores
        else "How many fluorophores to choose"
    )

    default_n = min(4, len(allowed_fluorophores))
    n_additional = st.sidebar.number_input(
        label_n,
        min_value=0 if fixed_fluorophores else 1,
        max_value=len(allowed_fluorophores),
        value=default_n,
        step=1,
        key="n_additional_pool",
    )

    required_count = len(fixed_fluorophores) + int(n_additional)

    if required_count == 0:
        st.info("Select at least one fixed fluorophore or choose at least one additional fluorophore.")
        st.stop()

    constrained_pool = sorted(set(fixed_fluorophores) | set(allowed_fluorophores))

    if not constrained_pool:
        st.error("No fluorophores are available after applying constraints.")
        st.stop()

    groups = {"Pool": constrained_pool}

    st.sidebar.caption(
        f"Final panel size = {len(fixed_fluorophores)} fixed "
        f"+ {int(n_additional)} additional = {required_count}."
    )

elif source_mode == "All fluorophores":
    pool = inventory_pool[:]
    if not pool:
        st.error("No fluorophores found in probe_fluor_map.yaml that also exist in dyes.yaml.")
        st.stop()

    use_pool = True
    all_available_fluors = sorted(set(pool))

    st.sidebar.subheader("Fluorophore constraints")

    fixed_fluorophores = st.sidebar.multiselect(
        "Fixed fluorophores",
        options=all_available_fluors,
        default=[],
        help=(
            "These fluorophores will be forced into the final selected panel. "
            "The optimizer will choose the remaining fluorophores around them."
        ),
        key="fixed_fluorophores_inventory",
    )

    candidate_options = [f for f in all_available_fluors if f not in fixed_fluorophores]

    allowed_fluorophores = st.sidebar.multiselect(
        "Candidate fluorophores for additional selection",
        options=candidate_options,
        default=candidate_options,
        help=(
            "The optimizer can only choose additional fluorophores from this list. "
            "Fixed fluorophores are always included."
        ),
        key="allowed_fluorophores_inventory",
    )

    label_n = (
        "How many additional fluorophores to choose"
        if fixed_fluorophores
        else "How many fluorophores to choose"
    )

    default_n = min(4, len(allowed_fluorophores))
    n_additional = st.sidebar.number_input(
        label_n,
        min_value=0 if fixed_fluorophores else 1,
        max_value=len(allowed_fluorophores),
        value=default_n,
        step=1,
        key="n_additional_inventory",
    )

    required_count = len(fixed_fluorophores) + int(n_additional)

    if required_count == 0:
        st.info("Select at least one fixed fluorophore or choose at least one additional fluorophore.")
        st.stop()

    constrained_pool = sorted(set(fixed_fluorophores) | set(allowed_fluorophores))

    if not constrained_pool:
        st.error("No fluorophores are available after applying constraints.")
        st.stop()

    groups = {"Pool": constrained_pool}

    st.sidebar.caption(
        f"Final panel size = {len(fixed_fluorophores)} fixed "
        f"+ {int(n_additional)} additional = {required_count}."
    )

elif source_mode == "EUB338 only":
    pool = _get_eub338_pool()
    if not pool:
        st.error("No candidates found for EUB 338 in probe_fluor_map.yaml.")
        st.stop()

    use_pool = True
    all_available_fluors = sorted(set(pool))

    st.sidebar.subheader("Fluorophore constraints")

    fixed_fluorophores = st.sidebar.multiselect(
        "Fixed fluorophores",
        options=all_available_fluors,
        default=[],
        help=(
            "These fluorophores will be forced into the final selected panel. "
            "The optimizer will choose the remaining fluorophores around them."
        ),
        key="fixed_fluorophores_eub338",
    )

    candidate_options = [f for f in all_available_fluors if f not in fixed_fluorophores]

    allowed_fluorophores = st.sidebar.multiselect(
        "Candidate fluorophores for additional selection",
        options=candidate_options,
        default=candidate_options,
        help=(
            "The optimizer can only choose additional fluorophores from this list. "
            "Fixed fluorophores are always included."
        ),
        key="allowed_fluorophores_eub338",
    )

    label_n = (
        "How many additional fluorophores to choose"
        if fixed_fluorophores
        else "How many fluorophores to choose"
    )

    default_n = min(4, len(allowed_fluorophores))
    n_additional = st.sidebar.number_input(
        label_n,
        min_value=0 if fixed_fluorophores else 1,
        max_value=len(allowed_fluorophores),
        value=default_n,
        step=1,
        key="n_additional_eub338",
    )

    required_count = len(fixed_fluorophores) + int(n_additional)

    if required_count == 0:
        st.info("Select at least one fixed fluorophore or choose at least one additional fluorophore.")
        st.stop()

    constrained_pool = sorted(set(fixed_fluorophores) | set(allowed_fluorophores))

    if not constrained_pool:
        st.error("No fluorophores are available after applying constraints.")
        st.stop()

    groups = {"Pool": constrained_pool}

    st.sidebar.caption(
        f"Final panel size = {len(fixed_fluorophores)} fixed "
        f"+ {int(n_additional)} additional = {required_count}."
    )

else:
    use_pool = False

    all_probes = sorted(probe_map.keys())

    pair_options = []
    pair_to_probe = {}
    pair_to_fluor = {}

    for p in all_probes:
        cands = [f for f in probe_map.get(p, []) if f in dye_db]
        for f in sorted(cands):
            pair = f"{p} – {f}"
            pair_options.append(pair)
            pair_to_probe[pair] = p
            pair_to_fluor[pair] = f

    st.sidebar.subheader("Probe constraints")

    fixed_probe_pairs = st.sidebar.multiselect(
        "Fixed probe–fluorophore pairs",
        options=pair_options,
        default=[],
        help=(
            "These exact probe–fluorophore pairs will be forced into the final design. "
            "For example, fixing 'EUB338 – AF488' means EUB338 will use AF488."
        ),
        key="fixed_probe_pairs",
    )

    fixed_probe_names = sorted({pair_to_probe[x] for x in fixed_probe_pairs})

    remaining_probe_options = [p for p in all_probes if p not in fixed_probe_names]

    picked_additional = st.sidebar.multiselect(
        "Choose additional probes",
        options=remaining_probe_options,
        default=[],
        help=(
            "For each additional probe, the optimizer will select one fluorophore. "
            "Fixed probe–fluorophore pairs are already included."
        ),
        key="picked_additional_probes",
    )

    picked = fixed_probe_names + picked_additional

    if not picked:
        st.info("Select at least one fixed probe–fluorophore pair or choose at least one additional probe.")
        st.stop()

    groups = {}

    for pair in fixed_probe_pairs:
        p = pair_to_probe[pair]
        f = pair_to_fluor[pair]
        groups[p] = [f]

    for p in picked_additional:
        cands = [f for f in probe_map.get(p, []) if f in dye_db]
        if cands:
            groups[p] = cands

    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml.")
        st.stop()

    required_count = None

def render_ai_result_panel(result_context):
    with st.expander("AI assistant for this result", expanded=False):
        st.caption(
            "AI assistance is optional. The fluorophore selection is computed by "
            "the FluoroSelect optimizer, not by the AI model."
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Explain this result", key=f"ai_explain_{result_context['run_id']}"):
                with st.spinner("Generating explanation..."):
                    text = explain_result(result_context)
                st.markdown(text)

        with col2:
            if st.button("Suggest improvements", key=f"ai_suggest_{result_context['run_id']}"):
                with st.spinner("Generating suggestions..."):
                    text = suggest_improvements(result_context)
                st.markdown(text)

        if st.button("Generate Methods paragraph", key=f"ai_methods_{result_context['run_id']}"):
            with st.spinner("Generating Methods draft..."):
                text = generate_methods_text(result_context)
            st.markdown(text)

        question = st.text_area(
            "Ask a question about this result",
            placeholder="Example: Why is this pair risky? Should I reduce the number of fluorophores?",
            key=f"ai_question_{result_context['run_id']}",
        )

        if st.button("Ask AI", key=f"ai_ask_{result_context['run_id']}"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Answering..."):
                    text = answer_light_question(question, ai_app_context, result_context)
                st.markdown(text)

# -------------------- Core runner --------------------
def run(groups, mode, laser_strategy, laser_list, spec_res_mode):
    required_count_local = required_count if use_pool else None

    def get_constraint_indices(labels):
        fixed_indices = []
        allowed_indices = None

        if use_pool:
            fixed_set = set(fixed_fluorophores or [])
            allowed_set = set(allowed_fluorophores or [])

            allowed_indices = []

            for j, lab in enumerate(labels):
                fluor = fluor_from_label(lab)
                if fluor in fixed_set:
                    fixed_indices.append(j)
                if fluor in allowed_set:
                    allowed_indices.append(j)

            return fixed_indices, allowed_indices

        fixed_pair_set = set(fixed_probe_pairs or [])

        for j, lab in enumerate(labels):
            if lab in fixed_pair_set:
                fixed_indices.append(j)

        return fixed_indices, None

    # ---------- EMISSION MODE ----------
    if mode == "Emission spectra":
        E_norm, labels, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra.")
            st.stop()

        fixed_indices, allowed_indices = get_constraint_indices(labels)
        try:
            sel_idx, _ = solve_lexicographic_k(
                E_norm,
                idx_groups,
                labels,
                levels=10,
                enforce_unique=True,
                required_count=required_count_local,
                fixed_indices=fixed_indices,
                allowed_indices=allowed_indices,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

        labels_sel_tmp = [labels[j] for j in sel_idx]
        order, _ = sorted_order_by_peak(labels_sel_tmp)
        sel_idx = [sel_idx[i] for i in order]

        worst_idx = select_worst_group_constrained(
            E_norm,
            labels,
            idx_groups,
            required_count_local,
            use_pool,
            fixed_fluorophores=fixed_fluorophores,
            allowed_fluorophores=allowed_fluorophores,
        )
        labels_worst_tmp = [labels[j] for j in worst_idx]
        order_w, _ = sorted_order_by_peak(labels_worst_tmp)
        worst_idx = [worst_idx[i] for i in order_w]

        colors = ensure_colors(len(sel_idx))

        if use_pool:
            fluors = [fluor_from_label(labels[j]) for j in sel_idx]
            st.subheader("Selected fluorophores (best)")
            html_two_row_table(
                "Slot",
                "Fluorophore",
                [f"Slot {i + 1}" for i in range(len(fluors))],
                fluors,
            )

            worst_fluors = [fluor_from_label(labels[j]) for j in worst_idx]
            st.markdown("**Worst fluorophores (same count)**")
            html_two_row_table(
                "Slot",
                "Fluorophore",
                [f"Slot {i + 1}" for i in range(len(worst_fluors))],
                worst_fluors,
            )
        else:
            sel_pairs = [labels[j] for j in sel_idx]
            st.subheader("Selected fluorophores (best)")
            html_two_row_table(
                "Probe",
                "Fluorophore",
                [s.split(" – ", 1)[0] for s in sel_pairs],
                [s.split(" – ", 1)[1] for s in sel_pairs],
            )

            worst_pairs = [labels[j] for j in worst_idx]
            st.markdown("**Worst fluorophores (same count)**")
            html_two_row_table(
                "Probe",
                "Fluorophore",
                [s.split(" – ", 1)[0] for s in worst_pairs],
                [s.split(" – ", 1)[1] for s in worst_pairs],
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

        # Spectra viewer
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

        prop_vals, acc_vals = compute_prop_and_accuracy(Atrue, Ahat)
        rmse_vals = []
        for r in range(len(names)):
            rmse_vals.append(np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r]) ** 2)))

        metric_header(
            "Per-fluorophore metrics",
            "RMSE: Root-mean-square error of the estimated abundance map for each fluorophore. "
            "Proportion: For each fluorophore, we look at pixels where its true abundance is nonzero "
            "and compute A_r / sum_k A_k, then average these ratios (ignoring pixels where the sum is zero). "
            "Accuracy: For each fluorophore, among pixels where its true abundance is nonzero, "
            "accuracy is the fraction of pixels where this fluorophore has the largest estimated abundance.",
        )
        render_metrics_table(names, rmse_vals, prop_vals, acc_vals)

result_context = {
    "run_id": "emission",
    "mode": mode,
    "selection_source": source_mode,
    "laser_strategy": laser_strategy,
    "lasers": laser_list,
    "spectral_resolution": spec_res_mode,
    "use_pool": use_pool,
    "fixed_probe_fluorophore_pairs": fixed_probe_pairs,
    "fixed_fluorophores": fixed_fluorophores,
    "allowed_fluorophores": allowed_fluorophores,
    "selected_labels": [labels[j] for j in sel_idx],
    "selected_fluorophores": [fluor_from_label(labels[j]) for j in sel_idx],
    "top_pairwise_similarities": [
        {
            "similarity": float(val),
            "label_1": a,
            "label_2": b,
            "fluorophore_pair": pair_only_fluor(a, b),
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

render_ai_result_panel(result_context)
return

    # ---------- PREDICTED MODE ----------
    if not laser_list:
        st.error("Please specify laser wavelengths.")
        st.stop()

    # Round A: provisional selection on emission-only spectra
    E0, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
    fixed_indices0, allowed_indices0 = get_constraint_indices(labels0)
    try:
        sel0, _ = solve_lexicographic_k(
            E0,
            idx0,
            labels0,
            levels=10,
            enforce_unique=True,
            required_count=required_count_local,
            fixed_indices=fixed_indices0,
            allowed_indices=allowed_indices0,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    A_labels = [labels0[j] for j in sel0]

    # (1) powers on provisional set
    if laser_strategy == "Simultaneous":
        powers_A, _ = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
    else:
        powers_A, _ = derive_powers_separate(wl, dye_db, A_labels, laser_list)

    # First build all candidates with lasers at 1 nm grid
    E_raw_all, E_norm_all, labels_all, idx_all = cached_build_effective_with_lasers(
        wl,
        dye_db,
        groups,
        laser_list,
        laser_strategy,
        powers_A,
    )

    # For selection + similarity: choose resolution
    if spec_res_mode == "33 detection channels (Valm lab)" and laser_strategy == "Simultaneous":
        E_raw_all_33 = cached_interpolate_E_on_channels(wl, E_raw_all, DETECTION_CHANNELS)
        E_raw_all_33 = apply_mbs_zeroing(E_raw_all_33, laser_strategy, spec_res_mode, laser_list)
        denom_all = np.linalg.norm(E_raw_all_33, axis=0, keepdims=True) + 1e-12
        E_norm_for_select = E_raw_all_33 / denom_all
    else:
        E_norm_for_select = E_norm_all

    # BEST group
    fixed_indices_all, allowed_indices_all = get_constraint_indices(labels_all)
    try:
        sel_idx, _ = solve_lexicographic_k(
            E_norm_for_select,
            idx_all,
            labels_all,
            levels=10,
            enforce_unique=True,
            required_count=required_count_local,
            fixed_indices=fixed_indices_all,
            allowed_indices=allowed_indices_all,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    final_labels = [labels_all[j] for j in sel_idx]

    # WORST group, same size, greedy heuristic
    worst_idx = select_worst_group_constrained(
        E_norm_for_select,
        labels_all,
        idx_all,
        required_count_local,
        use_pool,
        fixed_fluorophores=fixed_fluorophores,
        allowed_fluorophores=allowed_fluorophores,
    )
    worst_labels = [labels_all[j] for j in worst_idx]

    # (2) recalibrate on final set
    if laser_strategy == "Simultaneous":
        powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
    else:
        powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

    # Build final spectra at 1 nm grid, only selected set
    if use_pool:
        small_groups = {"Pool": [fluor_from_label(s) for s in final_labels]}
    else:
        small_groups = {}
        for s in final_labels:
            p, f = s.split(" – ", 1)
            small_groups.setdefault(p, []).append(f)

    E_raw_sel_1nm, E_norm_sel_1nm, labels_sel, _ = cached_build_effective_with_lasers(
        wl,
        dye_db,
        small_groups,
        laser_list,
        laser_strategy,
        powers,
    )

    # For display / simulation: choose final resolution
    if spec_res_mode == "33 detection channels (Valm lab)" and laser_strategy == "Simultaneous":
        E_raw_sel = cached_interpolate_E_on_channels(wl, E_raw_sel_1nm, DETECTION_CHANNELS)
        E_raw_sel = apply_mbs_zeroing(E_raw_sel, laser_strategy, spec_res_mode, laser_list)
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
    st.subheader("Selected fluorophores (with lasers, best)")
    fluors = [fluor_from_label(s) for s in labels_sel]
    html_two_row_table(
        "Slot",
        "Fluorophore",
        [f"Slot {i + 1}" for i in range(len(fluors))],
        fluors,
    )

    # Worst group, only as a contrast list
    if worst_labels:
        order_w, worst_labels = sorted_order_by_peak(worst_labels)
        worst_fluors = [fluor_from_label(s) for s in worst_labels]
        st.markdown("**Worst fluorophores (same count)**")
        html_two_row_table(
            "Slot",
            "Fluorophore",
            [f"Slot {i + 1}" for i in range(len(worst_fluors))],
            worst_fluors,
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
    if spec_res_mode == "33 detection channels (Valm lab)" and laser_strategy == "Simultaneous":
        E_chan = E_raw_sel / (B + 1e-12)
    else:
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

    prop_vals, acc_vals = compute_prop_and_accuracy(Atrue, Ahat)
    rmse_vals = []
    for r in range(len(names)):
        rmse_vals.append(np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r]) ** 2)))

    metric_header(
        "Per-fluorophore metrics",
        "RMSE: Root-mean-square error of the estimated abundance map for each fluorophore. "
        "Proportion: For each fluorophore, we look at pixels where its true abundance is nonzero "
        "and compute A_r / sum_k A_k, then average these ratios (ignoring pixels where the sum is zero). "
        "Accuracy: For each fluorophore, among pixels where its true abundance is nonzero, "
        "accuracy is the fraction of pixels where this fluorophore has the largest estimated abundance.",
    )
    render_metrics_table(names, rmse_vals, prop_vals, acc_vals)

result_context = {
    "run_id": "predicted",
    "mode": mode,
    "selection_source": source_mode,
    "laser_strategy": laser_strategy,
    "lasers": laser_list,
    "spectral_resolution": spec_res_mode,
    "use_pool": use_pool,
    "fixed_probe_fluorophore_pairs": fixed_probe_pairs,
    "fixed_fluorophores": fixed_fluorophores,
    "allowed_fluorophores": allowed_fluorophores,
    "selected_labels": labels_sel,
    "selected_fluorophores": [fluor_from_label(s) for s in labels_sel],
    "top_pairwise_similarities": [
        {
            "similarity": float(val),
            "label_1": a,
            "label_2": b,
            "fluorophore_pair": pair_only_fluor(a, b),
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

render_ai_result_panel(result_context)
return

st.caption(
    "AI-assisted features are used only for input parsing, lightweight Q&A, "
    "result explanation, and drafting suggestions. Optimization results are "
    "computed by the FluoroSelect algorithm."
)

# -------------------- Execute --------------------
run(groups, mode, laser_strategy, laser_list, spec_res_mode)
