# selection_ui.py
import streamlit as st


def render_sidebar_config(wl):
    """Render global configuration controls and return the selected state."""
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

        n_lasers = st.sidebar.number_input(
            "Number of lasers",
            1,
            8,
            4,
            1,
            key="num_lasers_input",
        )
        cols_l = st.sidebar.columns(2)
        defaults = [405, 488, 561, 639]
        for i in range(n_lasers):
            lam = cols_l[i % 2].number_input(
                f"Laser {i + 1} (nm)",
                int(wl.min()),
                int(max(700, wl.max())),
                defaults[i] if i < len(defaults) else int(wl.min()),
                1,
                key=f"laser_{i + 1}",
            )
            laser_list.append(int(lam))

    source_mode = st.sidebar.radio(
        "Selection source",
        ("By probes", "From readout pool", "All fluorophores", "EUB338 only"),
        key="source_radio",
    )

    k_show = st.sidebar.slider("Show top-K similarities", 5, 50, 10, 1, key="k_show_slider")

    return {
        "mode": mode,
        "laser_strategy": laser_strategy,
        "laser_list": laser_list,
        "spec_res_mode": spec_res_mode,
        "source_mode": source_mode,
        "k_show": k_show,
    }


def _pool_constraints(source_suffix, pool):
    """Common sidebar controls for pool-style fluorophore selection."""
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
        key=f"fixed_fluorophores_{source_suffix}",
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
        key=f"allowed_fluorophores_{source_suffix}",
    )

    label_n = (
        "How many additional fluorophores to choose"
        if fixed_fluorophores
        else "How many fluorophores to choose"
    )

    default_n = min(4, len(allowed_fluorophores))
    min_n = 0 if fixed_fluorophores else 1
    if len(allowed_fluorophores) == 0:
        min_n = 0

    n_additional = st.sidebar.number_input(
        label_n,
        min_value=min_n,
        max_value=len(allowed_fluorophores),
        value=max(min_n, default_n),
        step=1,
        key=f"n_additional_{source_suffix}",
    )

    required_count = len(fixed_fluorophores) + int(n_additional)

    if required_count == 0:
        st.info("Select at least one fixed fluorophore or choose at least one additional fluorophore.")
        st.stop()

    constrained_pool = sorted(set(fixed_fluorophores) | set(allowed_fluorophores))
    if not constrained_pool:
        st.error("No fluorophores are available after applying constraints.")
        st.stop()

    st.sidebar.caption(
        f"Final panel size = {len(fixed_fluorophores)} fixed "
        f"+ {int(n_additional)} additional = {required_count}."
    )

    return {
        "groups": {"Pool": constrained_pool},
        "use_pool": True,
        "required_count": required_count,
        "fixed_fluorophores": fixed_fluorophores,
        "allowed_fluorophores": allowed_fluorophores,
        "fixed_probe_pairs": [],
    }


def build_selection_groups(source_mode, probe_map, dye_db, readout_pool, inventory_pool, eub338_pool):
    """Render source-specific controls and build optimization groups/constraints."""
    if source_mode == "From readout pool":
        if not readout_pool:
            st.info("Readout pool not found (data/readout_fluorophores.yaml).")
            st.stop()
        return _pool_constraints("pool", readout_pool)

    if source_mode == "All fluorophores":
        if not inventory_pool:
            st.error("No fluorophores found in probe_fluor_map.yaml that also exist in dyes.yaml.")
            st.stop()
        return _pool_constraints("inventory", inventory_pool)

    if source_mode == "EUB338 only":
        if not eub338_pool:
            st.error("No candidates found for EUB 338 in probe_fluor_map.yaml.")
            st.stop()
        return _pool_constraints("eub338", eub338_pool)

    # By probes mode: fixed exact pairs + additional probes.
    all_probes = sorted(probe_map.keys())
    pair_options = []
    pair_to_probe = {}
    pair_to_fluor = {}

    for probe in all_probes:
        cands = [f for f in probe_map.get(probe, []) if f in dye_db]
        for fluor in sorted(cands):
            pair = f"{probe} – {fluor}"
            pair_options.append(pair)
            pair_to_probe[pair] = probe
            pair_to_fluor[pair] = fluor

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
        probe = pair_to_probe[pair]
        fluor = pair_to_fluor[pair]
        groups[probe] = [fluor]

    for probe in picked_additional:
        cands = [f for f in probe_map.get(probe, []) if f in dye_db]
        if cands:
            groups[probe] = cands

    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml.")
        st.stop()

    return {
        "groups": groups,
        "use_pool": False,
        "required_count": None,
        "fixed_fluorophores": [],
        "allowed_fluorophores": [],
        "fixed_probe_pairs": fixed_probe_pairs,
    }
