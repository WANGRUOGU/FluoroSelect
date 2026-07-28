# selection_ui.py
import re
import streamlit as st


SOFT_PENALTY_STRENGTHS = ("Low", "Medium", "High")
SOFT_PENALTY_WEIGHTS = {
    "Low": 0.005,
    "Medium": 0.02,
    "High": 0.08,
}


BRIGHTNESS_BALANCE_RATIOS = {
    "Off": None,
    "Weak": 5.0,
    "Strong": 2.5,
}


def _norm_probe_name(name):
    """Normalize probe names so 'EUB 338', 'EUB-338', and 'EUB338' match."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _choose_canonical_probe_name(names):
    """
    Choose one display name among equivalent probe aliases.

    Prefer compact names such as 'EUB338' over 'EUB 338'.
    """
    names = sorted(set(str(x).strip() for x in names if str(x).strip()))

    compact = [x for x in names if _norm_probe_name(x) == str(x).lower()]
    if compact:
        return sorted(compact, key=lambda x: (len(x), x.lower()))[0]

    return sorted(names, key=lambda x: (len(x), x.lower()))[0]


def _canonicalize_probe_map(probe_map, dye_db):
    """
    Merge probe aliases into one canonical probe.

    Example:
        'EUB 338' and 'EUB338' become one probe, preferably 'EUB338'.
    """
    aliases_by_norm = {}
    for probe in probe_map.keys():
        key = _norm_probe_name(probe)
        aliases_by_norm.setdefault(key, []).append(probe)

    canonical_by_norm = {
        key: _choose_canonical_probe_name(aliases)
        for key, aliases in aliases_by_norm.items()
    }

    alias_to_canonical = {}
    canonical_map = {}

    for raw_probe, fluor_list in probe_map.items():
        key = _norm_probe_name(raw_probe)
        canonical_probe = canonical_by_norm[key]
        alias_to_canonical[raw_probe] = canonical_probe

        cands = [f for f in fluor_list if isinstance(f, str) and f in dye_db]
        canonical_map.setdefault(canonical_probe, set()).update(cands)

    canonical_map = {probe: sorted(fluors) for probe, fluors in canonical_map.items()}
    return canonical_map, alias_to_canonical


def _canonicalize_probe_value(value, alias_to_canonical):
    """Convert a probe value to canonical form."""
    if value in alias_to_canonical:
        return alias_to_canonical[value]

    value_norm = _norm_probe_name(value)
    for alias, canonical in alias_to_canonical.items():
        if _norm_probe_name(alias) == value_norm:
            return canonical

    return value


def _canonicalize_pair_value(pair, alias_to_canonical):
    """Convert 'Probe – Fluor' to canonical probe name."""
    if " – " not in str(pair):
        return pair

    probe, fluor = str(pair).split(" – ", 1)
    probe = _canonicalize_probe_value(probe, alias_to_canonical)
    return f"{probe} – {fluor}"


def _sanitize_multiselect_state(key, options, value_mapper=None):
    """
    Remove stale/duplicate values from Streamlit multiselect session state.
    """
    if key not in st.session_state:
        return

    options_set = set(options)
    old_values = st.session_state.get(key) or []

    if not isinstance(old_values, list):
        old_values = [old_values]

    new_values = []
    seen = set()

    for value in old_values:
        mapped = value_mapper(value) if value_mapper is not None else value
        if mapped not in options_set:
            continue
        if mapped in seen:
            continue
        new_values.append(mapped)
        seen.add(mapped)

    st.session_state[key] = new_values


def render_sidebar_config(wl):
    """Render global configuration controls and return the selected state."""
    st.sidebar.header("Configuration")

    source_mode = st.sidebar.radio(
        "Selection source",
        ("By probes", "From readout pool", "All fluorophores", "EUB338 only"),
        key="source_radio",
    )

    panel_size = None
    if source_mode != "By probes":
        panel_size = st.sidebar.number_input(
            "Number of fluorophores",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            key=f"panel_size_{source_mode}",
            help="Total number of fluorophores in the selected panel.",
        )

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
    spec_res_mode = "1 nm"
    brightness_balance = "Off"

    if mode == "Predicted spectra":
        laser_strategy = st.sidebar.radio(
            "Laser usage",
            ("Simultaneous", "Separate"),
            key="laser_strategy_radio",
        )

        if laser_strategy == "Simultaneous":
            spec_res_mode = st.sidebar.radio(
                "Spectral resolution",
                ("1 nm", "9.8 nm"),
                key="spec_res_radio",
                help=(
                    "1 nm uses the full spectral grid. 9.8 nm uses the Valm-lab "
                    "detector-channel centers."
                ),
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

        brightness_balance = st.sidebar.radio(
            "Brightness balance",
            tuple(BRIGHTNESS_BALANCE_RATIOS),
            index=1,
            key="brightness_balance_radio",
            help=(
                "After laser-power calibration, the brightest panel member is set "
                "to relative brightness 1. Off applies no constraint. Weak and Strong "
                "require the dimmest selected fluorophore to have relative peak "
                "brightness of at least 0.2 and 0.4, respectively. "
                "FluoroSelect alternates panel selection and laser-power calibration "
                "for at most four iterations because each depends on the other."
            ),
        )

    k_show = st.sidebar.slider(
        "Show top-K similarities",
        5,
        50,
        5,
        1,
        key="k_show_slider",
    )

    return {
        "mode": mode,
        "laser_strategy": laser_strategy,
        "laser_list": laser_list,
        "spec_res_mode": spec_res_mode,
        "source_mode": source_mode,
        "panel_size": int(panel_size) if panel_size is not None else None,
        "similarity_metric": "Cosine similarity",
        "brightness_balance": brightness_balance,
        "max_brightness_ratio": BRIGHTNESS_BALANCE_RATIOS[brightness_balance],
        "k_show": k_show,
    }


def _soft_penalty_controls(source_suffix, fluor_options):
    """Render soft-penalty controls and return selected low-priority dyes."""
    fluor_options = sorted(set(fluor_options))

    if not fluor_options:
        return [], "Medium", SOFT_PENALTY_WEIGHTS["Medium"]

    st.sidebar.subheader("Soft penalty")

    _sanitize_multiselect_state(
        f"low_priority_fluorophores_{source_suffix}",
        fluor_options,
    )

    low_priority_fluorophores = st.sidebar.multiselect(
        "Lower-priority fluorophores",
        options=fluor_options,
        default=[],
        help=(
            "These fluorophores are not removed. The optimizer will avoid them "
            "when comparable alternatives exist, but can still select them if needed."
        ),
        key=f"low_priority_fluorophores_{source_suffix}",
    )

    soft_penalty_strength = "Medium"

    if low_priority_fluorophores:
        soft_penalty_strength = st.sidebar.radio(
            "Soft penalty strength",
            SOFT_PENALTY_STRENGTHS,
            index=1,
            key=f"soft_penalty_strength_{source_suffix}",
            help=(
                "Low mildly avoids lower-priority fluorophores. High avoids them "
                "more strongly, while still preserving the best worst-pair similarity."
            ),
        )
        st.sidebar.caption(
            "Soft penalty is applied only after the best worst-pair similarity is fixed."
        )

    soft_penalty_weight = SOFT_PENALTY_WEIGHTS.get(
        soft_penalty_strength,
        SOFT_PENALTY_WEIGHTS["Medium"],
    )

    return low_priority_fluorophores, soft_penalty_strength, soft_penalty_weight


def _pool_constraints(source_suffix, pool, panel_size):
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

    required_count = int(panel_size)
    n_additional = required_count - len(fixed_fluorophores)
    if n_additional < 0:
        st.error(
            f"Panel size is {required_count}, but {len(fixed_fluorophores)} "
            "fluorophores are fixed. Increase the panel size or remove fixed selections."
        )
        st.stop()
    if n_additional > len(allowed_fluorophores):
        st.error(
            f"Panel size {required_count} requires {n_additional} additional "
            f"fluorophores, but only {len(allowed_fluorophores)} candidates are allowed."
        )
        st.stop()

    constrained_pool = sorted(set(fixed_fluorophores) | set(allowed_fluorophores))
    if not constrained_pool:
        st.error("No fluorophores are available after applying constraints.")
        st.stop()

    st.sidebar.caption(
        f"Final panel size = {required_count}: {len(fixed_fluorophores)} fixed "
        f"+ {n_additional} selected by the optimizer."
    )

    low_priority_fluorophores, soft_penalty_strength, soft_penalty_weight = (
        _soft_penalty_controls(source_suffix, constrained_pool)
    )

    return {
        "groups": {"Pool": constrained_pool},
        "use_pool": True,
        "required_count": required_count,
        "fixed_fluorophores": fixed_fluorophores,
        "allowed_fluorophores": allowed_fluorophores,
        "fixed_probe_pairs": [],
        "low_priority_fluorophores": low_priority_fluorophores,
        "soft_penalty_strength": soft_penalty_strength,
        "soft_penalty_weight": soft_penalty_weight,
    }


def build_selection_groups(
    source_mode,
    probe_map,
    dye_db,
    readout_pool,
    inventory_pool,
    eub338_pool,
    panel_size=None,
):
    """Render source-specific controls and build optimization groups/constraints."""
    if source_mode == "From readout pool":
        if not readout_pool:
            st.info("Readout pool not found (data/readout_fluorophores.yaml).")
            st.stop()
        return _pool_constraints("pool", readout_pool, panel_size)

    if source_mode == "All fluorophores":
        if not inventory_pool:
            st.error(
                "No fluorophores found in probe_fluor_map.yaml that also exist in dyes.yaml."
            )
            st.stop()
        return _pool_constraints("inventory", inventory_pool, panel_size)

    if source_mode == "EUB338 only":
        if not eub338_pool:
            st.error("No candidates found for EUB338 in probe_fluor_map.yaml.")
            st.stop()
        return _pool_constraints("eub338", eub338_pool, panel_size)

    # By probes mode: canonicalized fixed exact pairs + additional probes.
    canonical_probe_map, alias_to_canonical = _canonicalize_probe_map(probe_map, dye_db)
    all_probes = sorted(canonical_probe_map.keys())

    pair_options = []
    pair_to_probe = {}
    pair_to_fluor = {}

    for probe in all_probes:
        cands = canonical_probe_map.get(probe, [])
        for fluor in sorted(cands):
            pair = f"{probe} – {fluor}"
            pair_options.append(pair)
            pair_to_probe[pair] = probe
            pair_to_fluor[pair] = fluor

    pair_options = sorted(pair_options)

    st.sidebar.subheader("Probe constraints")

    _sanitize_multiselect_state(
        "fixed_probe_pairs",
        pair_options,
        value_mapper=lambda x: _canonicalize_pair_value(x, alias_to_canonical),
    )

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

    fixed_probe_names = sorted(
        {pair_to_probe[pair] for pair in fixed_probe_pairs if pair in pair_to_probe}
    )

    remaining_probe_options = [probe for probe in all_probes if probe not in fixed_probe_names]

    _sanitize_multiselect_state(
        "picked_additional_probes",
        remaining_probe_options,
        value_mapper=lambda x: _canonicalize_probe_value(x, alias_to_canonical),
    )

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

    # Final dedupe by normalized probe name, protecting against stale session state.
    picked_additional_clean = []
    seen_probe_keys = set()

    for probe in picked_additional:
        canonical_probe = _canonicalize_probe_value(probe, alias_to_canonical)
        key = _norm_probe_name(canonical_probe)
        if key in seen_probe_keys:
            continue
        if canonical_probe in remaining_probe_options:
            picked_additional_clean.append(canonical_probe)
            seen_probe_keys.add(key)

    picked_additional = picked_additional_clean
    picked = fixed_probe_names + picked_additional

    if not picked:
        st.info(
            "Select at least one fixed probe–fluorophore pair or choose at least one additional probe."
        )
        st.stop()

    groups = {}

    for pair in fixed_probe_pairs:
        if pair not in pair_to_probe:
            continue
        probe = pair_to_probe[pair]
        fluor = pair_to_fluor[pair]
        groups[probe] = [fluor]

    for probe in picked_additional:
        cands = canonical_probe_map.get(probe, [])
        if cands:
            groups[probe] = cands

    # Last safety check: collapse any duplicate normalized probe names.
    safe_groups = {}
    for probe, cands in groups.items():
        key = _norm_probe_name(probe)
        if key in safe_groups:
            safe_groups[key]["cands"].update(cands)
        else:
            safe_groups[key] = {"probe": probe, "cands": set(cands)}

    groups = {item["probe"]: sorted(item["cands"]) for item in safe_groups.values()}

    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml.")
        st.stop()

    available_fluors_for_soft_penalty = sorted(
        {fluor for cands in groups.values() for fluor in cands}
    )

    low_priority_fluorophores, soft_penalty_strength, soft_penalty_weight = (
        _soft_penalty_controls("probes", available_fluors_for_soft_penalty)
    )

    return {
        "groups": groups,
        "use_pool": False,
        "required_count": None,
        "fixed_fluorophores": [],
        "allowed_fluorophores": [],
        "fixed_probe_pairs": fixed_probe_pairs,
        "low_priority_fluorophores": low_priority_fluorophores,
        "soft_penalty_strength": soft_penalty_strength,
        "soft_penalty_weight": soft_penalty_weight,
    }
