# ai_ui.py
import re
import streamlit as st

from ai_helper import (
    parse_user_request,
    explain_result,
    suggest_improvements,
    answer_light_question,
    generate_methods_text,
)


DEFAULT_MODE = "Emission spectra"
DEFAULT_LASERS = [488, 561, 639]
DEFAULT_LASER_STRATEGY = "Simultaneous"
DEFAULT_SPEC_RESOLUTION = "1 nm"
DEFAULT_SIMILARITY_METRIC = "Cosine similarity"
DEFAULT_SOFT_PENALTY_STRENGTH = "Medium"
DEFAULT_N_FLUOROPHORES = 4


def _norm_text(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def _choose_canonical_probe_name(names):
    """
    Choose one display name among equivalent probe aliases.

    Example:
        EUB 338 and EUB338 are treated as the same probe.
        Prefer compact names such as EUB338.
    """
    names = sorted(set(str(x).strip() for x in names if str(x).strip()))

    compact = [x for x in names if _norm_text(x) == str(x).lower()]

    if compact:
        return sorted(compact, key=lambda x: (len(x), x.lower()))[0]

    return sorted(names, key=lambda x: (len(x), x.lower()))[0]


def _build_probe_alias_maps(probe_map):
    """Build map from raw probe names to canonical probe names."""
    aliases_by_norm = {}

    for probe in probe_map.keys():
        key = _norm_text(probe)
        aliases_by_norm.setdefault(key, []).append(probe)

    canonical_by_norm = {
        key: _choose_canonical_probe_name(aliases)
        for key, aliases in aliases_by_norm.items()
    }

    alias_to_canonical = {}

    for probe in probe_map.keys():
        key = _norm_text(probe)
        alias_to_canonical[probe] = canonical_by_norm[key]

    return alias_to_canonical


def _dedupe_probe_names(probes, app_context):
    """
    Deduplicate probe names by normalized form.

    Example:
        EUB 338 and EUB338 are treated as the same probe.
    """
    alias_to_canonical = app_context.get("probe_alias_to_canonical", {})

    canonical_by_norm = {}

    for probe in app_context.get("probes", []):
        canonical_by_norm[_norm_text(probe)] = probe

    for alias, canonical in alias_to_canonical.items():
        canonical_by_norm[_norm_text(alias)] = canonical

    out = []
    seen = set()

    for probe in probes or []:
        key = _norm_text(probe)

        if key not in canonical_by_norm:
            continue

        canonical = canonical_by_norm[key]
        canonical_key = _norm_text(canonical)

        if canonical_key in seen:
            continue

        out.append(canonical)
        seen.add(canonical_key)

    return out


def build_ai_app_context(
    *,
    probe_map,
    dye_db,
    readout_pool,
    inventory_pool,
    eub338_pool,
):
    """
    Build a compact context for Gemini.

    This contains only allowed probes/fluorophores/settings, not raw spectra.
    Probe aliases such as EUB 338 and EUB338 are merged.
    """
    alias_to_canonical = _build_probe_alias_maps(probe_map)

    canonical_probe_to_fluors = {}

    for raw_probe, fluor_list in probe_map.items():
        canonical_probe = alias_to_canonical.get(raw_probe, raw_probe)

        cands = [f for f in fluor_list if isinstance(f, str) and f in dye_db]
        canonical_probe_to_fluors.setdefault(canonical_probe, set()).update(cands)

    probe_to_fluors = {
        probe: sorted(fluors)
        for probe, fluors in canonical_probe_to_fluors.items()
    }

    all_probes = sorted(probe_to_fluors.keys())
    all_probe_fluorophores = sorted(
        {fluor for fluors in probe_to_fluors.values() for fluor in fluors}
    )

    pair_options = []
    for probe in all_probes:
        for fluor in probe_to_fluors[probe]:
            pair_options.append(f"{probe} – {fluor}")

    return {
        "available_modes": ["Emission spectra", "Predicted spectra"],
        "available_laser_strategies": ["Simultaneous", "Separate"],
        "available_spectral_resolutions": ["1 nm", "9.8 nm"],
        "available_similarity_metrics": [
            "Cosine similarity",
            "Spectral overlap",
            "Pearson correlation",
            "Spectral angle similarity",
        ],
        "available_soft_penalty_strengths": [
            "Low",
            "Medium",
            "High",
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
        "probe_fluorophore_pairs": sorted(pair_options),
        "all_probe_fluorophores": all_probe_fluorophores,
        "probe_alias_to_canonical": alias_to_canonical,
        "readout_pool": readout_pool,
        "inventory_pool": inventory_pool,
        "eub338_pool": eub338_pool,
    }


def _find_known_fluors(user_text, fluor_options):
    text_norm = _norm_text(user_text)
    found = []

    for fluor in fluor_options:
        if _norm_text(fluor) in text_norm:
            found.append(fluor)

    return sorted(set(found), key=lambda x: (-len(x), x))


def _find_known_probes(user_text, probe_options):
    text_norm = _norm_text(user_text)
    found = []

    for probe in probe_options:
        if _norm_text(probe) in text_norm:
            found.append(probe)

    return sorted(set(found), key=lambda x: (-len(x), x))


def _extract_number(user_text):
    """
    Extract a requested count from phrases like:
        choose 4 fluorophores
        choose another 4 fluorophores
        select 5
    """
    text = user_text.lower()

    patterns = [
        r"(?:choose|select|pick)\s+(?:another\s+)?(\d+)",
        r"(\d+)\s+(?:more|additional|another)",
        r"another\s+(\d+)",
    ]

    for pat in patterns:
        match = re.search(pat, text)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                pass

    return None


def _mentions_predicted(user_text):
    text = user_text.lower()
    return any(
        key in text
        for key in [
            "predicted",
            "effective",
            "laser",
            "lasers",
            "excitation",
            "with 488",
            "with 561",
            "with 639",
        ]
    )


def _detect_similarity_metric(user_text):
    text = user_text.lower()

    if "spectral overlap" in text or "overlap" in text:
        return "Spectral overlap"
    if "pearson" in text or "correlation" in text or "corr" in text:
        return "Pearson correlation"
    if "spectral angle" in text or "angle" in text or "hyperspectral" in text:
        return "Spectral angle similarity"
    if "cosine" in text:
        return "Cosine similarity"

    return None


def _mentions_low_priority(user_text):
    text = user_text.lower()
    return any(
        key in text
        for key in [
            "avoid",
            "avoid using",
            "avoid if possible",
            "lower priority",
            "low priority",
            "deprioritize",
            "penalize",
            "soft penalty",
            "less preferred",
            "not preferred",
            "尽量不用",
            "降低优先级",
            "低优先级",
        ]
    )


def _detect_soft_penalty_strength(user_text):
    text = user_text.lower()

    if any(key in text for key in ["strongly", "strong", "high", "as much as possible"]):
        return "High"

    if any(key in text for key in ["slightly", "mild", "low", "a little"]):
        return "Low"

    if _mentions_low_priority(user_text):
        return "Medium"

    return None


def _mentions_fluorophore_selection(user_text):
    text = user_text.lower()
    return any(
        key in text
        for key in [
            "fluorophore",
            "fluorophores",
            "dye",
            "dyes",
            "another",
            "more",
            "additional",
            "fix",
            "fixed",
        ]
    )


def _mentions_probe_usage_without_fluorophore(user_text, found_probes, found_fluors):
    text = user_text.lower()
    use_words = ["use", "using", "with probes", "probe", "probes", "用", "使用"]

    if len(found_probes) >= 1 and not found_fluors:
        return any(w in text for w in use_words)

    return False


def _looks_like_selection_request(user_text):
    """Decide whether user text is a new selection request rather than result Q&A."""
    text = user_text.lower().strip()

    selection_keywords = [
        "use ",
        "using ",
        "select ",
        "choose ",
        "pick ",
        "fix ",
        "fixed ",
        "with ",
        "probe",
        "probes",
        "fluorophore",
        "fluorophores",
        "dye",
        "dyes",
        "laser",
        "lasers",
        "overlap",
        "cosine",
        "correlation",
        "pearson",
        "angle",
    ]

    question_keywords = [
        "why",
        "what",
        "which",
        "how",
        "should",
        "is this",
        "does this",
        "explain",
        "suggest",
        "risk",
        "risky",
    ]

    if any(k in text for k in selection_keywords):
        return True
    if any(k in text for k in question_keywords):
        return False

    return False


def normalize_ai_plan(plan, user_text, app_context):
    """Deterministic fallback rules after Gemini parsing."""
    plan = dict(plan or {})

    if not plan.get("mode"):
        plan["mode"] = DEFAULT_MODE

    if not plan.get("laser_strategy"):
        plan["laser_strategy"] = DEFAULT_LASER_STRATEGY

    if not plan.get("spectral_resolution"):
        plan["spectral_resolution"] = DEFAULT_SPEC_RESOLUTION

    if plan.get("spectral_resolution") == "1 nm (general)":
        plan["spectral_resolution"] = "1 nm"
    if plan.get("spectral_resolution") == "33 detection channels (Valm lab)":
        plan["spectral_resolution"] = "9.8 nm"

    metric_from_text = _detect_similarity_metric(user_text)
    if metric_from_text:
        plan["similarity_metric"] = metric_from_text

    if not plan.get("similarity_metric"):
        plan["similarity_metric"] = DEFAULT_SIMILARITY_METRIC

    if not plan.get("soft_penalty_strength"):
        plan["soft_penalty_strength"] = DEFAULT_SOFT_PENALTY_STRENGTH

    if not plan.get("lasers"):
        plan["lasers"] = DEFAULT_LASERS[:]

    all_fluors = sorted(set(app_context.get("inventory_pool", [])))
    all_probes = app_context.get("probes", [])

    found_fluors = _find_known_fluors(user_text, all_fluors)
    found_probes = _dedupe_probe_names(
        _find_known_probes(user_text, all_probes),
        app_context,
    )
    requested_n = _extract_number(user_text)

    if _mentions_low_priority(user_text) and found_fluors:
        existing_low_priority = list(plan.get("low_priority_fluorophores") or [])

        for fluor in found_fluors:
            if fluor not in existing_low_priority:
                existing_low_priority.append(fluor)

        plan["low_priority_fluorophores"] = existing_low_priority
        plan["soft_penalty_strength"] = (
            _detect_soft_penalty_strength(user_text)
            or plan.get("soft_penalty_strength")
            or DEFAULT_SOFT_PENALTY_STRENGTH
        )

    if _mentions_predicted(user_text):
        plan["mode"] = "Predicted spectra"
        if not plan.get("laser_strategy"):
            plan["laser_strategy"] = DEFAULT_LASER_STRATEGY
        if not plan.get("lasers"):
            plan["lasers"] = DEFAULT_LASERS[:]

    # Case A: "use EUB338 and ACT476" means By probes.
    if _mentions_probe_usage_without_fluorophore(user_text, found_probes, found_fluors):
        plan["selection_source"] = "By probes"
        plan["additional_probes"] = _dedupe_probe_names(found_probes, app_context)
        plan["fixed_fluorophores"] = []
        plan["fixed_probe_fluorophore_pairs"] = []
        plan["candidate_fluorophores"] = []
        plan["n_additional_fluorophores"] = None
        return plan

    # Case B: pool fluorophore selection.
    if _mentions_fluorophore_selection(user_text) or requested_n is not None:
        plan["selection_source"] = "All fluorophores"
        fixed_fluors = list(plan.get("fixed_fluorophores") or [])

        # Convert Gemini fixed pairs into fixed fluorophores for pool mode.
        for item in plan.get("fixed_probe_fluorophore_pairs") or []:
            fluor = item.get("fluorophore")
            if fluor in all_fluors and fluor not in fixed_fluors:
                fixed_fluors.append(fluor)

        for fluor in found_fluors:
            if fluor in all_fluors and fluor not in fixed_fluors:
                if "fix" in user_text.lower() or "fixed" in user_text.lower():
                    fixed_fluors.append(fluor)

        plan["fixed_fluorophores"] = fixed_fluors
        plan["fixed_probe_fluorophore_pairs"] = []

        if not plan.get("candidate_fluorophores"):
            plan["candidate_fluorophores"] = []

        if requested_n is not None:
            plan["n_additional_fluorophores"] = requested_n
        elif not isinstance(plan.get("n_additional_fluorophores"), int):
            plan["n_additional_fluorophores"] = DEFAULT_N_FLUOROPHORES

        return plan

    # Case C: probe names without fluorophore names.
    if found_probes and not found_fluors:
        plan["selection_source"] = "By probes"
        plan["additional_probes"] = _dedupe_probe_names(found_probes, app_context)
        plan["fixed_probe_fluorophore_pairs"] = []
        plan["fixed_fluorophores"] = []
        plan["candidate_fluorophores"] = []
        plan["n_additional_fluorophores"] = None
        return plan

    # Final fallback.
    if not plan.get("selection_source"):
        plan["selection_source"] = "All fluorophores"
        plan["fixed_fluorophores"] = []
        plan["candidate_fluorophores"] = []
        plan["n_additional_fluorophores"] = DEFAULT_N_FLUOROPHORES

    return plan


def apply_ai_plan_to_session_state(plan, app_context):
    """Apply parsed AI plan to Streamlit widget keys."""

    def valid(x, options):
        return x if x in options else None

    # Mode.
    mode_val = valid(plan.get("mode"), app_context["available_modes"])
    if mode_val:
        st.session_state["mode_radio"] = mode_val

    # Source.
    source_val = valid(plan.get("selection_source"), app_context["available_selection_sources"])
    if source_val:
        st.session_state["source_radio"] = source_val

    # Laser strategy.
    strategy_val = valid(
        plan.get("laser_strategy"),
        app_context["available_laser_strategies"],
    )
    if strategy_val:
        st.session_state["laser_strategy_radio"] = strategy_val

    # Spectral resolution.
    spec_val = plan.get("spectral_resolution")
    if spec_val == "1 nm (general)":
        spec_val = "1 nm"
    if spec_val == "33 detection channels (Valm lab)":
        spec_val = "9.8 nm"

    spec_val = valid(spec_val, app_context["available_spectral_resolutions"])
    if spec_val:
        st.session_state["spec_res_radio"] = spec_val

    # Similarity metric.
    metric_val = valid(plan.get("similarity_metric"), app_context["available_similarity_metrics"])
    if metric_val:
        st.session_state["similarity_metric_radio"] = metric_val

    # Lasers.
    lasers = plan.get("lasers") or []
    clean_lasers = []

    for x in lasers:
        try:
            clean_lasers.append(int(x))
        except Exception:
            pass

    if clean_lasers:
        st.session_state["num_lasers_input"] = len(clean_lasers)
        for i, lam in enumerate(clean_lasers, start=1):
            st.session_state[f"laser_{i}"] = int(lam)

    # By probes: fixed exact probe-fluorophore pairs.
    fixed_pairs = []

    for item in plan.get("fixed_probe_fluorophore_pairs") or []:
        probe = item.get("probe")
        fluor = item.get("fluorophore")

        canonical_probe_list = _dedupe_probe_names([probe], app_context)
        if not canonical_probe_list:
            continue

        canonical_probe = canonical_probe_list[0]
        pair = f"{canonical_probe} – {fluor}"

        if pair in app_context["probe_fluorophore_pairs"]:
            fixed_pairs.append(pair)

    fixed_pairs = list(dict.fromkeys(fixed_pairs))

    if fixed_pairs:
        st.session_state["fixed_probe_pairs"] = fixed_pairs
        st.session_state["source_radio"] = "By probes"

    # By probes: additional probes.
    additional_probes = _dedupe_probe_names(plan.get("additional_probes") or [], app_context)

    if additional_probes:
        st.session_state["picked_additional_probes"] = additional_probes
        st.session_state["source_radio"] = "By probes"

    # Pool modes.
    source_after = st.session_state.get(
        "source_radio",
        source_val or plan.get("selection_source") or "All fluorophores",
    )

    if source_after == "From readout pool":
        suffix = "pool"
        available_fluors = app_context["readout_pool"]
    elif source_after == "EUB338 only":
        suffix = "eub338"
        available_fluors = app_context["eub338_pool"]
    elif source_after == "By probes":
        suffix = "probes"
        available_fluors = app_context.get("all_probe_fluorophores", [])
    else:
        suffix = "inventory"
        available_fluors = app_context["inventory_pool"]

    low_priority_fluors = [
        f for f in (plan.get("low_priority_fluorophores") or [])
        if f in available_fluors
    ]

    if low_priority_fluors:
        st.session_state[f"low_priority_fluorophores_{suffix}"] = low_priority_fluors

        strength = plan.get("soft_penalty_strength") or DEFAULT_SOFT_PENALTY_STRENGTH
        if strength not in app_context.get("available_soft_penalty_strengths", []):
            strength = DEFAULT_SOFT_PENALTY_STRENGTH

        st.session_state[f"soft_penalty_strength_{suffix}"] = strength

    fixed_fluors = [
        f for f in (plan.get("fixed_fluorophores") or [])
        if f in available_fluors
    ]

    candidate_fluors_raw = plan.get("candidate_fluorophores") or []

    if candidate_fluors_raw:
        candidate_fluors = [
            f for f in candidate_fluors_raw
            if f in available_fluors and f not in fixed_fluors
        ]
    else:
        candidate_fluors = [f for f in available_fluors if f not in fixed_fluors]

    if source_after in ["From readout pool", "All fluorophores", "EUB338 only"]:
        st.session_state[f"fixed_fluorophores_{suffix}"] = fixed_fluors
        st.session_state[f"allowed_fluorophores_{suffix}"] = candidate_fluors

        n_add = plan.get("n_additional_fluorophores")

        if isinstance(n_add, int):
            n_add = max(0, min(n_add, len(candidate_fluors)))
        else:
            n_add = DEFAULT_N_FLUOROPHORES

        st.session_state[f"n_additional_{suffix}"] = n_add


def _submit_ai_input(app_context):
    user_text = st.session_state.get("ai_main_input", "").strip()

    if not user_text:
        return

    has_result = "last_result_context" in st.session_state
    is_new_selection = _looks_like_selection_request(user_text)

    # If result already exists, only treat input as result Q&A when the text
    # does NOT look like a new selection request.
    if (
        has_result
        and st.session_state.get("ai_input_mode") == "result_qa"
        and not is_new_selection
    ):
        try:
            result_context = st.session_state["last_result_context"]
            answer = answer_light_question(user_text, app_context, result_context)
            st.session_state["last_ai_answer"] = answer
        except Exception as exc:
            st.session_state["last_ai_answer"] = f"AI question failed: {exc}"
        finally:
            st.session_state["ai_main_input"] = ""

        return

    # Otherwise parse as a new selection request and apply to controls.
    try:
        plan_raw = parse_user_request(user_text, app_context)
        plan = normalize_ai_plan(plan_raw, user_text, app_context)
        apply_ai_plan_to_session_state(plan, app_context)

        st.session_state["last_ai_plan"] = plan
        st.session_state["ai_input_mode"] = "result_qa"
        st.session_state["last_ai_answer"] = None
        st.session_state["ai_status_message"] = "Applied AI selection request."

    except Exception as exc:
        st.session_state["ai_status_message"] = f"AI parsing failed: {exc}"

    finally:
        st.session_state["ai_main_input"] = ""


def render_ai_input_assistant(app_context):
    """Minimal ChatGPT-like AI box under the main title. Enter submits the text."""
    has_result = "last_result_context" in st.session_state
    mode = st.session_state.get("ai_input_mode", "selection")

    if has_result and mode == "result_qa":
        label = "Ask a question about this result"
        placeholder = "Example: Which selected pair is most risky, and what should I try next?"
    else:
        label = "Describe what you want to select"
        placeholder = (
            "Example: Use EUB338 and ACT476, avoid AF633 if possible, or "
            "fix EUB338 with AF488 and choose 4 fluorophores."
        )

    st.text_input(
        label,
        value="",
        placeholder=placeholder,
        key="ai_main_input",
        on_change=_submit_ai_input,
        args=(app_context,),
    )

    status = st.session_state.get("ai_status_message")

    if status:
        if status.startswith("AI parsing failed"):
            st.warning(status)
        else:
            st.caption(status)

    answer = st.session_state.get("last_ai_answer")
    if answer:
        st.markdown(answer)

    col1, _ = st.columns([1, 5])

    with col1:
        if st.button("New selection", key="ai_new_selection"):
            st.session_state["ai_input_mode"] = "selection"
            st.session_state.pop("last_ai_answer", None)
            st.session_state.pop("ai_status_message", None)
            st.rerun()


def render_ai_result_panel(result_context, app_context):
    """Compact AI actions below the result."""
    st.session_state["last_result_context"] = result_context

    with st.expander("AI suggestions", expanded=False):
        st.caption(
            "AI summarizes the optimizer result. "
            "The selected panel is computed by FluoroSelect, not by AI."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Explain", key=f"ai_explain_{result_context['run_id']}"):
                with st.spinner("Generating short explanation..."):
                    st.markdown(explain_result(result_context))

        with col2:
            if st.button("Suggest next step", key=f"ai_suggest_{result_context['run_id']}"):
                with st.spinner("Generating suggestions..."):
                    st.markdown(suggest_improvements(result_context))

        with col3:
            if st.button("Methods text", key=f"ai_methods_{result_context['run_id']}"):
                with st.spinner("Generating methods draft..."):
                    st.markdown(generate_methods_text(result_context))
