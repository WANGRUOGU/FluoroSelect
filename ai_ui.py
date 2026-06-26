# ai_ui.py
import streamlit as st

from ai_helper import (
    parse_user_request,
    explain_result,
    suggest_improvements,
    answer_light_question,
    generate_methods_text,
)

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
    """
    all_probes = sorted(probe_map.keys())

    probe_to_fluors = {}
    pair_options = []

    for probe in all_probes:
        cands = sorted([f for f in probe_map.get(probe, []) if f in dye_db])
        probe_to_fluors[probe] = cands

        for fluor in cands:
            pair_options.append(f"{probe} – {fluor}")

    return {
        "available_modes": [
            "Emission spectra",
            "Predicted spectra",
        ],
        "available_laser_strategies": [
            "Simultaneous",
            "Separate",
        ],
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
        "available_lasers_common": [
            405,
            445,
            488,
            514,
            561,
            594,
            633,
            639,
        ],
        "probes": all_probes,
        "probe_to_fluorophores": probe_to_fluors,
        "probe_fluorophore_pairs": pair_options,
        "readout_pool": readout_pool,
        "inventory_pool": inventory_pool,
        "eub338_pool": eub338_pool,
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
        candidate_fluors = [
            f for f in candidate_fluors_raw
            if f in available_fluors and f not in fixed_fluors
        ]
    else:
        candidate_fluors = [
            f for f in available_fluors
            if f not in fixed_fluors
        ]

    if source_after in ["From readout pool", "All fluorophores", "EUB338 only"]:
        st.session_state[f"fixed_fluorophores_{suffix}"] = fixed_fluors
        st.session_state[f"allowed_fluorophores_{suffix}"] = candidate_fluors

        n_add = plan.get("n_additional_fluorophores")
        if isinstance(n_add, int):
            n_add = max(0, min(n_add, len(candidate_fluors)))
            st.session_state[f"n_additional_{suffix}"] = n_add


def _submit_ai_input(app_context):
    user_text = st.session_state.get("ai_main_input", "").strip()
    if not user_text:
        return

    has_result = "last_result_context" in st.session_state

    # If result already exists, treat input as result Q&A.
    # Otherwise, treat input as selection-setting request.
    if has_result and st.session_state.get("ai_input_mode") == "result_qa":
        try:
            result_context = st.session_state["last_result_context"]
            answer = answer_light_question(user_text, app_context, result_context)
            st.session_state["last_ai_answer"] = answer
        except Exception as exc:
            st.session_state["last_ai_answer"] = f"AI question failed: {exc}"
        finally:
            st.session_state["ai_main_input"] = ""
        return

    # Default: parse selection request and apply to controls
    try:
        plan = parse_user_request(user_text, app_context)
        apply_ai_plan_to_session_state(plan, app_context)
        st.session_state["last_ai_plan"] = plan
        st.session_state["ai_input_mode"] = "result_qa"
        st.session_state["ai_status_message"] = "Applied AI selection request."
    except Exception as exc:
        st.session_state["ai_status_message"] = f"AI parsing failed: {exc}"
    finally:
        st.session_state["ai_main_input"] = ""


def render_ai_input_assistant(app_context):
    """
    Minimal ChatGPT-like AI box under the main title.
    Enter submits the text.
    """
    has_result = "last_result_context" in st.session_state
    mode = st.session_state.get("ai_input_mode", "selection")

    if has_result and mode == "result_qa":
        label = "Ask a question about this result"
        placeholder = "Example: Which selected pair is most risky, and what should I try next?"
    else:
        label = "Describe what you want to select"
        placeholder = (
            "Example: Fix EUB338 with AF488 and choose 4 additional probes "
            "using predicted spectra with 488, 561, and 639 nm lasers."
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

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("New selection", key="ai_new_selection"):
            st.session_state["ai_input_mode"] = "selection"
            st.session_state.pop("last_ai_answer", None)
            st.session_state.pop("ai_status_message", None)
            st.rerun()


def render_ai_result_panel(result_context, app_context):
    """
    Compact AI actions below the result.
    This is optional and concise.
    """
    st.session_state["last_result_context"] = result_context

    with st.expander("AI suggestions", expanded=False):
        st.caption(
            "AI summarizes the optimizer result. The selected panel is computed by FluoroSelect, not by AI."
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
