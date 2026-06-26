# ai_ui.py
import streamlit as st

from ai_helper import (
    answer_light_question,
    explain_result,
    generate_methods_text,
    parse_user_request,
    suggest_improvements,
)


def build_ai_app_context(probe_map, dye_db, readout_pool, inventory_pool, eub338_pool):
    all_probes = sorted(probe_map.keys())
    probe_to_fluors = {}
    pair_options = []

    for probe in all_probes:
        cands = sorted([f for f in probe_map.get(probe, []) if f in dye_db])
        probe_to_fluors[probe] = cands
        for fluor in cands:
            pair_options.append(f"{probe} – {fluor}")

    return {
        "available_modes": ["Emission spectra", "Predicted spectra"],
        "available_laser_strategies": ["Simultaneous", "Separate"],
        "available_spectral_resolutions": ["1 nm (general)", "33 detection channels (Valm lab)"],
        "available_selection_sources": ["By probes", "From readout pool", "All fluorophores", "EUB338 only"],
        "available_lasers_common": [405, 445, 488, 514, 561, 594, 633, 639],
        "probes": all_probes,
        "probe_to_fluorophores": probe_to_fluors,
        "probe_fluorophore_pairs": pair_options,
        "readout_pool": readout_pool,
        "inventory_pool": inventory_pool,
        "eub338_pool": eub338_pool,
    }


def apply_ai_plan_to_session_state(plan, app_context):
    def valid(x, options):
        return x if x in options else None

    mode_val = valid(plan.get("mode"), app_context["available_modes"])
    if mode_val:
        st.session_state["mode_radio"] = mode_val

    source_val = valid(plan.get("selection_source"), app_context["available_selection_sources"])
    if source_val:
        st.session_state["source_radio"] = source_val

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

    fixed_pairs = []
    for item in plan.get("fixed_probe_fluorophore_pairs") or []:
        probe = item.get("probe")
        fluor = item.get("fluorophore")
        pair = f"{probe} – {fluor}"
        if pair in app_context["probe_fluorophore_pairs"]:
            fixed_pairs.append(pair)

    if fixed_pairs:
        st.session_state["fixed_probe_pairs"] = fixed_pairs
        st.session_state["source_radio"] = "By probes"

    additional_probes = [p for p in (plan.get("additional_probes") or []) if p in app_context["probes"]]
    if additional_probes:
        st.session_state["picked_additional_probes"] = additional_probes
        st.session_state["source_radio"] = "By probes"

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

    if source_after in ["From readout pool", "All fluorophores", "EUB338 only"]:
        fixed_fluors = [f for f in (plan.get("fixed_fluorophores") or []) if f in available_fluors]

        candidate_raw = plan.get("candidate_fluorophores") or []
        if candidate_raw:
            candidate_fluors = [f for f in candidate_raw if f in available_fluors and f not in fixed_fluors]
        else:
            candidate_fluors = [f for f in available_fluors if f not in fixed_fluors]

        st.session_state[f"fixed_fluorophores_{suffix}"] = fixed_fluors
        st.session_state[f"allowed_fluorophores_{suffix}"] = candidate_fluors

        n_add = plan.get("n_additional_fluorophores")
        if isinstance(n_add, int):
            st.session_state[f"n_additional_{suffix}"] = max(0, min(n_add, len(candidate_fluors)))


def render_ai_input_assistant(app_context):
    with st.expander("AI input assistant", expanded=False):
        st.caption(
            "Optional: AI translates natural language into FluoroSelect settings. "
            "The optimizer still performs fluorophore selection."
        )

        user_ai_request = st.text_area(
            "Describe what you want to select",
            placeholder=(
                "Example: Fix EUB338 with AF488, then choose additional probes. "
                "Use predicted spectra with 488, 561, and 639 nm lasers."
            ),
            key="ai_user_request",
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            parse_clicked = st.button("Parse and apply", key="ai_parse_apply")

        with col2:
            st.caption(
                "Example: Fix EUB338 with AF488 and choose 4 additional probes "
                "using predicted spectra with 488, 561, and 639 nm lasers."
            )

        if parse_clicked:
            if not user_ai_request.strip():
                st.warning("Please enter a request first.")
            else:
                with st.spinner("Parsing request with Gemini..."):
                    try:
                        plan = parse_user_request(user_ai_request, app_context)
                        apply_ai_plan_to_session_state(plan, app_context)
                        st.session_state["last_ai_plan"] = plan
                        st.success("AI plan applied to the sidebar controls.")

                        if plan.get("warnings"):
                            st.warning("\n".join(plan["warnings"]))

                        st.rerun()
                    except Exception as exc:
                        st.error(f"AI parsing failed: {exc}")

        if "last_ai_plan" in st.session_state:
            st.caption("Last parsed AI plan")
            st.json(st.session_state["last_ai_plan"])


def render_ai_result_panel(result_context, app_context):
    with st.expander("AI assistant for this result", expanded=False):
        st.caption(
            "AI assistance is optional. The fluorophore selection is computed by "
            "the FluoroSelect optimizer, not by the AI model."
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Explain this result", key=f"ai_explain_{result_context['run_id']}"):
                with st.spinner("Generating explanation..."):
                    st.markdown(explain_result(result_context))

        with col2:
            if st.button("Suggest improvements", key=f"ai_suggest_{result_context['run_id']}"):
                with st.spinner("Generating suggestions..."):
                    st.markdown(suggest_improvements(result_context))

        if st.button("Generate Methods paragraph", key=f"ai_methods_{result_context['run_id']}"):
            with st.spinner("Generating Methods draft..."):
                st.markdown(generate_methods_text(result_context))

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
                    st.markdown(answer_light_question(question, app_context, result_context))
