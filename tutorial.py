import streamlit as st


def render_tutorial():
    st.header("FluoroSelect tutorial")
    st.markdown("""
### What can I do here?
Choose fluorophores that are easier to tell apart, while keeping the probes or
labels that you need. FluoroSelect minimizes the similarity of the hardest-to-
distinguish pair. AI helps fill in settings, but does not select the final panel.

### Choose your starting point
- **By probes:** choose named probes and one candidate fluorophore for each.
  This is useful for directly labeled probes. Fix a probe–fluorophore pair when
  you already use it or prefer it based on microscope experience.
- **From readout pool:** choose a specified number of labels from a shared list.
  This is useful for planning readout labels in a two-step workflow. The app
  selects fluorophores, not probe sequences, and does not validate readout chemistry.
- **All fluorophores:** choose from the fluorophores in the saved probe lists,
  including any probe options you add in this session.
- **EUB338 only:** choose from the labels listed for EUB338.

### Emission spectra or Predicted spectra?
**Emission spectra** compares spectral shapes after peak normalization. It is a
simple starting point when you want to compare overlap without modeling excitation.
**Predicted spectra** also uses the selected lasers, excitation spectra, quantum
yield and extinction coefficient. You can model simultaneous or separate excitation
and spectral sampling. These are model predictions, not measured microscope signals.
Brightness balancing is an optional constraint, not a guarantee of experimental brightness.

### What are similarities and top-K?
Cosine similarity compares two spectral shapes. Values near 1 mean more similar
shapes and potentially harder separation. The result lists the most similar pairs
first. The display count (previously called top-K) only controls how many pairs
you see. Five was a display default, not a scientific minimum. You can now choose
1–50. Showing more pairs gives more detail but a longer table. It does not change
the optimization, which considers every eligible pair.

### Planning a purchase
1. Open **Add probe / compare purchase options** on the Panel design page.
2. Enter the new probe or an existing probe name and select fluorophores you
   could order. Applying options replaces that probe's candidate list for this session.
3. In **By probes**, fix the probe–fluorophore pairs you want to keep. Include
   the purchase probe among the additional probes, then run the selection.
4. Compare the selected label and pair similarities. To compare another offer,
   change its candidate list or fix a particular candidate and rerun.

Only fluorophores with spectra in the app library can be compared. Added options
are hypothetical and are not saved to the shared inventory. Check vendor availability,
probe specificity, labeling chemistry and microscope performance before purchasing.

### How does the AI assistant work?
The assistant uses Google's Gemini API through the app owner's server-side API
key. Prompts and responses consume tokens under that API project's quota and
billing settings, not your personal ChatGPT allowance. We do not claim that usage
is free or unlimited. Your description and relevant app context are sent to Google,
so avoid sensitive information.

A **503** response means the model service is temporarily busy, even if you are
the only visitor to this app. A **429** response can mean a quota or rate limit.
The app retries temporary failures briefly, keeps failed input for retry, and
lets you continue with manual controls. AI is optional.
""")


def merge_probe_options(probe_map, custom, dye_db):
    from selection_ui import _canonicalize_probe_map, _norm_probe_name
    merged, _ = _canonicalize_probe_map(probe_map, dye_db)
    for name, candidates in custom.items():
        valid = sorted(set(f for f in candidates if f in dye_db))
        if not name.strip() or " – " in name or not valid:
            raise ValueError("Enter a probe name and at least one library fluorophore.")
        canonical = next((p for p in merged if _norm_probe_name(p) == _norm_probe_name(name)), name.strip())
        merged[canonical] = valid
    return merged


def render_purchase_options(probe_map, dye_db):
    with st.expander("Add probe / compare purchase options"):
        st.caption("Session-only candidates, not a verified vendor catalog. Fix existing pairs below to keep them.")
        with st.form("purchase_options"):
            name = st.text_input("New or existing probe name")
            candidates = st.multiselect("Fluorophores you could purchase", sorted(dye_db))
            submitted = st.form_submit_button("Apply probe options")
        if submitted:
            custom = dict(st.session_state.get("purchase_probe_options", {}))
            custom[name.strip()] = candidates
            try:
                merge_probe_options(probe_map, custom, dye_db)
            except ValueError as exc:
                st.warning(str(exc))
            else:
                st.session_state["purchase_probe_options"] = custom
                st.success("Options added. Choose this probe under By probes below.")
        custom = st.session_state.get("purchase_probe_options", {})
        for name, candidates in custom.items():
            st.caption(f"{name}: {', '.join(candidates)}")
        if st.button("Clear added probe options"):
            st.session_state.pop("purchase_probe_options", None)
            st.rerun()
    return merge_probe_options(probe_map, st.session_state.get("purchase_probe_options", {}), dye_db)
