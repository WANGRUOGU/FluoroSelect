import numpy as np
import streamlit as st

HIGH_SIMILARITY = 0.90  # Screening trigger, not an experimentally validated cutoff.


def rank_replacements(selected, labels, library, library_labels, fixed=(), threshold=HIGH_SIMILARITY):
    """Single-label replacements at unchanged settings; preserve the rest of the panel."""
    if len(labels) < 2:
        return []
    scores = selected.T @ selected
    np.fill_diagonal(scores, -np.inf)
    worst = float(np.max(scores))
    if worst < threshold:
        return []
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    rows = []
    for target in (i, j):
        if labels[target] in fixed or " – " not in labels[target]:
            continue
        probe, old = labels[target].split(" – ", 1)
        others = [k for k in range(len(labels)) if k != target]
        used = {labels[k].split(" – ", 1)[-1] for k in others}
        for c, label in enumerate(library_labels):
            dye = label.split(" – ", 1)[-1]
            spectrum = library[:, c]
            if dye == old or dye in used or not np.isfinite(spectrum).all() or np.linalg.norm(spectrum) < 1e-8:
                continue
            trial = selected.copy()
            trial[:, target] = spectrum
            similarities = trial.T @ trial
            np.fill_diagonal(similarities, -np.inf)
            new_worst = float(np.max(similarities))
            if new_worst < worst - 1e-8:
                rows.append({"Probe": probe, "Current fluorophore": old,
                             "Candidate fluorophore": dye,
                             "Current panel maximum similarity": worst,
                             "Predicted panel maximum similarity after replacement": new_worst})
    return sorted(rows, key=lambda r: r["Predicted panel maximum similarity after replacement"])


def render_purchase_advice(selected, labels, library, library_labels, fixed=()):
    if len(labels) < 2:
        return
    scores = selected.T @ selected
    np.fill_diagonal(scores, -np.inf)
    if np.max(scores) < HIGH_SIMILARITY:
        return
    st.subheader("Purchase candidates for the most similar pair")
    st.caption("Shown only when maximum cosine similarity is at least 0.90. This is a screening trigger, not a validated purchase cutoff. Fixed probe–fluorophore pairs are kept unchanged.")
    rows = rank_replacements(selected, labels, library, library_labels, fixed)
    if rows:
        st.dataframe(rows[:10], hide_index=True)
        st.info("Hypothetical labels from the spectral library, not verified vendor products. Other labels and current spectral settings are held fixed. Add a candidate under purchase options and rerun the full design to check brightness constraints and recalibrated laser powers. Confirm probe chemistry, availability and microscope performance before ordering.")
    else:
        st.caption("No single-label replacement improves the panel maximum while preserving fixed pairs. You may need to reconsider more than one label.")
