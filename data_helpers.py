# data_helpers.py
import json
import os
import re

import numpy as np
import streamlit as st
import yaml

from config import DETECTION_CHANNELS
from utils import build_effective_with_lasers


@st.cache_data(show_spinner=False)
def load_readout_pool(path, dye_names):
    """Load optional data/readout_fluorophores.yaml and keep dyes present in dyes.yaml."""
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    items = data.get("fluorophores", []) or []
    pool = sorted({s.strip() for s in items if isinstance(s, str) and s.strip()})
    dye_names = set(dye_names)
    return [f for f in pool if f in dye_names]


def get_inventory_from_probe_map(probe_map, dye_db):
    """Union of fluorophores in probe_fluor_map.yaml that also exist in dyes.yaml."""
    inv = set()
    for vals in probe_map.values():
        if not isinstance(vals, (list, tuple)):
            continue
        for f in vals:
            if isinstance(f, str):
                fs = f.strip()
                if fs and fs in dye_db:
                    inv.add(fs)
    return sorted(inv)


def get_eub338_pool(probe_map, dye_db):
    """Candidates under the EUB338 probe key, filtered to dyes.yaml presence."""
    targets = {"eub338", "eub 338", "eub-338"}

    def norm(s):
        return "".join(str(s).lower().split())

    for key in probe_map.keys():
        if norm(key) in targets:
            cands = [f for f in probe_map.get(key, []) if f in dye_db]
            return sorted({c.strip() for c in cands})

    def norm2(s):
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    for key in probe_map.keys():
        if norm2(key) == "eub338":
            cands = [f for f in probe_map.get(key, []) if f in dye_db]
            return sorted({c.strip() for c in cands})

    return []


def fluor_from_label(label: str) -> str:
    """Extract fluorophore name from 'Probe – Fluor' or return label itself."""
    return label.split(" – ", 1)[1] if " – " in label else label


def peak_wavelength_for_label(label: str, wl, dye_db) -> float:
    """Return emission peak wavelength for sorting labels from blue to red."""
    name = fluor_from_label(label)
    rec = dye_db.get(name)
    if rec is None:
        return float("inf")

    em = np.asarray(rec.get("emission", []), dtype=float)
    if em.size != len(wl) or np.max(em) <= 0:
        return float("inf")

    return float(wl[int(np.argmax(em))])


def sorted_order_by_peak(labels_list, wl, dye_db):
    peaks = [peak_wavelength_for_label(lbl, wl, dye_db) for lbl in labels_list]
    order = np.argsort(peaks)
    sorted_labels = [labels_list[i] for i in order]
    return order, sorted_labels


@st.cache_data(show_spinner=False)
def cached_build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers):
    """Cached wrapper around utils.build_effective_with_lasers."""
    # Make Streamlit cache depend on groups/powers in a stable way.
    _ = json.dumps({k: sorted(v) for k, v in sorted(groups.items())}, ensure_ascii=False)
    _ = tuple(sorted(laser_list)), laser_strategy, tuple(np.asarray(powers, float)) if powers is not None else None
    return build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers)


@st.cache_data(show_spinner=False)
def cached_interpolate_E_on_channels(wl, spectra_cols, chan_centers_nm):
    spectra_cols = np.asarray(spectra_cols, dtype=float)

    if spectra_cols.ndim == 1:
        spectra_cols = spectra_cols[:, None]

    _, n_cols = spectra_cols.shape
    out = np.zeros((len(chan_centers_nm), n_cols), dtype=float)

    for j in range(n_cols):
        y = spectra_cols[:, j]
        out[:, j] = np.interp(chan_centers_nm, wl, y, left=float(y[0]), right=float(y[-1]))

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def apply_mbs_zeroing(E_raw_on_det, laser_strategy, spec_res_mode, laser_list):
    """
    In simultaneous 9.8 nm detector-channel mode with lasers [405, 488, 561, 639],
    zero out MBS-blocked detection channels.
    """
    # Backward compatibility with the previous UI label.
    if spec_res_mode not in {"9.8 nm", "33 detection channels (Valm lab)"}:
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
    out = np.array(E_raw_on_det, copy=True)

    if out.ndim == 2:
        out[mask, :] = 0.0

    return out
