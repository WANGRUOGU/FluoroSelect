# app.py
import streamlit as st

from ai_ui import build_ai_app_context, render_ai_input_assistant
from config import DYES_YAML, PROBE_MAP_YAML, READOUT_POOL_YAML
from data_helpers import get_eub338_pool, get_inventory_from_probe_map, load_readout_pool
from runners import run_fluoroselect
from selection_ui import build_selection_groups, render_sidebar_config
from utils import load_dyes_yaml, load_probe_fluor_map


st.set_page_config(page_title="Fluorophore Selection", layout="wide")
st.title("Fluorophore Selection for Multiplexed Imaging")

# -------------------- Data --------------------
wl, dye_db = load_dyes_yaml(DYES_YAML)
probe_map = load_probe_fluor_map(PROBE_MAP_YAML)
readout_pool = load_readout_pool(READOUT_POOL_YAML, list(dye_db.keys()))
inventory_pool = get_inventory_from_probe_map(probe_map, dye_db)
eub338_pool = get_eub338_pool(probe_map, dye_db)

# -------------------- AI input assistant --------------------
ai_app_context = build_ai_app_context(
    probe_map=probe_map,
    dye_db=dye_db,
    readout_pool=readout_pool,
    inventory_pool=inventory_pool,
    eub338_pool=eub338_pool,
)
render_ai_input_assistant(ai_app_context)

# -------------------- User controls --------------------
config = render_sidebar_config(wl)
selection = build_selection_groups(
    source_mode=config["source_mode"],
    probe_map=probe_map,
    dye_db=dye_db,
    readout_pool=readout_pool,
    inventory_pool=inventory_pool,
    eub338_pool=eub338_pool,
)

# -------------------- Execute --------------------
run_fluoroselect(
    wl=wl,
    dye_db=dye_db,
    groups=selection["groups"],
    config=config,
    constraints=selection,
    app_context=ai_app_context,
)

st.caption(
    "AI-assisted features are optional and are used only for input parsing, "
    "lightweight Q&A, result explanation, and drafting suggestions. "
    "Optimization results are computed by the FluoroSelect algorithm."
)
