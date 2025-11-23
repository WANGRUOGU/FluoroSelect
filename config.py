# config.py
import numpy as np

# Paths for data
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"

# Detection channels for Valm lab (33 channels)
DETECTION_CHANNELS = np.array([
    414, 423, 432, 441, 450, 459, 468, 477, 486,
    494, 503, 512, 521, 530, 539, 548, 557, 566,
    575, 583, 592, 601, 610, 619, 628, 637, 646,
    655, 664, 673, 681, 690, 717
], dtype=float)

# Default RGB colors for up to 8 dyes; beyond that, generate procedurally
DEFAULT_COLORS = np.array([
    [0.95, 0.25, 0.25],  # red-ish
    [0.25, 0.65, 0.95],  # blue-ish
    [0.25, 0.85, 0.35],  # green-ish
    [0.90, 0.70, 0.20],  # yellow-ish
    [0.80, 0.40, 0.80],  # purple-ish
    [0.25, 0.80, 0.80],  # cyan-ish
    [0.85, 0.50, 0.35],  # orange-ish
    [0.60, 0.60, 0.60],  # gray
], dtype=float)
