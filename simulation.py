"""Compatibility imports. Simulation implementations live in sim_core."""
from sim_core import (
    to_uint8_gray, argmax_labelmap, nls_unmix,
    spectral_angle_classify_and_estimate, colorize_single, colorize_composite,
    simulate_rods_and_unmix, simulate_balanced_pixels,
)
