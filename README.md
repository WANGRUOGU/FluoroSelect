# FluoroSelect

FluoroSelect is a Streamlit application for constraint-aware fluorophore panel
design. It can select a requested number of fluorophores from a common pool or
assign one globally unique fluorophore to each probe from probe-specific
candidate lists.

## Brightness balance

In **Predicted spectra** mode, the sidebar offers three brightness-balance
settings based on the panel-specific laser-power calibration:

- **Off:** no brightness constraint.
- **Weak:** dimmest relative peak brightness must be at least 0.2.
- **Strong:** dimmest relative peak brightness must be at least 0.4.

The brightest member of the current panel is normalized to 1. FluoroSelect then
alternates laser-power calibration and constrained panel selection for at most
four iterations, updating that reference after each selected panel. Each newly
selected panel is immediately recalibrated and accepted as soon as it satisfies
the selected minimum brightness under its own laser powers; the panel does not
need to repeat in two consecutive iterations.

## Simulated classification and abundance estimation

The rod images illustrate the selected spectra. The quantitative table uses the
manuscript evaluation separately: exactly 100 pixels per dye, one dye per pixel,
uniform abundances from 0.5 to 1, five independent noise replicates, and a peak
expected single-channel count of 25 before Poisson sampling. The number of pixels
therefore grows with panel size. These are simulations, not biological measurements.

Each pixel is classified by spectral angle (cosine similarity), followed by a
one-dimensional nonnegative least-squares abundance fit for the classified dye.
Zero-photon pixels are unclassified and count as errors. Other dye estimates are zero.
RMSE is computed only on pixels truly containing the corresponding dye, not over
the background. The interface also reports macro accuracy, worst-class accuracy,
true-class RMSE and worst-class RMSE. Worst-class endpoints are computed within
each replicate before averaging the five results.

Spectra are nonnegative. Negative baseline tails in the supplied profiles are
clipped to zero when loading, consistently for selection and image simulation.

## Selection objective

Both app modes minimize maximum pairwise cosine similarity under the requested
constraints. A second solve uses a power-weighted sum of pairwise similarities
and optional candidate preferences, without worsening the primary value.
Fixed inclusions count toward the total panel size. Probe assignments use one
candidate per probe and never reuse the same dye. Solver failures and incomplete
or nonintegral outputs are rejected, rather than rounded into a suggested panel.
Brightness feasibility at one set of calibrated powers is not a proof about
every possible panel and power configuration.

## Run locally

Use Python 3.12 and install the supplied requirements. PuLP is pinned to 3.3.0
because this model uses the PuLP 3.x API and bundled CBC. PuLP 4 removes
`PULP_CBC_CMD` and changes model construction, so upgrading PuLP alone is not supported.
The CBC backend is initialized when solving, rather than during module import.

Online app: https://choosefluorophore.streamlit.app/

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the optimizer tests with:

```bash
python -m unittest discover -s tests -v
```
