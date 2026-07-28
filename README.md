# FluoroSelect

FluoroSelect is a Streamlit application for constraint-aware fluorophore panel
design. It can select a requested number of fluorophores from a common pool or
assign one globally unique fluorophore to each probe from probe-specific
candidate lists.

## Brightness balance

In **Predicted spectra** mode, the sidebar offers four brightness-balance
settings based on the panel-specific laser-power calibration:

- **Off:** no brightness constraint.
- **Weak:** dimmest relative peak brightness must be at least 0.0625.
- **Medium:** dimmest relative peak brightness must be at least 0.125.
- **Strong:** dimmest relative peak brightness must be at least 0.25.

The brightest member of the current panel is normalized to 1. FluoroSelect then
alternates laser-power calibration and constrained panel selection for at most
four iterations, updating that reference after each selected panel. A result is
displayed only when the final stable panel, after recalibrating laser powers,
satisfies the selected minimum brightness.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the optimizer tests with:

```bash
python -m unittest discover -s tests -v
```
