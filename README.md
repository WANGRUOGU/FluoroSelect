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

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the optimizer tests with:

```bash
python -m unittest discover -s tests -v
```
