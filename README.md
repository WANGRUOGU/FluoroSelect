# FluoroSelect

FluoroSelect is a Streamlit application for constraint-aware fluorophore panel
design. It can select a requested number of fluorophores from a common pool or
assign one globally unique fluorophore to each probe from probe-specific
candidate lists.

## Brightness balance

In **Predicted spectra** mode, the sidebar offers four brightness-balance
settings based on the panel-specific laser-power calibration:

- **Off:** no brightness constraint.
- **Weak:** brightest-to-dimmest predicted peak brightness must be at most 16x.
- **Medium:** the ratio must be at most 8x.
- **Strong:** the ratio must be at most 4x.

FluoroSelect alternates laser-power calibration and constrained panel selection
until the panel is stable. A result is displayed only when the final panel,
after recalibrating laser powers, satisfies the selected brightness limit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the optimizer tests with:

```bash
python -m unittest discover -s tests -v
```
