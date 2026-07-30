# FluoroSelect project handoff

Last updated: 2026-07-30

## How to resume this project in Codex

1. Clone `https://github.com/WANGRUOGU/FluoroSelect.git`.
2. Check out `codex/brightness-balance` unless these changes have already been merged.
3. Open the repository folder in Codex.
4. Ask Codex to read this file, `README.md`, and the current Git diff/log before making changes.
5. Preserve user changes and verify which branch Streamlit deploys before expecting web updates.

Suggested opening prompt on a new computer:

> Continue the FluoroSelect project. Read PROJECT_HANDOFF.md and inspect the current branch and recent commits first. Treat the decisions in the handoff as current unless I explicitly revise them.

## Project objective

FluoroSelect is a public Streamlit application for designing fluorophore panels for multiplexed spectral imaging. The intended manuscript is a Software article for PLOS Computational Biology.

The app has two principal selection modes:

1. Select a user-specified number of fluorophores from a shared library.
2. Assign one unique fluorophore to each probe from probe-specific candidate lists.

Both modes use combinatorial optimization based on spectral similarity. The application supports fixed selections, probe-specific candidates, fluorophore uniqueness, candidate restrictions, and soft penalties.

## Current panel-selection model

- Spectral confusability is represented by cosine similarity (equivalently spectral-angle similarity for nonnegative normalized spectra).
- The optimization is lexicographic: first minimize the worst selected pairwise similarity, then improve the remaining top-heavy pairwise scores without sacrificing the optimum worst-pair value.
- The similarity metric is fixed in the interface; the user-facing similarity-metric selector was removed.
- The default number of displayed pairwise scores is 5.
- The results heading is `Top pairwise scores` and does not append `Cosine similarity`.
- All worst-panel/worst-fluorophore calculations and displays were removed.

## Brightness and laser-power model

Predicted spectra incorporate excitation, emission, quantum yield, extinction coefficient, selected lasers, and panel-dependent laser-power calibration.

Brightness balance is an optional hard constraint based on the dimmest selected fluorophore relative to the brightest, which is normalized to 1:

- Off: no brightness constraint.
- Weak: minimum relative peak brightness 0.2.
- Strong: minimum relative peak brightness 0.4.

Weak is the default. Medium was removed.

Panel selection and laser-power calibration depend on one another. The app alternates them for at most four iterations. A newly selected panel is recalibrated under its own laser powers and accepted immediately when its dimmest member satisfies the chosen threshold. Repeating the same panel in two consecutive iterations is not required. Repeated-panel cycles and failure to find a feasible panel are detected.

## Current simulated image analysis

The biological-image simulation assumes one dominant fluorophore per foreground pixel.

1. Generate nonoverlapping synthetic rod-like objects with spatially varying abundance.
2. Form the clean spectral image using the selected effective spectra.
3. Scale the clean image to a peak expected count of 50.
4. Sample Poisson shot noise. The 50 is the simulation count scale, not the final 8-bit display maximum; rendered RGB images still use 0-255.
5. Classify every pixel using spectral angle/cosine similarity after L2 normalization.
6. For the winning fluorophore only, estimate abundance with closed-form one-dimensional nonnegative least squares:

   `a_hat = max(0, dot(y, s_j) / dot(s_j, s_j))`

7. Set all nonwinning fluorophore abundances to zero.

This is deliberately not general dense spectral unmixing. The classification step is scale-invariant, while abundance fitting uses the original brightness-bearing spectrum.

## Current simulation metrics

The per-fluorophore table contains only:

- RMSE, calculated from estimated versus true abundance maps.
- Accuracy, calculated directly from spectral-angle classification labels among pixels where that fluorophore is truly present.

Accuracy is displayed as a percentage, for example `99.0%`. The former `Proportion` metric was removed.

## Current interface decisions

- `Selection source` is at the top of the sidebar.
- For pool-style sources, `Number of fluorophores` appears immediately below Selection source.
- This number means final total panel size, including fixed selections.
- Probe-based mode derives panel size from selected probes and does not show that control.
- Result-page AI suggestions were removed. The optional AI input/Q&A assistant remains.

## Manuscript positioning

Target journal: PLOS Computational Biology, Software article.

The manuscript should present FluoroSelect primarily as a constraint-aware fluorophore panel-design tool, not as a general unmixing package. Spectral-angle classification and one-dimensional NNLS are downstream simulation/evaluation components used to assess candidate panels under the single-label-pixel assumption.

The abstract should emphasize:

- the panel-design problem;
- unequal probe-specific fluorophore availability;
- combinatorial constraints;
- pairwise spectral separability;
- brightness-aware, panel-dependent laser-power calibration;
- accessible browser-based implementation;
- feasibility demonstrated using simulations and real biological samples.

The user prefers a qualitative abstract without numerical results. Do not invent performance numbers. PLOS-specific availability information will eventually need the public app URL, source-code URL, OS/dependency statement, MIT license, archived release/DOI, and support channel.

PLOS Software articles are expected to include an Abstract, Author Summary, Introduction, Design and Implementation, Results, and Availability and Future Directions, with reproducible software, test data, parameters, documentation, and an OSI-compliant license.

## Proposed validation experiments

### 1. Metric-to-task concordance

Generate many candidate panels across multiple library and panel sizes. For every panel calculate:

- maximum pairwise cosine similarity;
- mean and top-k pairwise similarity;
- spectral-matrix condition number;
- Hotspot Matrix/SIF summaries from the UDS framework;
- macro classification accuracy;
- worst-fluorophore accuracy;
- dimmest-fluorophore accuracy;
- abundance RMSE.

Use Spearman rank correlation, top-panel enrichment, and selection regret to determine which design metric best predicts the downstream single-label classification task.

Do not describe this as `pairwise similarity versus UDS`. UDS is a phenomenon. The fair metric comparison is pairwise cosine similarity versus condition number versus Hotspot/SIF-based scores.

### 2. Objective ablation

Compare random selection and optimization of:

- maximum pairwise similarity only;
- mean pairwise similarity;
- top-k aggregate similarity;
- the current FluoroSelect lexicographic objective;
- condition number;
- maximum SIF/Hotspot score.

This should establish whether the current minimax-then-top-heavy objective improves worst-class and macro classification performance.

### 3. Mixture-sparsity crossover

Vary active fluorophores per pixel from one to two, three, and dense mixtures. Use the current classifier plus one-dimensional NNLS for the one-label condition and sparse/general unmixing for mixed pixels. Test the hypothesis that pairwise similarity is most appropriate for single-label images, whereas global collinearity metrics become more informative as mixtures become denser.

### 4. Brightness ablation

Vary peak expected counts (for example 25, 50, 100, and 255) and dimmest-to-brightest ratios (for example 1.0, 0.8, 0.4, 0.2, 0.1, and 0.05). Compare Off, Weak, and Strong brightness balance.

Primary endpoints should include worst-class accuracy, dimmest-fluorophore accuracy, macro accuracy, and abundance RMSE. The correct physical interpretation is that dim fluorophores yield fewer detected photons and therefore higher relative shot noise, approximately proportional to `1/sqrt(lambda)`; they do not necessarily produce more absolute noise.

### 5. Validate relative-brightness prediction

Use single-fluorophore controls under known laser configurations. Compare predicted and measured relative brightness, rankings, brightness ratios, and identification of the dimmest fluorophore. The intended claim is relative ranking within an instrument configuration, not prediction of absolute photon counts.

### 6. Laser-power iteration ablation

Compare emission-only selection, predicted spectra with equal/fixed laser powers, one-pass calibration, iterative panel-power recalibration, and iterative calibration plus brightness constraints. Report performance, feasibility, iterations, panel changes, and runtime.

### 7. Optimizer correctness and scalability

For small problems, compare against exhaustive enumeration to verify the optimum and all constraints. For larger libraries, vary library size, panel size, probe count, candidates per probe, and constraint count; report runtime and feasibility detection.

### 8. Biological validation

Compare matched panels selected by FluoroSelect, a condition-number or SIF objective, and a random or manually designed baseline. Control targets, sample, acquisition settings, laser configuration, exposure, and analysis pipeline. Suitable endpoints include classification accuracy against single-label ground truth, per-fluorophore recall, false-positive rate, abundance agreement, biological-object identification, and repeatability.

### 9. Robustness

Perturb spectral wavelength, channel scaling, background/autofluorescence, laser power, photobleaching, and reference-versus-measured spectra. Quantify performance loss and panel-selection stability.

## Reference PDFs discussed

These files were supplied locally and may need OneDrive synchronization on another computer:

- `C:\Users\Ruogu Wang\OneDrive - University at Albany - SUNY\Documents\articles and books\Measurement and prediction of UDS in spectral flow cytometry panels.pdf`
- `C:\Users\Ruogu Wang\OneDrive - University at Albany - SUNY\Documents\articles and books\Ferrer-Fontetal.-2020-DesignandOptimizationProtocolforHigh-DimensionalImmunophenotypingAssaysusingSpectralFlowCytome2.pdf`

Relevant interpretation from the UDS manuscript: cosine similarity is pairwise; condition number is a full-panel global sensitivity measure; the Hotspot Matrix and SIF are panel-specific collinearity/variance-inflation measures. They were developed for linear spectral unmixing and should be compared fairly with FluoroSelect under tasks with different mixture sparsity.

## Repository state at handoff

Remote: `https://github.com/WANGRUOGU/FluoroSelect.git`

Working branch: `codex/brightness-balance`

Important commits before this handoff:

- `4937cf1` - iterative brightness-balance constraints;
- `22a2bd3` - simplified iteration;
- `7d29afa` - accept self-calibrated feasible panels;
- `0a836f4` - streamlined panel controls and results;
- `2120260` - increased simulated shot noise using peak expected count 50;
- `61ec160` - spectral-angle classification and one-dimensional NNLS abundance estimation.

The deployed Streamlit app will not change merely because this branch is pushed. Confirm the branch configured in Streamlit and merge or redeploy as appropriate.

## Immediate next steps

1. Merge or open a pull request from `codex/brightness-balance` when ready.
2. Confirm the public Streamlit URL and deployed branch.
3. Create a versioned release and archive it with Zenodo before manuscript submission.
4. Add complete user documentation, test data, reproducible parameters, and a support channel.
5. Finalize the experimental plan before writing Results.
6. Draft the qualitative abstract and PLOS Author Summary without inventing results.
