# Planet Hunters: Find New Worlds in Starlight 🌌

**Hack4Hack ML Challenge 2026 — Round 1**

A machine learning project focused on classifying stellar light curves into eight astronomical categories using time-series feature engineering, stellar metadata, centroid measurements, and ensemble learning.

## Overview

Stars observed by space telescopes produce time-series measurements of their brightness, known as *light curves*. These signals can reveal different types of stellar behavior, including planetary transits, stellar variability, and eclipsing binary systems.

The objective of this challenge was to build a multiclass classification model that identifies the underlying category of each star from its observed data.

This project explores a complete machine learning workflow, including data preprocessing, feature engineering, model evaluation, ensemble learning, and test-set prediction.

## Problem Statement

The objective of the Planet Hunters challenge was to classify stars into one of eight categories based on their observed light curves, centroid measurements, and stellar metadata.

This is a **multiclass classification problem** in which each star must be assigned a single target label.

The model was evaluated using the **macro-averaged F1 score**, which calculates the F1 score independently for each class and then takes the unweighted average. This metric gives equal importance to every class, making performance on less frequent categories important.

### Target Classes

| Class | Description |
|---|---|
| `quiet_star` | Stars exhibiting relatively stable brightness |
| `starspots` | Stars showing variability associated with stellar spots |
| `eclipsing_binary` | Binary star systems exhibiting periodic eclipses |
| `planet` | Stars with brightness signals associated with planetary transits |
| `pulsating_star` | Stars exhibiting brightness variations caused by pulsations |
| `flare_star` | Stars exhibiting sudden increases in brightness |
| `instrument_glitch` | Observations affected by instrumental anomalies |
| `background_binary` | Signals associated with background eclipsing binary systems |

## Dataset

The dataset provided for the challenge contains stellar observations for supervised training and test-time prediction.

### Dataset Overview

| Dataset | Number of Stars | Original Flux Measurements per Star |
|---|---:|---:|
| Training set | 18,000 | 1,315 |
| Test set | 7,000 | 1,315 |

### Data Components

The dataset includes the following files:

| File | Description |
|---|---|
| `train_stars.csv` | Training star metadata and target labels |
| `test_stars.csv` | Test star metadata |
| `train_flux.npy` | Training light-curve flux measurements |
| `test_flux.npy` | Test light-curve flux measurements |
| `train_centroid.npy` | Training centroid measurements |
| `test_centroid.npy` | Test centroid measurements |
| `train_cadences.csv` | Training cadence information and quality flags |
| `test_cadences.csv` | Test cadence information and quality flags |
| `sample_submission.csv` | Example submission file specifying the required output format |

### Stellar Metadata

The star metadata includes properties such as:

- `tmag` — TESS magnitude
- `teff` — Effective temperature
- `logg` — Surface gravity
- `radius` — Stellar radius
- `contamination` — Contamination measurement
- `camera` and `ccd` — Instrument configuration identifiers

The training metadata also contains the target label used for supervised learning.

### Data Quality Filtering

Cadence quality flags were used to filter observations, retaining measurements where `quality == 0`.

After filtering, the flux arrays contained:

- **Training set:** 1,221 measurements per star
- **Test set:** 1,219 measurements per star

Flux and centroid measurements were filtered consistently with their corresponding cadence information. Feature extraction was performed separately on the training and test datasets to accommodate their different numbers of retained cadences.

## Methodology

The project follows a structured machine learning workflow, starting with raw stellar observations and ending with multiclass predictions for the test dataset.

### 1. Data Loading and Quality Filtering

- Loaded flux measurements, centroid arrays, cadence information, and stellar metadata.
- Applied cadence quality filtering using the supplied quality flags.
- Filtered flux and centroid measurements consistently with the corresponding cadence indices.
- Processed training and test data separately to accommodate differences in the number of valid observations.

### 2. Feature Engineering

Raw light curves and stellar metadata were transformed into numerical features suitable for traditional machine learning models.

The engineered features capture brightness distributions, temporal behavior, periodic patterns, centroid variations, potential instrumental artifacts, and transit-like dips.

### 3. Feature Assembly

Features extracted from different sources were combined into a single tabular representation for each star.

The final feature matrix contained **40 features per star**. Training and test feature matrices were checked for matching columns, consistent ordering, and finite numerical values.

### 4. Model Training and Validation

- Used a stratified 80/20 train-validation split.
- Evaluated models using macro-averaged F1 score.
- Compared individual classifiers and probability-based ensembles.
- Used validation results to guide model selection and blending decisions.

### 5. Final Prediction

After selecting the best-performing ensemble, the models were retrained on the full labeled training dataset.

Class probabilities from the individual models were aligned and combined using weighted averaging. The class with the highest final probability was selected as the predicted label.

The resulting predictions were formatted according to the provided sample submission structure.

## Feature Engineering

Feature engineering was a key component of the solution. Instead of feeding raw light curves directly into traditional machine learning classifiers, the observations were transformed into **40 numerical features** describing stellar behavior, temporal patterns, and instrumental effects.

### Feature Groups

| Feature Group | Number of Features | Purpose |
|---|---:|---|
| Flux Statistics | 12 | Capture brightness distribution and variability |
| Stellar Metadata | 5 | Represent stellar properties and contamination |
| Temporal Features | 3 | Describe trends and changes between consecutive observations |
| Centroid Features | 8 | Capture positional variations and centroid movement |
| Periodic Features | 5 | Represent recurring patterns using autocorrelation |
| Artifact Features | 3 | Capture potential instrumental anomalies |
| Transit/Dip Features | 4 | Characterize flux dips and transit-like behavior |
| **Total** | **40** | |

### 1. Flux Statistics

Statistical features were extracted from the filtered flux measurements to summarize the brightness distribution and variability of each star.

These features provide a compact representation of the light curve without retaining every individual observation.

### 2. Stellar Metadata

Five metadata features were included:

- TESS magnitude (`tmag`)
- Effective temperature (`teff`)
- Surface gravity (`logg`)
- Stellar radius (`radius`)
- Contamination (`contamination`)

These measurements provide additional context that may help distinguish between stellar classes.

### 3. Temporal Features

Three features were selected to characterize changes over time:

- `trend_slope`
- `diff_abs_median`
- `diff_std`

These capture the overall flux trend and the magnitude and variability of successive flux differences.

### 4. Centroid Features

Eight features were extracted from the centroid measurements to represent positional behavior and movement.

Centroid information can provide additional evidence for distinguishing astrophysical signals from instrumental effects or background sources.

### 5. Periodic Features

Five autocorrelation-based features were extracted to capture recurring patterns in the light curves.

Periodic behavior is relevant to classes such as eclipsing binaries, pulsating stars, and stars exhibiting rotational variability.

### 6. Artifact Features

Three features were designed to capture patterns associated with potential instrumental anomalies.

These features complement the brightness statistics and centroid measurements when distinguishing instrumental effects from genuine stellar variability.

### 7. Transit and Dip Features

Four features were extracted to characterize flux dips and related light-curve behavior.

These features provide additional information for identifying transit-like events and distinguishing them from other forms of stellar variability.

### Feature Assembly

All feature groups were concatenated into a single feature matrix. The same feature definitions and column ordering were maintained for the training and test datasets.

The resulting 40-dimensional representation was used to train and evaluate the classification models.

## Model Selection and Training

Multiple machine learning classifiers were evaluated to identify an effective approach for the eight-class stellar classification task. Model selection was based on validation performance using the macro-averaged F1 score.

### Models Evaluated

#### 1. HistGradientBoostingClassifier

A histogram-based gradient boosting classifier was used as the baseline model.

It learns nonlinear relationships between engineered features and target classes while providing an efficient approach for training on tabular data.

**Configuration:**
- Maximum iterations: `300`
- Learning rate: `0.06`
- Class weighting: `balanced`
- Random state: `0`

#### 2. ExtraTreesClassifier

An Extra Trees ensemble was evaluated to capture patterns through randomized decision trees.

It provides a different learning strategy from gradient boosting, making it a useful candidate for probability blending.

**Configuration:**
- Number of estimators: `500`
- Minimum samples per leaf: `2`
- Maximum features: `0.9`
- Class weighting: `balanced`
- Random state: `42`

#### 3. CatBoostClassifier

CatBoost was evaluated as an additional gradient-boosting model. Its predictions were incorporated into the final ensemble to combine its classification behavior with that of the other models.

**Configuration of the final model:**
- Depth: `5`
- Learning rate: `0.05`
- Maximum iterations: `2,500`
- Loss function: `MultiClass`
- Class weighting: `Balanced`
- Random seed: `42`

### Ensemble Learning

Rather than relying on a single classifier, probability predictions from multiple models were combined using weighted averaging.

First, HistGradientBoosting and ExtraTrees predictions were blended:

$$
P_{\text{HGB-ET}} = 0.75P_{\text{HGB}} + 0.25P_{\text{ET}}
$$

The resulting probabilities were then combined with CatBoost predictions:

$$
P_{\text{final}} = 0.60P_{\text{HGB-ET}} + 0.40P_{\text{CatBoost}}
$$

The class with the highest final probability was selected as the predicted label.

### Why Use an Ensemble?

The models use different tree-based learning strategies and can capture different relationships within the engineered features. Combining their probability outputs was intended to improve generalization compared with relying on one classifier alone.

The blending weights were selected based on local validation experiments. The ensemble performed better on the local validation split than the individual models tested during this iteration.

## Validation Results and Experiments

Model performance was evaluated using a stratified 80/20 train-validation split, with macro-averaged F1 as the evaluation metric.

The training data was divided into:
- **Training subset:** 14,400 stars
- **Validation subset:** 3,600 stars

### Local Validation Results

| Model / Experiment | Validation Macro-F1 |
|---|---:|
| HistGradientBoosting baseline | 0.5308 |
| ExtraTrees standalone model | 0.5091 |
| HistGradientBoosting + ExtraTrees blend | 0.5339 |
| CatBoost standalone model | 0.5272 |
| Three-model ensemble | **0.5366** |

The three-model ensemble achieved the highest macro-F1 score among the configurations tested locally.

### Public Leaderboard Results

Several submissions were evaluated on the public leaderboard.

| Submission | Public Macro-F1 |
|---|---:|
| HistGradientBoosting baseline | 0.3989 |
| Model with camera and CCD features | 0.3845 |
| Starspots probability adjustment | 0.3962 |
| HistGradientBoosting + ExtraTrees blend | 0.4055 |
| Final CatBoost ensemble | **0.4175** |

The final ensemble achieved a public leaderboard score of **0.4175**, the best result among the submissions tested for this project.

### Experiments and Observations

- **Ensemble learning:** Combining HistGradientBoosting and ExtraTrees improved local validation performance over either model alone.
- **Additional model diversity:** Incorporating CatBoost improved the local validation score to 0.5366 and produced the strongest public submission tested.
- **Metadata experiments:** Adding camera and CCD identifiers reduced the public score in the tested configuration.
- **Class probability adjustment:** Increasing the probability assigned to the `starspots` class did not improve local validation performance, so the adjustment was not retained.
- **Train-test distribution shift:** Domain-classification experiments produced ROC-AUC values of approximately 0.83–0.94 in different setups. This suggests that the training and test datasets may differ substantially, although it does not establish the cause of the public leaderboard performance gap.

### Key Observation

The local validation score was higher than the public leaderboard score. This highlights an important challenge in machine learning competitions: strong performance on a single validation split does not necessarily translate into equivalent performance on unseen data.

More systematic validation, deeper investigation of distribution differences, and targeted experiments would be valuable next steps.

## Limitations and Future Improvements

Although the final ensemble improved the public leaderboard score over the initial baseline, several limitations remain.

### Limitations

- **Validation and leaderboard gap:** The local validation macro-F1 score was 0.5366, while the best public leaderboard score was 0.4175. This indicates that performance on the local validation split did not fully generalize to the leaderboard dataset.
- **Potential distribution shift:** Domain-classification experiments suggested differences between the training and test datasets. Further investigation is needed to understand the nature and impact of these differences.
- **Hand-engineered features:** The models relied on a compact set of 40 engineered features. Some useful patterns in the original light curves may not have been captured by these features.
- **Limited validation strategy:** Model selection relied primarily on a single stratified train-validation split. Results may vary across different splits.
- **Limited experimental time:** The final solution was developed under the challenge deadline, restricting the time available for systematic error analysis and more extensive experimentation.

### Future Improvements

#### 1. Stronger Validation

Use stratified cross-validation to obtain more reliable performance estimates and determine whether model improvements remain consistent across multiple splits.

#### 2. Class-Level Error Analysis

Inspect per-class precision, recall, F1 scores, and confusion matrices to identify which stellar categories are frequently misclassified.

#### 3. Distribution Shift Investigation

Compare feature distributions between the training and test datasets, investigate possible sources of distribution shift, and evaluate whether any proposed mitigation method improves validation performance.

#### 4. Improved Feature Engineering

Explore additional light-curve characteristics, including more robust periodicity measurements, transit-shape descriptors, and features that capture temporal structure.

#### 5. Systematic Model Experiments

Compare models and feature groups through controlled experiments. Record each experiment's configuration, validation results, and public leaderboard score to distinguish genuine improvements from random variation.

#### 6. Better Probability Calibration and Ensembling

Evaluate whether model probabilities are reliable and whether alternative blending weights improve performance across multiple validation splits rather than only one.

### Lessons Learned

This challenge reinforced the importance of understanding the data and evaluation metric before expanding model complexity.

A stronger solution requires more than trying additional algorithms. It depends on reliable validation, targeted error analysis, controlled experiments, and evidence that each change improves generalization.

## Project Structure

The project separates feature engineering code, experimental notebooks, raw datasets, and generated submission files.

```
text
planet-hunter/
├── data/
│   └── raw/
│       ├── sample_submission.csv
│       ├── train_stars.csv
│       ├── test_stars.csv
│       ├── train_flux.npy
│       ├── test_flux.npy
│       ├── train_centroid.npy
│       ├── test_centroid.npy
│       ├── train_cadences.csv
│       └── test_cadences.csv
├── notebooks/
│   └── artifacts/
│       ├── submission.csv
│       ├── submission_2.csv
│       ├── submission_3_starspots.csv
│       ├── submission_4_blend.csv
│       └── submission_5_catboost_blend.csv
├── src/
│   └── features/
│       ├── centroid.py
│       ├── periodic.py
│       ├── artifacts.py
│       └── dip.py
└── README.md
```
## Setup and Reproducibility

### Prerequisites

The project uses Python and common machine learning libraries for data processing, feature engineering, model training, and evaluation.

You will need:

- Python
- NumPy and pandas for data processing
- scikit-learn for model training and evaluation
- CatBoost for gradient boosting
- Jupyter Notebook or JupyterLab for interactive experimentation

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd planet-hunter
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment.

   **Windows:**

   ```bash
   .venv\Scripts\activate
   ```

   **Linux/macOS:**

   ```bash
   source .venv/bin/activate
   ```

4. Install the required libraries:

   ```bash
   pip install numpy pandas scikit-learn catboost jupyter
   ```

### Dataset Setup

Place the challenge-provided files in the `data/raw/` directory, preserving their expected filenames and formats.

The dataset includes CSV metadata and cadence files alongside NumPy arrays containing flux and centroid measurements.

### Running the Experiments

Open the project notebooks in Jupyter and execute the workflow in the required order:

1. Load the training and test datasets.
2. Apply cadence quality filtering.
3. Extract and assemble the engineered features.
4. Train and evaluate the candidate models.
5. Combine model probability predictions using the selected ensemble weights.
6. Generate predictions and format the submission file according to `sample_submission.csv`.

### Reproducibility Notes

- Use the documented random seeds and model configurations where applicable.
- Keep training and test feature columns aligned.
- Record validation scores and leaderboard results for each experiment.
- Preserve the original dataset files to ensure experiments use consistent inputs.

**Note:** The installation commands list the main libraries used in the project, but they are not a verified, version-pinned environment. For exact reproducibility, add a `requirements.txt` containing the dependencies and versions confirmed by your working environment.

## Results Summary

### Final Performance

| Metric | Result |
|---|---:|
| Training samples | 18,000 |
| Test samples | 7,000 |
| Engineered features | 40 |
| Number of target classes | 8 |
| Best local validation macro-F1 | 0.5366 |
| Best public leaderboard score | **0.4175** |

### Final Ensemble

The final submission combined three tree-based classifiers using weighted probability averaging:

- HistGradientBoosting
- ExtraTrees
- CatBoost

The final ensemble used the following weights:

- HistGradientBoosting: 45%
- ExtraTrees: 15%
- CatBoost: 40%

### Key Takeaways

- Feature engineering provided a structured representation of stellar light curves and metadata.
- Combining multiple classifiers improved performance over the initial baseline in the experiments conducted.
- Adding more features did not always improve leaderboard performance.
- The gap between local validation and public leaderboard results highlighted the importance of robust validation and investigating potential distribution shift.

The project provided practical experience in time-series feature engineering, multiclass classification, ensemble learning, model evaluation, and iterative experimentation under competition constraints.


## Acknowledgements

This project was developed as part of the **Hack4Hack ML Challenge 2026 — Round 1: Planet Hunters: Find New Worlds in Starlight**.

Thanks to the challenge organizers for providing the dataset and the opportunity to explore machine learning techniques for stellar classification.

## Disclaimer

This project was developed for educational purposes and as part of a machine learning competition. The classification results reflect the experiments conducted during the challenge and should not be interpreted as validated scientific findings or confirmed astronomical discoveries.