# ML Benchmarking

This project started as a small benchmark and grew into a way to ask why a model is wrong, not only how high its score is.

Phase 1 trains the same models on two datasets and writes the metrics and error report. Phase 2 lets you pick the dataset and models from a config file, the command line, or a Streamlit page, using that same training and scoring code. Phase 3 takes a finished experiment and looks at the held-out test set: which groups of rows do worse, which mistakes the model made confidently, how the score changes if the numeric inputs are nudged, and which features a tree model relies on.

The datasets are downloaded by the code (scikit-learn and OpenML). They are not stored in git.

## How to run

From the project root, after the setup below:

```bash
python -m src.run_phase1
```

```bash
python main.py
```

```bash
python main.py --dataset breast_cancer --models logistic_regression random_forest
```

```bash
python main.py --dataset adult --models all
```

```bash
python main.py --no-diagnostics
```

```bash
python -m src.run_diagnostics
```

```bash
python -m src.run_diagnostics --dataset breast_cancer --models logistic_regression
```

```bash
python -m src.run_robustness --dataset breast_cancer --models logistic_regression random_forest
```

```bash
streamlit run app.py
```

```bash
python -m unittest discover -s tests
```

`python -m src.run_phase2` is the same command as `python main.py`. The default `config.json` has diagnostics turned on, so `python main.py` trains or reuses the experiment and then runs the Phase 3 checks. `--no-diagnostics` stops after scoring. `--diagnostics` turns those checks on if the config had them off. `python -m src.run_diagnostics` always runs them. `python -m src.run_robustness` only does the noise test. The Streamlit page calls `run_phase2`, so **Run Experiment** trains and scores but does not run the diagnostic report.

## How the code is organized

```
main.py / app.py
        │
  configuration (config.json)
        │
     registry
        │
    experiment.py
      /        \
Phase 1 engine  results
      \        /
    visualization
          │
      Streamlit
```

`python main.py` calls `run_configured_experiment`. If diagnostics are off, that calls `run_phase2`, which calls `run_experiment`. Training still goes through the Phase 1 pipeline: load the data, fit only the models you asked for, score them with `evaluate_model`, and run `analyze_errors`. The dashboard reads those saved results. It does not train or rescore on its own.

`python -m src.run_phase1` is the original full grid. It writes its own three files under `outputs/metrics`, `outputs/errors`, and `outputs/reports`. Later experiments do not overwrite those.

When diagnostics are on, the extra steps are:

```
Model
  ↓
Predictions
  ↓
Error slicing
  ↓
Hard examples
  ↓
Robustness
  ↓
Interpretability
  ↓
Diagnostic report
```

Slicing, hard examples, and the report use the predictions that were already saved. The robustness step loads that same fitted model and calls `predict` on copies of the test rows. It does not fit the model again. Feature importance reads `feature_importances_` when the model has it. Logistic regression does not, so that part is marked not supported and the rest of the run continues.

## Project structure

```
ML Project/
├── app.py                 # Streamlit UI
├── main.py                # experiment CLI
├── config.json            # default experiment
├── src/
│   ├── data/              # dataset loading
│   ├── models/            # baseline classifiers
│   ├── training/          # fit one pipeline on the training split
│   ├── evaluation/        # metric computation
│   ├── analysis/          # error analysis and the Phase 1 report
│   ├── experiments/       # registry, config, experiment, visualization
│   └── diagnostics/       # Phase 3 slicing, hard examples, robustness, explanations, report
├── data/                  # downloaded data (not committed)
├── models/                # Phase 1 pipelines (not committed)
├── outputs/
│   ├── metrics/           # Phase 1 metrics.json
│   ├── errors/            # Phase 1 error_report.json
│   ├── reports/           # Phase 1 summary.md
│   ├── experiments/       # one directory per experiment
│   ├── predictions/       # prediction CSVs
│   └── diagnostics/       # Phase 3 JSON, markdown, and plots
├── tests/
└── requirements.txt
```

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets

| Name | Source | Class 0 | Class 1 |
|---|---|---|---|
| `breast_cancer` | `sklearn.datasets.load_breast_cancer` | malignant | benign |
| `adult` | OpenML `adult` (version 2) | `<=50K` | `>50K` |

Both splits are stratified. The default `test_size` is `0.2` and the default `random_state` is `42`. Preprocessing is fit on the training rows only. Breast cancer features are passed through as they are. For Adult, numeric columns are filled with the training median and then scaled. Categorical columns are filled with the training mode and one-hot encoded.

The class names come from the experiment result. The command line, the report, and the dashboard all read that mapping, so the labels are not typed in by hand in those places.

## Models

Every dataset uses the same three baselines. None of them are tuned.

| Name | Estimator |
|---|---|
| `logistic_regression` | `LogisticRegression(C=1.0, max_iter=5000)` |
| `random_forest` | `RandomForestClassifier(n_estimators=100)` |
| `gradient_boosting` | `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)` |

Each estimator gets `random_state`. There is no hyperparameter search.

## Phase 1

```bash
python -m src.run_phase1
```

This trains all six dataset/model combinations at `random_state=42` and `test_size=0.2`, then writes:

- `outputs/metrics/metrics.json`
- `outputs/errors/error_report.json`
- `outputs/reports/summary.md`

The fitted pipelines are saved as `models/{dataset}__{model}.joblib`. The summary is written from those results.

## Phase 2

### config.json

`config.json` is the default experiment. A command-line flag replaces the matching field. If you leave the `diagnostics` block out, diagnostics stay off. The individual switches still default to true, so turning diagnostics on later runs all of them.

```json
{
  "dataset": "breast_cancer",
  "models": ["logistic_regression", "random_forest", "gradient_boosting"],
  "random_state": 42,
  "test_size": 0.2,
  "output_dir": "outputs/experiments",
  "diagnostics": {
    "enabled": true,
    "slicing": true,
    "hard_examples": true,
    "robustness": true,
    "explanations": true,
    "report": true,
    "plots": true
  }
}
```

`models` can be one name, several names, or `["all"]`. `all` means every model in the registry. If `random_state` or `test_size` is missing, the defaults are `42` and `0.2`. Names are checked before any training starts. An unknown diagnostic key is an error. Setting `plots` to false skips the figures, but the JSON and markdown files are still written.

### Command line

```bash
python main.py
```

```bash
python main.py --dataset breast_cancer --models logistic_regression random_forest
```

```bash
python main.py --dataset adult --models all
```

```bash
python main.py --no-diagnostics
```

The other flags are `--random-state`, `--test-size`, `--config`, and `--diagnostics`. `--models all` trains every registered model. An unknown dataset, an unknown model, or an empty model list prints a short error and stops, without a traceback.

### Streamlit

```bash
streamlit run app.py
```

The sidebar is filled from the same dataset and model lists. You can select more than one model. Random state starts at `42` and test size at `0.2`. **Run Experiment** uses `run_phase2`, so it trains and scores only.

Before a run, the page explains what it is about to do. After a run it shows the configuration, including `experiment_id`, the class labels from the result, the metric table, the ROC curves, one confusion matrix per model, and the folder where the files were saved.

### Reproducibility

The split and the classifiers use the `random_state` you set, and scoring uses the same `test_size` as training. Running the same dataset, models, seed, and test size again gives the same metrics, predictions, confusion matrices, and ROC data. JSON keeps the full metric values. The command line prints four decimal places and tells you to open `metrics.json` for the rest.

The noise test uses that same seed. If you do not pass a seed list, the seeds are `random_state`, `random_state + 1`, and so on, five of them by default. A noise level of `0` copies the test rows and does not draw random noise. The same seeds and noise levels give the same noisy scores. Diagnostic file names do not include the seed, so a later run with a different seed replaces those shared files. The experiment folder name does include the seed and the test size.

### Metrics

`evaluate_model` scores the held-out test set:

| Metric | Source |
|---|---|
| `accuracy` | `predict` |
| `precision` | `predict`, positive class `1` |
| `recall` | `predict`, positive class `1` |
| `f1` | `predict`, positive class `1` |
| `roc_auc` | probability of class `1` |
| `brier_score` | probability of class `1` |

Accuracy, precision, recall, and F1 use the model's normal 0.5 cutoff. If a model cannot give probabilities, scoring stops instead of inventing a number. Slice metrics and the robustness scores use those same four label metrics. The noise test does not recompute ROC-AUC or Brier score.

The metric table in the dashboard is one row per selected model and one column per metric above. The numbers are the ones `evaluate_model` already stored.

The ROC plot puts every selected model on one figure. The points come from the stored test labels and the stored probability of class 1, with `pos_label=1`. The legend shows the stored `roc_auc`.

Each model also gets a confusion matrix from `analyze_errors`. The layout is `[[TN, FP], [FN, TP]]` for labels `[0, 1]`. The tick labels use the experiment's class names, for example `0 = malignant` and `1 = benign`, or `0 = <=50K` and `1 = >50K`.

### Experiment files

Each run writes `outputs/experiments/<experiment_id>/`. The id is built from the dataset, the models in the order you asked for, `random_state`, and `test_size`. For example:

```
outputs/experiments/breast_cancer__logistic_regression+random_forest__rs42__ts0p2/
├── metadata.json
├── metrics.json
├── error_analysis.json
├── predictions.json
├── breast_cancer__logistic_regression.joblib
└── breast_cancer__random_forest.joblib
```

`metadata.json` stores the dataset, models, `random_state`, `test_size`, class labels, and `experiment_id`. A different dataset or model list gets its own folder. Running the same id again replaces that folder.

## Phase 3

The question for this phase is where the model fails on the test set it was not trained on. If `predictions.json`, `metadata.json`, and the model files are already saved, the run uses them. It trains only when those files are missing.

Each prediction row has `sample_id`, `y_true`, `y_pred`, `probability`, `confidence`, and `correct`. For these models, `probability` is the probability of class `1`, and `confidence` is `max(p, 1 - p)`.

### Error slicing

Slicing groups the saved test predictions by the original feature values. Numeric columns are split into quantile bins, four by default. A numeric column that never changes becomes one slice. Categorical columns are grouped by category. Missing values get their own slice, labeled `(missing)`. Each slice stores how many rows it has, plus accuracy, precision, recall, and F1, and how those differ from the full test set.

A slice with fewer than 20 rows is kept but marked unreliable. Its deltas are left empty, and it is not used when ranking the worst slices. A bigger slice that has no class 1 rows is also left out of that ranking. Binary F1 is 0 there even if every prediction is correct, so it would look like a failure when it is not one. The worst slices are the ones that remain, ordered by the largest drop in F1. A lower score in a slice means those rows did worse together. It does not mean the feature caused the mistakes.

### Hard examples

This step ranks the saved predictions. It does not call the model again. Wrong rows are sorted by confidence, and all rows are sorted by uncertainty. Both lists keep 20 rows by default. For these binary probabilities, uncertainty is `1 - 2 * abs(p - 0.5)`. That is 1 when `p = 0.5` and 0 when `p` is 0 or 1. Each example includes the original feature values for that `sample_id`.

The summary also counts four groups, using a confidence cutoff of 0.75: correct and confident, correct and uncertain, wrong and confident, and wrong and uncertain. The ranked wrong-row list is the most confident mistakes, even if some of them fall under 0.75. The written report only calls one a high-confidence error when its confidence is at least that cutoff.

### Robustness

This loads the saved pipeline and scores it on copies of the test features. Gaussian noise is added only to numeric columns. At a noise level of `0.10`, the noise scale is `0.10` times that column's sample standard deviation on the test rows. The default levels are `0`, `0.05`, `0.10`, `0.20`, and `0.30`, and the default is five seeds. Categorical columns and boolean columns are left alone. The original test table is not changed.

For each level the file stores the mean and sample standard deviation of accuracy, precision, recall, and F1, and how far the mean fell from the clean baseline. The baseline is `predict` on an unmodified copy of the same test rows.

`python -m src.run_robustness` runs only this step.

### Feature importance

Random forest and gradient boosting expose `feature_importances_`. The names come from the fitted preprocessor, through `get_feature_names_out`, so they are the columns the classifier actually saw, including one-hot columns. They are sorted by importance, and ties are broken by the feature name. Logistic regression does not have `feature_importances_`. That result is saved as `not_supported` with an empty list. The code does not swap in coefficients. SHAP is not installed and is not imported.

### Putting the pieces together

For one model, the summary tries to line up important features with weak slices. A transformed name is matched to a raw feature if it is that feature, or if it starts with the feature name plus an underscore, which is how the one-hot columns are named. If more than one raw name fits, the longer one wins. Grouped importance is the sum of those columns. A feature is listed when it is in the top five by that sum and its worst reliable slice has a negative F1 delta. The drop is called substantial when the F1 delta is `-0.05` or lower. The sentences in the report are filled from those stored fields. If the model has no importance scores, the join is marked not supported.

Model comparison is separate. It compares models inside one dataset, not across datasets. It copies the stored overall F1, the worst-slice F1 delta, the count of high-confidence errors, the mean uncertainty, the F1 drop at each noise level, and feature importance when that model has it. A missing number stays missing. A "best" or "worst" model is named only when at least two models have that measurement. A tie stays a tie. Feature importance is kept as a within-model ranking. It is not used as a score for deciding which model is better.

### Reports and plots

`failure_analysis.md` is written from the stored diagnostic result. It does not recompute the metrics. The sections are the experiment setup, overall performance, failure slices, high-confidence errors, the most uncertain predictions, robustness, feature importance, and a short summary. Numbers in the markdown are shown to four decimal places. `robustness_report.json` and `model_comparison.json` keep the full stored values.

Plots are ordinary matplotlib figures. They are saved when `diagnostics.plots` is true and there is something to draw. If a plot has no data, that file is skipped. Turning plots off, or skipping one figure, does not stop the JSON or markdown from being written.

### Example from a real run

This is a shortened piece of the report for `breast_cancer` / `logistic_regression`, with `random_state` 42 and `test_size` 0.2. The full tables are longer. The markdown rounds to four decimal places, which is why a count can look like `29.0000`.

```text
## Overall Performance

| dataset | model | accuracy | precision | recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| breast_cancer | logistic_regression | 0.9649 | 0.9595 | 0.9861 | 0.9726 | 0.9954 |

## 6. Diagnostic Summary

- The largest failure slice is worst concavity Q4: (0.376, 0.939] (29 samples), with F1 0.7500 compared with overall F1 0.9726 (delta -0.2226).
- 2 incorrect and confident predictions are recorded. A notable high-confidence error is sample 541: actual 1 (benign), predicted 0 (malignant), confidence 0.7803.
- The largest robustness degradation is an F1 drop of 0.0989 at noise level 0.3000. Mean F1 is 0.8737 and the standard deviation is 0.0174.
- Feature importance is not supported. This model does not expose feature_importances_.
```

The matching robustness rows for that model are:

| noise level | mean F1 | standard deviation | performance drop |
| --- | --- | --- | --- |
| 0.0000 | 0.9726 | 0.0000 | 0.0000 |
| 0.3000 | 0.8737 | 0.0174 | 0.0989 |

### Where the diagnostic files go

Per-model files are named `{dataset}_{model}`. The seed is not part of the name.

```
outputs/predictions/
└── breast_cancer_logistic_regression_predictions.csv
outputs/diagnostics/
├── slicing/breast_cancer_logistic_regression_slices.json
├── hard_examples/breast_cancer_logistic_regression_hard_examples.json
├── robustness/breast_cancer_logistic_regression_robustness.json
├── explanations/breast_cancer_logistic_regression_feature_importance.json
├── summaries/breast_cancer_logistic_regression_diagnostic_summary.json
├── plots/
│   ├── breast_cancer_logistic_regression_slice_f1.png
│   ├── breast_cancer_logistic_regression_robustness_curve.png
│   ├── breast_cancer_logistic_regression_confidence_distribution.png
│   ├── breast_cancer_random_forest_feature_importance.png
│   └── breast_cancer_robustness_comparison.png
├── failure_analysis.md
├── robustness_report.json
└── model_comparison.json
```

`failure_analysis.md`, `robustness_report.json`, and `model_comparison.json` are shared. The next diagnostic run replaces them. The robustness comparison plot is written when at least two models have a curve. The feature-importance plot is written only for models that have `feature_importances_`.

### What these results do not show

A worse slice, or a feature that is both important and weak in one slice, is a pattern in the saved test set. It is not proof that the feature caused the errors.

Slices with fewer than 20 rows can be noisy. They are saved and marked unreliable, and they are left out of the worst-slice ranking.

Adding Gaussian noise to numeric columns is only one way to test robustness. Categories are not changed. The result also depends on which columns are noised, the noise levels, the seeds, and the choice to scale the noise by each test column's sample standard deviation.

`feature_importances_` is a built-in ranking, not a full explanation of why the model decided a particular row. Logistic regression is reported as not supported. SHAP would be a later addition. It is not part of this project and is not a required install.

## Tests

From the project root:

```bash
python -m unittest discover -s tests
```
