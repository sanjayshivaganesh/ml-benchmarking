# ML Benchmarking

Two layers share one training and evaluation engine.

**Phase 1 — Static ML Benchmarking System.** Fixed datasets, fixed model configurations, and written reports of metrics and errors.

**Phase 2 — Interactive ML Experimentation Framework.** One dataset and one or more registered models, chosen from `config.json`, the command line, or the Streamlit dashboard. Phase 2 calls the same loader, trainer, metrics, and error analysis as Phase 1.

Datasets are fetched programmatically (scikit-learn and OpenML) and are not committed.

## Architecture

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

`python main.py` and `streamlit run app.py` both call `run_phase2`, which calls `run_experiment`. That function loads the dataset, trains only the selected models, scores them with `evaluate_model`, and runs `analyze_errors`. The dashboard reads those results. It does not load data, train, or recompute metrics on its own.

Phase 1 is still `python -m src.run_phase1`. It trains the full dataset/model grid and writes its own three artifacts. A Phase 2 run does not overwrite them.

## Project structure

```
ML Project/
├── app.py                 # Streamlit UI
├── main.py                # Phase 2 CLI
├── config.json            # default experiment
├── src/
│   ├── data/              # dataset loading
│   ├── models/            # baseline classifiers
│   ├── training/          # fit one pipeline on the training split
│   ├── evaluation/        # metric computation
│   ├── analysis/          # error analysis and the Phase 1 report
│   └── experiments/       # registry, config, experiment, visualization
├── data/                  # downloaded data (not committed)
├── models/                # Phase 1 pipelines (not committed)
├── outputs/
│   ├── metrics/           # Phase 1 metrics.json
│   ├── errors/            # Phase 1 error_report.json
│   ├── reports/           # Phase 1 summary.md
│   └── experiments/       # one directory per Phase 2 run
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

Both use a stratified split. The default `test_size` is `0.2` and the default `random_state` is `42`. Preprocessing is fit only on the training split. Breast cancer features pass through. Adult numeric columns are median-imputed and scaled; categorical columns are mode-imputed and one-hot encoded.

Class names are stored on the experiment result. The CLI, the report, and the dashboard read that mapping.

## Models

The same untuned baselines are used for every dataset:

| Name | Estimator |
|---|---|
| `logistic_regression` | `LogisticRegression(C=1.0, max_iter=5000)` |
| `random_forest` | `RandomForestClassifier(n_estimators=100)` |
| `gradient_boosting` | `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)` |

Each estimator receives `random_state`. There is no hyperparameter search.

## Phase 1

From the project root, with the virtual environment activated:

```bash
python -m src.run_phase1
```

This trains all six dataset/model pipelines at `random_state=42` and `test_size=0.2`, then writes:

- `outputs/metrics/metrics.json`
- `outputs/errors/error_report.json`
- `outputs/reports/summary.md`

Fitted pipelines are saved as `models/{dataset}__{model}.joblib`. The summary is rendered from the generated results.

## Phase 2

### config.json

`config.json` is the default experiment. The CLI overrides any field it is given.

```json
{
  "dataset": "breast_cancer",
  "models": ["logistic_regression", "random_forest", "gradient_boosting"],
  "random_state": 42,
  "test_size": 0.2,
  "output_dir": "outputs/experiments"
}
```

`models` may be one name, several names, or `["all"]`. `all` expands to the model registry. `random_state` defaults to `42` and `test_size` to `0.2` when those keys are omitted. Names are checked before training starts.

### CLI

```bash
python main.py
```

```bash
python main.py --dataset breast_cancer --models logistic_regression random_forest
```

```bash
python main.py --dataset adult --models all
```

Other flags: `--random-state`, `--test-size`, and `--config`. A present flag replaces the matching `config.json` value. `--models all` trains every registered model. An unknown dataset, an unknown model, or an empty model list prints a short error and does not dump a traceback.

`python -m src.run_phase2` is the same entry point.

### Streamlit dashboard

```bash
streamlit run app.py
```

The sidebar lists datasets and models from the registry. Several models can be selected. Random state starts at `42` and test size at `0.2`. **Run Experiment** uses the same configuration and `run_phase2` path as the CLI.

Before a run, the page explains the experiment. After a run it shows the configuration (including `experiment_id`), the class labels from the result, the metric table, the ROC comparison, one confusion matrix per selected model, and the directory where artifacts were saved.

### Reproducibility

The split and the classifiers use the requested `random_state`. The scored split uses the same `test_size` as training. Running the same dataset, models, `random_state`, and `test_size` again produces equivalent metrics, predictions, confusion matrices, and ROC data. JSON stores full-precision metric values. The CLI prints four decimal places and points at `metrics.json` for the full values.

### Evaluation metrics

`evaluate_model` computes these scores on the held-out test set:

| Metric | Source |
|---|---|
| `accuracy` | `predict` |
| `precision` | `predict`, positive class `1` |
| `recall` | `predict`, positive class `1` |
| `f1` | `predict`, positive class `1` |
| `roc_auc` | probability of class `1` |
| `brier_score` | probability of class `1` |

Label metrics use the estimator’s default 0.5 threshold. A model without probabilities raises rather than returning a stand-in score.

### Metric comparison

The comparison table has one row per selected model and one column per metric above. Values are the stored `evaluate_model` outputs.

### ROC curves

One plot shows every selected model. Curve points come from the stored test labels and class-1 probabilities, with `pos_label=1`, which is the same positive class used for ROC-AUC. The legend prints the stored `roc_auc`.

### Confusion matrices

Each selected model gets one matrix from `analyze_errors`. The layout is `[[TN, FP], [FN, TP]]` for labels `[0, 1]`. Tick labels use the experiment’s class names, such as `0 = malignant` and `1 = benign`, or `0 = <=50K` and `1 = >50K`.

### Experiment output structure

Each run writes `outputs/experiments/<experiment_id>/`. The id is determined by the dataset, the selected models in request order, `random_state`, and `test_size`. For example:

```
outputs/experiments/breast_cancer__logistic_regression+random_forest__rs42__ts0p2/
├── metadata.json
├── metrics.json
├── error_analysis.json
├── predictions.json
├── breast_cancer__logistic_regression.joblib
└── breast_cancer__random_forest.joblib
```

`metadata.json` records the dataset, models, `random_state`, `test_size`, class labels, and `experiment_id`. A different dataset or model list gets a different directory. Repeating an id replaces that directory only.

## Tests

From the project root:

```bash
python -m unittest discover -s tests
```
