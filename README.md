# ML Benchmarking

A reproducible pipeline for training classification models across multiple datasets and producing structured evaluation and failure-analysis reports.

Phase 1 is a static benchmarking and failure-analysis engine: fixed datasets, fixed model configurations, and written reports of metrics and errors. Dataset loading, baseline training, evaluation, failure analysis, and the Phase 1 summary report are implemented.

Datasets should be fetched programmatically where possible (for example through scikit-learn dataset loaders) rather than committed to the repository.

## Project structure

```
ML Project/
├── src/
│   ├── data/          # dataset loading
│   ├── models/        # model definitions
│   ├── training/      # training entry points
│   ├── evaluation/    # metric computation
│   ├── analysis/      # error analysis and reports
│   └── utils/         # shared I/O helpers
├── data/              # downloaded raw datasets (not committed)
├── models/            # saved model artifacts (not committed)
├── outputs/
│   ├── metrics/
│   ├── errors/
│   └── reports/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Phase 1

From the project root, with the virtual environment activated:

```bash
python -m src.run_phase1
```

The command trains all six dataset/model pipelines with `random_state=42`, then writes `outputs/metrics/metrics.json`, `outputs/errors/error_report.json`, and `outputs/reports/summary.md`. The summary is rendered from those results.
