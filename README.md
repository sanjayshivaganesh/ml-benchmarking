# ML Benchmarking

A reproducible pipeline for training classification models across multiple datasets and producing structured evaluation and failure-analysis reports.

Phase 1 is a static benchmarking and failure-analysis engine: fixed datasets, fixed model configurations, and written reports of metrics and errors. This repository currently contains the project layout only. Dataset loading, training, evaluation, and report generation are not implemented yet.

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
