# Zeek — Network Anomaly Detection Pipeline

A production-grade, end-to-end machine learning pipeline for **network intrusion detection** that operates in both **supervised** and **unsupervised** modes. It ingests raw Zeek network logs or standard benchmark NIDS datasets (CIC-IDS 2017/2018, NSL-KDD, UNSW-NB15), engineers high-signal features, trains models, evaluates performance, and outputs scored anomaly reports.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Single-Dataset Detection](#single-dataset-detection)
  - [Multi-Dataset Training](#multi-dataset-training)
  - [Model Evaluation](#model-evaluation)
  - [Smoke Testing](#smoke-testing)
- [Pipeline Modes](#pipeline-modes)
  - [Supervised Mode](#supervised-mode-random-forest--xgboost)
  - [Unsupervised Mode](#unsupervised-mode-isolation-forest)
  - [Inference Mode](#inference-mode-saved-model)
- [Feature Engineering](#feature-engineering)
- [Supported Datasets](#supported-datasets)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Model Performance](#model-performance)
- [Requirements](#requirements)

---

## Key Features

| Capability | Description |
|---|---|
| **Dual-Mode Detection** | Automatically selects supervised (Random Forest) or unsupervised (Isolation Forest) path based on label availability |
| **Multi-Dataset Training** | Train a single unified model across heterogeneous NIDS datasets with automatic feature alignment |
| **Rich Feature Engineering** | 15+ derived features including packet rates, TCP flag ratios, port-category indicators, and IAT normalization |
| **Threshold Tuning** | Precision-recall curve-based threshold optimization to maximize F1-score |
| **Cross-Validation** | 5-fold stratified CV with per-fold threshold tuning for robust generalization estimates |
| **Class Imbalance Handling** | SMOTE oversampling + `balanced` class weights for skewed attack/benign distributions |
| **Multi-Format Ingestion** | Zeek `.log` (TSV with `#fields`), CSV, and TSV files with automatic encoding detection (UTF-8, Latin-1, CP1252) |
| **Benchmark Adapters** | Plug-and-play adapters for CIC-IDS 2017/2018, NSL-KDD, and UNSW-NB15 with column normalization |
| **Comprehensive Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Cohen's Kappa, FPR/FNR, per-attack-type recall |

---

## Architecture Overview
<img width="617" height="856" alt="image" src="https://github.com/user-attachments/assets/0e7372fa-85c4-4ae8-b6c8-d3738a702c8b" />


---

## Project Structure

```
Zeek/
├── main.py                          # Primary entry point — full detection pipeline
├── evaluate_metrics.py              # Comprehensive model evaluation with CV & reporting
├── train_multi.py                   # CLI for multi-dataset training strategies
├── smoke_test.py                    # Quick validation on a single CIC-IDS file
├── requirements.txt                 # Python dependencies
│
├── feature_engineering/
│   ├── extract_features.py          # Protocol-aware feature extraction (HTTP, DNS, SSL)
│   ├── derive_features.py           # Engineered features (packet rates, flag ratios, etc.)
│   └── feature_aligner.py           # Cross-dataset feature space alignment
│
├── models/
│   ├── random_forest.py             # Supervised Random Forest classifier
│   ├── xgboost_model.py             # Supervised XGBoost classifier
│   └── isolation_forest.py          # Unsupervised Isolation Forest
│
├── detection/
│   └── detect_anomalies.py          # Anomaly DataFrame builder & CSV writer
│
├── training/
│   ├── __init__.py
│   └── multi_dataset_trainer.py     # Multi-dataset training (combined/per-dataset/cross-eval)
│
├── utils/
│   ├── preprocess.py                # Zeek log parsing, encoding, numeric coercion
│   ├── dataset_registry.py          # Auto-discovery of benchmark datasets
│   └── dataset_adapters.py          # CIC-IDS, NSL-KDD, UNSW-NB15 adapters
│
└── data/                            # (gitignored – user-provided)
    ├── raw_zeek_logs/               # Place raw logs or benchmark CSVs here
    ├── results/                     # Pipeline outputs (anomalies, reports, metrics)
    └── models/                      # Saved model artifacts (.pkl)
```

---

## Installation

### Prerequisites

- **Python** 3.10+
- **pip** package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/srinivaskokkula1234/Zeek.git
cd Zeek

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Place your Zeek logs or a benchmark dataset under data/raw_zeek_logs/
mkdir -p data/raw_zeek_logs

# 2. Run anomaly detection
python main.py

# 3. Check results
#    → data/results/anomalies.csv          (flagged anomalies)
#    → data/results/top_suspicious.csv     (top 100 suspicious records)
#    → data/results/model_metrics.json     (performance metrics, if labels exist)
```

---

## Usage Guide

### Single-Dataset Detection

The primary entry point is `main.py`, which runs the full pipeline:

```bash
# Automatic mode selection (supervised if labels exist, unsupervised otherwise)
python main.py

# Use a previously saved multi-dataset model for inference
python main.py --use-saved-model --model-dir data/models

# Force unsupervised mode by setting contamination ratio
IF_CONTAMINATION=0.05 python main.py
```

**Pipeline steps:**
1. Load all data from `data/raw_zeek_logs/`
2. Extract protocol-specific features (HTTP, DNS, SSL, or generic)
3. If labels exist → train Random Forest, tune threshold, score records
4. If no labels → train Isolation Forest (or load saved model), score records
5. Save anomalies and top-100 suspicious records to `data/results/`

### Multi-Dataset Training

Train a unified model across multiple benchmark datasets:

```bash
# Combined training (one model on all datasets)
python train_multi.py --strategy combined --verbose

# Per-dataset training (one model per dataset)
python train_multi.py --strategy per_dataset

# Cross-evaluation (leave-one-dataset-out F1 matrix)
python train_multi.py --strategy cross_eval

# Use a fraction of each dataset for faster experimentation
python train_multi.py --strategy combined --sample-frac 0.3 --verbose
```

**Training strategies:**

| Strategy | Description | Output |
|---|---|---|
| `combined` | Concatenate all datasets (aligned features), 80/20 stratified split, single RF | `combined_rf_model.pkl` + `feature_aligner.pkl` |
| `per_dataset` | Train separate RF per dataset | `<name>_rf_model.pkl` per dataset |
| `cross_eval` | Leave-one-dataset-out: train on N-1, test on held-out | F1 cross-evaluation matrix |

### Model Evaluation

Run a comprehensive evaluation with cross-validation, threshold sweep, and feature importance:

```bash
python evaluate_metrics.py
```

**Outputs:**
- `data/results/metrics_report.json` — machine-readable full report
- `data/results/evaluation_report.md` — human-readable Markdown report
- `data/results/feature_importance.csv` — top features ranked by importance (supervised)
- `data/results/threshold_sweep.csv` — threshold vs. F1/precision/recall (supervised)

### Smoke Testing

Validate the full pipeline on a single CIC-IDS file:

```bash
python smoke_test.py
```

Runs on 50,000 sampled rows and asserts F1 > 0.50 as a sanity check.

---

## Pipeline Modes

### Supervised Mode (Random Forest / XGBoost)

Activated automatically when a `Label` column is detected (e.g., CIC-IDS datasets).

- **Model:** `RandomForestClassifier` with 300 trees, balanced class weights
- **Feature enrichment:** Derived features (packet rates, flag ratios, port categories)
- **Threshold tuning:** Precision-recall curve sweep to maximize F1
- **Evaluation:** 5-fold stratified CV, per-attack-type recall breakdown, confusion matrix

### Unsupervised Mode (Isolation Forest)

Fallback when no labels are available (e.g., real-world Zeek logs).

- **Model:** `IsolationForest` with `contamination="auto"`
- **Scoring:** Decision function inverted so higher score = more anomalous
- **Label convention:** `anomaly_label = -1` (anomaly), `1` (normal)

### Inference Mode (Saved Model)

When a pre-trained combined model exists in `data/models/`:

```bash
python main.py --use-saved-model --model-dir data/models
```

Uses `FeatureAligner` to align incoming data to the training feature space, then scores with the saved Random Forest.

---

## Feature Engineering

### Base Features (Protocol-Specific)

| Protocol | Features |
|---|---|
| **Conn (Generic)** | `duration`, `bytes_sent`, `bytes_received`, `packet_count`, `connection_state` |
| **HTTP** | `http_method`, `http_response_code`, `http_uri_length` |
| **DNS** | `dns_query_length`, `dns_answer_count`, `dns_ttl` |
| **SSL/HTTPS** | `ssl_version`, `ssl_cipher`, `cert_validity` |

### Derived Features

| Feature | Formula / Description |
|---|---|
| `pkt_rate` | (fwd_pkts + bwd_pkts) / duration |
| `fwd_bwd_ratio` | fwd_pkts / bwd_pkts |
| `bytes_per_pkt` | total_bytes / total_pkts |
| `bytes_per_sec` | total_bytes / duration |
| `syn_ratio` | SYN flags / total_pkts |
| `fin_ratio` | FIN flags / total_pkts |
| `rst_ratio` | RST flags / total_pkts |
| `psh_ratio` | PSH flags / total_pkts |
| `ack_ratio` | ACK flags / total_pkts |
| `is_well_known_port` | dst_port < 1024 |
| `is_http_https` | dst_port ∈ {80, 443, 8080, 8443} |
| `is_dns_port` | dst_port == 53 |
| `is_high_port` | dst_port > 49151 |
| `iat_mean_norm` | Flow IAT Mean / Flow IAT Std |
| `fwd_pkt_len_ratio` | Fwd Pkt Len Max / Fwd Pkt Len Mean |

---

## Supported Datasets

The pipeline auto-discovers and adapts the following benchmark datasets when placed in `data/raw_zeek_logs/`:

| Dataset | Detection Heuristic | Adapter |
|---|---|---|
| **CIC-IDS 2017** | `Label` + `Flow Duration` + `Total Fwd Packets` columns | Generic pipeline |
| **CIC-IDS 2018** | `Timestamp` + `Dst Port` + `Label` columns | `adapt_cicids2018` |
| **NSL-KDD** | Filenames containing `KDDTrain` / `KDDTest` | `adapt_nslkdd` |
| **UNSW-NB15** | `attack_cat` column or filenames with `UNSW` + `training`/`testing` | `adapt_unswnb15` |
| **Zeek Logs** | Native `.log` files with `#fields` header (conn, http, dns, ssl) | Built-in parser |

---

## Output Files

| File | Description |
|---|---|
| `data/results/anomalies.csv` | All records flagged as anomalous (`anomaly_label = -1`) |
| `data/results/top_suspicious.csv` | Top 100 most suspicious records by anomaly score |
| `data/results/model_metrics.json` | Detection metrics (accuracy, F1, ROC-AUC, etc.) |
| `data/results/metrics_report.json` | Full evaluation report (from `evaluate_metrics.py`) |
| `data/results/evaluation_report.md` | Human-readable evaluation report |
| `data/results/feature_importance.csv` | Feature importance rankings (supervised only) |
| `data/results/threshold_sweep.csv` | Threshold vs F1/precision/recall sweep |
| `data/models/combined_rf_model.pkl` | Trained combined Random Forest model |
| `data/models/feature_aligner.pkl` | Feature alignment mapping for inference |
| `data/models/multi_dataset_metrics.json` | Multi-dataset training report |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `IF_CONTAMINATION` | `"auto"` | Isolation Forest contamination ratio (float or `"auto"`) |

### CLI Arguments (`main.py`)

| Argument | Default | Description |
|---|---|---|
| `--use-saved-model` / `--no-use-saved-model` | `true` | Use saved multi-dataset model for inference |
| `--model-dir` | `data/models` | Directory containing saved model artifacts |

### CLI Arguments (`train_multi.py`)

| Argument | Default | Description |
|---|---|---|
| `--raw-dir` | `data/raw_zeek_logs` | Root directory to scan for datasets |
| `--output-dir` | `data/models` | Save directory for trained models |
| `--strategy` | `combined` | Training strategy (`combined`, `per_dataset`, `cross_eval`) |
| `--sample-frac` | `1.0` | Fraction of each dataset to use (0 < frac ≤ 1) |
| `--verbose` | `false` | Enable verbose output |

---

## Model Performance

### Training Configuration

A unified **Random Forest** classifier was trained across **all 5 datasets** simultaneously using `train_all_datasets.py`:

| Parameter | Value |
|---|---|
| Model | `RandomForestClassifier` |
| Trees | 200 |
| Class weight | `balanced` |
| Total records | **680,000** (20K sampled per file) |
| Features | 329 (aligned across datasets) |
| Train / Test split | 80 / 20 stratified |
| Decision threshold | 0.55 (optimised for F1) |
| Cross-validation | 3-fold stratified |

---

### Global Metrics (20% Held-Out Test Set)

| Metric | Value |
|---|---|
| **Accuracy** | **0.9977** |
| **Precision** | **0.9953** |
| **Recall** | **0.9961** |
| **F1-Score** | **0.9957** |
| **ROC-AUC** | **0.9999** |
| **PR-AUC** | **0.9999** |
| MCC | 0.9941 |
| Cohen's Kappa | 0.9941 |
| FPR | 0.0017 |
| FNR | 0.0039 |
| TP / TN / FP / FN | 35,873 / 99,817 / 168 / 142 |

---

### Per-Dataset Metrics

| Dataset | F1-Score | ROC-AUC | Precision | Recall | FPR | Test Records |
|---|---|---|---|---|---|---|
| **CIC-IDS 2017** | **0.9985** | 0.9999 | 1.0000 | 0.9969 | 0.0000 | 24,220 |
| **CIC-IDS 2018** | **0.9994** | 1.0000 | 0.9996 | 0.9992 | 0.0009 | 32,097 |
| **UNSW-NB15** | **0.9928** | 0.9999 | 0.9899 | 0.9957 | 0.0040 | 19,885 |
| **NSL-KDD** | **0.9888** | 0.9995 | 0.9908 | 0.9869 | 0.0100 | 7,911 |
| **CTU-13** | **0.9811** | 0.9999 | 0.9877 | 0.9746 | 0.0002 | 51,887 |

---

### 3-Fold Cross-Validation

| Fold | F1-Score | ROC-AUC |
|---|---|---|
| 1 | 0.9954 | 0.9999 |
| 2 | 0.9952 | 0.9999 |
| 3 | 0.9951 | 0.9999 |
| **Mean ± Std** | **0.9952 ± 0.0001** | **0.9999** |

The extremely low standard deviation (0.0001) across folds confirms the model generalises robustly across all five heterogeneous datasets.

---

### Unsupervised Mode — Isolation Forest

Evaluated against CIC-IDS ground truth with no label information used during training:

| Metric | Value |
|---|---|
| Accuracy | 0.7611 |
| Precision | 0.3629 |
| Recall | 0.0806 |
| F1-Score | 0.1319 |
| ROC-AUC | 0.5198 |

Performance is substantially lower than the supervised path, which is expected — Isolation Forest has no access to ground truth during training. Use this mode only when labels or a pre-trained model are unavailable (e.g., live Zeek traffic).

---

## Requirements

| Package | Version |
|---|---|
| numpy | ≥1.24, <3.0 |
| pandas | ≥2.0, <3.0 |
| scikit-learn | ≥1.4, <2.0 |
| xgboost | ≥2.0, <3.0 |
| imbalanced-learn | ≥0.11, <1.0 |
| joblib | ≥1.3 |
| tqdm | ≥4.65 |

---

## License

This project is developed as part of an academic/research initiative for network intrusion detection using machine learning.
