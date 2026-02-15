## Network Anomaly Detection System (Zeek + Isolation Forest)

This project implements a modular, machine-learning–based network anomaly detection system using **Zeek logs** and an **Isolation Forest** model.

It consumes Zeek HTTP / HTTPS / DNS logs, extracts protocol-aware numerical features, trains an unsupervised Isolation Forest model, and outputs anomaly scores and labels per record.

---

### Project Structure

- `data/raw_zeek_logs/`: input Zeek logs (`*.log` or CSV). Files with `http`, `dns`, `https` or `ssl` in the name are processed.
- `data/processed/`: (reserved for future use).
- `data/results/anomalies.csv`: anomaly detection output.
- `feature_engineering/extract_features.py`: feature extraction and encoding.
- `models/isolation_forest.py`: Isolation Forest training and scoring.
- `detection/detect_anomalies.py`: combines metadata with model outputs and writes CSV.
- `utils/preprocess.py`: Zeek log loading and basic normalization utilities.
- `main.py`: single entry point for running the full pipeline.

---

### Features

**Common features**

- `duration`
- `bytes_sent`
- `bytes_received`
- `packet_count`
- `connection_state` (label-encoded)

**HTTP-specific**

- `http_method` (label-encoded)
- `http_response_code`
- `http_uri_length`

**DNS-specific**

- `dns_query_length`
- `dns_answer_count`
- `dns_ttl`

**HTTPS-specific**

- `ssl_version` (label-encoded)
- `ssl_cipher` (label-encoded)
- `cert_validity` (approximate certificate lifetime in days, if available)

All features are converted to numeric form; missing values are filled with `0`. Categorical fields are label encoded.

---

### Model

- **Algorithm**: `IsolationForest`
- **Parameters**:
  - `n_estimators = 100`
  - `contamination = 0.05`
  - `random_state = 42`
- **Output**:
  - `anomaly_score`: higher means more anomalous
  - `anomaly_label`: `1` (normal), `-1` (anomaly)

---

### Installation

From the project root:

```bash
pip install -r requirements.txt
```

---

### Preparing Zeek Logs

Place your Zeek logs under `data/raw_zeek_logs/`. The pipeline supports:

- Raw Zeek `.log` files with `#fields` headers (tab-separated).
- CSV/TSV files with standard headers.

Filename hints are used to infer protocol:

- `*http*` (excluding `*https*`) → HTTP
- `*dns*` → DNS
- `*https*` or `*ssl*` → HTTPS

Only these protocols are processed by default.

---

### Running the Pipeline

From the project root:

```bash
python main.py
```

Execution flow:

1. Load Zeek logs from `data/raw_zeek_logs/`.
2. Preprocess logs and extract features.
3. Train the Isolation Forest model (unsupervised).
4. Score all records and assign anomaly labels.
5. Save results to `data/results/anomalies.csv` with:
   - `timestamp`
   - `source_ip`
   - `destination_ip`
   - `protocol`
   - `anomaly_score`
   - `anomaly_label`

---

### Notes

- Designed for large logs: all heavy operations are vectorized with pandas / NumPy.
- Dependencies are minimal and pinned in `requirements.txt`.
- The code is organized into short, focused functions to keep the pipeline easy to extend (e.g., adding new protocols or features).

