import os
from pathlib import Path

import pandas as pd

from detection.detect_anomalies import build_anomaly_dataframe, save_anomalies_csv
from feature_engineering.extract_features import aggregate_features_from_directory
from models.isolation_forest import score_anomalies, train_isolation_forest


def run_pipeline() -> None:
    """
    Full execution flow:
      1. Load Zeek logs from data/raw_zeek_logs
      2. Extract and encode features
      3. Train Isolation Forest
      4. Run detection
      5. Save anomalies to data/results/anomalies.csv
    """
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"
    raw_dir = data_root / "raw_zeek_logs"
    results_dir = data_root / "results"
    anomalies_path = results_dir / "anomalies.csv"
    top_suspicious_path = results_dir / "top_suspicious.csv"

    # Ensure expected directory structure exists
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading and extracting features from: {raw_dir}")
    metadata_df, features_df, _encoder = aggregate_features_from_directory(str(raw_dir))

    if features_df.empty:
        print("[!] No supported Zeek logs found or no features extracted.")
        return

    print(f"[+] Training Isolation Forest on {len(features_df)} records...")
    model = train_isolation_forest(features_df)

    print("[+] Scoring anomalies...")
    anomaly_score, anomaly_label = score_anomalies(model, features_df)

    print("[+] Building anomaly output DataFrame...")
    full_df = build_anomaly_dataframe(metadata_df, anomaly_score, anomaly_label)

    # Keep only anomalous records (-1) in anomalies.csv
    anomalies_df = full_df[full_df["anomaly_label"] == -1].reset_index(drop=True)
    print(f"[+] Saving anomalous records to: {anomalies_path}")
    save_anomalies_csv(anomalies_df, str(anomalies_path))

    # Additionally, save the top-N most suspicious records by anomaly_score,
    # regardless of label, to top_suspicious.csv for inspection.
    TOP_N = 100
    top_df = full_df.sort_values("anomaly_score", ascending=False).head(TOP_N)
    print(f"[+] Saving top {TOP_N} most suspicious records to: {top_suspicious_path}")
    save_anomalies_csv(top_df, str(top_suspicious_path))

    print("[+] Anomaly detection complete.")


if __name__ == "__main__":
    run_pipeline()

