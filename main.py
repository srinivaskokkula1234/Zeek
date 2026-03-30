import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from detection.detect_anomalies import build_anomaly_dataframe, save_anomalies_csv
from feature_engineering.extract_features import aggregate_features_from_directory
from models.isolation_forest import score_anomalies, train_isolation_forest


def _compute_detection_metrics(metadata_df: pd.DataFrame, anomaly_score, anomaly_label) -> dict:
    """
    Compute anomaly-detection metrics against CIC-IDS ground truth `Label`
    when available.

    Label convention assumed:
      - "BENIGN" => normal (y_true = 0)
      - anything else => attack/anomaly (y_true = 1)

    Pred convention:
      - anomaly_label == -1 => predicted anomaly (y_pred = 1)
      - anomaly_label ==  1 => predicted normal (y_pred = 0)
    """
    if "Label" not in metadata_df.columns or len(metadata_df) == 0:
        return {}

    y_true_binary = (
        metadata_df["Label"]
        .fillna("BENIGN")
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda lbl: 0 if lbl == "BENIGN" else 1)
        .values
    )
    y_pred_binary = (anomaly_label == -1).astype(int)

    # Precision/recall/F1 for anomaly class (positive = 1)
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, pos_label=1, average="binary", zero_division=0
    )

    metrics = {
        "n_records": int(len(metadata_df)),
        "n_benign": int((y_true_binary == 0).sum()),
        "n_attack": int((y_true_binary == 1).sum()),
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "anomaly_precision": float(precision),
        "anomaly_recall": float(recall),
        "anomaly_f1": float(f1),
    }

    # ROC-AUC / PR-AUC require both classes present
    if len(np.unique(y_true_binary)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_binary, anomaly_score))
        except Exception:
            pass
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true_binary, anomaly_score))
        except Exception:
            pass

    return metrics


def run_pipeline() -> None:
    """
    Full execution flow:
      1. Load Zeek logs (or generic CSVs) from data/raw_zeek_logs
      2. Extract and encode features
      3. If ground-truth Labels exist (e.g. CIC-IDS dataset):
           a. Add derived features (packet rates, flag ratios, port indicators …)
           b. Train supervised Random Forest classifier
           c. Tune decision threshold to maximise F1
           d. Score all records
         Otherwise (real Zeek logs without labels):
           a. Auto-estimate contamination from data
           b. Train Isolation Forest (unsupervised)
           c. Score all records
      4. Save anomalies to data/results/anomalies.csv
      5. Save top-100 suspicious records to data/results/top_suspicious.csv
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--use-saved-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use saved multi-dataset model (default: true).",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models",
        help="Directory containing combined_rf_model.pkl and feature_aligner.pkl",
    )
    args, _unknown = parser.parse_known_args()

    # Optional contamination override for the unsupervised path
    contamination = os.environ.get("IF_CONTAMINATION", "auto")

    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"
    raw_dir = data_root / "raw_zeek_logs"
    results_dir = data_root / "results"
    anomalies_path = results_dir / "anomalies.csv"
    top_suspicious_path = results_dir / "top_suspicious.csv"
    metrics_path = results_dir / "model_metrics.json"

    # Ensure expected directory structure exists
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading and extracting features from: {raw_dir}")
    metadata_df, features_df, _encoder = aggregate_features_from_directory(str(raw_dir))

    if features_df.empty:
        print("[!] No supported Zeek logs found or no features extracted.")
        return

    # ------------------------------------------------------------------
    # Branch: supervised vs. unsupervised
    # ------------------------------------------------------------------
    if "Label" in metadata_df.columns:
        # ── Supervised path — ground-truth labels available ────────────
        print("[+] Ground-truth labels detected — using supervised Random Forest.")
        from feature_engineering.derive_features import add_derived_features
        from models.random_forest import (
            score_supervised,
            train_random_forest,
            tune_threshold,
        )

        # Combine metadata + features for joint feature engineering.
        # Drop any features_df columns that duplicate a metadata column to
        # avoid pd.concat producing duplicate column names (which causes
        # downstream pd.to_numeric / Series attribute errors).
        meta_reset = metadata_df.reset_index(drop=True)
        feats_reset = features_df.reset_index(drop=True)
        overlap = [c for c in feats_reset.columns if c in meta_reset.columns]
        if overlap:
            feats_reset = feats_reset.drop(columns=overlap)
        combined = pd.concat([meta_reset, feats_reset], axis=1)
        enriched = add_derived_features(combined)

        # Build feature matrix (numeric only, excluding label/id columns)
        DROP = {"Label", "timestamp", "source_ip", "destination_ip", "protocol"}
        feat_cols = [
            c for c in enriched.columns
            if c not in DROP and pd.api.types.is_numeric_dtype(enriched[c])
        ]
        X = (
            enriched[feat_cols]
            .replace([float("inf"), float("-inf")], 0)
            .fillna(0)
        )
        y = metadata_df["Label"].fillna("BENIGN").astype(str)

        print(f"[+] Training Random Forest on {len(X):,} records "
              f"({X.shape[1]} features) …")
        model_rf, _ = train_random_forest(X, y)

        print("[+] Tuning decision threshold …")
        threshold = tune_threshold(model_rf, X, y)
        print(f"    Optimal threshold: {threshold:.4f}")

        print("[+] Scoring records …")
        proba, _ = score_supervised(model_rf, X)
        anomaly_label = np.where(proba >= threshold, -1, 1)  # IF convention
        anomaly_score = proba
    else:
        # ── Unsupervised / inference path — no labels (real Zeek logs) ───
        # If a saved multi-dataset supervised model exists, use it for scoring.
        model_dir = (project_root / args.model_dir).resolve()
        combined_model_path = model_dir / "combined_rf_model.pkl"
        aligner_path = model_dir / "feature_aligner.pkl"

        if args.use_saved_model and combined_model_path.exists() and aligner_path.exists():
            print(f"[+] Using saved combined model from: {combined_model_path}")
            import joblib

            from feature_engineering.feature_aligner import FeatureAligner

            model = joblib.load(combined_model_path)
            aligner = FeatureAligner.load(str(aligner_path))
            X_aligned = aligner.transform(features_df.fillna(0))
            proba = model.predict_proba(X_aligned.values)[:, 1]
            anomaly_score = proba
            anomaly_label = np.where(proba >= 0.5, -1, 1)
        else:
            print("[+] Using unsupervised Isolation Forest.")
            if isinstance(contamination, str) and contamination != "auto":
                contamination = float(contamination)
            model = train_isolation_forest(
                features_df,
                contamination=contamination,
            )
            anomaly_score, anomaly_label = score_anomalies(model, features_df)

    # ------------------------------------------------------------------
    # Build output DataFrames and save
    # ------------------------------------------------------------------
    print("[+] Building anomaly output DataFrame …")
    full_df = build_anomaly_dataframe(metadata_df, anomaly_score, anomaly_label)

    # ------------------------------------------------------------------
    # Metrics (only if ground-truth labels exist)
    # ------------------------------------------------------------------
    metrics = _compute_detection_metrics(metadata_df, anomaly_score, anomaly_label)
    if metrics:
        import json

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("[+] Model metrics (vs CIC-IDS Label):")
        for k, v in metrics.items():
            if k != "n_records":
                print(f"    {k}: {v}")
        print(f"    n_records: {metrics['n_records']}")

    # Keep only anomalous records (-1) in anomalies.csv
    anomalies_df = full_df[full_df["anomaly_label"] == -1].reset_index(drop=True)

    # Preserve the original anomalies.csv output schema exactly.
    # Some datasets (e.g., CIC-IDS) do not provide timestamp/source/dest;
    # we still emit these columns as NA for schema stability.
    schema_cols = [
        "timestamp",
        "source_ip",
        "destination_ip",
        "protocol",
        "anomaly_score",
        "anomaly_label",
    ]
    for col in schema_cols:
        if col not in anomalies_df.columns:
            anomalies_df[col] = pd.NA
    anomalies_df = anomalies_df[schema_cols]

    print(f"[+] Saving {len(anomalies_df):,} anomalous records to: {anomalies_path}")
    save_anomalies_csv(anomalies_df, str(anomalies_path))

    # Top-N most suspicious records (any label) for quick inspection
    TOP_N = 100
    top_df = full_df.sort_values("anomaly_score", ascending=False).head(TOP_N)
    print(f"[+] Saving top {TOP_N} most suspicious records to: {top_suspicious_path}")
    save_anomalies_csv(top_df, str(top_suspicious_path))

    print("[+] Anomaly detection complete.")


if __name__ == "__main__":
    run_pipeline()
