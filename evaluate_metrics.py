"""
evaluate_metrics.py
====================
Evaluate the Isolation Forest anomaly detection model by re-running the full
pipeline on the existing data and computing supervised metrics where the
ground-truth 'Label' column is available (CIC-IDS 2017 dataset).

Metrics computed
----------------
Supervised (using ground-truth anomaly label from dataset):
  - Confusion Matrix  (TP, FP, TN, FN)
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, Average Precision (PR-AUC)
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa
  - False Positive Rate (FPR), False Negative Rate (FNR)

Unsupervised / operational:
  - Total records processed
  - Total anomalies detected (anomaly_label == -1)
  - Anomaly rate (contamination fraction realised)
  - Anomaly score distribution (mean, std, min, max, percentiles)
  - Per-attack-type breakdown (recall per class in detected anomalies)

Output
------
  data/results/metrics_report.json   – machine-readable
  data/results/evaluation_report.md  – human-readable markdown report
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from detection.detect_anomalies import build_anomaly_dataframe
from feature_engineering.extract_features import aggregate_features_from_directory
from models.isolation_forest import score_anomalies, train_isolation_forest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_zeek_logs"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON = RESULTS_DIR / "metrics_report.json"
REPORT_MD = RESULTS_DIR / "evaluation_report.md"


def main():
    print("[+] Loading and extracting features …")
    t0 = time.perf_counter()
    metadata_df, features_df, _encoder = aggregate_features_from_directory(str(RAW_DIR))
    load_time = time.perf_counter() - t0

    if features_df.empty:
        print("[!] No features found – aborting.")
        return

    n_total = len(features_df)
    n_features = features_df.shape[1]
    print(f"    Records loaded : {n_total:,}")
    print(f"    Feature count  : {n_features}")

    # -----------------------------------------------------------------------
    # Train & score
    # -----------------------------------------------------------------------
    print("[+] Training Isolation Forest …")
    t1 = time.perf_counter()
    model = train_isolation_forest(features_df)
    train_time = time.perf_counter() - t1

    print("[+] Scoring …")
    t2 = time.perf_counter()
    anomaly_score, anomaly_label = score_anomalies(model, features_df)
    infer_time = time.perf_counter() - t2

    full_df = build_anomaly_dataframe(metadata_df, anomaly_score, anomaly_label)

    # -----------------------------------------------------------------------
    # Operational / unsupervised metrics
    # -----------------------------------------------------------------------
    n_anomaly = int((anomaly_label == -1).sum())
    n_normal = int((anomaly_label == 1).sum())
    anomaly_rate = n_anomaly / n_total

    score_stats = {
        "mean": float(np.mean(anomaly_score)),
        "std": float(np.std(anomaly_score)),
        "min": float(np.min(anomaly_score)),
        "max": float(np.max(anomaly_score)),
        "p25": float(np.percentile(anomaly_score, 25)),
        "p50": float(np.percentile(anomaly_score, 50)),
        "p75": float(np.percentile(anomaly_score, 75)),
        "p95": float(np.percentile(anomaly_score, 95)),
        "p99": float(np.percentile(anomaly_score, 99)),
    }

    # -----------------------------------------------------------------------
    # Supervised metrics  (available because CIC-IDS has a 'Label' column)
    # -----------------------------------------------------------------------
    supervised_metrics = None
    class_breakdown = {}

    if "Label" in full_df.columns:
        labels_raw = full_df["Label"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        gt_binary = (labels_raw != "BENIGN").astype(int)   # 1 = attack, 0 = benign
        pred_binary = (full_df["anomaly_label"] == -1).astype(int)  # 1 = anomaly

        tn, fp, fn, tp = confusion_matrix(gt_binary, pred_binary, labels=[0, 1]).ravel()

        precision   = precision_score(gt_binary, pred_binary, zero_division=0)
        recall      = recall_score(gt_binary, pred_binary, zero_division=0)
        f1          = f1_score(gt_binary, pred_binary, zero_division=0)
        accuracy    = accuracy_score(gt_binary, pred_binary)
        mcc         = matthews_corrcoef(gt_binary, pred_binary)
        kappa       = cohen_kappa_score(gt_binary, pred_binary)
        fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr         = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # ROC-AUC & PR-AUC use continuous anomaly score
        # Higher anomaly_score → more likely anomalous  ✓
        # Align score ordering with gt_binary (attack=1)
        try:
            roc_auc = roc_auc_score(gt_binary, anomaly_score)
        except Exception:
            roc_auc = None

        try:
            pr_auc = average_precision_score(gt_binary, anomaly_score)
        except Exception:
            pr_auc = None

        supervised_metrics = {
            "true_positives":  int(tp),
            "true_negatives":  int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "accuracy":        round(accuracy, 6),
            "precision":       round(precision, 6),
            "recall":          round(recall, 6),
            "f1_score":        round(f1, 6),
            "specificity":     round(specificity, 6),
            "fpr":             round(fpr, 6),
            "fnr":             round(fnr, 6),
            "mcc":             round(mcc, 6),
            "cohen_kappa":     round(kappa, 6),
            "roc_auc":         round(roc_auc, 6) if roc_auc is not None else None,
            "pr_auc":          round(pr_auc, 6)  if pr_auc  is not None else None,
        }

        # Per-class breakdown
        all_attack_types = labels_raw.unique().tolist()
        for attack in sorted(all_attack_types):
            mask = (labels_raw == attack)
            total_cls = int(mask.sum())
            detected  = int((pred_binary[mask] == 1).sum())
            class_breakdown[attack] = {
                "total":    total_cls,
                "detected": detected,
                "recall":   round(detected / total_cls, 6) if total_cls > 0 else 0.0,
            }

    # -----------------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------------
    timing = {
        "data_load_seconds":      round(load_time, 3),
        "model_train_seconds":    round(train_time, 3),
        "inference_seconds":      round(infer_time, 3),
        "throughput_records_per_second": round(n_total / infer_time, 1) if infer_time > 0 else None,
    }

    # -----------------------------------------------------------------------
    # Model hyper-parameters
    # -----------------------------------------------------------------------
    hyperparams = {
        "n_estimators":  model.n_estimators,
        "contamination": model.contamination,
        "random_state":  model.random_state,
        "n_features":    n_features,
    }

    # -----------------------------------------------------------------------
    # Assemble JSON report
    # -----------------------------------------------------------------------
    report = {
        "dataset": {
            "raw_dir":        str(RAW_DIR),
            "total_records":  n_total,
            "total_anomalies": n_anomaly,
            "total_normal":   n_normal,
            "anomaly_rate":   round(anomaly_rate, 6),
        },
        "model_hyperparameters": hyperparams,
        "anomaly_score_distribution": score_stats,
        "supervised_metrics": supervised_metrics,
        "per_class_breakdown": class_breakdown,
        "timing": timing,
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] JSON metrics saved to: {METRICS_JSON}")

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    lines = []
    lines.append("# Isolation Forest – Model Evaluation Report\n")
    lines.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    lines.append("---\n")
    lines.append("## 1. Dataset Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Raw data directory | `{RAW_DIR}` |")
    lines.append(f"| Total records | **{n_total:,}** |")
    lines.append(f"| Records flagged as anomaly | **{n_anomaly:,}** |")
    lines.append(f"| Records flagged as normal | **{n_normal:,}** |")
    lines.append(f"| Realised anomaly rate | **{anomaly_rate:.2%}** |")
    lines.append("")

    lines.append("## 2. Model Hyper-parameters\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    for k, v in hyperparams.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## 3. Anomaly Score Distribution\n")
    lines.append(f"| Statistic | Value |")
    lines.append(f"|-----------|-------|")
    for k, v in score_stats.items():
        lines.append(f"| {k} | {v:.6f} |")
    lines.append("")

    if supervised_metrics:
        m = supervised_metrics
        lines.append("## 4. Supervised Classification Metrics\n")
        lines.append("> Ground-truth: `BENIGN` → negative (0);  all other labels → positive / attack (1).\n")
        lines.append("### 4a. Confusion Matrix\n")
        lines.append("```")
        lines.append("                  Predicted")
        lines.append("                  Normal    Anomaly")
        lines.append(f"  Actual Normal   TN={m['true_negatives']:>8,}   FP={m['false_positives']:>8,}")
        lines.append(f"  Actual Attack   FN={m['false_negatives']:>8,}   TP={m['true_positives']:>8,}")
        lines.append("```\n")

        lines.append("### 4b. Core Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        metric_rows = [
            ("Accuracy",                         f"{m['accuracy']:.4f}"),
            ("Precision (attack class)",          f"{m['precision']:.4f}"),
            ("Recall / Sensitivity (attack)",     f"{m['recall']:.4f}"),
            ("F1-Score (attack class)",           f"{m['f1_score']:.4f}"),
            ("Specificity (benign recall)",       f"{m['specificity']:.4f}"),
            ("False Positive Rate (FPR)",         f"{m['fpr']:.4f}"),
            ("False Negative Rate (FNR)",         f"{m['fnr']:.4f}"),
            ("Matthews Correlation Coefficient",  f"{m['mcc']:.4f}"),
            ("Cohen's Kappa",                     f"{m['cohen_kappa']:.4f}"),
            ("ROC-AUC",                           f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"),
            ("PR-AUC (Avg. Precision)",           f"{m['pr_auc']:.4f}"  if m['pr_auc']  is not None else "N/A"),
        ]
        for row_name, row_val in metric_rows:
            lines.append(f"| {row_name} | **{row_val}** |")
        lines.append("")

    if class_breakdown:
        lines.append("## 5. Per-Attack-Type Breakdown\n")
        lines.append("| Label | Total Records | Detected as Anomaly | Recall |")
        lines.append("|-------|--------------|---------------------|--------|")
        for cls, info in sorted(class_breakdown.items()):
            lines.append(
                f"| {cls} | {info['total']:,} | {info['detected']:,} | {info['recall']:.4f} |"
            )
        lines.append("")

    lines.append("## 6. Timing & Throughput\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Data loading time | {timing['data_load_seconds']:.2f} s |")
    lines.append(f"| Model training time | {timing['model_train_seconds']:.2f} s |")
    lines.append(f"| Inference time | {timing['inference_seconds']:.2f} s |")
    if timing['throughput_records_per_second']:
        lines.append(f"| Throughput | {timing['throughput_records_per_second']:,.1f} records/s |")
    lines.append("")

    lines.append("## 7. Observations & Interpretation\n")
    if supervised_metrics:
        m = supervised_metrics
        lines.append(
            f"- The model **detected {m['true_positives']:,} out of "
            f"{m['true_positives'] + m['false_negatives']:,} actual attacks** "
            f"(Recall = {m['recall']:.2%})."
        )
        lines.append(
            f"- It raised **{m['false_positives']:,} false alarms** on benign traffic "
            f"(FPR = {m['fpr']:.2%})."
        )
        if m['roc_auc'] is not None:
            lines.append(
                f"- ROC-AUC of **{m['roc_auc']:.4f}** indicates "
                + ("strong" if m['roc_auc'] >= 0.80 else "moderate" if m['roc_auc'] >= 0.65 else "weak")
                + " discrimination ability between normal and attack traffic."
            )
        lines.append(
            "- Isolation Forest is an *unsupervised* method trained without labels; "
            "performance is sensitive to the `contamination` hyper-parameter (currently "
            f"`{model.contamination}`)."
        )
        lines.append(
            "- Consider tuning `contamination` to the known attack ratio in your deployment "
            "environment to balance precision vs. recall."
        )

    report_text = "\n".join(lines)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[+] Markdown report saved to: {REPORT_MD}")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("           EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total records    : {n_total:,}")
    print(f"  Anomalies found  : {n_anomaly:,}  ({anomaly_rate:.2%})")
    if supervised_metrics:
        m = supervised_metrics
        print(f"  Accuracy         : {m['accuracy']:.4f}")
        print(f"  Precision        : {m['precision']:.4f}")
        print(f"  Recall           : {m['recall']:.4f}")
        print(f"  F1-Score         : {m['f1_score']:.4f}")
        print(f"  ROC-AUC          : {m['roc_auc']}")
        print(f"  PR-AUC           : {m['pr_auc']}")
        print(f"  MCC              : {m['mcc']:.4f}")
        print(f"  FPR              : {m['fpr']:.4f}")
        print(f"  FNR              : {m['fnr']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
