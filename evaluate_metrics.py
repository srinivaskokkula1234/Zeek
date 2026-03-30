"""
evaluate_metrics.py
====================
Evaluate the anomaly-detection model by re-running the full pipeline on
the existing data and computing metrics.

When ground-truth ``Label`` column is present (CIC-IDS datasets):
  - Supervised Random Forest is trained and evaluated.
  - 5-fold stratified cross-validation reports mean ± std F1.
  - A probability threshold sweep (0.10 → 0.90) records F1 at each step.
  - Top-15 feature importances are printed and saved.

When no labels are present (raw Zeek logs):
  - Unsupervised Isolation Forest is used (「contamination='auto'」).
  - Only operational / unsupervised metrics are reported.

Output files
------------
  data/results/metrics_report.json       – machine-readable full report
  data/results/evaluation_report.md      – human-readable Markdown
  data/results/feature_importance.csv    – top features (supervised only)
  data/results/threshold_sweep.csv       – threshold vs F1 (supervised only)
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
from sklearn.model_selection import StratifiedKFold

from detection.detect_anomalies import build_anomaly_dataframe
from feature_engineering.derive_features import add_derived_features
from feature_engineering.extract_features import aggregate_features_from_directory
from models.isolation_forest import score_anomalies, train_isolation_forest
from models.random_forest import score_supervised, train_random_forest, tune_threshold


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_zeek_logs"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON    = RESULTS_DIR / "metrics_report.json"
REPORT_MD       = RESULTS_DIR / "evaluation_report.md"
FEAT_IMP_CSV    = RESULTS_DIR / "feature_importance.csv"
THRESHOLD_CSV   = RESULTS_DIR / "threshold_sweep.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binarise(labels_series: pd.Series) -> np.ndarray:
    """Return 0 = BENIGN, 1 = attack from a raw Label series."""
    return (
        labels_series.fillna("BENIGN")
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda lbl: 0 if lbl == "BENIGN" else 1)
        .values
    )


def _supervised_metrics(gt_binary, pred_binary, proba) -> dict:
    """Compute a full suite of supervised classification metrics."""
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

    try:
        roc_auc = roc_auc_score(gt_binary, proba)
    except Exception:
        roc_auc = None

    try:
        pr_auc = average_precision_score(gt_binary, proba)
    except Exception:
        pr_auc = None

    return {
        "true_positives":  int(tp),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "accuracy":        round(float(accuracy), 6),
        "precision":       round(float(precision), 6),
        "recall":          round(float(recall), 6),
        "f1_score":        round(float(f1), 6),
        "specificity":     round(float(specificity), 6),
        "fpr":             round(float(fpr), 6),
        "fnr":             round(float(fnr), 6),
        "mcc":             round(float(mcc), 6),
        "cohen_kappa":     round(float(kappa), 6),
        "roc_auc":         round(float(roc_auc), 6) if roc_auc is not None else None,
        "pr_auc":          round(float(pr_auc), 6)  if pr_auc  is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[+] Loading and extracting features …")
    t0 = time.perf_counter()
    metadata_df, features_df, _encoder = aggregate_features_from_directory(str(RAW_DIR))
    load_time = time.perf_counter() - t0

    if features_df.empty:
        print("[!] No features found – aborting.")
        return

    n_total    = len(features_df)
    n_features_raw = features_df.shape[1]
    print(f"    Records loaded : {n_total:,}")
    print(f"    Feature count  : {n_features_raw}")

    has_labels = "Label" in metadata_df.columns

    # -----------------------------------------------------------------------
    # === SUPERVISED PATH ===
    # -----------------------------------------------------------------------
    if has_labels:
        print("[+] Labels found — running supervised evaluation (Random Forest).")

        # Build enriched feature matrix.
        # Drop features_df columns that duplicate a metadata_df column so
        # pd.concat never creates duplicate column names (which makes
        # column indexing return a DataFrame instead of a Series).
        meta_r = metadata_df.reset_index(drop=True)
        feats_r = features_df.reset_index(drop=True)
        _overlap = [c for c in feats_r.columns if c in meta_r.columns]
        if _overlap:
            feats_r = feats_r.drop(columns=_overlap)
        combined = pd.concat([meta_r, feats_r], axis=1)
        enriched = add_derived_features(combined)

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
        y_raw = metadata_df["Label"].fillna("BENIGN").astype(str)
        y_bin = _binarise(y_raw)

        n_features = X.shape[1]
        print(f"    Enriched feature count: {n_features}")

        # ── Train (full dataset) ─────────────────────────────────────────
        print("[+] Training Random Forest …")
        t1 = time.perf_counter()
        model_rf, _ = train_random_forest(X, y_raw)
        train_time = time.perf_counter() - t1

        # ── Threshold tuning ─────────────────────────────────────────────
        print("[+] Tuning decision threshold …")
        threshold = tune_threshold(model_rf, X, y_raw)
        print(f"    Optimal threshold: {threshold:.4f}")

        # ── Full-dataset scoring ─────────────────────────────────────────
        print("[+] Scoring …")
        t2 = time.perf_counter()
        proba, _ = score_supervised(model_rf, X)
        infer_time = time.perf_counter() - t2

        pred_bin = (proba >= threshold).astype(int)
        anomaly_label_if = np.where(pred_bin == 1, -1, 1)  # IF convention

        # Full Supervised metrics
        supervised_metrics = _supervised_metrics(y_bin, pred_bin, proba)

        # ── Threshold sweep ──────────────────────────────────────────────
        print("[+] Running threshold sweep (0.10 → 0.90) …")
        sweep_rows = []
        for thr in np.arange(0.10, 0.91, 0.05):
            thr = round(float(thr), 2)
            p_bin = (proba >= thr).astype(int)
            f1_thr = f1_score(y_bin, p_bin, zero_division=0)
            prec   = precision_score(y_bin, p_bin, zero_division=0)
            rec    = recall_score(y_bin, p_bin, zero_division=0)
            sweep_rows.append({"threshold": thr, "f1": round(f1_thr, 6),
                                "precision": round(prec, 6), "recall": round(rec, 6)})
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(THRESHOLD_CSV, index=False)
        print(f"[+] Threshold sweep saved to: {THRESHOLD_CSV}")

        # ── 5-fold cross-validation ──────────────────────────────────────
        print("[+] Running 5-fold stratified cross-validation …")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bin), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr_raw    = y_raw.iloc[train_idx]
            y_val_bin   = y_bin[val_idx]

            fold_model, _ = train_random_forest(X_tr, y_tr_raw)
            fold_thr       = tune_threshold(fold_model, X_tr, y_tr_raw)
            fold_proba, _  = score_supervised(fold_model, X_val)
            fold_pred      = (fold_proba >= fold_thr).astype(int)
            fold_f1        = f1_score(y_val_bin, fold_pred, zero_division=0)
            cv_f1_scores.append(fold_f1)
            print(f"    Fold {fold}: F1 = {fold_f1:.4f}  (threshold={fold_thr:.4f})")

        cv_mean = float(np.mean(cv_f1_scores))
        cv_std  = float(np.std(cv_f1_scores))
        print(f"    CV F1: {cv_mean:.4f} ± {cv_std:.4f}")

        # ── Feature importance ───────────────────────────────────────────
        print("[+] Extracting feature importances …")
        importances = model_rf.feature_importances_
        imp_df = (
            pd.DataFrame({"feature": feat_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        imp_df.to_csv(FEAT_IMP_CSV, index=False)
        print(f"[+] Feature importance saved to: {FEAT_IMP_CSV}")

        print("\n  Top 15 Features:")
        for _, row in imp_df.head(15).iterrows():
            bar = "█" * int(row["importance"] * 400)
            print(f"    {row['feature']:<45} {row['importance']:.6f}  {bar}")

        # ── Per-class breakdown ──────────────────────────────────────────
        labels_upper = y_raw.str.strip().str.upper()
        class_breakdown = {}
        for attack in sorted(labels_upper.unique()):
            mask      = (labels_upper == attack).values
            total_cls = int(mask.sum())
            detected  = int((pred_bin[mask] == 1).sum())
            class_breakdown[attack] = {
                "total":    total_cls,
                "detected": detected,
                "recall":   round(detected / total_cls, 6) if total_cls > 0 else 0.0,
            }

        # Operational stats
        n_anomaly = int((anomaly_label_if == -1).sum())
        n_normal  = int((anomaly_label_if == 1).sum())
        anomaly_rate = n_anomaly / n_total

        # Score stats (on proba)
        score_stats = {
            "mean": float(np.mean(proba)),
            "std":  float(np.std(proba)),
            "min":  float(np.min(proba)),
            "max":  float(np.max(proba)),
            "p25":  float(np.percentile(proba, 25)),
            "p50":  float(np.percentile(proba, 50)),
            "p75":  float(np.percentile(proba, 75)),
            "p95":  float(np.percentile(proba, 95)),
            "p99":  float(np.percentile(proba, 99)),
        }

        timing = {
            "data_load_seconds":              round(load_time, 3),
            "model_train_seconds":            round(train_time, 3),
            "inference_seconds":              round(infer_time, 3),
            "throughput_records_per_second":  round(n_total / infer_time, 1) if infer_time > 0 else None,
        }

        hyperparams = {
            "model":         "RandomForestClassifier",
            "n_estimators":  model_rf.n_estimators,
            "class_weight":  str(model_rf.class_weight),
            "n_features":    n_features,
            "threshold":     round(threshold, 6),
        }

        report = {
            "dataset": {
                "raw_dir":         str(RAW_DIR),
                "total_records":   n_total,
                "total_anomalies": n_anomaly,
                "total_normal":    n_normal,
                "anomaly_rate":    round(anomaly_rate, 6),
            },
            "model_hyperparameters":        hyperparams,
            "anomaly_score_distribution":   score_stats,
            "supervised_metrics":           supervised_metrics,
            "cross_validation": {
                "folds":     5,
                "f1_scores": [round(s, 6) for s in cv_f1_scores],
                "mean_f1":   round(cv_mean, 6),
                "std_f1":    round(cv_std, 6),
            },
            "per_class_breakdown": class_breakdown,
            "timing":              timing,
        }

    # -----------------------------------------------------------------------
    # === UNSUPERVISED PATH ===
    # -----------------------------------------------------------------------
    else:
        print("[+] No labels — running unsupervised Isolation Forest evaluation.")
        n_features = features_df.shape[1]

        print("[+] Training Isolation Forest …")
        t1 = time.perf_counter()
        model = train_isolation_forest(features_df, contamination="auto")
        train_time = time.perf_counter() - t1

        print("[+] Scoring …")
        t2 = time.perf_counter()
        anomaly_score, anomaly_label_if = score_anomalies(model, features_df)
        infer_time = time.perf_counter() - t2

        n_anomaly   = int((anomaly_label_if == -1).sum())
        n_normal    = int((anomaly_label_if == 1).sum())
        anomaly_rate = n_anomaly / n_total

        score_stats = {
            "mean": float(np.mean(anomaly_score)),
            "std":  float(np.std(anomaly_score)),
            "min":  float(np.min(anomaly_score)),
            "max":  float(np.max(anomaly_score)),
            "p25":  float(np.percentile(anomaly_score, 25)),
            "p50":  float(np.percentile(anomaly_score, 50)),
            "p75":  float(np.percentile(anomaly_score, 75)),
            "p95":  float(np.percentile(anomaly_score, 95)),
            "p99":  float(np.percentile(anomaly_score, 99)),
        }
        timing = {
            "data_load_seconds":             round(load_time, 3),
            "model_train_seconds":           round(train_time, 3),
            "inference_seconds":             round(infer_time, 3),
            "throughput_records_per_second": round(n_total / infer_time, 1) if infer_time > 0 else None,
        }
        hyperparams = {
            "model":         "IsolationForest",
            "n_estimators":  model.n_estimators,
            "contamination": str(model.contamination),
            "random_state":  model.random_state,
            "n_features":    n_features,
        }
        supervised_metrics = None
        class_breakdown    = {}
        proba              = anomaly_score  # for unified printing below

        report = {
            "dataset": {
                "raw_dir":         str(RAW_DIR),
                "total_records":   n_total,
                "total_anomalies": n_anomaly,
                "total_normal":    n_normal,
                "anomaly_rate":    round(anomaly_rate, 6),
            },
            "model_hyperparameters":       hyperparams,
            "anomaly_score_distribution":  score_stats,
            "supervised_metrics":          None,
            "cross_validation":            None,
            "per_class_breakdown":         {},
            "timing":                      timing,
        }

    # -----------------------------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------------------------
    with open(METRICS_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] JSON metrics saved to: {METRICS_JSON}")

    # -----------------------------------------------------------------------
    # Build Markdown report
    # -----------------------------------------------------------------------
    lines = []
    lines.append("# Anomaly Detection – Model Evaluation Report\n")
    lines.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("---\n")

    lines.append("## 1. Dataset Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Raw data directory | `{RAW_DIR}` |")
    lines.append(f"| Total records | **{n_total:,}** |")
    lines.append(f"| Records flagged as anomaly | **{n_anomaly:,}** |")
    lines.append(f"| Records flagged as normal  | **{n_normal:,}** |")
    lines.append(f"| Realised anomaly rate | **{n_anomaly/n_total:.2%}** |")
    lines.append("")

    lines.append("## 2. Model Hyper-parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for k, v in report["model_hyperparameters"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## 3. Anomaly Score Distribution\n")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
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
            ("Accuracy",                        f"{m['accuracy']:.4f}"),
            ("Precision (attack class)",         f"{m['precision']:.4f}"),
            ("Recall / Sensitivity (attack)",    f"{m['recall']:.4f}"),
            ("F1-Score (attack class)",          f"{m['f1_score']:.4f}"),
            ("Specificity (benign recall)",      f"{m['specificity']:.4f}"),
            ("False Positive Rate (FPR)",        f"{m['fpr']:.4f}"),
            ("False Negative Rate (FNR)",        f"{m['fnr']:.4f}"),
            ("Matthews Correlation Coefficient", f"{m['mcc']:.4f}"),
            ("Cohen's Kappa",                    f"{m['cohen_kappa']:.4f}"),
            ("ROC-AUC",  f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"),
            ("PR-AUC",   f"{m['pr_auc']:.4f}"  if m["pr_auc"]  is not None else "N/A"),
        ]
        for row_name, row_val in metric_rows:
            lines.append(f"| {row_name} | **{row_val}** |")
        lines.append("")

        if "cross_validation" in report and report["cross_validation"]:
            cv = report["cross_validation"]
            lines.append("### 4c. 5-Fold Cross-Validation F1\n")
            lines.append("| Fold | F1 |")
            lines.append("|------|----|")
            for i, s in enumerate(cv["f1_scores"], 1):
                lines.append(f"| {i} | {s:.4f} |")
            lines.append(f"| **Mean** | **{cv['mean_f1']:.4f} ± {cv['std_f1']:.4f}** |")
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
    if timing.get("throughput_records_per_second"):
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
        if m["roc_auc"] is not None:
            disc = "strong" if m["roc_auc"] >= 0.80 else "moderate" if m["roc_auc"] >= 0.65 else "weak"
            lines.append(
                f"- ROC-AUC of **{m['roc_auc']:.4f}** indicates {disc} "
                "discrimination ability between normal and attack traffic."
            )
        lines.append(
            "- Random Forest (supervised) is used because ground-truth labels are available. "
            "For unlabelled real Zeek logs the pipeline falls back to Isolation Forest."
        )
    else:
        lines.append(
            "- **Unsupervised Isolation Forest** was used (no Label column found). "
            "For datasets with labels (CIC-IDS, NSL-KDD, UNSW-NB15) the pipeline "
            "automatically switches to the supervised Random Forest path."
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
    print(f"  Model            : {report['model_hyperparameters']['model']}")
    print(f"  Total records    : {n_total:,}")
    print(f"  Anomalies found  : {n_anomaly:,}  ({n_anomaly/n_total:.2%})")
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
        if "cross_validation" in report and report["cross_validation"]:
            cv = report["cross_validation"]
            print(f"  CV F1 (5-fold)   : {cv['mean_f1']:.4f} ± {cv['std_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
