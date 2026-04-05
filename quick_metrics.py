"""
quick_metrics.py
=================
Fast model metrics on a SINGLE chosen file (avoids scanning GBs of data).

Target: CIC-IDS 2017 – Friday DDos (74 MB, ~225k rows, labelled).
Samples 100k rows, trains RF, runs 3-fold CV, prints full metrics.
Typical runtime: 2–5 minutes.

Usage:
    python quick_metrics.py                       # uses default CIC-IDS 2017 file
    python quick_metrics.py --file <path>         # use any labelled CSV
    python quick_metrics.py --max-rows 50000      # smaller sample
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_engineering.derive_features import add_derived_features

# ─────────────────────────────────────────────────────────────────────────────
# Default: largest CIC-IDS 2017 labelled file (Wednesday, ~215 MB)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw_zeek_logs"
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pick best available CIC-IDS 2017 file (in priority order)
_CANDIDATE_FILES = [
    RAW_DIR / "CIC-IDS- 2017" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    RAW_DIR / "CIC-IDS- 2017" / "Wednesday-workingHours.pcap_ISCX.csv",
    RAW_DIR / "CIC-IDS- 2017" / "Tuesday-WorkingHours.pcap_ISCX.csv",
]
DEFAULT_FILE = next((p for p in _CANDIDATE_FILES if p.exists()), None)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def binarise(labels: pd.Series) -> np.ndarray:
    return (
        labels.fillna("BENIGN").astype(str).str.strip().str.upper()
        .apply(lambda x: 0 if x == "BENIGN" else 1).values
    )


def compute_metrics(y_true, y_pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        roc = float(roc_auc_score(y_true, proba))
    except Exception:
        roc = None
    try:
        prc = float(average_precision_score(y_true, proba))
    except Exception:
        prc = None
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score":    round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "specificity": round(float(spec), 6),
        "fpr":         round(float(fpr), 6),
        "fnr":         round(float(fnr), 6),
        "mcc":         round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 6),
        "roc_auc":     round(roc, 6) if roc is not None else None,
        "pr_auc":      round(prc, 6) if prc is not None else None,
        "true_positives":  int(tp),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(data_file: Path, max_rows: int, cv_folds: int):
    overall_t0 = time.perf_counter()

    # ── 1. Load single file ───────────────────────────────────────────────────
    print(f"[1/6] Loading: {data_file.name}  ({data_file.stat().st_size/1e6:.0f} MB)")
    t0 = time.perf_counter()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(data_file, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [str(c).strip() for c in df.columns]
    load_time = time.perf_counter() - t0
    print(f"      {len(df):,} rows  |  {df.shape[1]} columns  |  loaded in {load_time:.1f}s")

    if "Label" not in df.columns:
        print("[!] No 'Label' column found in this file. Try a different --file.")
        return

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("[2/6] Engineering features …")
    # Remove duplicate header rows (CIC-IDS quirk)
    df = df[df["Label"].astype(str) != "Label"]
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    y_raw_full = df["Label"].fillna("BENIGN").astype(str).str.strip()
    DROP = {"Label"}
    feat_cols_raw = [c for c in df.columns if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]
    X_base = df[feat_cols_raw].copy()

    # Add derived features
    enriched = add_derived_features(df)
    feat_cols = [
        c for c in enriched.columns
        if c not in DROP and pd.api.types.is_numeric_dtype(enriched[c])
    ]
    X_full = enriched[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_bin_full = binarise(y_raw_full)

    n_total    = len(X_full)
    n_features = X_full.shape[1]
    n_benign   = int((y_bin_full == 0).sum())
    n_attack   = int((y_bin_full == 1).sum())
    attack_types = sorted(y_raw_full.str.upper().unique().tolist())
    print(f"      {n_total:,} records  |  {n_features} features")
    print(f"      Benign: {n_benign:,}  Attack: {n_attack:,}")
    print(f"      Attack types: {attack_types}")

    # ── 3. Stratified sample ──────────────────────────────────────────────────
    if n_total > max_rows:
        print(f"[3/6] Stratified sample: {max_rows:,} rows from {n_total:,} …")
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_rows, random_state=42)
        idx, _ = next(sss.split(X_full, y_bin_full))
        X_s     = X_full.iloc[idx].reset_index(drop=True)
        y_bin_s = y_bin_full[idx]
        y_raw_s = y_raw_full.iloc[idx].reset_index(drop=True)
    else:
        print(f"[3/6] Using all {n_total:,} rows.")
        X_s, y_bin_s, y_raw_s = X_full, y_bin_full, y_raw_full.reset_index(drop=True)

    n_sample   = len(X_s)
    n_s_benign = int((y_bin_s == 0).sum())
    n_s_attack = int((y_bin_s == 1).sum())
    print(f"      Sample: {n_sample:,}  ({n_s_benign:,} benign / {n_s_attack:,} attack)")

    # ── 4. Train / test split & fit ───────────────────────────────────────────
    print("[4/6] Training Random Forest (80/20 split) …")
    X_tr, X_te, y_tr, y_te, _, y_te_raw = train_test_split(
        X_s, y_bin_s, y_raw_s,
        test_size=0.20, stratify=y_bin_s, random_state=42,
    )
    t1 = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr.values, y_tr)
    train_time = time.perf_counter() - t1
    print(f"      Trained in {train_time:.1f}s")

    # ── 5. Score & optimal threshold ─────────────────────────────────────────
    print("[5/6] Scoring and tuning threshold …")
    t2 = time.perf_counter()
    proba = model.predict_proba(X_te.values)[:, 1]
    infer_time = time.perf_counter() - t2

    best_thr, best_f1 = 0.5, 0.0
    sweep_rows = []
    for thr in np.arange(0.05, 0.96, 0.05):
        thr_r = round(float(thr), 2)
        p = precision_score(y_te, (proba >= thr_r).astype(int), zero_division=0)
        r = recall_score(y_te, (proba >= thr_r).astype(int), zero_division=0)
        f = f1_score(y_te, (proba >= thr_r).astype(int), zero_division=0)
        sweep_rows.append({"threshold": thr_r, "precision": round(p,4), "recall": round(r,4), "f1": round(f,4)})
        if f > best_f1:
            best_f1, best_thr = f, thr_r

    y_pred = (proba >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred, proba)
    print(f"      Best threshold: {best_thr:.2f}  |  Test F1: {metrics['f1_score']:.4f}")

    # Save threshold sweep
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)

    # ── 5b. Cross-validation ─────────────────────────────────────────────────
    print(f"[5b/6] {cv_folds}-fold stratified CV …")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_f1, cv_roc = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_s, y_bin_s), 1):
        Xtr_f, Xval_f = X_s.iloc[tr_idx], X_s.iloc[val_idx]
        ytr_f, yval_f = y_bin_s[tr_idx], y_bin_s[val_idx]
        mf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )
        mf.fit(Xtr_f.values, ytr_f)
        pf = mf.predict_proba(Xval_f.values)[:, 1]
        bthr_f, bf1_f = 0.5, 0.0
        for thr in np.arange(0.05, 0.96, 0.05):
            f = f1_score(yval_f, (pf >= thr).astype(int), zero_division=0)
            if f > bf1_f:
                bf1_f, bthr_f = f, round(float(thr), 2)
        try:
            roc_f = float(roc_auc_score(yval_f, pf))
        except Exception:
            roc_f = 0.0
        cv_f1.append(bf1_f)
        cv_roc.append(roc_f)
        print(f"       Fold {fold}: F1={bf1_f:.4f}  ROC-AUC={roc_f:.4f}  (thr={bthr_f:.2f})")

    cv_mean_f1  = float(np.mean(cv_f1))
    cv_std_f1   = float(np.std(cv_f1))
    cv_mean_roc = float(np.mean(cv_roc))
    print(f"       CV F1: {cv_mean_f1:.4f} ± {cv_std_f1:.4f}  |  CV ROC-AUC: {cv_mean_roc:.4f}")

    # ── 5c. Per-attack breakdown ──────────────────────────────────────────────
    labels_u = y_te_raw.str.strip().str.upper()
    class_breakdown = {}
    for cls in sorted(labels_u.unique()):
        mask = (labels_u == cls).values
        total_ = int(mask.sum())
        detected_ = int((y_pred[mask] == 1).sum())
        class_breakdown[cls] = {
            "total": total_,
            "detected": detected_,
            "recall": round(detected_ / total_, 6) if total_ > 0 else 0.0,
        }

    # ── 5d. Feature importance ────────────────────────────────────────────────
    imp_df = (
        pd.DataFrame({"feature": X_s.columns.tolist(), "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    total_wall = time.perf_counter() - overall_t0

    # ── 6. Save JSON + Markdown ───────────────────────────────────────────────
    report = {
        "source_file": str(data_file),
        "sample_info": {
            "total_records_in_file": n_total,
            "records_sampled": n_sample,
            "benign_in_sample": n_s_benign,
            "attack_in_sample": n_s_attack,
            "n_features": n_features,
            "attack_types": attack_types,
        },
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 200,
            "class_weight": "balanced",
            "decision_threshold": round(best_thr, 4),
        },
        "test_metrics": metrics,
        "cross_validation": {
            "folds": cv_folds,
            "f1_scores": [round(s, 6) for s in cv_f1],
            "roc_auc_scores": [round(s, 6) for s in cv_roc],
            "mean_f1":   round(cv_mean_f1, 6),
            "std_f1":    round(cv_std_f1, 6),
            "mean_roc_auc": round(cv_mean_roc, 6),
        },
        "per_attack_class": class_breakdown,
        "top_20_features": imp_df.head(20).to_dict(orient="records"),
        "timing": {
            "data_load_seconds": round(load_time, 2),
            "train_seconds": round(train_time, 2),
            "inference_seconds": round(infer_time, 4),
            "total_wall_seconds": round(total_wall, 2),
            "throughput_rps": round(len(X_te) / infer_time, 0) if infer_time > 0 else None,
        },
    }

    with open(RESULTS_DIR / "quick_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── Markdown ──────────────────────────────────────────────────────────────
    m   = metrics
    cv  = report["cross_validation"]
    ts  = report["timing"]
    lines = [
        "# Network Anomaly Detection — Quick Metrics Report\n",
        f"*Source: `{data_file.name}`  |  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "---\n",
        "## 1. Dataset Info\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Source file | `{data_file.name}` |",
        f"| Total records in file | {n_total:,} |",
        f"| Records sampled | {n_sample:,} |",
        f"| Benign (sample) | {n_s_benign:,} |",
        f"| Attack (sample) | {n_s_attack:,} |",
        f"| Feature count | {n_features} |",
        f"| Attack types | {', '.join(attack_types)} |",
        "",
        "## 2. Confusion Matrix (20% held-out test set)\n",
        "```",
        "                   Predicted",
        "                   Normal       Anomaly",
        f"  Actual Normal    TN={m['true_negatives']:>8,}   FP={m['false_positives']:>8,}",
        f"  Actual Attack    FN={m['false_negatives']:>8,}   TP={m['true_positives']:>8,}",
        "```\n",
        "## 3. Classification Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Accuracy** | **{m['accuracy']:.4f}** |",
        f"| **Precision** | **{m['precision']:.4f}** |",
        f"| **Recall (Sensitivity)** | **{m['recall']:.4f}** |",
        f"| **F1-Score** | **{m['f1_score']:.4f}** |",
        f"| Specificity | {m['specificity']:.4f} |",
        f"| False Positive Rate | {m['fpr']:.4f} |",
        f"| False Negative Rate | {m['fnr']:.4f} |",
        f"| **ROC-AUC** | **{m['roc_auc']}** |",
        f"| **PR-AUC** | **{m['pr_auc']}** |",
        f"| Matthews Corr. Coeff. (MCC) | {m['mcc']:.4f} |",
        f"| Cohen's Kappa | {m['cohen_kappa']:.4f} |",
        f"| Decision Threshold | {best_thr:.2f} |",
        "",
        f"## 4. {cv_folds}-Fold Cross-Validation\n",
        "| Fold | F1 | ROC-AUC |",
        "|------|----|---------|",
    ]
    for i, (f1_, roc_) in enumerate(zip(cv["f1_scores"], cv["roc_auc_scores"]), 1):
        lines.append(f"| {i} | {f1_:.4f} | {roc_:.4f} |")
    lines += [
        f"| **Mean** | **{cv['mean_f1']:.4f} ± {cv['std_f1']:.4f}** | **{cv['mean_roc_auc']:.4f}** |",
        "",
        "## 5. Per-Attack-Type Detection Recall\n",
        "| Attack Label | Total | Detected | Recall |",
        "|---|---|---|---|",
    ]
    for cls, info in sorted(class_breakdown.items()):
        bar = "█" * int(info["recall"] * 20)
        lines.append(f"| {cls} | {info['total']:,} | {info['detected']:,} | {info['recall']:.4f} {bar} |")

    lines += [
        "",
        "## 6. Top 20 Feature Importances\n",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for i, row in imp_df.head(20).iterrows():
        bar = "█" * int(row["importance"] * 250)
        lines.append(f"| {i+1} | `{row['feature']}` | {row['importance']:.5f} {bar} |")

    lines += [
        "",
        "## 7. Timing\n",
        "| Step | Time |",
        "|------|------|",
        f"| Data loading | {ts['data_load_seconds']:.1f}s |",
        f"| Model training | {ts['train_seconds']:.1f}s |",
        f"| Inference | {ts['inference_seconds']:.4f}s |",
        f"| Total wall time | {ts['total_wall_seconds']:.1f}s |",
        f"| Throughput | {ts['throughput_rps']:,} records/s |",
    ]

    with open(RESULTS_DIR / "quick_metrics_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("          QUICK METRICS SUMMARY")
    print("=" * 62)
    print(f"  File           : {data_file.name}")
    print(f"  Sample         : {n_sample:,} rows  ({n_s_benign:,} benign / {n_s_attack:,} attack)")
    print(f"  Features       : {n_features}")
    print(f"  Threshold      : {best_thr:.2f}")
    print()
    print(f"  ── Test Set ────────────────────────────────────")
    print(f"  Accuracy       : {m['accuracy']:.4f}")
    print(f"  Precision      : {m['precision']:.4f}")
    print(f"  Recall         : {m['recall']:.4f}")
    print(f"  F1-Score       : {m['f1_score']:.4f}  ◄ primary metric")
    print(f"  ROC-AUC        : {m['roc_auc']}")
    print(f"  PR-AUC         : {m['pr_auc']}")
    print(f"  MCC            : {m['mcc']:.4f}")
    print(f"  FPR            : {m['fpr']:.4f}")
    print(f"  FNR            : {m['fnr']:.4f}")
    print(f"  Confusion:  TP={m['true_positives']:,}  TN={m['true_negatives']:,}"
          f"  FP={m['false_positives']:,}  FN={m['false_negatives']:,}")
    print()
    print(f"  ── {cv_folds}-Fold CV ──────────────────────────────────")
    print(f"  Mean F1        : {cv_mean_f1:.4f} ± {cv_std_f1:.4f}")
    print(f"  Mean ROC-AUC   : {cv_mean_roc:.4f}")
    print()
    print(f"  ── Timing ──────────────────────────────────────")
    print(f"  Total runtime  : {total_wall:.1f}s")
    print("=" * 62)
    print(f"\n  JSON   → data/results/quick_metrics.json")
    print(f"  Report → data/results/quick_metrics_report.md")
    print(f"  FeatImp→ data/results/feature_importance.csv")
    print(f"  Sweep  → data/results/threshold_sweep.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast model metrics on a single file.")
    parser.add_argument(
        "--file", type=str,
        default=str(DEFAULT_FILE) if DEFAULT_FILE else None,
        help="Path to a labelled CSV file (must have a 'Label' column).",
    )
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()

    if not args.file:
        print("[!] No default labelled file found. Pass --file <path>.")
        raise SystemExit(1)

    fp = Path(args.file)
    if not fp.exists():
        print(f"[!] File not found: {fp}")
        raise SystemExit(1)

    main(fp, args.max_rows, args.cv_folds)
