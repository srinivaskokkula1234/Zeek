"""
train_all_datasets.py
=====================
Train the Random Forest classifier on ALL available labelled datasets:

  - CIC-IDS 2017  (8 CSV files, 'Label' column)
  - CIC-IDS 2018  (10 CSV files, 'Label' column)
  - UNSW-NB15     (4 raw CSVs + pre-split sets, 'label' / 'attack_cat' column)
  - NSL-KDD       (KDDTrain+ txt/arff, column 42 = label)
  - CTU-13        (parquet binetflow files, 'Label' column)

Strategy
--------
  1. Each dataset family has its own loader that normalises the label column
     to a unified string: "BENIGN" (negative) or attack-name (positive).
  2. Up to `--max-rows-per-file` rows are sampled from each file to keep
     runtime manageable (default 50 000).
  3. All samples are concatenated and a single *global* Random Forest is
     trained and evaluated with 3-fold CV.
  4. Per-dataset and per-attack-type metrics are reported.
  5. Results are saved to data/results/multi_dataset_metrics.json and
     data/results/multi_dataset_report.md.

Usage
-----
  python train_all_datasets.py
  python train_all_datasets.py --max-rows-per-file 30000 --cv-folds 5
  python train_all_datasets.py --datasets cic2017 cic2018   # subset only
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Optional

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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

from feature_engineering.derive_features import add_derived_features

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw_zeek_logs"
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CIC2017_DIR  = RAW_DIR / "CIC-IDS- 2017"
CIC2018_DIR  = RAW_DIR / "CIC-IDS 2018"
UNSW_DIR     = RAW_DIR / "UNSW-NB15"
NSL_DIR      = RAW_DIR / "NSL-KDD"
CTU_DIR      = RAW_DIR / "CTU-13"

# ─────────────────────────────────────────────────────────────────────────────
# NSL-KDD column definitions
# ─────────────────────────────────────────────────────────────────────────────
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

# ─────────────────────────────────────────────────────────────────────────────
# CTU-13 binetflow label normalisation
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_ctu_label(raw: str) -> str:
    """Map CTU-13 flow direction labels to BENIGN / attack-type."""
    s = str(raw).strip().upper()
    if "BACKGROUND" in s or s in ("", "NAN"):
        return "BENIGN"
    if "NORMAL" in s:
        return "BENIGN"
    # Labels look like: "flow=From-Botnet-V42-..."
    if "BOTNET" in s:
        return "BOT"
    if "FROM-NORMAL" in s:
        return "BENIGN"
    return "BOT"  # any labelled flow in CTU-13 that is not normal is botnet


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_csv_robust(path: Path) -> Optional[pd.DataFrame]:
    """Try multiple encodings; return None on failure."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            print(f"      [!] Could not read {path.name}: {exc}")
            return None
    print(f"      [!] All encodings failed for {path.name}")
    return None


def binarise(labels: pd.Series) -> np.ndarray:
    """BENIGN → 0, anything else → 1."""
    return (
        labels.fillna("BENIGN").astype(str).str.strip().str.upper()
        .apply(lambda x: 0 if x == "BENIGN" else 1).values
    )


def compute_metrics(y_true, y_pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:   roc = float(roc_auc_score(y_true, proba))
    except Exception: roc = None
    try:   prc = float(average_precision_score(y_true, proba))
    except Exception: prc = None
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


def _sample(X: pd.DataFrame, y_bin: np.ndarray, y_raw: pd.Series, max_rows: int):
    """Stratified sample if over max_rows."""
    n = len(X)
    if n <= max_rows:
        return X.reset_index(drop=True), y_bin, y_raw.reset_index(drop=True)
    n_classes = len(np.unique(y_bin))
    if n_classes < 2:
        idx = np.random.choice(n, max_rows, replace=False)
        return X.iloc[idx].reset_index(drop=True), y_bin[idx], y_raw.iloc[idx].reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_rows, random_state=42)
    idx, _ = next(sss.split(X, y_bin))
    return X.iloc[idx].reset_index(drop=True), y_bin[idx], y_raw.iloc[idx].reset_index(drop=True)


def _engineer_and_pick_features(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Apply feature engineering; return (X_numeric, y_raw_labels)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop duplicate header rows (CIC-IDS quirk)
    if label_col in df.columns:
        df = df[df[label_col].astype(str) != label_col]
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    y_raw = df[label_col].fillna("BENIGN").astype(str).str.strip() if label_col in df.columns else pd.Series(["BENIGN"] * len(df))

    drop = {label_col, "difficulty_level", "id", "srcip", "dstip",
            "sport", "dsport", "stime", "ltime", "Timestamp",
            "Flow ID", "Source IP", "Destination IP", "StartTime", "Dur",
            "SrcAddr", "DstAddr", "Proto", "Dir", "State", "sTos", "dTos"}
    feat_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X_base = df[feat_cols].copy()

    # Derived features (silently skips missing source columns)
    enriched = add_derived_features(df)
    feat_cols_e = [c for c in enriched.columns if c not in drop and pd.api.types.is_numeric_dtype(enriched[c])]
    X = enriched[feat_cols_e].replace([np.inf, -np.inf], 0).fillna(0)
    return X, y_raw.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_cic_ids(directory: Path, dataset_name: str, max_rows: int) -> list[dict]:
    """Load CIC-IDS 2017 or 2018 CSV files (Label column)."""
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"  [!] No CSV files found in {directory}")
        return []

    chunks = []
    for fpath in csv_files:
        size_mb = fpath.stat().st_size / 1e6
        print(f"  → {fpath.name}  ({size_mb:.0f} MB) …", end="", flush=True)
        df = _read_csv_robust(fpath)
        if df is None:
            continue
        df.columns = [str(c).strip() for c in df.columns]

        # Detect label column (CIC-IDS 2018 uses ' Label' with a space)
        label_col = None
        for candidate in ("Label", " Label", "label"):
            if candidate in df.columns:
                label_col = candidate
                df.rename(columns={candidate: "Label"}, inplace=True)
                break
        if label_col is None:
            print(f" ✗ no Label column")
            continue

        if len(df) == 0:
            print(f" ✗ empty")
            continue

        X, y_raw = _engineer_and_pick_features(df, "Label")
        y_bin    = binarise(y_raw)

        # Skip files that are all-benign (no attack signal to learn)
        if y_bin.sum() == 0:
            print(f" skipped (all BENIGN)")
            continue

        X_s, y_s, y_r_s = _sample(X, y_bin, y_raw, max_rows)
        n_b = int((y_s == 0).sum())
        n_a = int((y_s == 1).sum())
        print(f" {len(X_s):,} rows  ({n_b:,} benign / {n_a:,} attack)")
        chunks.append({
            "dataset": dataset_name,
            "file": fpath.name,
            "X": X_s,
            "y_bin": y_s,
            "y_raw": y_r_s,
        })
    return chunks


def load_unsw_nb15(directory: Path, max_rows: int) -> list[dict]:
    """
    Load UNSW-NB15.  Uses the pre-split training + testing sets which have
    proper headers and a 'label' (0/1) + 'attack_cat' column.
    Falls back to raw CSVs (no header – infers from feature CSV).
    """
    chunks = []

    # Prefer the pre-split sets (they have headers)
    for fname in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
        fpath = directory / fname
        if not fpath.exists():
            continue
        size_mb = fpath.stat().st_size / 1e6
        print(f"  → {fname}  ({size_mb:.0f} MB) …", end="", flush=True)
        df = _read_csv_robust(fpath)
        if df is None:
            continue
        df.columns = [str(c).strip() for c in df.columns]

        # label column: 'label' (0/1) and 'attack_cat' (attack name)
        if "attack_cat" in df.columns and "label" in df.columns:
            # Build unified Label: BENIGN if label==0, else attack_cat
            df["Label"] = df.apply(
                lambda r: "BENIGN" if str(r["label"]).strip() == "0"
                else str(r["attack_cat"]).strip().upper() or "ATTACK",
                axis=1,
            )
        elif "label" in df.columns:
            df["Label"] = df["label"].apply(lambda x: "BENIGN" if str(x).strip() == "0" else "ATTACK")
        else:
            print(" ✗ no label column")
            continue

        X, y_raw = _engineer_and_pick_features(df, "Label")
        y_bin    = binarise(y_raw)
        if y_bin.sum() == 0:
            print(f" skipped (all BENIGN)")
            continue

        X_s, y_s, y_r_s = _sample(X, y_bin, y_raw, max_rows)
        n_b = int((y_s == 0).sum())
        n_a = int((y_s == 1).sum())
        print(f" {len(X_s):,} rows  ({n_b:,} benign / {n_a:,} attack)")
        chunks.append({"dataset": "UNSW-NB15", "file": fname, "X": X_s, "y_bin": y_s, "y_raw": y_r_s})

    # Also try raw UNSW-NB15_*.csv files if they have headers
    for fpath in sorted(directory.glob("UNSW-NB15_*.csv")):
        if "features" in fpath.name.lower() or "list" in fpath.name.lower():
            continue
        size_mb = fpath.stat().st_size / 1e6
        print(f"  → {fpath.name}  ({size_mb:.0f} MB) …", end="", flush=True)
        df = _read_csv_robust(fpath)
        if df is None:
            continue
        df.columns = [str(c).strip() for c in df.columns]

        # Raw files sometimes have no header → check first row for numbers
        first_row_numeric = pd.to_numeric(df.iloc[0], errors="coerce").notna().mean() > 0.7
        if first_row_numeric and "label" not in df.columns:
            # Assign NSL-KDD-style positional columns from features file
            feat_csv = directory / "NUSW-NB15_features.csv"
            if feat_csv.exists():
                feat_df   = pd.read_csv(feat_csv, encoding="latin-1")
                col_names = [str(r).strip() for r in feat_df.iloc[:, 1]]
                if len(col_names) == df.shape[1]:
                    df.columns = col_names
                else:
                    df.columns = [f"f{i}" for i in range(df.shape[1])]
            else:
                df.columns = [f"f{i}" for i in range(df.shape[1])]

        # Find label
        label_col = None
        for cand in ("label", "Label", "attack_cat"):
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            print(" ✗ no label column (raw file)")
            continue

        if label_col == "attack_cat":
            df["Label"] = df["attack_cat"].apply(
                lambda x: "BENIGN" if str(x).strip().upper() in ("", "NAN", "NONE", "NORMAL") else str(x).strip().upper()
            )
        else:
            df["Label"] = df[label_col].apply(lambda x: "BENIGN" if str(x).strip() in ("0", "normal") else "ATTACK")

        X, y_raw = _engineer_and_pick_features(df, "Label")
        y_bin    = binarise(y_raw)
        if y_bin.sum() == 0:
            print(f" skipped (all BENIGN)")
            continue

        X_s, y_s, y_r_s = _sample(X, y_bin, y_raw, max_rows)
        n_b = int((y_s == 0).sum())
        n_a = int((y_s == 1).sum())
        print(f" {len(X_s):,} rows  ({n_b:,} benign / {n_a:,} attack)")
        chunks.append({"dataset": "UNSW-NB15", "file": fpath.name, "X": X_s, "y_bin": y_s, "y_raw": y_r_s})

    return chunks


def load_nsl_kdd(directory: Path, max_rows: int) -> list[dict]:
    """Load NSL-KDD KDDTrain+.txt (comma-separated, no header)."""
    chunks = []
    for fname in ("KDDTrain+.txt", "KDDTest+.txt"):
        fpath = directory / fname
        if not fpath.exists():
            # Try sub-folder
            fpath = directory / "nsl-kdd" / fname
        if not fpath.exists():
            continue
        size_mb = fpath.stat().st_size / 1e6
        print(f"  → {fname}  ({size_mb:.0f} MB) …", end="", flush=True)
        try:
            df = pd.read_csv(fpath, header=None, names=NSL_KDD_COLUMNS, low_memory=False)
        except Exception as exc:
            print(f" ✗ {exc}")
            continue

        df.columns = [str(c).strip() for c in df.columns]
        # Normalise label: 'normal' → BENIGN, else use attack name upper-cased
        df["Label"] = df["label"].apply(
            lambda x: "BENIGN" if str(x).strip().lower() == "normal" else str(x).strip().upper()
        )
        # One-hot encode categorical columns
        cat_cols = ["protocol_type", "service", "flag"]
        df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])

        X, y_raw = _engineer_and_pick_features(df, "Label")
        y_bin    = binarise(y_raw)
        if y_bin.sum() == 0:
            print(f" skipped (all BENIGN)")
            continue

        X_s, y_s, y_r_s = _sample(X, y_bin, y_raw, max_rows)
        n_b = int((y_s == 0).sum())
        n_a = int((y_s == 1).sum())
        print(f" {len(X_s):,} rows  ({n_b:,} benign / {n_a:,} attack)")
        chunks.append({"dataset": "NSL-KDD", "file": fname, "X": X_s, "y_bin": y_s, "y_raw": y_r_s})
    return chunks


def load_ctu13(directory: Path, max_rows: int) -> list[dict]:
    """Load CTU-13 binetflow Parquet files."""
    chunks = []
    for fpath in sorted(directory.glob("*.parquet")):
        size_mb = fpath.stat().st_size / 1e6
        print(f"  → {fpath.name}  ({size_mb:.0f} MB) …", end="", flush=True)
        try:
            df = pd.read_parquet(fpath)
        except Exception as exc:
            print(f" ✗ {exc}")
            continue

        df.columns = [str(c).strip() for c in df.columns]

        # CTU-13 label column is 'Label' (flow direction string)
        label_col = None
        for cand in ("Label", "label", "class", "Class"):
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            print(" ✗ no Label column")
            continue

        df["Label"] = df[label_col].apply(_normalise_ctu_label)
        y_bin_check = binarise(df["Label"])
        if y_bin_check.sum() == 0:
            print(f" skipped (all BENIGN)")
            continue

        # One-hot encode string columns (Proto, Dir, State, etc.)
        cat_cols = [c for c in df.columns if df[c].dtype == object and c not in ("Label", label_col)]
        df = pd.get_dummies(df, columns=cat_cols[:10])  # limit to avoid explosion

        X, y_raw = _engineer_and_pick_features(df, "Label")
        y_bin    = binarise(y_raw)

        X_s, y_s, y_r_s = _sample(X, y_bin, y_raw, max_rows)
        n_b = int((y_s == 0).sum())
        n_a = int((y_s == 1).sum())
        print(f" {len(X_s):,} rows  ({n_b:,} benign / {n_a:,} attack)")
        chunks.append({"dataset": "CTU-13", "file": fpath.name, "X": X_s, "y_bin": y_s, "y_raw": y_r_s})
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def align_columns(chunks: list[dict]) -> list[dict]:
    """
    Align all feature matrices to a common column set using union + zero-fill.
    Required because different datasets have different numeric features.
    """
    all_cols = set()
    for c in chunks:
        all_cols.update(c["X"].columns.tolist())
    all_cols = sorted(all_cols)

    for c in chunks:
        X = c["X"]
        missing = [col for col in all_cols if col not in X.columns]
        for m in missing:
            X[m] = 0.0
        c["X"] = X[all_cols]
    return chunks, all_cols


def find_best_threshold(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Grid-search optimal F1 threshold."""
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.05):
        f = f1_score(y_true, (proba >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, round(float(thr), 2)
    return best_thr


def run_training(chunks: list[dict], all_cols: list[str], cv_folds: int) -> dict:
    """Concatenate all chunks, train global RF, evaluate."""

    # ── Build combined matrices ───────────────────────────────────────────────
    X_all   = pd.concat([c["X"] for c in chunks], ignore_index=True)
    y_all   = np.concatenate([c["y_bin"] for c in chunks])
    y_raw_all = pd.concat([c["y_raw"] for c in chunks], ignore_index=True)
    ds_labels = np.concatenate([[c["dataset"]] * len(c["X"]) for c in chunks])

    n_total   = len(X_all)
    n_benign  = int((y_all == 0).sum())
    n_attack  = int((y_all == 1).sum())
    n_features = X_all.shape[1]

    print(f"\n  Combined: {n_total:,} rows  |  {n_features} features")
    print(f"  Benign: {n_benign:,}  Attack: {n_attack:,}")
    print(f"  Attack types: {sorted(y_raw_all.str.upper().unique().tolist())}")

    # ── Train / test split ────────────────────────────────────────────────────
    print("\n[Train] 80/20 stratified split …")
    X_tr, X_te, y_tr, y_te, ds_tr, ds_te, _, y_te_raw = train_test_split(
        X_all, y_all, ds_labels, y_raw_all,
        test_size=0.20, stratify=y_all, random_state=42,
    )
    t_train_0 = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr.values, y_tr)
    train_time = time.perf_counter() - t_train_0
    print(f"  Trained in {train_time:.1f}s")

    # ── Global test metrics ───────────────────────────────────────────────────
    t_inf_0 = time.perf_counter()
    proba   = model.predict_proba(X_te.values)[:, 1]
    inf_time = time.perf_counter() - t_inf_0

    best_thr = find_best_threshold(proba, y_te)
    y_pred   = (proba >= best_thr).astype(int)
    global_metrics = compute_metrics(y_te, y_pred, proba)
    print(f"  Best threshold: {best_thr:.2f}  |  Test F1: {global_metrics['f1_score']:.4f}")

    # ── Per-dataset metrics ───────────────────────────────────────────────────
    per_dataset = {}
    for ds in sorted(set(ds_te)):
        mask = (ds_te == ds)
        if mask.sum() < 5:
            continue
        X_ds, y_ds, p_ds = X_te[mask], y_te[mask], proba[mask]
        thr_ds = find_best_threshold(p_ds, y_ds)
        yp_ds  = (p_ds >= thr_ds).astype(int)
        try:
            per_dataset[ds] = compute_metrics(y_ds, yp_ds, p_ds)
            per_dataset[ds]["n_test_records"] = int(mask.sum())
            per_dataset[ds]["best_threshold"]  = thr_ds
        except Exception:
            pass

    # ── Per-attack-type breakdown ─────────────────────────────────────────────
    labels_u = y_te_raw.str.strip().str.upper()
    class_breakdown = {}
    for cls in sorted(labels_u.unique()):
        mask = (labels_u == cls).values
        total_   = int(mask.sum())
        detected_ = int((y_pred[mask] == 1).sum())
        class_breakdown[cls] = {
            "total":    total_,
            "detected": detected_,
            "recall":   round(detected_ / total_, 6) if total_ > 0 else 0.0,
        }

    # ── Cross-validation ──────────────────────────────────────────────────────
    print(f"\n[CV] {cv_folds}-fold stratified cross-validation …")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_f1, cv_roc = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        Xtr_f  = X_all.iloc[tr_idx]
        Xval_f = X_all.iloc[val_idx]
        ytr_f  = y_all[tr_idx]
        yval_f = y_all[val_idx]
        mf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )
        mf.fit(Xtr_f.values, ytr_f)
        pf     = mf.predict_proba(Xval_f.values)[:, 1]
        thr_f  = find_best_threshold(pf, yval_f)
        f1_f   = f1_score(yval_f, (pf >= thr_f).astype(int), zero_division=0)
        try:   roc_f = float(roc_auc_score(yval_f, pf))
        except Exception: roc_f = 0.0
        cv_f1.append(f1_f)
        cv_roc.append(roc_f)
        print(f"   Fold {fold}: F1={f1_f:.4f}  ROC-AUC={roc_f:.4f}  (thr={thr_f:.2f})")

    # ── Feature importance ────────────────────────────────────────────────────
    imp_df = (
        pd.DataFrame({"feature": all_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(RESULTS_DIR / "multi_dataset_feature_importance.csv", index=False)

    return {
        "n_total": n_total,
        "n_benign": n_benign,
        "n_attack": n_attack,
        "n_features": n_features,
        "best_threshold": best_thr,
        "global_metrics": global_metrics,
        "per_dataset_metrics": per_dataset,
        "per_attack_class": class_breakdown,
        "cross_validation": {
            "folds": cv_folds,
            "f1_scores":    [round(s, 6) for s in cv_f1],
            "roc_auc_scores": [round(s, 6) for s in cv_roc],
            "mean_f1":      round(float(np.mean(cv_f1)), 6),
            "std_f1":       round(float(np.std(cv_f1)), 6),
            "mean_roc_auc": round(float(np.mean(cv_roc)), 6),
        },
        "top_20_features": imp_df.head(20).to_dict(orient="records"),
        "timing": {
            "train_seconds":     round(train_time, 2),
            "inference_seconds": round(inf_time, 4),
            "throughput_rps":    round(len(X_te) / inf_time) if inf_time > 0 else None,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown_report(report: dict, dataset_summary: list[dict]) -> None:
    m   = report["global_metrics"]
    cv  = report["cross_validation"]
    ts  = report["timing"]
    lines = [
        "# Multi-Dataset Network Anomaly Detection — Training Report\n",
        f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "---\n",
        "## 1. Datasets Loaded\n",
        "| Dataset | File | Benign | Attack | Total |",
        "|---------|------|--------|--------|-------|",
    ]
    for ds in dataset_summary:
        lines.append(f"| {ds['dataset']} | {ds['file']} | {ds['n_benign']:,} | {ds['n_attack']:,} | {ds['total']:,} |")

    lines += [
        "",
        f"**Combined total:** {report['n_total']:,} records · {report['n_features']} features · "
        f"{report['n_benign']:,} benign / {report['n_attack']:,} attack\n",
        "---\n",
        "## 2. Global Classification Metrics (20% held-out test set)\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Accuracy** | **{m['accuracy']:.4f}** |",
        f"| **Precision** | **{m['precision']:.4f}** |",
        f"| **Recall** | **{m['recall']:.4f}** |",
        f"| **F1-Score** | **{m['f1_score']:.4f}** |",
        f"| Specificity | {m['specificity']:.4f} |",
        f"| False Positive Rate | {m['fpr']:.4f} |",
        f"| False Negative Rate | {m['fnr']:.4f} |",
        f"| **ROC-AUC** | **{m['roc_auc']}** |",
        f"| **PR-AUC** | **{m['pr_auc']}** |",
        f"| MCC | {m['mcc']:.4f} |",
        f"| Cohen's Kappa | {m['cohen_kappa']:.4f} |",
        f"| Decision Threshold | {report['best_threshold']:.2f} |",
        f"| TP | {m['true_positives']:,} |",
        f"| TN | {m['true_negatives']:,} |",
        f"| FP | {m['false_positives']:,} |",
        f"| FN | {m['false_negatives']:,} |",
        "",
        "---\n",
        "## 3. Per-Dataset Metrics\n",
        "| Dataset | F1 | ROC-AUC | Precision | Recall | N (test) |",
        "|---------|----|---------|-----------|----|------|",
    ]
    for ds_name, ds_m in report["per_dataset_metrics"].items():
        lines.append(
            f"| {ds_name} | {ds_m['f1_score']:.4f} | {ds_m['roc_auc']} "
            f"| {ds_m['precision']:.4f} | {ds_m['recall']:.4f} | {ds_m['n_test_records']:,} |"
        )

    lines += [
        "",
        f"## 4. {cv['folds']}-Fold Cross-Validation\n",
        "| Fold | F1 | ROC-AUC |",
        "|------|----|---------| ",
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
    for cls, info in sorted(report["per_attack_class"].items()):
        bar = "█" * int(info["recall"] * 20)
        lines.append(f"| {cls} | {info['total']:,} | {info['detected']:,} | {info['recall']:.4f} {bar} |")

    lines += [
        "",
        "## 6. Top 20 Feature Importances\n",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for i, row in enumerate(report["top_20_features"], 1):
        bar = "█" * int(row["importance"] * 200)
        lines.append(f"| {i} | `{row['feature']}` | {row['importance']:.5f} {bar} |")

    lines += [
        "",
        "## 7. Timing\n",
        "| Step | Value |",
        "|------|-------|",
        f"| Training | {ts['train_seconds']:.1f}s |",
        f"| Inference | {ts['inference_seconds']:.4f}s |",
        f"| Throughput | {ts['throughput_rps']:,} records/s |",
    ]

    with open(RESULTS_DIR / "multi_dataset_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report → data/results/multi_dataset_report.md")


def print_summary(report: dict) -> None:
    m  = report["global_metrics"]
    cv = report["cross_validation"]
    print()
    print("=" * 65)
    print("       MULTI-DATASET TRAINING SUMMARY")
    print("=" * 65)
    print(f"  Total records   : {report['n_total']:,}")
    print(f"  Features        : {report['n_features']}")
    print(f"  Threshold       : {report['best_threshold']:.2f}")
    print()
    print(f"  ── Global Test Metrics────────────────────────────────")
    print(f"  Accuracy        : {m['accuracy']:.4f}")
    print(f"  Precision       : {m['precision']:.4f}")
    print(f"  Recall          : {m['recall']:.4f}")
    print(f"  F1-Score        : {m['f1_score']:.4f}  ◄ primary")
    print(f"  ROC-AUC         : {m['roc_auc']}")
    print(f"  PR-AUC          : {m['pr_auc']}")
    print(f"  MCC             : {m['mcc']:.4f}")
    print(f"  FPR             : {m['fpr']:.4f}")
    print(f"  FNR             : {m['fnr']:.4f}")
    print(f"  TP={m['true_positives']:,}  TN={m['true_negatives']:,}  FP={m['false_positives']:,}  FN={m['false_negatives']:,}")
    print()
    print(f"  ── {cv['folds']}-Fold CV ────────────────────────────────────")
    print(f"  Mean F1         : {cv['mean_f1']:.4f} ± {cv['std_f1']:.4f}")
    print(f"  Mean ROC-AUC    : {cv['mean_roc_auc']:.4f}")
    print()
    print(f"  ── Per-Dataset ──────────────────────────────────────")
    for ds_name, ds_m in report["per_dataset_metrics"].items():
        print(f"  {ds_name:<18}: F1={ds_m['f1_score']:.4f}  ROC-AUC={ds_m['roc_auc']}")
    print("=" * 65)
    print(f"\n  JSON   → data/results/multi_dataset_metrics.json")
    print(f"  Report → data/results/multi_dataset_report.md")
    print(f"  FeatImp→ data/results/multi_dataset_feature_importance.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

DATASET_LOADERS = {
    "cic2017": ("CIC-IDS 2017", lambda mr: load_cic_ids(CIC2017_DIR, "CIC-IDS 2017", mr)),
    "cic2018": ("CIC-IDS 2018", lambda mr: load_cic_ids(CIC2018_DIR, "CIC-IDS 2018", mr)),
    "unsw":    ("UNSW-NB15",    lambda mr: load_unsw_nb15(UNSW_DIR, mr)),
    "nslkdd":  ("NSL-KDD",      lambda mr: load_nsl_kdd(NSL_DIR, mr)),
    "ctu13":   ("CTU-13",       lambda mr: load_ctu13(CTU_DIR, mr)),
}


def main():
    parser = argparse.ArgumentParser(description="Train RF on all datasets.")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_LOADERS.keys()),
        default=list(DATASET_LOADERS.keys()),
        help="Which dataset families to include (default: all).",
    )
    parser.add_argument("--max-rows-per-file", type=int, default=50_000,
                        help="Max rows sampled from each file (default: 50000).")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Number of CV folds (default: 3).")
    args = parser.parse_args()

    wall_t0 = time.perf_counter()

    # ── Load all datasets ─────────────────────────────────────────────────────
    all_chunks: list[dict] = []
    dataset_summary: list[dict] = []

    for key in args.datasets:
        ds_name, loader_fn = DATASET_LOADERS[key]
        print(f"\n{'─'*60}")
        print(f"  Loading {ds_name} …")
        print(f"{'─'*60}")
        chunks = loader_fn(args.max_rows_per_file)
        for c in chunks:
            n_b = int((c["y_bin"] == 0).sum())
            n_a = int((c["y_bin"] == 1).sum())
            dataset_summary.append({
                "dataset": c["dataset"],
                "file":    c["file"],
                "n_benign": n_b,
                "n_attack": n_a,
                "total":   len(c["X"]),
            })
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\n[!] No data loaded. Check dataset paths.")
        return

    total_loaded = sum(len(c["X"]) for c in all_chunks)
    print(f"\n{'─'*60}")
    print(f"  Total chunks loaded: {len(all_chunks)}")
    print(f"  Total rows         : {total_loaded:,}")
    print(f"{'─'*60}")

    # ── Align feature columns across datasets ─────────────────────────────────
    print("\n[Align] Unifying feature columns across datasets …")
    all_chunks, all_cols = align_columns(all_chunks)
    print(f"  Unified feature set: {len(all_cols)} columns")

    # ── Train & evaluate ──────────────────────────────────────────────────────
    print("\n[Train & Evaluate]")
    report = run_training(all_chunks, all_cols, args.cv_folds)

    wall_total = time.perf_counter() - wall_t0
    report["timing"]["total_wall_seconds"] = round(wall_total, 1)
    report["dataset_summary"] = dataset_summary

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "multi_dataset_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    write_markdown_report(report, dataset_summary)
    print_summary(report)

    print(f"\n  Total wall time: {wall_total/60:.1f} min")


if __name__ == "__main__":
    main()
