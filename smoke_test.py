"""
smoke_test.py  –  validates all pipeline fixes on a single CIC-IDS file.
Run:  python smoke_test.py
"""
import sys, traceback
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from feature_engineering.extract_features import extract_protocol_features
from feature_engineering.derive_features import add_derived_features
from models.random_forest import train_random_forest, score_supervised, tune_threshold

FP = "data/raw_zeek_logs/CIC-IDS- 2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"


def run():
    print(f"[1] Loading CIC-IDS file ...")
    meta, feats = extract_protocol_features(FP, "GENERIC")
    meta_cols_str = list(meta.columns)
    print(f"    meta cols     : {meta_cols_str}")
    print(f"    feats shape   : {feats.shape}")
    has_lbl = "Label" in meta.columns
    print(f"    Label present : {has_lbl}")

    print("[2] Dedup-safe concat …")
    meta_r  = meta.reset_index(drop=True)
    feats_r = feats.reset_index(drop=True)
    overlap = [c for c in feats_r.columns if c in meta_r.columns]
    print(f"    overlapping cols: {overlap}")
    if overlap:
        feats_r = feats_r.drop(columns=overlap)
    combined = pd.concat([meta_r, feats_r], axis=1)
    print(f"    combined shape  : {combined.shape}")
    assert not combined.columns.duplicated().any(), "Duplicate columns still present!"

    print("[3] Deriving features …")
    enriched = add_derived_features(combined)
    new_cols = [c for c in enriched.columns if c not in combined.columns]
    print(f"    enriched shape  : {enriched.shape}")
    print(f"    new cols added  : {new_cols}")

    DROP = {"Label", "timestamp", "source_ip", "destination_ip", "protocol"}
    feat_cols = [
        c for c in enriched.columns
        if c not in DROP and pd.api.types.is_numeric_dtype(enriched[c])
    ]
    X = enriched[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = meta["Label"].fillna("BENIGN").astype(str)

    # Sample 50 000 rows for a quick sanity check
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X), min(50_000, len(X)), replace=False)
    X_s, y_s = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    print(f"[4] Training RF on {len(X_s):,} rows × {X_s.shape[1]} features …")
    model, _ = train_random_forest(X_s, y_s)

    print("[5] Tuning threshold …")
    thr = tune_threshold(model, X_s, y_s)
    print(f"    threshold: {thr:.4f}")

    print("[6] Scoring …")
    proba, _ = score_supervised(model, X_s)
    y_bin = (y_s.str.strip().str.upper() != "BENIGN").astype(int).values
    pred  = (proba >= thr).astype(int)
    f1    = f1_score(y_bin, pred, zero_division=0)
    auc   = roc_auc_score(y_bin, proba)
    print(f"    F1={f1:.4f}   ROC-AUC={auc:.4f}")

    assert f1 > 0.50, f"F1 unexpectedly low: {f1:.4f}"
    print("\n✅  SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
