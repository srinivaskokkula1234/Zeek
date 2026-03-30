"""
training/multi_dataset_trainer.py
=================================
Train supervised models across multiple heterogeneous datasets.

Supports:
  - combined: train one model on concatenated datasets (aligned features)
  - per_dataset: train one model per dataset
  - cross_eval: leave-one-dataset-out evaluation matrix
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from feature_engineering.derive_features import add_derived_features
from feature_engineering.feature_aligner import FeatureAligner
from feature_engineering.extract_features import aggregate_features_from_directory
from utils.dataset_adapters import adapt_cicids2018, adapt_nslkdd, adapt_unswnb15
from utils.dataset_registry import discover_datasets


def _binarise(labels: pd.Series) -> np.ndarray:
    """Return y in {0,1} where 1 = attack/anomaly."""
    return (
        labels.fillna("BENIGN")
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda x: 0 if x == "BENIGN" else 1)
        .values
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> Dict:
    """Compute core supervised metrics."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, score))
        except Exception:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, score))
        except Exception:
            out["pr_auc"] = None
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _per_attack_breakdown(labels: pd.Series, y_pred: np.ndarray) -> Dict[str, Dict]:
    """Per attack-type recall breakdown using raw string labels."""
    labels_u = labels.fillna("BENIGN").astype(str).str.strip().str.upper()
    out: Dict[str, Dict] = {}
    for cls in sorted(labels_u.unique()):
        mask = (labels_u == cls).values
        total = int(mask.sum())
        detected = int((y_pred[mask] == 1).sum())
        out[cls] = {
            "total": total,
            "detected": detected,
            "recall": float(detected / total) if total > 0 else 0.0,
        }
    return out


def _train_rf(X: pd.DataFrame, y_bin: np.ndarray) -> RandomForestClassifier:
    """Train a balanced Random Forest with fixed hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.values, y_bin)
    return model


def _maybe_smote(X: pd.DataFrame, y_bin: np.ndarray, verbose: bool) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Apply SMOTE only if minority class < 10% of majority.
    Returns (X_res, y_res, info_dict).
    """
    n0 = int((y_bin == 0).sum())
    n1 = int((y_bin == 1).sum())
    minority = min(n0, n1)
    majority = max(n0, n1)
    info = {"applied": False, "n_benign": n0, "n_attack": n1}

    if minority == 0 or majority == 0:
        return X, y_bin, info

    if minority / majority < 0.10:
        if verbose:
            print(f"[+] Applying SMOTE (minority/majority={minority}/{majority}) …")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X.values, y_bin)
        X_res = pd.DataFrame(X_res, columns=X.columns)
        info["applied"] = True
        info["n_benign_after"] = int((y_res == 0).sum())
        info["n_attack_after"] = int((y_res == 1).sum())
        return X_res, y_res, info

    return X, y_bin, info


def _load_dataset(descriptor: Dict, sample_frac: float, verbose: bool) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and adapt a dataset descriptor into (X, labels).
    """
    dtype = descriptor["type"]
    paths = descriptor["paths"]

    if dtype == "cicids2018":
        return adapt_cicids2018(paths["directory"], sample_frac=sample_frac)
    if dtype == "nslkdd":
        return adapt_nslkdd(paths["train"], paths["test"], sample_frac=sample_frac)
    if dtype == "unswnb15":
        return adapt_unswnb15(paths["train"], paths["test"], sample_frac=sample_frac)
    if dtype == "cicids2017":
        # Use existing generic CIC-IDS 2017 handling by reading directory as raw logs
        metadata, feats, _ = aggregate_features_from_directory(paths["directory"])
        if "Label" not in metadata.columns:
            raise ValueError("CIC-IDS 2017 directory does not include Label column.")
        labels = metadata["Label"].fillna("BENIGN").astype(str)
        X = feats.fillna(0)
        return X, labels

    # generic fallback: try existing feature extraction directory
    metadata, feats, _ = aggregate_features_from_directory(paths["directory"])
    if "Label" in metadata.columns:
        labels = metadata["Label"].fillna("BENIGN").astype(str)
    else:
        labels = pd.Series(["BENIGN"] * len(feats))
    return feats.fillna(0), labels


def train_on_all_datasets(
    raw_dir: str,
    output_dir: str,
    strategy: str = "combined",
    sample_frac: float = 1.0,
    verbose: bool = False,
) -> Tuple[object, FeatureAligner, Dict]:
    """
    Train models on all discovered datasets.

    Parameters
    ----------
    raw_dir : str
        Root directory to scan for datasets.
    output_dir : str
        Directory to save trained models and reports.
    strategy : str, optional
        One of {"combined","per_dataset","cross_eval"}. Default: "combined".
    sample_frac : float, optional
        Fraction in (0,1] to sample from each dataset for faster runs.
    verbose : bool, optional
        Verbose logging.

    Returns
    -------
    (model, aligner, metrics_dict)
        For "combined": a trained RF model + aligner + metrics.
        For other strategies: see return described in train_multi.py (handled there).
    """
    os.makedirs(output_dir, exist_ok=True)
    discovered = discover_datasets(raw_dir)
    # De-duplicate registrations (especially NSL-KDD which may be discovered twice)
    keep = []
    seen_keys = set()
    for d in discovered:
        d_type = d.get("type")
        name = d.get("name")
        paths = d.get("paths", {}) or {}
        # Key based on type + sorted path values to avoid duplicates
        path_values = sorted([str(v) for v in paths.values()])
        key = (d_type, name, tuple(path_values))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        keep.append(d)
    discovered = keep

    if strategy not in {"combined", "per_dataset", "cross_eval"}:
        raise ValueError("strategy must be one of: combined, per_dataset, cross_eval")

    # Load/adapt each dataset
    X_list: List[pd.DataFrame] = []
    y_list: List[pd.Series] = []
    names: List[str] = []

    for d in tqdm(discovered, desc="Adapt datasets", unit="dataset"):
        name = d["name"]
        try:
            X, y = _load_dataset(d, sample_frac=sample_frac, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[!] Skipping {name}: {e}")
            continue

        # Add derived features where possible
        combined = pd.concat([X.reset_index(drop=True)], axis=1)
        combined = add_derived_features(combined)
        # Ensure numeric dtype + reduce memory
        combined = combined.select_dtypes(include=["number"]).astype(np.float32, copy=False)
        X = combined

        X_list.append(X)
        y_list.append(y.reset_index(drop=True))
        names.append(name)

    if not X_list:
        raise ValueError("No datasets could be loaded/adapted from raw_dir.")

    aligner = FeatureAligner()
    X_aligned_list = aligner.fit_transform(X_list)
    # Safety cast for memory
    X_aligned_list = [Xd.select_dtypes(include=["number"]).astype(np.float32, copy=False) for Xd in X_aligned_list]

    report: Dict = {"strategy": strategy, "datasets": {}, "alignment": {"n_features": aligner.n_features}}

    def _eval_one(model: RandomForestClassifier, X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict:
        y_true = _binarise(y_eval)
        proba = model.predict_proba(X_eval.values)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        return {
            "metrics": _metrics(y_true, y_pred, proba),
            "per_attack": _per_attack_breakdown(y_eval, y_pred),
        }

    # ------------------------------------------------------------
    # Strategy: combined
    # ------------------------------------------------------------
    if strategy == "combined":
        X_all = pd.concat(X_aligned_list, axis=0, ignore_index=True)
        y_all = pd.concat(y_list, axis=0, ignore_index=True)
        y_bin = _binarise(y_all)

        # Stratified train/test split
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X_all, y_bin))
        X_train = X_all.iloc[train_idx]
        y_train = y_bin[train_idx]
        y_train_raw = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_test_raw = y_all.iloc[test_idx]

        X_train_sm, y_train_sm, smote_info = _maybe_smote(X_train, y_train, verbose=verbose)
        if isinstance(X_train_sm, pd.DataFrame):
            X_train_sm = X_train_sm.astype(np.float32, copy=False)
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.astype(np.float32, copy=False)

        if verbose:
            print("[+] Training combined Random Forest …")
        model = _train_rf(X_train_sm, y_train_sm)

        # Evaluate on combined test split
        y_true = _binarise(y_test_raw)
        proba = model.predict_proba(X_test.values)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        combined_metrics = _metrics(y_true, y_pred, proba)

        report["combined"] = {
            "smote": smote_info,
            "test_metrics": combined_metrics,
        }

        # Per-dataset evaluation using the same model
        for name, Xd, yd in zip(names, X_aligned_list, y_list):
            ev = _eval_one(model, Xd, yd)
            report["datasets"][name] = ev

        # Save
        import joblib

        model_path = os.path.join(output_dir, "combined_rf_model.pkl")
        aligner_path = os.path.join(output_dir, "feature_aligner.pkl")
        joblib.dump(model, model_path)
        aligner.save(aligner_path)

        with open(os.path.join(output_dir, "multi_dataset_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return model, aligner, report

    # ------------------------------------------------------------
    # Strategy: per_dataset
    # ------------------------------------------------------------
    if strategy == "per_dataset":
        import joblib

        models_out: Dict[str, Dict] = {}
        for name, Xd, yd in zip(names, X_aligned_list, y_list):
            y_bin = _binarise(yd)
            X_train_sm, y_train_sm, smote_info = _maybe_smote(Xd, y_bin, verbose=verbose)
            model = _train_rf(X_train_sm, y_train_sm)
            ev = _eval_one(model, Xd, yd)
            models_out[name] = {"smote": smote_info, **ev}
            joblib.dump(model, os.path.join(output_dir, f"{name}_rf_model.pkl"))

        aligner_path = os.path.join(output_dir, "feature_aligner.pkl")
        aligner.save(aligner_path)

        report["per_dataset"] = models_out
        with open(os.path.join(output_dir, "multi_dataset_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return models_out, aligner, report

    # ------------------------------------------------------------
    # Strategy: cross_eval
    # ------------------------------------------------------------
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, test_name in enumerate(names):
        # Train on all others
        X_train_parts = [X_aligned_list[j] for j in range(len(names)) if j != i]
        y_train_parts = [y_list[j] for j in range(len(names)) if j != i]
        X_train = pd.concat(X_train_parts, axis=0, ignore_index=True)
        y_train_raw = pd.concat(y_train_parts, axis=0, ignore_index=True)
        y_train = _binarise(y_train_raw)

        X_train_sm, y_train_sm, _smote_info = _maybe_smote(X_train, y_train, verbose=verbose)
        model = _train_rf(X_train_sm, y_train_sm)

        # Evaluate on held-out dataset
        X_test = X_aligned_list[i]
        y_test_raw = y_list[i]
        y_true = _binarise(y_test_raw)
        proba = model.predict_proba(X_test.values)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        matrix.loc[test_name, "TRAIN=ALL-OTHERS"] = f1
        report["datasets"][test_name] = {
            "f1_cross_eval": f1,
            "metrics": _metrics(y_true, y_pred, proba),
        }

    with open(os.path.join(output_dir, "multi_dataset_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return matrix, aligner, report

