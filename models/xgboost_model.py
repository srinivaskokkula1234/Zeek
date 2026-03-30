"""
models/xgboost_model.py
=======================
Supervised XGBoost classifier for network anomaly detection.

Provides the same (anomaly_score, anomaly_label) API as the
Isolation Forest module so the pipeline is interchangeable.

Label convention (shared with Isolation Forest):
  anomaly_label == -1  →  attack / anomaly
  anomaly_label ==  1  →  normal / benign
  anomaly_score        →  P(attack), higher = more suspicious

Install dependency (if missing):
  pip install xgboost>=2.0,<3.0
"""

from typing import Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier  # noqa: F401 – checked at import time
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


def _check_xgboost() -> None:
    """Raise a descriptive ImportError if xgboost is not installed."""
    if not _XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is required but is not installed. "
            "Install it with:\n\n"
            "    pip install 'xgboost>=2.0,<3.0'\n"
        )


def train_xgboost(
    features_df: pd.DataFrame,
    labels_series: pd.Series,
    max_depth: int = 6,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple["XGBClassifier", float]:
    """
    Train an XGBoost classifier on labelled network-flow data.

    Class imbalance is handled via ``scale_pos_weight`` computed as
    ``n_benign / n_attack`` so that attack samples are up-weighted.

    Parameters
    ----------
    features_df : pd.DataFrame
        Numeric feature matrix.
    labels_series : pd.Series
        Raw string labels.  "BENIGN" → 0, everything else → 1.
    max_depth : int, optional
        Maximum depth of each tree.  Default: 6.
    n_estimators : int, optional
        Number of boosting rounds.  Default: 300.
    learning_rate : float, optional
        Step size for boosting.  Default: 0.05.
    random_state : int, optional
        Reproducibility seed.  Default: 42.

    Returns
    -------
    model : XGBClassifier
        Fitted classifier.
    scale_pos_weight : float
        The ``scale_pos_weight`` value that was used (benign / attack ratio).

    Raises
    ------
    ImportError
        If ``xgboost`` is not installed.
    ValueError
        If the feature matrix is empty or lengths mismatch.
    """
    _check_xgboost()
    from xgboost import XGBClassifier  # local import after guard

    if features_df.empty:
        raise ValueError("Feature matrix is empty; cannot train XGBoost.")
    if len(features_df) != len(labels_series):
        raise ValueError(
            f"Feature matrix ({len(features_df)} rows) and label series "
            f"({len(labels_series)} rows) must have the same length."
        )

    # Binarise labels
    y_binary = (
        labels_series.fillna("BENIGN")
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda lbl: 0 if lbl == "BENIGN" else 1)
        .values
    )

    n_benign = int((y_binary == 0).sum())
    n_attack = int((y_binary == 1).sum())
    # Avoid division by zero when all records are one class
    spw = n_benign / max(n_attack, 1)

    model = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(features_df.values, y_binary)
    return model, spw


def score_xgboost(
    model: "XGBClassifier",
    features_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score records using a trained XGBoost model.

    Returns results in the Isolation Forest convention so that downstream
    code is fully interchangeable.

    Parameters
    ----------
    model : XGBClassifier
        Fitted classifier returned by ``train_xgboost``.
    features_df : pd.DataFrame
        Numeric feature matrix.
    threshold : float, optional
        Probability cut-off above which a record is an anomaly.
        Default: 0.5.

    Returns
    -------
    anomaly_score : np.ndarray
        P(attack) for each record; higher = more suspicious.
    anomaly_label : np.ndarray
        1 = normal, -1 = anomaly  (Isolation Forest convention).

    Raises
    ------
    ImportError
        If ``xgboost`` is not installed.
    """
    _check_xgboost()

    if features_df.empty:
        return np.array([]), np.array([])

    proba = model.predict_proba(features_df.values)[:, 1]
    anomaly_label = np.where(proba >= threshold, -1, 1)
    return proba, anomaly_label
