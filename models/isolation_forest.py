"""
models/isolation_forest.py
===========================
Unsupervised Isolation Forest for network anomaly detection.

This is the fallback model used when no ground-truth labels are
available (e.g. live Zeek logs).  When labels *are* available use the
supervised models in ``random_forest.py`` or ``xgboost_model.py`` for
much higher accuracy.

Label convention:
  anomaly_label == -1  →  anomaly
  anomaly_label ==  1  →  normal
  anomaly_score        →  higher = more anomalous (inverted decision fn.)
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    features: pd.DataFrame,
    n_estimators: int = 100,
    contamination: Union[float, str] = "auto",
    random_state: int = 42,
    true_labels: Optional[pd.Series] = None,
) -> IsolationForest:
    """
    Train an IsolationForest model on the provided feature matrix.

    Parameters
    ----------
    features : pd.DataFrame
        Numeric feature matrix.
    n_estimators : int, optional
        Number of trees in the forest.  Default: 100.
    contamination : float or "auto", optional
        Expected proportion of anomalies.  When ``true_labels`` is
        provided, the actual attack ratio is computed from the labels and
        used (capped to [0.01, 0.49]).  When ``contamination="auto"``
        and no labels are provided, scikit-learn's automatic threshold
        is used.  Default: ``"auto"``.
    random_state : int, optional
        Reproducibility seed.  Default: 42.
    true_labels : pd.Series, optional
        Optional ground-truth label series (e.g. from the CIC-IDS
        ``Label`` column).  "BENIGN" is treated as the normal class;
        everything else counts as an attack.  When given, the actual
        attack fraction overrides ``contamination``.

    Returns
    -------
    IsolationForest
        Fitted model.
    """
    if features.empty:
        raise ValueError("Feature matrix is empty; cannot train IsolationForest.")

    # Derive contamination from real label distribution when available
    if true_labels is not None:
        y_binary = (
            true_labels.fillna("BENIGN")
            .astype(str)
            .str.strip()
            .str.upper()
            .apply(lambda lbl: 0 if lbl == "BENIGN" else 1)
        )
        attack_ratio = float(y_binary.mean())
        # Clamp to a range sklearn accepts
        contamination = float(np.clip(attack_ratio, 0.01, 0.49))
        print(f"    [IF] Computed contamination from labels: {contamination:.4f}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features.values)
    return model


def score_anomalies(
    model: IsolationForest, features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute anomaly scores and labels for the given feature matrix.

    Parameters
    ----------
    model : IsolationForest
        Fitted model returned by ``train_isolation_forest``.
    features : pd.DataFrame
        Numeric feature matrix.

    Returns
    -------
    anomaly_score : np.ndarray
        Higher value means more anomalous (inverted decision function).
    anomaly_label : np.ndarray
        1 = normal, -1 = anomaly (matches IsolationForest convention).
    """
    if features.empty:
        return np.array([]), np.array([])

    # decision_function: larger → more normal; invert to get anomaly score.
    decision_values = model.decision_function(features.values)
    anomaly_score = -decision_values
    anomaly_label = model.predict(features.values)
    return anomaly_score, anomaly_label
