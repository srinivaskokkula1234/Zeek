"""
models/random_forest.py
=======================
Supervised Random Forest classifier for network anomaly detection.

Provides the same (anomaly_score, anomaly_label) API as the Isolation
Forest module so that the rest of the pipeline is interchangeable.

Label convention (shared with Isolation Forest):
  anomaly_label == -1  →  attack / anomaly
  anomaly_label ==  1  →  normal / benign
  anomaly_score        →  P(attack), higher = more suspicious
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder


def train_random_forest(
    features_df: pd.DataFrame,
    labels_series: pd.Series,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """
    Train a balanced Random Forest classifier on labelled data.

    Parameters
    ----------
    features_df : pd.DataFrame
        Numeric feature matrix (rows = records, columns = features).
    labels_series : pd.Series
        Raw string labels aligned with features_df.
        "BENIGN" is treated as the negative class (0); everything else
        is treated as an attack (1).
    n_estimators : int, optional
        Number of trees.  Default: 300.
    random_state : int, optional
        Reproducibility seed.  Default: 42.

    Returns
    -------
    model : RandomForestClassifier
        Fitted classifier with ``predict_proba`` available.
    label_encoder : LabelEncoder
        Fitted encoder used to binarise ``labels_series``
        (classes_: [0, 1] where 1 = attack).
    """
    if features_df.empty:
        raise ValueError("Feature matrix is empty; cannot train Random Forest.")
    if len(features_df) != len(labels_series):
        raise ValueError(
            f"Feature matrix ({len(features_df)} rows) and label series "
            f"({len(labels_series)} rows) must have the same length."
        )

    # Binarise labels: BENIGN → 0, attack → 1
    y_binary = (
        labels_series.fillna("BENIGN")
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda lbl: 0 if lbl == "BENIGN" else 1)
        .values
    )

    le = LabelEncoder()
    le.fit(y_binary)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features_df.values, y_binary)
    return model, le


def score_supervised(
    model: RandomForestClassifier,
    features_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score records using a trained model.

    Returns results in the same convention as ``score_anomalies`` in the
    Isolation Forest module so that downstream code is interchangeable.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted classifier returned by ``train_random_forest``.
    features_df : pd.DataFrame
        Numeric feature matrix.
    threshold : float, optional
        Probability cut-off above which a record is classified as an
        anomaly (attack).  Default: 0.5.

    Returns
    -------
    anomaly_score : np.ndarray
        P(attack) for each record; higher = more suspicious.
    anomaly_label : np.ndarray
        1 = normal, -1 = anomaly  (Isolation Forest convention).
    """
    if features_df.empty:
        return np.array([]), np.array([])

    proba = model.predict_proba(features_df.values)[:, 1]  # P(attack)
    anomaly_label = np.where(proba >= threshold, -1, 1)
    return proba, anomaly_label


def tune_threshold(
    model: RandomForestClassifier,
    features_df: pd.DataFrame,
    labels_series: pd.Series,
) -> float:
    """
    Find the probability threshold that maximises F1 score on known labels.

    Uses ``precision_recall_curve`` so no additional data is needed beyond
    the training set; suitable for a final threshold calibration step.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted classifier.
    features_df : pd.DataFrame
        Feature matrix (same split used for training is acceptable as a
        quick calibration; use a held-out set for unbiased estimation).
    labels_series : pd.Series
        Ground-truth labels ("BENIGN" = 0, everything else = 1).

    Returns
    -------
    float
        Optimal probability threshold in [0, 1].
    """
    proba = model.predict_proba(features_df.values)[:, 1]

    y_binary = (
        labels_series.fillna("BENIGN")
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda lbl: 0 if lbl == "BENIGN" else 1)
        .values
    )

    precision, recall, thresholds = precision_recall_curve(y_binary, proba)

    # Compute F1 for each threshold (avoid division by zero)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )

    best_idx = int(np.argmax(f1_scores[:-1]))  # thresholds has one fewer element
    best_threshold = float(thresholds[best_idx])
    return best_threshold
