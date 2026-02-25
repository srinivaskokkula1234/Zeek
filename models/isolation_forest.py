from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    features: pd.DataFrame,
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train an IsolationForest model on the provided feature matrix.
    """
    if features.empty:
        raise ValueError("Feature matrix is empty; cannot train IsolationForest.")

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

    Returns:
      - anomaly_score: higher means more anomalous
      - anomaly_label: 1 (normal) or -1 (anomaly), matching IsolationForest
    """
    if features.empty:
        return np.array([]), np.array([])

    # decision_function: larger -> more normal. Invert to get an "anomaly score".
    decision_values = model.decision_function(features.values)
    anomaly_score = -decision_values
    anomaly_label = model.predict(features.values)
    return anomaly_score, anomaly_label

