import os
from typing import Tuple

import numpy as np
import pandas as pd


def build_anomaly_dataframe(
    metadata: pd.DataFrame,
    anomaly_score: np.ndarray,
    anomaly_label: np.ndarray,
) -> pd.DataFrame:
    """
    Combine metadata and model outputs into a single DataFrame with:
      - timestamp
      - source_ip
      - destination_ip
      - protocol
      - anomaly_score
      - anomaly_label
    """
    if metadata.empty or anomaly_score.size == 0:
        return pd.DataFrame()

    if len(metadata) != len(anomaly_score):
        raise ValueError(
            f"Metadata length ({len(metadata)}) and anomaly scores length "
            f"({len(anomaly_score)}) do not match."
        )

    # Start from metadata so any context columns (e.g. Label, Flow Duration)
    # are preserved, then append anomaly score/label.
    result = metadata.copy().reset_index(drop=True)
    result["anomaly_score"] = anomaly_score
    result["anomaly_label"] = anomaly_label

    # Sort by anomaly score (descending: most anomalous first)
    result = result.sort_values(by="anomaly_score", ascending=False).reset_index(drop=True)
    return result


def save_anomalies_csv(df: pd.DataFrame, output_path: str) -> str:
    """
    Save anomalies DataFrame to CSV. Ensures parent directory exists.
    Returns the absolute path to the saved file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return os.path.abspath(output_path)

