"""
utils/dataset_adapters.py
==========================
Dataset-specific loaders/adapters that normalise features + labels into a
common convention compatible with the existing supervised pipeline:

  - labels: "BENIGN" for normal traffic, otherwise a descriptive attack name
  - features: numeric DataFrame suitable for scikit-learn models

Each adapter returns:
  (features_df: pd.DataFrame, labels_series: pd.Series)
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Column mapping constants (common schema)
# ---------------------------------------------------------------------------

CIC_TO_COMMON = {
    "Dst Port": "destination_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "Flow Bytes/s": "bytes_per_sec",
    "Flow Packets/s": "packets_per_sec",
    "SYN Flag Count": "syn_count",
    "FIN Flag Count": "fin_count",
    "RST Flag Count": "rst_count",
}

NSL_TO_COMMON = {
    "duration": "flow_duration",
    "src_bytes": "fwd_bytes",
    "dst_bytes": "bwd_bytes",
    "count": "fwd_packets",
    "srv_count": "bwd_packets",
}

UNSW_TO_COMMON = {
    "dur": "flow_duration",
    "sbytes": "fwd_bytes",
    "dbytes": "bwd_bytes",
    "spkts": "fwd_packets",
    "dpkts": "bwd_packets",
    "rate": "packets_per_sec",
    "sload": "bytes_per_sec",
    "synack": "syn_count",
}


# ---------------------------------------------------------------------------
# CIC-IDS 2018
# ---------------------------------------------------------------------------

def adapt_cicids2018(directory: str, sample_frac: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Adapt CIC-IDS 2018 processed CSV files to (features, labels).

    Parameters
    ----------
    directory : str
        Directory containing multiple CIC-IDS 2018 CSV files.
    sample_frac : float, optional
        Fraction in (0, 1] of each file/chunk to sample for faster runs.
        Default: 1.0 (use all rows).

    Returns
    -------
    (pd.DataFrame, pd.Series)
        features_df (numeric), labels_series ("BENIGN" or attack name).
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"CIC-IDS 2018 directory not found: {directory}")

    csv_paths: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, fn))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {directory}")

    sample_frac = float(sample_frac)
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0, 1].")

    frames: List[pd.DataFrame] = []
    for path in tqdm(sorted(csv_paths), desc="Load CIC-IDS2018 CSVs", unit="file"):
        # Memory safety: stream in chunks for very large files
        try:
            reader = pd.read_csv(path, chunksize=200_000, low_memory=False)
        except TypeError:
            reader = [pd.read_csv(path, low_memory=False)]

        for chunk in reader:
            # Strip whitespace from columns
            chunk.columns = [str(c).strip() for c in chunk.columns]

            if "Label" not in chunk.columns:
                continue

            # Drop duplicate header rows embedded in file
            chunk = chunk[chunk["Label"].astype(str) != "Label"]

            # Sample
            if sample_frac < 1.0 and len(chunk) > 0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)

            frames.append(chunk)

    if not frames:
        raise ValueError("No usable rows found for CIC-IDS 2018 (missing Label?).")

    df = pd.concat(frames, axis=0, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalise label: uppercase+strip; BENIGN if equals BENIGN
    labels = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda x: "BENIGN" if x == "BENIGN" else x)
    )

    # Apply common mapping before dropping
    df.rename(columns=CIC_TO_COMMON, inplace=True)

    drop_cols = [c for c in ["Timestamp", "Label"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, labels


# ---------------------------------------------------------------------------
# NSL-KDD
# ---------------------------------------------------------------------------

NSL_COLS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "class",
    "difficulty",
]


def adapt_nslkdd(
    train_path: str, test_path: str, sample_frac: float = 1.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Adapt NSL-KDD train/test files to (features, labels).

    Parameters
    ----------
    train_path : str
        Path to KDDTrain+.txt (or equivalent).
    test_path : str
        Path to KDDTest+.txt (or equivalent).
    sample_frac : float, optional
        Fraction in (0, 1] to sample from the concatenated dataset.
        Default: 1.0.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        features_df (numeric), labels_series ("BENIGN" or ATTACKNAME).
    """
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"NSL-KDD file not found: {p}")

    sample_frac = float(sample_frac)
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0, 1].")

    # NSL-KDD is comma-separated in most mirrors; allow auto-detect
    train_df = pd.read_csv(train_path, header=None, names=NSL_COLS, sep=None, engine="python")
    test_df = pd.read_csv(test_path, header=None, names=NSL_COLS, sep=None, engine="python")
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    if sample_frac < 1.0 and len(df) > 0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # Labels
    labels = (
        df["class"]
        .fillna("normal")
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda x: "BENIGN" if x == "normal" else x.upper())
    )

    # Encode categorical columns
    cat_cols = ["protocol_type", "service", "flag"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("NA").astype(str))

    # Apply common mapping
    df.rename(columns=NSL_TO_COMMON, inplace=True)

    # Drop non-features
    X = df.drop(columns=["class", "difficulty"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, labels


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------

def adapt_unswnb15(
    train_path: str, test_path: str, sample_frac: float = 1.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Adapt UNSW-NB15 train/test CSVs to (features, labels).

    Parameters
    ----------
    train_path : str
        Path to UNSW NB15 training CSV.
    test_path : str
        Path to UNSW NB15 testing CSV.
    sample_frac : float, optional
        Fraction in (0, 1] to sample from the concatenated dataset.
        Default: 1.0.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        features_df (numeric), labels_series ("BENIGN" or ATTACKCAT).
    """
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"UNSW-NB15 file not found: {p}")

    sample_frac = float(sample_frac)
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0, 1].")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    if sample_frac < 1.0 and len(df) > 0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # Build labels from attack_cat
    if "attack_cat" in df.columns:
        raw = df["attack_cat"]
    else:
        raw = pd.Series([pd.NA] * len(df), index=df.index)

    raw_s = raw.fillna("").astype(str).str.strip()
    labels = raw_s.apply(
        lambda x: "BENIGN" if (x == "" or x.lower() == "normal") else x.upper()
    )

    # Encode categorical columns
    for col in ["proto", "service", "state"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("NA").astype(str))

    # Apply common mapping
    df.rename(columns=UNSW_TO_COMMON, inplace=True)

    # Drop non-feature columns
    drop_cols = {"id", "attack_cat", "label", "srcip", "dstip"}
    drop_cols |= {c for c in df.columns if str(c).lower() in {"stime", "ltime"}}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Keep only numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, labels

