import os
from typing import Tuple

import pandas as pd


def _parse_zeek_tsv(path: str) -> pd.DataFrame:
    """
    Parse a raw Zeek .log file (tab-separated with #fields header)
    into a pandas DataFrame with proper column names.
    Falls back to a generic CSV parser if Zeek headers are not present.
    """
    field_names = None

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#fields"):
                    parts = line.strip().split()
                    # '#fields' <name1> <name2> ...
                    field_names = parts[1:]
                    break
    except FileNotFoundError:
        raise

    if field_names:
        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            header=None,
            names=field_names,
            na_values=["-"],
        )
    else:
        # Assume a regular CSV/TSV with a header row
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            na_values=["-"],
        )

    return df


def load_zeek_log(path: str) -> pd.DataFrame:
    """
    Load a Zeek log (http/dns/ssl/conn, .log or .csv) into a DataFrame.

    - Supports raw Zeek .log files with '#fields' header.
    - Supports CSV/TSV files with standard headers.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Zeek log not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".log":
        df = _parse_zeek_tsv(path)
    else:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            na_values=["-"],
        )

    # Strip whitespace from column names to better match known fields
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize missing values
    df = df.fillna(pd.NA)
    return df


def build_common_columns(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """
    Build a minimal, protocol-agnostic set of common columns:
      - timestamp
      - source_ip
      - destination_ip
      - protocol
    """
    ts_col_candidates = ["ts", "timestamp", "time"]
    src_col_candidates = ["id.orig_h", "src_ip", "source_ip"]
    dst_col_candidates = ["id.resp_h", "dst_ip", "destination_ip"]

    def _first_existing(columns) -> str:
        for c in columns:
            if c in df.columns:
                return c
        return None

    ts_col = _first_existing(ts_col_candidates)
    src_col = _first_existing(src_col_candidates)
    dst_col = _first_existing(dst_col_candidates)

    common = pd.DataFrame(index=df.index)
    common["timestamp"] = df[ts_col] if ts_col else pd.Series([pd.NA] * len(df), index=df.index)
    common["source_ip"] = df[src_col] if src_col else pd.Series([pd.NA] * len(df), index=df.index)
    common["destination_ip"] = df[dst_col] if dst_col else pd.Series([pd.NA] * len(df), index=df.index)
    common["protocol"] = protocol

    return common


def ensure_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    """
    Ensure given columns are numeric. Missing columns are created as zeros.
    Non-convertible values are coerced to NaN and then filled with 0.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def train_test_split_unsupervised(
    features: pd.DataFrame, train_fraction: float = 0.8, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple splitter for unsupervised training:
      - Shuffle rows
      - Take a fraction for training, rest for evaluation/detection
    """
    if features.empty:
        return features, features

    shuffled = features.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_train = max(1, int(len(shuffled) * train_fraction))
    train = shuffled.iloc[:n_train].reset_index(drop=True)
    test = shuffled.iloc[n_train:].reset_index(drop=True)
    return train, test

