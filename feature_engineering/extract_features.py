from typing import Dict, List, Tuple

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.preprocess import build_common_columns, ensure_numeric, load_zeek_log


class FeatureEncoder:
    """
    Lightweight wrapper to manage label encoders for multiple categorical columns.
    """

    def __init__(self) -> None:
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame, categorical_cols: List[str]) -> None:
        for col in categorical_cols:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            # Work on string representation to be robust
            le.fit(df[col].astype(str).fillna("NA"))
            self.encoders[col] = le

    def transform(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in categorical_cols:
            # If we never fitted an encoder for this column, drop it (if present)
            # to avoid leaving non-numeric strings like 'NA' in the feature matrix.
            if col not in self.encoders:
                if col in df.columns:
                    df = df.drop(columns=[col])
                continue

            # Ensure column exists; if missing, fill with a default category.
            if col not in df.columns:
                df[col] = "NA"

            le = self.encoders[col]
            values = df[col].astype(str).fillna("NA")
            df[col] = le.transform(values)
        return df


def _common_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build protocol-agnostic numeric features:
      - duration
      - bytes_sent
      - bytes_received
      - packet_count
      - connection_state (categorical, handled later)
    """
    data = pd.DataFrame(index=df.index)

    # Duration
    if "duration" in df.columns:
        data["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    else:
        data["duration"] = 0

    # Bytes sent / received
    # Prefer conn.log style fields, fall back to HTTP/DNS approximations
    sent_candidates = ["orig_bytes", "request_body_len", "request_len"]
    recv_candidates = ["resp_bytes", "response_body_len", "response_len", "answer_len"]

    def _first_existing(col_names):
        for c in col_names:
            if c in df.columns:
                return c
        return None

    sent_col = _first_existing(sent_candidates)
    recv_col = _first_existing(recv_candidates)

    data["bytes_sent"] = (
        pd.to_numeric(df[sent_col], errors="coerce").fillna(0) if sent_col else 0
    )
    data["bytes_received"] = (
        pd.to_numeric(df[recv_col], errors="coerce").fillna(0) if recv_col else 0
    )

    # Packet counts
    if "orig_pkts" in df.columns or "resp_pkts" in df.columns:
        orig_pkts = pd.to_numeric(df.get("orig_pkts", 0), errors="coerce").fillna(0)
        resp_pkts = pd.to_numeric(df.get("resp_pkts", 0), errors="coerce").fillna(0)
        data["packet_count"] = orig_pkts + resp_pkts
    else:
        data["packet_count"] = 0

    # Connection state (kept as raw, encoded later)
    data["connection_state"] = df.get("conn_state", "NA")

    return data


def _http_specific(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(index=df.index)
    # Ensure we always work with Series aligned to df.index
    if "method" in df.columns:
        method_series = df["method"]
    else:
        method_series = pd.Series(["UNKNOWN"] * len(df), index=df.index)
    data["http_method"] = method_series
    # Zeek HTTP uses 'status_code' for response code
    if "status_code" in df.columns:
        status_series = df["status_code"]
    else:
        status_series = pd.Series([0] * len(df), index=df.index)
    data["http_response_code"] = status_series
    # URI length
    if "uri" in df.columns:
        uri_series = df["uri"]
    else:
        uri_series = pd.Series([""] * len(df), index=df.index)
    data["http_uri_length"] = uri_series.fillna("").astype(str).str.len()
    return data


def _dns_specific(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(index=df.index)
    if "query" in df.columns:
        query_series = df["query"]
    else:
        query_series = pd.Series([""] * len(df), index=df.index)
    data["dns_query_length"] = query_series.fillna("").astype(str).str.len()

    # answers: space or comma-separated list; count elements
    if "answers" in df.columns:
        answers = df["answers"]
    else:
        answers = pd.Series([""] * len(df), index=df.index)
    answers = answers.fillna("").astype(str)
    data["dns_answer_count"] = answers.apply(
        lambda x: 0 if not x or x == "-" else len([p for p in str(x).split(",") if p])
    )

    # TTL: use first TTL from 'TTLs' column if present
    if "TTLs" in df.columns:
        ttls = df["TTLs"]
    else:
        ttls = pd.Series([""] * len(df), index=df.index)
    ttls = ttls.fillna("").astype(str)
    data["dns_ttl"] = ttls.apply(
        lambda x: float(str(x).split(",")[0]) if x and x != "-" else 0.0
    )

    return ensure_numeric(data, ["dns_query_length", "dns_answer_count", "dns_ttl"])


def _https_specific(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(index=df.index)

    # Zeek ssl.log style fields
    if "version" in df.columns:
        version_series = df["version"]
    else:
        version_series = pd.Series(["UNKNOWN"] * len(df), index=df.index)
    data["ssl_version"] = version_series

    if "cipher" in df.columns:
        cipher_series = df["cipher"]
    else:
        cipher_series = pd.Series(["UNKNOWN"] * len(df), index=df.index)
    data["ssl_cipher"] = cipher_series

    # Approximate certificate validity if not_before/not_after exist (from x509 logs)
    not_before = df.get("cert_valid_not_before", pd.NA)
    not_after = df.get("cert_valid_not_after", pd.NA)

    if not_before is not pd.NA and not_after is not pd.NA:
        try:
            nb = pd.to_datetime(not_before, errors="coerce")
            na = pd.to_datetime(not_after, errors="coerce")
            validity_days = (na - nb).dt.total_seconds() / 86400.0
            data["cert_validity"] = validity_days.fillna(0)
        except Exception:
            data["cert_validity"] = 0.0
    else:
        data["cert_validity"] = 0.0

    return ensure_numeric(data, ["cert_validity"])


def extract_protocol_features(
    log_path: str, protocol: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a single Zeek log and extract:
      - metadata (timestamp/source_ip/destination_ip/protocol)
      - numeric feature matrix for the chosen protocol
    """
    df = load_zeek_log(log_path)
    protocol_upper = protocol.upper()

    if protocol_upper == "GENERIC":
        # Generic CSV-based datasets such as CIC-IDS:
        # - use all numeric columns (except label) as features
        # - keep key descriptive columns (e.g. Label, Flow Duration, Total Fwd Packets)
        #   in metadata so they appear in the results.
        metadata = pd.DataFrame(index=df.index)

        # Carry through useful identifying/context columns when present
        for col in [
            "Label",
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
        ]:
            if col in df.columns:
                metadata[col] = df[col]

        metadata["protocol"] = protocol_upper

        # Use numeric columns as features, excluding the label if numeric
        features = df.select_dtypes(include=["number"]).copy()
        if "Label" in features.columns:
            features = features.drop(columns=["Label"])
    else:
        # Zeek-style logs: build common + protocol-specific features
        metadata = build_common_columns(df, protocol_upper)
        features = _common_numeric_features(df)

        if protocol_upper == "HTTP":
            proto_feats = _http_specific(df)
        elif protocol_upper == "DNS":
            proto_feats = _dns_specific(df)
        elif protocol_upper in ("HTTPS", "SSL"):
            proto_feats = _https_specific(df)
        else:
            proto_feats = pd.DataFrame(index=df.index)

        features = pd.concat([features, proto_feats], axis=1)

        # Ensure all numeric fields well-formed; categorical left for encoding
        numeric_cols = [
            "duration",
            "bytes_sent",
            "bytes_received",
            "packet_count",
            "http_response_code",
            "http_uri_length",
            "dns_query_length",
            "dns_answer_count",
            "dns_ttl",
            "cert_validity",
        ]
        features = ensure_numeric(features, [c for c in numeric_cols if c in features.columns])

    return metadata, features


def aggregate_features_from_directory(
    raw_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureEncoder]:
    """
    Scan a directory of Zeek logs and build:
      - combined metadata
      - combined numeric feature matrix (fully encoded)
      - fitted FeatureEncoder instance

    Expected file name hints:
      - 'http'  in filename -> HTTP log
      - 'dns'   in filename -> DNS log
      - 'ssl' or 'https' in filename -> HTTPS log
    """
    all_metadata: List[pd.DataFrame] = []
    all_features: List[pd.DataFrame] = []

    # Walk the directory tree so we can handle datasets inside
    # nested folders (e.g., CIC-IDS, CTU-13).
    for root, _dirs, files in os.walk(raw_dir):
        for entry in sorted(files):
            path = os.path.join(root, entry)

            lower = entry.lower()
            _, ext = os.path.splitext(lower)

            # Accept generic CSV / LOG / TSV files even if they don't
            # contain protocol keywords in the filename.
            if "http" in lower and "https" not in lower:
                protocol = "HTTP"
            elif "dns" in lower:
                protocol = "DNS"
            elif "https" in lower or "ssl" in lower:
                protocol = "HTTPS"
            elif ext in {".csv", ".log", ".tsv"}:
                protocol = "GENERIC"
            else:
                # Unsupported or non-data file
                continue

            metadata, feats = extract_protocol_features(path, protocol)
            all_metadata.append(metadata)
            all_features.append(feats)

    if not all_features:
        return (
            pd.DataFrame(columns=["timestamp", "source_ip", "destination_ip", "protocol"]),
            pd.DataFrame(),
            FeatureEncoder(),
        )

    metadata_df = pd.concat(all_metadata, axis=0).reset_index(drop=True)
    features_df = pd.concat(all_features, axis=0).reset_index(drop=True)

    # Categorical columns to encode
    categorical_cols = [
        "connection_state",
        "http_method",
        "ssl_version",
        "ssl_cipher",
    ]

    encoder = FeatureEncoder()
    encoder.fit(features_df, categorical_cols)
    features_df = encoder.transform(features_df, categorical_cols)

    # Replace any remaining NaNs with zeros for model consumption
    features_df = features_df.fillna(0)

    return metadata_df, features_df, encoder

