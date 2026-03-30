"""
feature_engineering/derive_features.py
=======================================
Derived / engineered features for network anomaly detection.

All features are computed from existing columns found in CIC-IDS style
datasets (and equivalent Zeek-enriched DataFrames).  Each feature is
added only when the required source column(s) are present; if they are
absent the feature is silently skipped so the function is safe to call
on any DataFrame.

Usage
-----
    from feature_engineering.derive_features import add_derived_features
    enriched_df = add_derived_features(combined_df)
"""

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
    """
    Return the first existing column in ``candidates`` as a numeric Series,
    or ``None`` if none of the candidates exist in ``df``.

    Only a proper DataFrame column is accepted (no fallback scalars).
    """
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return None


def _safe_div(
    numerator: pd.Series, denominator: pd.Series, eps: float = 1e-9
) -> pd.Series:
    """Element-wise safe division: ``numerator / (denominator + eps)``."""
    return numerator / (denominator + eps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add high-signal derived features to a network-flow DataFrame.

    The function is completely idempotent regarding missing columns: it
    silently skips any feature whose source columns are absent so it can
    safely be called on any DataFrame (Zeek, CIC-IDS, UNSW-NB15, etc.).

    Features added
    --------------
    Packet- / byte-rate features:
      pkt_rate          – packets per second
      fwd_bwd_ratio     – forward-to-backward packet ratio
      bytes_per_pkt     – average payload per packet
      bytes_per_sec     – throughput in bytes/second

    TCP flag ratios (relative to total packet count):
      syn_ratio, fin_ratio, rst_ratio, psh_ratio, ack_ratio

    Port-category indicators (binary integers):
      is_well_known_port – destination port < 1024
      is_http_https      – port in {80, 443, 8080, 8443}
      is_dns_port        – port == 53
      is_high_port       – port > 49151 (ephemeral)

    Inter-arrival / length ratios:
      iat_mean_norm      – IAT mean normalised by IAT std
      fwd_pkt_len_ratio  – max / mean forward packet length

    Parameters
    ----------
    df : pd.DataFrame
        Combined metadata + feature DataFrame.

    Returns
    -------
    pd.DataFrame
        New DataFrame with derived columns appended.  ``inf`` / ``-inf``
        values are replaced with 0; ``NaN`` values are filled with 0.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Packet / byte-rate features
    # ------------------------------------------------------------------
    fwd_pkts   = _col(df, "Total Fwd Packets", "fwd_pkts", "orig_pkts")
    bwd_pkts   = _col(df, "Total Backward Packets", "bwd_pkts", "resp_pkts")
    duration   = _col(df, "Flow Duration", "duration")
    total_bytes = _col(df,
                       "Total Length of Fwd Packets",
                       "orig_ip_bytes", "orig_bytes", "bytes_sent")

    # Derive total packets from fwd + bwd when both are available
    if fwd_pkts is not None and bwd_pkts is not None:
        total_pkts: Optional[pd.Series] = fwd_pkts + bwd_pkts
    elif fwd_pkts is not None:
        total_pkts = fwd_pkts
    elif bwd_pkts is not None:
        total_pkts = bwd_pkts
    else:
        total_pkts = None

    if fwd_pkts is not None and bwd_pkts is not None and duration is not None:
        df["pkt_rate"] = _safe_div(fwd_pkts + bwd_pkts, duration)

    if fwd_pkts is not None and bwd_pkts is not None:
        df["fwd_bwd_ratio"] = _safe_div(fwd_pkts, bwd_pkts)

    if total_bytes is not None and total_pkts is not None:
        df["bytes_per_pkt"] = _safe_div(total_bytes, total_pkts)

    if total_bytes is not None and duration is not None:
        df["bytes_per_sec"] = _safe_div(total_bytes, duration)

    # ------------------------------------------------------------------
    # 2. TCP flag ratios
    # ------------------------------------------------------------------
    _FLAG_MAP = {
        "syn_ratio": ("SYN Flag Count",),
        "fin_ratio": ("FIN Flag Count",),
        "rst_ratio": ("RST Flag Count",),
        "psh_ratio": ("PSH Flag Count",),
        "ack_ratio": ("ACK Flag Count",),
    }

    if total_pkts is not None:
        for feat_name, col_candidates in _FLAG_MAP.items():
            flag_col = _col(df, *col_candidates)
            if flag_col is not None:
                df[feat_name] = _safe_div(flag_col, total_pkts)

    # ------------------------------------------------------------------
    # 3. Port-category indicators
    # ------------------------------------------------------------------
    port_col = _col(df, "Destination Port", "id.resp_p", "dst_port", "dest_port")
    if port_col is not None:
        df["is_well_known_port"] = (port_col < 1024).astype(int)
        df["is_http_https"]      = port_col.isin({80, 443, 8080, 8443}).astype(int)
        df["is_dns_port"]        = (port_col == 53).astype(int)
        df["is_high_port"]       = (port_col > 49151).astype(int)

    # ------------------------------------------------------------------
    # 4. IAT / length ratios
    # ------------------------------------------------------------------
    iat_mean = _col(df, "Flow IAT Mean",  "flow_iat_mean")
    iat_std  = _col(df, "Flow IAT Std",   "flow_iat_std")
    if iat_mean is not None and iat_std is not None:
        df["iat_mean_norm"] = _safe_div(iat_mean, iat_std)

    fwd_len_max  = _col(df, "Fwd Packet Length Max",  "fwd_pkt_len_max")
    fwd_len_mean = _col(df, "Fwd Packet Length Mean", "fwd_pkt_len_mean")
    if fwd_len_max is not None and fwd_len_mean is not None:
        df["fwd_pkt_len_ratio"] = _safe_div(fwd_len_max, fwd_len_mean)

    # ------------------------------------------------------------------
    # 5. Sanitise: replace ±inf and NaN with 0
    # ------------------------------------------------------------------
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df
