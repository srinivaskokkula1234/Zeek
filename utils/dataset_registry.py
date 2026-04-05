"""
utils/dataset_registry.py
========================
Discovery utilities for datasets under data/raw_zeek_logs/.

The discovery function returns a list of dataset descriptors, each with:
  - name: human readable name
  - type: one of {"cicids2018","nslkdd","unswnb15","cicids2017","generic"}
  - paths: dict of relevant file paths
  - adapter: callable adapter function

Heuristics are designed to be fast (lightweight header reads).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import pandas as pd

from utils.dataset_adapters import adapt_cicids2018, adapt_ctu13, adapt_nslkdd, adapt_unswnb15


def _peek_csv_columns(path: str, nrows: int = 5) -> List[str]:
    """Read a few rows and return stripped column names, or [] on failure."""
    try:
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
        return [str(c).strip() for c in df.columns]
    except Exception:
        return []


def _find_first_csv(directory: str) -> Optional[str]:
    for root, _dirs, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith(".csv"):
                return os.path.join(root, fn)
    return None


def discover_datasets(raw_dir: str) -> List[Dict]:
    """
    Scan raw_dir recursively and return dataset descriptors.

    Parameters
    ----------
    raw_dir : str
        Root directory containing downloaded datasets and/or Zeek logs.

    Returns
    -------
    List[Dict]
        List of dataset dicts. Each dict contains keys:
        name, type, paths, adapter.
    """
    if not os.path.isdir(raw_dir):
        return []

    datasets: List[Dict] = []

    # Consider each immediate child directory as a dataset candidate.
    # Also allow raw_dir itself if files are directly inside.
    candidates = []
    entries = [os.path.join(raw_dir, e) for e in os.listdir(raw_dir)]
    for p in entries:
        if os.path.isdir(p):
            candidates.append(p)
    candidates.append(raw_dir)

    seen = set()
    for cand in candidates:
        cand = os.path.abspath(cand)
        if cand in seen:
            continue
        seen.add(cand)

        # ------------------------------------------------------------
        # Heuristic: NSL-KDD
        # ------------------------------------------------------------
        train_path = None
        test_path = None
        for root, _dirs, files in os.walk(cand):
            for fn in files:
                lower = fn.lower()
                if "kddtrain" in lower:
                    train_path = os.path.join(root, fn)
                if "kddtest" in lower:
                    test_path = os.path.join(root, fn)
        if train_path and test_path:
            datasets.append(
                {
                    "name": "NSL-KDD",
                    "type": "nslkdd",
                    "paths": {"train": train_path, "test": test_path},
                    "adapter": adapt_nslkdd,
                }
            )
            continue

        # ------------------------------------------------------------
        # Heuristic: UNSW-NB15
        # ------------------------------------------------------------
        unsw_train = None
        unsw_test = None
        for root, _dirs, files in os.walk(cand):
            for fn in files:
                lower = fn.lower()
                if "unsw" in lower and "training" in lower and lower.endswith(".csv"):
                    unsw_train = os.path.join(root, fn)
                if "unsw" in lower and "testing" in lower and lower.endswith(".csv"):
                    unsw_test = os.path.join(root, fn)
        # If not named clearly, look for attack_cat column in a CSV
        if not (unsw_train and unsw_test):
            first_csv = _find_first_csv(cand)
            if first_csv:
                cols = _peek_csv_columns(first_csv)
                if "attack_cat" in cols:
                    # Best effort: find two CSVs in the directory tree
                    csvs = []
                    for root, _dirs, files in os.walk(cand):
                        for fn in files:
                            if fn.lower().endswith(".csv"):
                                csvs.append(os.path.join(root, fn))
                    if len(csvs) >= 2:
                        unsw_train, unsw_test = sorted(csvs)[:2]
        if unsw_train and unsw_test:
            datasets.append(
                {
                    "name": "UNSW-NB15",
                    "type": "unswnb15",
                    "paths": {"train": unsw_train, "test": unsw_test},
                    "adapter": adapt_unswnb15,
                }
            )
            continue

        # ------------------------------------------------------------
        # Heuristic: CTU-13
        # ------------------------------------------------------------
        # Repo layout may contain CTU-13 as either CSV or parquet
        # (e.g. *.binetflow.parquet). We register based on the directory
        # name containing "ctu" and the presence of at least one data file.
        if "ctu" in os.path.basename(cand).lower() or "ctu" in cand.lower():
            has_ctu_file = False
            for root, _dirs, files in os.walk(cand):
                for fn in files:
                    if fn.lower().endswith((".csv", ".parquet")):
                        has_ctu_file = True
                        break
                if has_ctu_file:
                    break

            if has_ctu_file:
                datasets.append(
                    {
                        "name": "CTU-13",
                        "type": "ctu13",
                        "paths": {"directory": cand},
                        "adapter": adapt_ctu13,
                    }
                )
                continue

        # ------------------------------------------------------------
        # Heuristic: CIC-IDS 2018
        # ------------------------------------------------------------
        first_csv = _find_first_csv(cand)
        if first_csv:
            cols = _peek_csv_columns(first_csv)
            cols_set = set(cols)
            # CIC-IDS 2018: Timestamp + Dst Port + Label
            if {"Timestamp", "Dst Port", "Label"}.issubset(cols_set):
                datasets.append(
                    {
                        "name": "CIC-IDS 2018",
                        "type": "cicids2018",
                        "paths": {"directory": cand},
                        "adapter": adapt_cicids2018,
                    }
                )
                continue

            # --------------------------------------------------------
            # Heuristic: CIC-IDS 2017
            # --------------------------------------------------------
            if {"Label", "Flow Duration", "Total Fwd Packets"}.issubset(cols_set):
                # CIC-IDS 2017 already works via existing GENERIC path, but
                # we still register it explicitly for multi-dataset training.
                datasets.append(
                    {
                        "name": "CIC-IDS 2017",
                        "type": "cicids2017",
                        "paths": {"directory": cand},
                        "adapter": None,  # handled by existing pipeline in trainer
                    }
                )
                continue

        # ------------------------------------------------------------
        # Fallback: generic
        # ------------------------------------------------------------
        first_csv = _find_first_csv(cand)
        if first_csv:
            cols = _peek_csv_columns(first_csv)
            if "Label" not in set(cols) and "label" not in set(cols):
                # Avoid registering large unlabeled directories (e.g. raw DNS logs).
                continue

        datasets.append(
            {
                "name": os.path.basename(cand) or "generic",
                "type": "generic",
                "paths": {"directory": cand},
                "adapter": None,
            }
        )

    return datasets

