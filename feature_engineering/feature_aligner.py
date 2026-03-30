"""
feature_engineering/feature_aligner.py
=====================================
Align heterogeneous feature matrices across datasets.

The aligner fits on the union of columns across multiple DataFrames and
then transforms each DataFrame to have the same columns in the same
order, filling missing columns with zeros.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import ClassVar, List

import pandas as pd


@dataclass
class FeatureAligner:
    """
    Align feature DataFrames to a shared column space.
    """

    all_columns: List[str] = field(default_factory=list)
    n_features: int = 0

    def fit(self, feature_dfs: List[pd.DataFrame]) -> None:
        """
        Fit the aligner on a list of feature matrices by storing the union
        of all column names (sorted for determinism).
        """
        cols = set()
        for df in feature_dfs:
            # Only keep numeric feature columns to avoid bloating the union.
            numeric_cols = df.select_dtypes(include=["number"]).columns
            cols.update(map(str, numeric_cols))
        self.all_columns = sorted(cols)
        self.n_features = len(self.all_columns)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a feature matrix:
          - add missing columns (filled with 0)
          - drop extra columns not in the fitted column set
          - reorder columns to match the fitted order
        """
        if not self.all_columns:
            raise ValueError("FeatureAligner is not fitted (all_columns is empty).")

        # Use reindex to avoid repeated column insertion (which fragments
        # DataFrames and can cause huge temporary allocations).
        out = df.reindex(columns=self.all_columns, fill_value=0)

        # Keep output numeric and memory-friendly for model training.
        # Adapters should already produce numeric features; this is a safety net.
        out = out.fillna(0)
        try:
            return out.astype("float32", copy=False)
        except (ValueError, TypeError):
            out = out.apply(pd.to_numeric, errors="coerce").fillna(0)
            return out.astype("float32", copy=False)

    def fit_transform(self, feature_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Fit on the union of columns and return aligned DataFrames.
        """
        self.fit(feature_dfs)
        return [self.transform(df) for df in feature_dfs]

    def save(self, path: str) -> None:
        """
        Persist aligner column order to disk using pickle.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"all_columns": self.all_columns}, f)

    @classmethod
    def load(cls, path: str) -> "FeatureAligner":
        """
        Load an aligner from disk.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        aligner = cls(all_columns=list(payload.get("all_columns", [])))
        aligner.n_features = len(aligner.all_columns)
        return aligner

