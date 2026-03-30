"""
train_multi.py
==============
Standalone entrypoint for multi-dataset training.

Examples:
  python train_multi.py --strategy combined --verbose
  python train_multi.py --strategy cross_eval --sample-frac 0.3
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from training.multi_dataset_trainer import train_on_all_datasets
from utils.dataset_registry import discover_datasets


def main() -> int:
    """
    CLI entrypoint. Returns process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw_zeek_logs", help="Raw datasets directory.")
    parser.add_argument("--output-dir", default="data/models", help="Directory to save trained models.")
    parser.add_argument(
        "--strategy",
        default="combined",
        choices=["combined", "per_dataset", "cross_eval"],
        help="Training strategy.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of each dataset to use (0 < frac <= 1).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    try:
        discovered = discover_datasets(args.raw_dir)
        print("[+] Discovered datasets:")
        for d in discovered:
            print(f"    {d['name']}  — type={d['type']}")

        model, aligner, report = train_on_all_datasets(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            strategy=args.strategy,
            sample_frac=args.sample_frac,
            verbose=args.verbose,
        )

        print("[+] Training complete.")
        if args.strategy == "combined":
            m = report.get("combined", {}).get("test_metrics", {})
            if m:
                print("[+] Results (combined test split):")
                print(f"    F1      : {m.get('f1')}")
                print(f"    ROC-AUC : {m.get('roc_auc')}")
                print(f"    Recall  : {m.get('recall')}")
        elif args.strategy == "cross_eval" and isinstance(model, pd.DataFrame):
            print("[+] Cross-eval F1 matrix:")
            print(model.to_string())

        return 0
    except Exception as e:
        print(f"[!] ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

