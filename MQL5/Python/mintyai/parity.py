"""Prove that features.py and MintyAIFeatures.mqh compute the same thing.

Run this after any change to either feature engine, and before trusting
any backtest:

    python -m mintyai.parity --bars ../../Files/MintyAI/EURUSD_H1_bars.csv

A model trained on Python features but fed MQL5 features at runtime will
produce confident, meaningless predictions. This is the check that
catches that, and it is not optional.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from . import data as D
from . import features as F


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MQL5 <-> Python feature parity check")
    p.add_argument("--bars", required=True, help="path to *_bars.csv")
    p.add_argument("--dump", default=None, help="path to *_features_mql5.csv")
    p.add_argument("--tol", type=float, default=2e-4,
                   help="max allowed absolute difference per feature")
    args = p.parse_args(argv)

    dump_path = args.dump or args.bars.replace("_bars.csv", "_features_mql5.csv")
    if not os.path.exists(dump_path):
        print(f"[parity] no MQL5 dump at {dump_path}")
        print("[parity] run MintyAIExport.mq5 with InpDumpFeatures=true first")
        return 2

    bars = D.load_bars(args.bars)
    py = F.build_features(
        bars.open, bars.high, bars.low, bars.close, bars.tick_volume, bars.time
    )

    sig_meta = bars.meta.get("feature_signature")
    if sig_meta is not None:
        ok = int(sig_meta) == F.feature_signature()
        print(f"[parity] signature MQL5={sig_meta} Python={F.feature_signature()} "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            print("[parity] the two feature engines do not even agree on their "
                  "feature list; fix that before comparing values")
            return 1

    mql = pd.read_csv(dump_path)
    idx = {int(t): i for i, t in enumerate(bars.time)}
    rows = [idx.get(int(t), -1) for t in mql["time"]]
    keep = np.array([r >= 0 for r in rows])
    if not keep.any():
        print("[parity] no overlapping timestamps between dump and bars")
        return 1

    rows = np.array(rows)[keep]
    mql_vals = mql.loc[keep, F.FEATURE_NAMES].to_numpy(dtype=np.float64)
    py_vals = py[rows].astype(np.float64)

    # the first bars of the dump depend on history the exporter did not
    # include, so compare only warmed-up rows
    warm = rows >= F.WARMUP
    if warm.sum() == 0:
        print("[parity] dump contains no warmed-up rows; export more history")
        return 1
    mql_vals, py_vals = mql_vals[warm], py_vals[warm]
    print(f"[parity] comparing {len(mql_vals)} warmed-up rows")

    diff = np.abs(mql_vals - py_vals)
    worst_per_feat = diff.max(axis=0)

    failed = []
    for i, name in enumerate(F.FEATURE_NAMES):
        status = "ok" if worst_per_feat[i] <= args.tol else "FAIL"
        if status == "FAIL":
            failed.append(name)
        print(f"  {name:<16} max|diff| {worst_per_feat[i]:.3e}  {status}")

    if failed:
        print(f"\n[parity] FAILED for: {', '.join(failed)}")
        print("[parity] the MQL5 and Python feature engines have diverged. "
              "Do not train or trade until this passes.")
        return 1

    print(f"\n[parity] PASS - worst difference {worst_per_feat.max():.3e} "
          f"(tolerance {args.tol:.1e})")
    print("[parity] note: the exporter writes 8 decimals, so differences "
          "around 1e-8 are formatting, not logic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
