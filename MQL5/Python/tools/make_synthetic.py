"""Generate synthetic bars in MintyAIExport.mq5's exact output format.

Used to smoke-test the training pipeline without a terminal attached, and
to sanity-check that the model can learn a signal that is known to be
there. The series carries a deliberate autocorrelation in returns, so a
working pipeline should find a positive out-of-sample edge on it. If it
does not, the bug is in the code, not in the market.

    python tools/make_synthetic.py --bars 60000 --out ../Files/MintyAI/SYNTH_H1

Real trading decisions must never be based on this data.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=int, default=60000)
    p.add_argument("--out", required=True, help="output prefix (no _bars.csv)")
    p.add_argument("--phi", type=float, default=0.15,
                   help="return autocorrelation - the learnable signal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=int, default=1_262_304_000)  # 2010-01-01 UTC
    p.add_argument("--symbol", default="SYNTH")
    p.add_argument("--timeframe", default="H1")
    p.add_argument("--digits", type=int, default=5)
    p.add_argument("--spread-points", type=int, default=8)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    n = args.bars
    step = 3600

    ts = args.start + np.arange(n, dtype=np.int64) * step
    hour = (ts % 86400) / 3600.0

    # volatility clusters (GARCH-ish) plus an intraday session profile
    sigma = np.empty(n)
    sigma[0] = 3e-4
    shock = rng.standard_normal(n)
    for i in range(1, n):
        sigma[i] = np.sqrt(
            1e-9 + 0.90 * sigma[i - 1] ** 2 + 0.06 * (sigma[i - 1] * shock[i - 1]) ** 2
        )
    session = 1.0 + 0.5 * np.sin(2.0 * np.pi * (hour - 6.0) / 24.0)
    sigma = sigma * session

    # AR(1) returns: this is the edge the model is supposed to discover
    r = np.zeros(n)
    eps = rng.standard_normal(n) * sigma
    for i in range(1, n):
        r[i] = args.phi * r[i - 1] + eps[i]

    close = 1.10 * np.exp(np.cumsum(r))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    # intrabar extremes consistent with the bar's own move
    wick = np.abs(rng.standard_normal(n)) * sigma * close * 0.8
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * sigma * close * 0.8
    low = np.minimum(low, np.minimum(open_, close))
    high = np.maximum(high, np.maximum(open_, close))

    vol = (rng.gamma(3.0, 300.0, n) * session).astype(np.int64) + 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bars_path = args.out + "_bars.csv"
    with open(bars_path, "w", encoding="ascii", newline="") as fh:
        fh.write("time,open,high,low,close,tick_volume,real_volume,spread\n")
        for i in range(n):
            fh.write(
                f"{ts[i]},{open_[i]:.8f},{high[i]:.8f},{low[i]:.8f},"
                f"{close[i]:.8f},{vol[i]},0,{args.spread_points}\n"
            )

    point = 10.0 ** (-args.digits)
    meta_path = args.out + "_meta.csv"
    with open(meta_path, "w", encoding="ascii", newline="") as fh:
        fh.write("key,value\n")
        fh.write(f"symbol,{args.symbol}\n")
        fh.write(f"timeframe,{args.timeframe}\n")
        fh.write(f"digits,{args.digits}\n")
        fh.write(f"point,{point:.10f}\n")
        fh.write(f"tick_size,{point:.10f}\n")
        fh.write("tick_value,1.0000000000\n")
        fh.write("volume_min,0.0100\n")
        fh.write("volume_step,0.0100\n")
        fh.write(f"spread_current,{args.spread_points}\n")

    print(f"wrote {bars_path} ({n} bars) and {meta_path}")
    print(f"embedded return autocorrelation phi={args.phi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
