"""Write a placeholder Include/MintyAI/MintyAIModelConfig.mqh.

The EA includes that header, so it cannot compile before a model exists.
This generates a syntactically valid stand-in with the correct feature
signature, letting you compile and inspect the EA immediately. It points
at a model file that does not exist yet, so the EA will fail cleanly at
OnInit until you actually train:

    python -m mintyai.placeholder --symbol EURUSD --timeframe H1
"""

from __future__ import annotations

import argparse
import os

from . import features as F
from .train import INCLUDE_DIR, write_mqh


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="generate a placeholder model config")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", default="H1")
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--barrier-atr", type=float, default=3.0)
    args = p.parse_args(argv)

    base = f"mintyai_{args.symbol}_{args.timeframe}"
    path = os.path.join(INCLUDE_DIR, "MintyAIModelConfig.mqh")

    write_mqh(
        path,
        trained_at="NOT TRAINED - placeholder",
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_bars=0,
        date_from="-",
        date_to="-",
        feature_version=F.FEATURE_VERSION,
        feature_count=F.FEATURE_COUNT,
        signature=F.feature_signature(),
        window=args.window,
        model_file="MintyAI\\\\" + base + ".onnx",
        barrier_atr=f"{args.barrier_atr}",
        horizon=args.horizon,
        min_prob="0.42",
        edge_mult="0.5",
        wf_trades=0,
        wf_total_r="0.00",
        wf_tstat="0.000",
    )
    print(f"wrote placeholder {path}")
    print("The EA will now compile, but will refuse to start until "
          f"MQL5\\Files\\MintyAI\\{base}.onnx exists. Train to produce it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
