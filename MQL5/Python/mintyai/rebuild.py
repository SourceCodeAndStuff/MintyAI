"""Put every model already on disk back into the EA - no retraining.

    python -m mintyai.rebuild
    python -m mintyai.rebuild --skip BTCUSD,ETHUSD,SOLUSD,XRPUSD,ADAUSD,DOGEUSD
    python -m mintyai.rebuild --symbols EURUSD,GBPUSD,XAUUSD

Every training run writes <base>.onnx and <base>_report.json together, so
the pair on disk is always consistent. This reads every report it can find,
rebuilds the config table around the graphs those reports describe, and
writes MintyAIModelConfig.mqh. Nothing is retrained and no weights are
loaded - the graphs are handed to the compiler exactly as they are.

Use it to recover a table that a single-symbol training run overwrote, or
to change WHICH of the trained symbols ship without touching any of them.

A symbol is only included if it still describes the same bet as the rest of
the build - same feature signature, timeframe, window, horizon and barrier.
Those constants are compiled into the EA once, for every model at once, so
a model fitted to different ones would price a trade the EA is not placing.
Anything excluded is named, with the reason.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

from . import train as T


def dominant_definition() -> tuple[dict, list]:
    """The trade definition most of the trained symbols agree on.

    Picking the majority rather than the newest means one experimental run
    at a different barrier cannot drag the whole table onto its settings and
    evict everything else.
    """
    seen = []
    for path in sorted(glob.glob(os.path.join(T.FILES_DIR, "mintyai_*_report.json"))):
        if "_pooled_" in os.path.basename(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        a = r.get("args") or {}
        if not os.path.exists(os.path.join(
                T.FILES_DIR, os.path.basename(path)[: -len("_report.json")] + ".onnx")):
            continue
        seen.append({
            "timeframe": r.get("timeframe", "H1"),
            "window": a.get("window"),
            "horizon": a.get("horizon"),
            "barrier_atr": a.get("barrier_atr"),
            "slippage_points": a.get("slippage_points", 2.0),
            "kelly_scale": a.get("kelly_scale", 0.5),
            "risk_pcts": a.get("risk_pcts"),
            "bars": a.get("bars", ""),
        })

    if not seen:
        return {}, []

    key = Counter((s["timeframe"], s["window"], s["horizon"], s["barrier_atr"])
                  for s in seen).most_common(1)[0][0]
    winner = next(s for s in seen
                  if (s["timeframe"], s["window"], s["horizon"], s["barrier_atr"]) == key)
    return winner, seen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Rebuild MintyAIModelConfig.mqh from the models already on "
                    "disk, without retraining anything")
    p.add_argument("--symbols", default=None,
                   help="comma-separated whitelist; default is everything found")
    p.add_argument("--skip", default=None,
                   help="comma-separated symbols to leave out of the build")
    a = p.parse_args(argv)

    winner, seen = dominant_definition()
    if not winner:
        print(f"[mintyai] no trained models found in {T.FILES_DIR}")
        return 1

    caps = winner.get("risk_pcts")
    if isinstance(caps, str):
        caps = [float(x) for x in caps.split(",")]
    if not caps:
        caps = T.load_deploy_defaults().get("risk_pcts", T.DEFAULT_RISK_PCTS)

    meta = T.read_export_meta(winner.get("bars", ""))
    ns = argparse.Namespace(window=winner["window"], horizon=winner["horizon"],
                            barrier_atr=winner["barrier_atr"],
                            slippage_points=winner["slippage_points"])
    shared = T.shared_config(ns, caps, winner["timeframe"],
                             int(float(meta.get("server_utc_offset_h", 0) or 0)))

    print(f"[mintyai] building on the definition {len(seen)} report(s) were found "
          f"under: {winner['timeframe']}, window {winner['window']}, horizon "
          f"{winner['horizon']}, barrier {winner['barrier_atr']} ATR")

    entries = T.carry_forward(shared, [])

    want = {s.strip().upper() for s in a.symbols.split(",")} if a.symbols else None
    drop = {s.strip().upper() for s in a.skip.split(",")} if a.skip else set()
    if want is not None:
        missing = want - {e["symbol"].upper() for e in entries}
        if missing:
            print("[mintyai] asked for, but not trained (or not compatible): "
                  + ", ".join(sorted(missing)))
        entries = [e for e in entries if e["symbol"].upper() in want]
    if drop:
        entries = [e for e in entries if e["symbol"].upper() not in drop]

    if not entries:
        print("[mintyai] nothing left to write; config left untouched.")
        return 1

    T.write_config(shared, entries, merge=False)
    print("[mintyai] " + ", ".join(e["symbol"] for e in entries))
    print(f"[mintyai] {len(entries)} model(s) x ~1.8 MB will be embedded. "
          "Recompile MintyAI.mq5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
