"""Train a network for every symbol MintyAIExportAll.mq5 downloaded.

    python -m mintyai.train_all

It reads MQL5\\Files\\MintyAI\\ and trains whatever it finds, one symbol at
a time - there is no symbol list to keep in sync. Each symbol gets its own
ONNX graph and its own gate, hours and stakes, and the whole lot is written
into one MintyAIModelConfig.mqh that the EA embeds on the next compile.

A shared network across all symbols was tried and measured worse. In
principle pooling should help - the features are scale-free, so instruments
are comparable, and there is far more data. In practice the shared function
came out a compromise that priced none of them sharply: its probabilities
collapsed toward the base rate (median top class 0.387 against a 0.333
floor), the gate search could find no exposure worth taking, and the
backtest degraded. A specialist per instrument beat it.

The three risk presets are one model evaluated at three stakes; the cap
never enters the loss, so it cannot change the weights.

Options worth knowing:
    --symbols EURUSD,GBPUSD   train only these
    --skip EURJPY             exclude some
    --keep-going              carry on if one symbol fails
    --folds 0                 skip walk-forward (much faster, no honest number)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback

from . import train as T


def discover(files_dir: str, timeframe: str | None) -> list:
    """Every *_bars.csv the export scripts left behind."""
    found = []
    for path in sorted(glob.glob(os.path.join(files_dir, "*_bars.csv"))):
        stem = os.path.basename(path)[: -len("_bars.csv")]
        if "_" not in stem:
            continue
        symbol, _, tf = stem.rpartition("_")
        if not symbol:
            continue
        if timeframe and tf.upper() != timeframe.upper():
            continue
        found.append({"symbol": symbol, "timeframe": tf, "bars": path})
    return found


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train every exported symbol, one after another")
    p.add_argument("--timeframe", default="H1",
                   help="only train exports of this timeframe (default H1)")
    p.add_argument("--symbols", default=None,
                   help="comma-separated subset to train")
    p.add_argument("--skip", default=None,
                   help="comma-separated symbols to leave out")
    p.add_argument("--keep-going", action="store_true",
                   help="carry on when a symbol fails instead of stopping")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be trained and exit")
    p.add_argument("--no-merge", action="store_true",
                   help="write a config holding ONLY the symbols trained in this "
                        "run, dropping every other symbol out of the EA")
    # forwarded to the single-symbol trainer
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--barrier-atr", type=float, default=None)
    # model size / training knobs, forwarded verbatim to the single-symbol trainer
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--dilations", default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-trades", type=int, default=None)
    p.add_argument("--spread-points", type=float, default=None)
    p.add_argument("--slippage-points", type=float, default=None)
    p.add_argument("--risk-pcts", default=None)
    p.add_argument("--kelly-scale", type=float, default=None)
    p.add_argument("--reverse-split", action="store_true", default=None)
    p.add_argument("--cpu", action="store_true", default=None,
                   help="train on the CPU even if a CUDA device is present")
    p.add_argument("--search-attempts", type=int, default=None)
    p.add_argument("--target-tstat", type=float, default=None)
    return p


def split_list(spec) -> set:
    if not spec:
        return set()
    return {s.strip().upper() for s in str(spec).split(",") if s.strip()}


def main(argv=None) -> int:
    a = build_argparser().parse_args(argv)

    jobs = discover(T.FILES_DIR, a.timeframe)
    only, skip = split_list(a.symbols), split_list(a.skip)
    if only:
        jobs = [j for j in jobs if j["symbol"].upper() in only]
    if skip:
        jobs = [j for j in jobs if j["symbol"].upper() not in skip]

    if not jobs:
        print(f"[mintyai] no *_bars.csv found in {T.FILES_DIR}")
        print("[mintyai] run Scripts\\MintyAI\\MintyAIExportAll.mq5 in MetaTrader first.")
        return 1

    print(f"[mintyai] found {len(jobs)} symbol(s) in {T.FILES_DIR}:")
    for j in jobs:
        size_mb = os.path.getsize(j["bars"]) / 1e6
        print(f"           {j['symbol']:<12} {j['timeframe']:<4} {size_mb:6.1f} MB")
    if a.dry_run:
        return 0

    # one argument set, reused per symbol so every model is comparable
    forwarded = []
    for flag in ("folds", "epochs", "window", "horizon", "barrier_atr",
                 "spread_points", "slippage_points", "risk_pcts", "kelly_scale",
                 "search_attempts", "target_tstat",
                 "channels", "dilations", "dropout", "batch_size", "lr",
                 "seed", "min_trades"):
        val = getattr(a, flag)
        if val is not None:
            forwarded += ["--" + flag.replace("_", "-"), str(val)]
    if a.reverse_split:
        forwarded += ["--reverse-split"]
    if a.cpu:
        forwarded += ["--cpu"]

    base_args = T.build_argparser().parse_args(["--bars", jobs[0]["bars"]] + forwarded)
    risk_pcts = T.parse_risk_pcts(base_args.risk_pcts)
    print("[mintyai] risk presets: " + ", ".join(
        f"{n} {p:g}%" for n, p in zip(T.RISK_NAMES, risk_pcts)))

    device = T.pick_device(bool(a.cpu))
    entries, failed = [], []
    t_start = time.time()

    for i, job in enumerate(jobs, 1):
        print("\n" + "=" * 72)
        print(f"[mintyai] [{i}/{len(jobs)}] {job['symbol']} {job['timeframe']}")
        print("=" * 72)

        args = T.build_argparser().parse_args(["--bars", job["bars"]] + forwarded)
        try:
            if args.search_attempts > 1:
                e = T.search_until_signal(args, device, risk_pcts,
                                          args.search_attempts, args.target_tstat)
            else:
                e = T.run_pipeline(args, device, risk_pcts)
            entries.append(T.strip_internals(e))
        except (SystemExit, RuntimeError, ValueError, OSError) as exc:
            print(f"[mintyai] {job['symbol']} FAILED: {exc}")
            if not a.keep_going:
                traceback.print_exc()
                print("[mintyai] stopping. Pass --keep-going to train the rest anyway.")
                return 1
            failed.append(job["symbol"])

    if not entries:
        print("[mintyai] nothing trained successfully; config left untouched.")
        return 1

    T.write_config(T.shared_config(base_args, risk_pcts, entries[0]["timeframe"],
                                   entries[0].get("utc_offset_h", 0)), entries,
                   merge=not a.no_merge)

    mins = (time.time() - t_start) / 60.0
    print("\n" + "=" * 72)
    print(f"[mintyai] trained {len(entries)} symbol(s) in {mins:.1f} min")
    if failed:
        print(f"[mintyai] failed: {', '.join(failed)}")

    print("\n[mintyai] walk-forward summary (the honest number):")
    print(f"           {'symbol':<12} {'trades':>8} {'total R':>10} {'t':>7}   verdict")
    for e in entries:
        verdict = "tradable" if e["wf_tstat"] > 2.0 else "noise - do not trade"
        print(f"           {e['symbol']:<12} {e['wf_trades']:>8} "
              f"{e['wf_total_r']:>10.1f} {e['wf_tstat']:>7.2f}   {verdict}")

    good = [e for e in entries if e["wf_tstat"] > 2.0]
    if not good:
        print("\n[mintyai] NO symbol showed a statistically meaningful edge "
              "out-of-sample. The model is embedded and selectable, but trading "
              "it is an experiment, not a strategy.")
    else:
        print(f"\n[mintyai] {len(good)} of {len(entries)} symbols cleared t > 2: "
              + ", ".join(e["symbol"] for e in good))

    print("\n[mintyai] done. Recompile MintyAI.mq5 to embed the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
