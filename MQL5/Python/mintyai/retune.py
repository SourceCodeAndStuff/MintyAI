"""Recompute the config from models already trained - no retraining.

    python -m mintyai.retune

Changing a risk preset normally means running train_all again, which with
--search-attempts picks a different seed and hands you a DIFFERENT model.
When the model you have is the one you want, that is exactly the wrong
trade. This reloads the saved .pt files, recomputes everything that is
downstream of the weights - the entry gate, the tradable hours, and the
three risk rows at whatever mintyai_deploy.json now says - and rewrites
MintyAIModelConfig.mqh around the same graphs.

Everything here is deterministic given the weights and the data, so the
gate and hours come out identical to the training run that produced them.
Only the risk rows move, because only the caps changed.

Walk-forward figures are carried over from each symbol's report, since
they describe folds this does not re-run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

from . import data as D
from . import features as F
from . import train as T
from .model import MintyAINet


def retune_symbol(pt_path: str, risk_pcts: list, force_cpu: bool = False) -> dict | None:
    base = os.path.basename(pt_path)[: -len(".pt")]
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])

    if not os.path.exists(a.bars):
        print(f"[mintyai] {base}: its bars file is gone ({a.bars}) - skipped.")
        return None

    bars = D.load_bars(a.bars)
    if int(bars.meta.get("feature_signature", 0)) != F.feature_signature():
        print(f"[mintyai] {base}: feature signature has moved on since this "
              "model was trained - skipped. Retrain it.")
        return None

    ds = D.build_dataset(
        bars, window=a.window, horizon=a.horizon, barrier_atr=a.barrier_atr,
        commission_per_lot=a.commission_per_lot, slippage_points=a.slippage_points,
        spread_override_points=a.spread_points)

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model = MintyAINet(n_features=F.FEATURE_COUNT, window=a.window,
                       channels=a.channels, dropout=a.dropout,
                       dilations=T.parse_dilations(
                           getattr(a, "dilations", "1,2,4,8,16"))).to(device)
    model.load_state_dict(ck["state_dict"])   # temperature rides in the buffers
    model.eval()

    feats_gpu = torch.as_tensor(ds.feats, dtype=torch.float32, device=device)
    embargo = a.window + a.horizon
    train_idx, val_idx, _ = D.purged_split(ds, a.train_frac, a.val_frac, embargo)

    probs = T.predict_probs(model, feats_gpu, val_idx, a.window, use_amp=a.amp)
    risk = T.predict_risk(model, feats_gpu, val_idx, a.window, use_amp=a.amp)

    # Reuse the gate and hours the training run actually shipped rather than
    # re-deriving them. They are downstream of the same weights, so they
    # ought to come out identical - but when two gate settings are nearly
    # tied on validation, bfloat16 autocast is enough to flip which one wins
    # and the retune would silently hand back a different strategy. Only the
    # caps are meant to move here.
    rpt_path = os.path.join(T.FILES_DIR, base + "_report.json")
    shipped = {}
    if os.path.exists(rpt_path):
        with open(rpt_path, encoding="utf-8") as fh:
            shipped = json.load(fh)

    if all(k in shipped for k in ("min_prob", "edge_mult", "hour_mask")):
        min_prob = float(shipped["min_prob"])
        edge_mult = float(shipped["edge_mult"])
        hour_mask = int(shipped["hour_mask"])
        source = "from the training run"
    else:
        min_prob, edge_mult, _ = T.search_gate(probs, ds, val_idx, a.min_trades)
        hour_mask = T.learn_hour_mask(probs, ds, val_idx, min_prob, edge_mult)
        source = "RE-DERIVED - no report found, may differ from what shipped"

    # same derivation the trainer uses, off the walk-forward carried in the
    # report - so retuning and retraining agree on the caps
    caps = list(risk_pcts)
    wf_pre = shipped.get("walk_forward", {})
    f_star = T.kelly_star(wf_pre) if wf_pre else None
    if f_star and T.load_deploy_defaults().get("risk_from_kelly", True):
        caps = T.kelly_presets(f_star, T.parse_multipliers(",".join(
            str(x) for x in T.load_deploy_defaults().get(
                "kelly_multipliers", [0.5, 1.0, 2.0]))))
        print(f"[mintyai]   Kelly f* = {f_star * 100:.2f}% per trade; growth "
              f"turns negative beyond {f_star * 200:.2f}%")

    rows = []
    for name, cap in zip(T.RISK_NAMES, caps):
        lim = T.learn_risk_limits(probs, risk, ds, val_idx, min_prob, edge_mult,
                                  kelly_scale=a.kelly_scale, force_cap_pct=cap)
        rows.append(lim)
        print(f"[mintyai]   {name:<6} cap {lim['max_risk_pct']:6.2f}%  "
              f"validation drawdown {lim['sim_dd_pct']:5.1f}%  "
              f"halts {lim['daily_loss_pct']:.1f}%/{lim['max_dd_pct']:.1f}%")

    # walk-forward describes folds we are not re-running; carry it over
    wf = shipped.get("walk_forward", {"trades": 0, "total_r": 0.0, "tstat": 0.0})

    symbol = ds.meta.get("symbol", "?")
    print(f"[mintyai] {symbol}: gate p>={min_prob:.2f} edge>={edge_mult:.2f}x, "
          f"hours {bin(hour_mask).count('1')}/24 ({source})  "
          f"walk-forward t={wf.get('tstat', 0.0):+.2f} (carried over)")

    return {
        "symbol": symbol,
        "timeframe": ds.meta.get("timeframe", "H1"),
        "utc_offset_h": int(float(bars.meta.get("server_utc_offset_h", 0) or 0)),
        "file": base + ".onnx",
        "min_prob": min_prob,
        "edge_mult": edge_mult,
        "kelly_scale": a.kelly_scale,
        "hour_mask": hour_mask,
        "risk_pcts": caps,
        "wf_trades": wf.get("trades", 0),
        "wf_total_r": round(wf.get("total_r", 0.0), 2),
        "wf_tstat": round(wf.get("tstat", 0.0), 3),
        "risk": rows,
        "_args": a,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Rebuild the config from saved models, without retraining")
    p.add_argument("--symbols", default=None, help="comma-separated subset")
    p.add_argument("--cpu", action="store_true",
                   help="run on the CPU even if a CUDA device is present")
    p.add_argument("--risk-pcts", default=None,
                   help="override the three caps; default reads mintyai_deploy.json")
    a = p.parse_args(argv)

    risk_pcts = T.parse_risk_pcts(
        a.risk_pcts if a.risk_pcts else
        ",".join(str(x) for x in T.load_deploy_defaults().get(
            "risk_pcts", T.DEFAULT_RISK_PCTS)))
    print("[mintyai] risk presets: " + ", ".join(
        f"{n} {v:g}%" for n, v in zip(T.RISK_NAMES, risk_pcts)))

    want = {s.strip().upper() for s in a.symbols.split(",")} if a.symbols else None
    pts = sorted(glob.glob(os.path.join(T.FILES_DIR, "mintyai_*.pt")))
    pts = [q for q in pts if "_pooled_" not in q]
    if want:
        pts = [q for q in pts
               if os.path.basename(q).split("_")[1].upper() in want]
    if not pts:
        print(f"[mintyai] no saved models found in {T.FILES_DIR}")
        return 1

    entries = []
    for pt in pts:
        e = retune_symbol(pt, risk_pcts, force_cpu=a.cpu)
        if e:
            entries.append(e)
    if not entries:
        print("[mintyai] nothing retuned; config left untouched.")
        return 1

    args0 = entries[0].pop("_args")
    for e in entries:
        e.pop("_args", None)
    T.write_config(T.shared_config(args0, entries[0].get("risk_pcts", risk_pcts),
                                   entries[0]["timeframe"],
                                   entries[0].get("utc_offset_h", 0)), entries)
    print(f"[mintyai] retuned {len(entries)} symbol(s) - the graphs are "
          "untouched. Recompile MintyAI.mq5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
