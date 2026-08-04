"""Train MintyAI on the GPU, evaluate it honestly, export it for MetaTrader.

Typical use:

    python -m mintyai.train --bars ../../Files/MintyAI/EURUSD_H1_bars.csv

What it does, in order:
  1. builds features + triple-barrier labels
  2. runs an anchored walk-forward evaluation (the honest number)
  3. trains the deployable model on train+val, calibrates it on val
  4. grid-searches the trading gate on validation data
  5. exports ONNX, verifies ONNX Runtime agrees with PyTorch
  6. writes Include/MintyAI/MintyAIModelConfig.mqh so the EA cannot load a
     model that disagrees with its own feature engine
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from . import features as F
from .model import MintyAINet, count_parameters

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MQL5_ROOT = os.path.abspath(os.path.join(PKG_DIR, "..", ".."))
FILES_DIR = os.path.join(MQL5_ROOT, "Files", "MintyAI")
INCLUDE_DIR = os.path.join(MQL5_ROOT, "Include", "MintyAI")

# How much to stake is a deployment decision, not a property of the model,
# so it lives in a file that retraining does not touch. Without this, every
# retrain silently reverts the risk cap to whatever the drawdown solve
# produces - which is exactly the trap a generated config invites.
DEPLOY_PATH = os.path.join(MQL5_ROOT, "Python", "mintyai_deploy.json")


def load_deploy_defaults() -> dict:
    if not os.path.exists(DEPLOY_PATH):
        return {}
    try:
        with open(DEPLOY_PATH, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError) as exc:
        print(f"[mintyai] WARNING ignoring {DEPLOY_PATH}: {exc}")
        return {}


# --------------------------------------------------------------------------
# batching: the whole feature matrix fits in VRAM, so windows are gathered
# on the GPU instead of being shuttled across PCIe every step
# --------------------------------------------------------------------------


def make_windows(feats_gpu: torch.Tensor, idx: torch.Tensor, window: int) -> torch.Tensor:
    """idx = bar indices (window end, inclusive) -> [B, W, F]."""
    offsets = torch.arange(-window + 1, 1, device=feats_gpu.device)
    gather = idx.unsqueeze(1) + offsets.unsqueeze(0)  # [B, W]
    return feats_gpu[gather]


@torch.no_grad()
def predict_probs(
    model: MintyAINet, feats_gpu: torch.Tensor, idx: np.ndarray, window: int,
    batch: int = 8192, use_amp: bool = True,
) -> np.ndarray:
    model.eval()
    out = np.empty((idx.shape[0], model.n_classes), dtype=np.float64)
    dev = feats_gpu.device
    for s in range(0, idx.shape[0], batch):
        chunk = torch.as_tensor(idx[s : s + batch], device=dev, dtype=torch.long)
        x = make_windows(feats_gpu, chunk, window)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            p, _ = model(x)
        out[s : s + chunk.shape[0]] = p.float().cpu().numpy()
    return out


@torch.no_grad()
def predict_risk(
    model: MintyAINet, feats_gpu: torch.Tensor, idx: np.ndarray, window: int,
    batch: int = 8192, use_amp: bool = True,
) -> np.ndarray:
    """[n, 2] = (mu, sigma) of the long-side outcome in R units."""
    model.eval()
    out = np.empty((idx.shape[0], 2), dtype=np.float64)
    dev = feats_gpu.device
    for s in range(0, idx.shape[0], batch):
        chunk = torch.as_tensor(idx[s : s + batch], device=dev, dtype=torch.long)
        x = make_windows(feats_gpu, chunk, window)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, rk = model(x)
        rk = rk.float().cpu().numpy()
        out[s : s + chunk.shape[0], 0] = rk[:, 0]
        out[s : s + chunk.shape[0], 1] = np.exp(rk[:, 1])
    return out


@torch.no_grad()
def predict_logits(
    model: MintyAINet, feats_gpu: torch.Tensor, idx: np.ndarray, window: int,
    batch: int = 8192,
) -> np.ndarray:
    model.eval()
    out = np.empty((idx.shape[0], model.n_classes), dtype=np.float64)
    dev = feats_gpu.device
    for s in range(0, idx.shape[0], batch):
        chunk = torch.as_tensor(idx[s : s + batch], device=dev, dtype=torch.long)
        x = make_windows(feats_gpu, chunk, window)
        out[s : s + chunk.shape[0]] = model.logits(x).float().cpu().numpy()
    return out


# --------------------------------------------------------------------------
# the only metric that matters: money, after costs
# --------------------------------------------------------------------------


def gate_signals(
    probs: np.ndarray, ds: D.Dataset, idx: np.ndarray,
    min_prob: float, edge_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The EA's entry rule, bar by bar.

    Returns (go_long, go_short, r) where r is the outcome of whichever side
    fired, in R units. R = one barrier width: a trade that hits its
    take-profit earns +1R minus costs, one that hits its stop loses 1R plus
    costs.
    """
    b = ds.barrier[idx]
    cost = ds.cost[idx]
    close = ds.close[idx]
    exitp = ds.exit_price[idx]

    p_dn, p_up = probs[:, 0], probs[:, 2]
    ev_long = b * (p_up - p_dn) - cost
    ev_short = b * (p_dn - p_up) - cost
    thresh = edge_mult * cost

    go_long = (ev_long > thresh) & (p_up >= min_prob)
    go_short = (ev_short > thresh) & (p_dn >= min_prob)

    pnl = np.zeros(idx.shape[0], dtype=np.float64)
    pnl[go_long] = (exitp - close - cost)[go_long]
    pnl[go_short] = (close - exitp - cost)[go_short]

    return go_long, go_short, pnl / np.maximum(b, 1e-12)


def gate_pnl(
    probs: np.ndarray, ds: D.Dataset, idx: np.ndarray,
    min_prob: float, edge_mult: float,
) -> dict:
    """Score the entry rule on a set of bars."""
    go_long, go_short, r_all = gate_signals(probs, ds, idx, min_prob, edge_mult)

    traded = go_long | go_short
    n = int(traded.sum())
    if n == 0:
        return {
            "trades": 0, "total_r": 0.0, "mean_r": 0.0, "tstat": 0.0,
            "hit_rate": 0.0, "exposure": 0.0, "long_frac": 0.0,
        }

    r = r_all[traded]
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    return {
        "trades": n,
        "total_r": float(r.sum()),
        "mean_r": float(r.mean()),
        "tstat": float(r.mean() / sd * math.sqrt(n)) if sd > 1e-12 else 0.0,
        "hit_rate": float((r > 0).mean()),
        "exposure": float(n / idx.shape[0]),
        "long_frac": float(go_long.sum() / n),
    }


MIN_PROB_GRID = [0.34, 0.38, 0.42, 0.46, 0.50, 0.55]
EDGE_GRID = [0.0, 0.25, 0.5, 1.0, 1.5]


def search_gate(probs, ds, idx, min_trades: int) -> tuple[float, float, dict]:
    """Pick the entry gate on validation data, scored by t-statistic.

    t-stat rather than total R: a handful of lucky trades can dominate a
    sum, and this choice is made on data the deployed model will not see
    again.
    """
    best = (0.42, 0.5, None)
    best_score = -1e18
    for mp in MIN_PROB_GRID:
        for em in EDGE_GRID:
            m = gate_pnl(probs, ds, idx, mp, em)
            if m["trades"] < min_trades:
                continue
            if m["tstat"] > best_score:
                best_score = m["tstat"]
                best = (mp, em, m)
    if best[2] is None:
        # nothing cleared the minimum-trades bar; fall back to the loosest gate
        best = (MIN_PROB_GRID[0], 0.0, gate_pnl(probs, ds, idx, MIN_PROB_GRID[0], 0.0))
    return best


# --------------------------------------------------------------------------
# what used to be EA inputs, measured on validation data instead
# --------------------------------------------------------------------------

# Half Kelly by default. Full Kelly maximises log growth only when the edge
# estimate is exact, and ours is a model output. Halving costs a quarter of
# the growth rate and roughly halves the drawdown.
KELLY_SCALE = 0.5

# The cap is solved for, not picked off a grid: the largest per-trade risk
# whose validation equity curve stays inside the drawdown budget. This is
# the number that decides how much the EA stakes, so --dd-target is the
# lever for trading bigger. Both are training-time decisions; the EA still
# exposes nothing.
DD_TARGET = 0.20

# Bisection bounds for that solve, as fractions of equity per trade.
CAP_LO, CAP_HI = 0.0005, 0.10

MIN_TRADES_PER_HOUR = 25

# Bounds for the temperature search. A genuine calibration lands near 1 -
# 0.65 and 1.42 are ordinary. Hitting either edge is a diagnosis, not a value.
LO_T, HI_T = 0.25, 4.0

# Past these a halt stops meaning "worse than the model has ever shown" and
# starts meaning "the account is gone anyway".
HALT_HARD_DAILY = 50.0
HALT_HARD_TOTAL = 90.0


# Bounds on a derived preset. Below the floor a stake is pointless; above
# the ceiling no measured edge justifies it whatever Kelly says.
PRESET_FLOOR_PCT = 0.05
PRESET_CEIL_PCT = 20.0


def kelly_star(m: dict) -> float | None:
    """Growth-optimal fraction of equity per trade, from a set of results.

    For a bet returning r per unit staked, log growth is maximised at
    f* = E[r] / E[r^2]. Beyond 2 x f* the second-order term overtakes the
    first and growth goes NEGATIVE - the account bleeds even though every
    trade has positive expectancy. That threshold is the reason to derive
    these rather than pick round numbers.

    sd is recovered from the t-statistic, which is how the trades were
    summarised: t = mean / sd * sqrt(n).
    """
    n = m.get("trades", 0)
    t = m.get("tstat", 0.0)
    if n < 30 or abs(t) < 1e-9:
        return None
    mean = m.get("total_r", 0.0) / n
    sd = mean / t * math.sqrt(n)
    er2 = sd * sd + mean * mean
    if er2 <= 0.0 or mean <= 0.0:
        return None                      # no edge to size against
    return mean / er2


def kelly_presets(f_star: float, multipliers: list) -> list:
    return [round(min(max(f_star * mult * 100.0, PRESET_FLOOR_PCT),
                      PRESET_CEIL_PCT), 3)
            for mult in multipliers]


def kelly_fractions(
    risk_out: np.ndarray, ds: D.Dataset, idx: np.ndarray,
    go_long: np.ndarray, go_short: np.ndarray, kelly_scale: float = KELLY_SCALE,
) -> np.ndarray:
    """The fraction of equity to risk per bar - the EA's exact arithmetic."""
    mu = risk_out[:, 0]
    sigma = np.maximum(risk_out[:, 1], 1e-3)
    cost_r = ds.cost[idx] / np.maximum(ds.barrier[idx], 1e-12)
    edge = np.where(go_long, mu - cost_r, np.where(go_short, -mu - cost_r, 0.0))
    return np.clip(edge / (sigma ** 2), 0.0, 1.0) * kelly_scale


def solve_cap(ft: np.ndarray, rt: np.ndarray, dd_target: float) -> float:
    """Largest per-trade cap whose equity curve stays inside dd_target.

    Bisection rather than a grid: with most trades pinned to the cap the
    drawdown is close to linear in it, so a coarse grid throws away a third
    of the available stake between one step and the next.
    """
    def dd(cap: float) -> float:
        eq = np.cumprod(1.0 + np.minimum(ft, cap) * rt)
        return float((1.0 - eq / np.maximum.accumulate(eq)).max())

    if dd(CAP_HI) <= dd_target:
        return CAP_HI
    if dd(CAP_LO) > dd_target:
        return CAP_LO

    lo, hi = CAP_LO, CAP_HI
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if dd(mid) <= dd_target:
            lo = mid
        else:
            hi = mid
    return lo


def learn_hour_mask(
    probs: np.ndarray, ds: D.Dataset, idx: np.ndarray,
    min_prob: float, edge_mult: float,
) -> int:
    """24-bit mask of server hours the model is allowed to trade.

    An hour is switched off only on evidence: it has to have produced at
    least MIN_TRADES_PER_HOUR validation trades AND lost money over them.
    Thin hours stay enabled, because banning on three trades is fitting
    noise, which is the failure mode a session-hours input invites.
    """
    go_long, go_short, r = gate_signals(probs, ds, idx, min_prob, edge_mult)
    traded = go_long | go_short
    hours = ((ds.time[idx] // 3600) % 24).astype(np.int64)

    mask = 0
    for h in range(24):
        sel = traded & (hours == h)
        n = int(sel.sum())
        if n < MIN_TRADES_PER_HOUR or float(r[sel].mean()) > 0.0:
            mask |= (1 << h)
    return mask


def learn_risk_limits(
    probs: np.ndarray, risk_out: np.ndarray, ds: D.Dataset, idx: np.ndarray,
    min_prob: float, edge_mult: float,
    dd_target: float = DD_TARGET, kelly_scale: float = KELLY_SCALE,
    force_cap_pct: float | None = None,
) -> dict:
    """Size validation trades exactly as the EA will, then read the limits
    off the resulting equity curve.

    force_cap_pct sets the per-trade cap directly and skips the drawdown
    solve. The drawdown is then whatever it turns out to be, and is
    reported rather than bounded.
    """
    fallback = {"max_risk_pct": round(CAP_LO * 100.0, 3), "daily_loss_pct": 3.0,
                "max_dd_pct": 20.0, "sim_dd_pct": 0.0, "sim_trades": 0,
                "mean_risk_pct": 0.0, "capped_frac": 0.0}

    go_long, go_short, r = gate_signals(probs, ds, idx, min_prob, edge_mult)
    traded = go_long | go_short
    if int(traded.sum()) == 0:
        return fallback

    f_all = kelly_fractions(risk_out, ds, idx, go_long, go_short, kelly_scale)
    ft, rt, tt = f_all[traded], r[traded], ds.time[idx][traded]

    cap = (force_cap_pct / 100.0) if force_cap_pct is not None \
          else solve_cap(ft, rt, dd_target)

    f = np.minimum(ft, cap)
    eq = np.cumprod(1.0 + f * rt)
    sim_dd = float((1.0 - eq / np.maximum.accumulate(eq)).max()) * 100.0

    # worst single day, as a percentage of that day's starting equity
    days = (tt // 86400).astype(np.int64)
    ret = f * rt
    worst_day = 0.0
    for d in np.unique(days):
        worst_day = max(worst_day, float(-ret[days == d].sum() * 100.0))

    # Halts sit at twice the worst the model has actually shown. Tighter and
    # normal variance trips them; looser and they are decoration.
    #
    # The ceilings scale with the stake. A fixed 10% daily halt against a 10%
    # per-trade cap would stop the EA on its first ordinary loss, which is
    # not risk control - it is a position limit wearing a halt's clothes.
    # At 3x the cap it takes three losing trades in a day to trip.
    cap_pct = cap * 100.0
    daily_ceiling = max(10.0, 3.0 * cap_pct)
    total_ceiling = max(40.0, 8.0 * cap_pct)

    # v2.0 PARITY: the ceilings are the raw per-stake ones, with no guard
    # that the halt clears the drawdown the model actually produced. That
    # guard is why 2.0 and this build disagreed on the halt levels, so it is
    # off. The consequence is real and worth knowing: where 2 x sim_dd
    # exceeds the ceiling, the halt sits BELOW the model's own measured
    # drawdown and will fire during ordinary trading.
    if sim_dd * 2.0 > total_ceiling:
        print(
            f"[mintyai]   NOTE at {cap_pct:g}% the halt lands at "
            f"{min(max(sim_dd * 2.0, 5.0), total_ceiling):.1f}% while validation "
            f"drew down {sim_dd:.1f}% - the halt will trip in normal trading. "
            "This is 2.0's behaviour, restored on request."
        )

    return {
        "max_risk_pct": round(cap_pct, 3),
        "daily_loss_pct": round(min(max(worst_day * 2.0, 1.0), daily_ceiling), 2),
        "max_dd_pct": round(min(max(sim_dd * 2.0, 5.0), total_ceiling), 2),
        "sim_dd_pct": round(sim_dd, 2),
        "sim_trades": int(traded.sum()),
        "mean_risk_pct": round(float(f.mean() * 100.0), 4),
        # how often the cap bound rather than Kelly - if this is near 1.0,
        # --dd-target is what decides the stake, not the model
        "capped_frac": round(float((ft >= cap).mean()), 3),
    }


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


WEIGHT_CLIP = 4.0


def class_weights(labels: np.ndarray, idx: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Damped inverse-frequency weights, clipped.

    The clip matters: with tight barriers the 'flat' class can be almost
    empty, and an unclipped inverse frequency hands it a weight in the
    hundreds. The loss then optimises almost entirely for an outcome that
    never happens, and the model learns nothing about direction.
    """
    counts = np.bincount(labels[idx], minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (counts.sum() / counts) ** 0.5
    w = w / w.mean()
    return np.clip(w, 1.0 / WEIGHT_CLIP, WEIGHT_CLIP)


def train_model(
    ds: D.Dataset, feats_gpu: torch.Tensor, train_idx: np.ndarray,
    val_idx: np.ndarray, args, device: str, log_prefix: str = "",
) -> tuple[MintyAINet, dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dilations = parse_dilations(getattr(args, "dilations", "1,2,4,8,16"))
    model = MintyAINet(
        n_features=F.FEATURE_COUNT,
        window=args.window,
        channels=args.channels,
        dilations=dilations,
        dropout=args.dropout,
    ).to(device)

    # normalisation from TRAIN bars only - using all bars would leak the
    # future's volatility regime into the past
    # Statistics from the TRAINING ROWS themselves, not from a slice of the
    # array: with several symbols pooled, row order is symbol-major and a
    # positional slice would take its stats from whichever symbols happen to
    # sit at the front.
    train_rows = ds.feats[train_idx]
    model.set_normalisation(train_rows.mean(axis=0), train_rows.std(axis=0))

    w = torch.as_tensor(class_weights(ds.labels, train_idx), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.25
    )

    y_all = torch.as_tensor(ds.labels, dtype=torch.long, device=device)
    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)

    # Regression target for the risk head: the long-side outcome in R units.
    # Short is the mirror image, so one target covers both directions.
    r_long = (ds.exit_price - ds.close) / np.maximum(ds.barrier, 1e-12)
    r_long = np.clip(np.nan_to_num(r_long, nan=0.0, posinf=0.0, neginf=0.0), -3.0, 3.0)
    r_all = torch.as_tensor(r_long, dtype=torch.float32, device=device)

    best_score, best_state, best_epoch, stale = (-1e18, -1e18), None, -1, 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(train_t.shape[0], device=device)
        total_loss, nb = 0.0, 0
        for s in range(0, steps_per_epoch * args.batch_size, args.batch_size):
            sel = train_t[perm[s : s + args.batch_size]]
            if sel.shape[0] < 8:
                continue
            x = make_windows(feats_gpu, sel, args.window)
            y = y_all[sel]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                logits, risk = model.heads(x)
                mu, log_sigma = risk[:, 0], risk[:, 1]
                # Gaussian NLL on the realised outcome. Learning sigma is the
                # point: it is what the class probabilities cannot say and
                # what Kelly divides by.
                z = (r_all[sel] - mu) / torch.exp(log_sigma)
                loss_risk = (0.5 * z * z + log_sigma).mean()
                loss = criterion(logits, y) + args.risk_weight * loss_risk
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += float(loss.detach())
            nb += 1

        probs = predict_probs(model, feats_gpu, val_idx, args.window, use_amp=args.amp)
        m = gate_pnl(probs, ds, val_idx, args.es_min_prob, args.es_edge_mult)
        val_nll = float(
            -np.log(np.clip(probs[np.arange(len(val_idx)), ds.labels[val_idx]], 1e-12, None)).mean()
        )

        # Select on risk-adjusted edge, not accuracy: a model can be more
        # accurate and still lose money. Break ties on likelihood, because
        # early on the gate fires zero trades every epoch and a pure t-stat
        # would freeze the very first set of weights.
        score = (m["tstat"], -val_nll)
        history.append({"epoch": epoch, "loss": total_loss / max(nb, 1),
                        "val_nll": val_nll, **m})

        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"{log_prefix}epoch {epoch:3d} loss {total_loss / max(nb, 1):.4f} "
                f"nll {val_nll:.4f} val trades {m['trades']:5d} R {m['total_r']:8.1f} "
                f"t {m['tstat']:5.2f} hit {m['hit_rate']:.3f}"
            )
        if stale >= args.patience:
            print(f"{log_prefix}early stop at epoch {epoch} (best {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_score": best_score, "history": history}


def calibrate_temperature(model: MintyAINet, feats_gpu, ds, val_idx, window) -> float:
    """Temperature scaling on validation logits.

    Uncalibrated probabilities make the expected-value gate meaningless:
    the EA multiplies them by a barrier size and compares against real
    money, so they have to mean what they say.
    """
    logits = torch.as_tensor(predict_logits(model, feats_gpu, val_idx, window), dtype=torch.float32)
    labels = torch.as_tensor(ds.labels[val_idx], dtype=torch.long)
    nll = nn.CrossEntropyLoss()

    # v2.0 PARITY: LBFGS with a wide clamp, restored on request.
    #
    # Known failure mode, left in deliberately: on a nearly flat objective
    # LBFGS's inverse-Hessian step blows up and it returns something like
    # T=4e-5, which is not a minimum - NLL there is far WORSE than at T=1.
    # Clamping then ships a 20x sharpening of the logits as if it were a
    # calibration. The bounded grid search that replaced this is in git
    # history if you want it back.
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nll(logits / torch.exp(log_t), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(log_t).item())

    # How much separation is there to calibrate in the first place? A model
    # that has collapsed toward the base rate emits near-identical logits
    # for all three classes; the NLL objective is then almost flat in T and
    # the optimiser's answer is arbitrary rather than informative.
    spread = float((logits - logits.mean(dim=1, keepdim=True)).std())

    # v2.0 PARITY: clamp into [0.05, 20] rather than refusing the answer.
    lo, hi = 0.05, 20.0
    if not math.isfinite(t) or t < lo or t > hi:
        clamped = min(max(t if math.isfinite(t) else 1.0, lo), hi)
        print(
            f"[mintyai] WARNING temperature scaling wanted T={t:.4g}; clamped to "
            f"{clamped:.4g}. Logit spread is {spread:.4f}; NLL is "
            f"{float(nll(logits / clamped, labels)):.5f} at the clamp vs "
            f"{float(nll(logits, labels)):.5f} at T=1 - if the clamp is worse, "
            "LBFGS diverged and this is not a calibration."
        )
        return clamped
    return t


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def export_onnx(model: MintyAINet, path: str, window: int, device: str) -> None:
    model.eval().cpu()
    dummy = torch.zeros(1, window, F.FEATURE_COUNT, dtype=torch.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = ["probs", "risk"]
    try:
        torch.onnx.export(
            model, (dummy,), path,
            input_names=["features"], output_names=names,
            opset_version=17, dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model, (dummy,), path,
            input_names=["features"], output_names=names, opset_version=17,
        )

    # Torch leaves the output batch dimension symbolic even for a static
    # export. MetaTrader binds output buffers by shape, so pin them rather
    # than relying on OnnxSetOutputShape to paper over it.
    import onnx

    m = onnx.load(path)
    widths = [model.n_classes, 2]
    changed = False
    for out, width in zip(m.graph.output, widths):
        dims = out.type.tensor_type.shape.dim
        if len(dims) == 2:
            dims[0].ClearField("dim_param")
            dims[0].dim_value = 1
            dims[1].ClearField("dim_param")
            dims[1].dim_value = width
            changed = True
    if changed:
        onnx.checker.check_model(m)
        onnx.save(m, path)

    model.to(device)


def verify_onnx(model: MintyAINet, path: str, ds: D.Dataset, idx: np.ndarray, window: int) -> float:
    """Run the exported graph through ONNX Runtime and diff against PyTorch.

    MetaTrader runs this same graph, so a mismatch here is a mismatch live.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    sample = idx[:: max(1, len(idx) // 512)][:512]
    model.eval().cpu()

    worst = 0.0
    for bar in sample:
        w = ds.windows[int(bar) - window + 1]  # [F, W]
        x = np.ascontiguousarray(w.T[None, :, :].astype(np.float32))  # [1, W, F]
        got_p, got_r = sess.run(["probs", "risk"], {"features": x})
        with torch.no_grad():
            want_p, want_r = model(torch.from_numpy(x))
        worst = max(
            worst,
            float(np.abs(got_p - want_p.numpy()).max()),
            float(np.abs(got_r - want_r.numpy()).max()),
        )
    return worst




# --------------------------------------------------------------------------
# the EA's config header: one table covering every model in the build
# --------------------------------------------------------------------------

# The three presets the EA offers. Network weights are identical across
# them - the risk cap never enters the loss - so a symbol is trained once
# and then evaluated at each cap. Training three times would burn three
# times the GPU for byte-identical graphs.
DEFAULT_RISK_PCTS = [2.0, 5.0, 10.0]
RISK_NAMES = ["LOW", "MEDIUM", "HIGH"]

CONFIG_PATH = os.path.join(INCLUDE_DIR, "MintyAIModelConfig.mqh")


def mql_ident(symbol: str) -> str:
    """A symbol name as a legal MQL5 enum member.

    Broker names carry characters an identifier cannot: EURUSD.pro,
    EURUSD-ECN, BRENT_OIL. The enum's displayed label keeps the real name;
    only the identifier is scrubbed.
    """
    safe = "".join(c if c.isalnum() else "_" for c in symbol)
    return "MINTYAI_SYM_" + safe.upper()


def render_config(shared: dict, entries: list) -> str:
    """Build MintyAIModelConfig.mqh for any number of symbols."""
    n = len(entries)
    risk_pcts = shared["risk_pcts"]

    out = []
    w = out.append

    w("//+------------------------------------------------------------------+")
    w("//|                                            MintyAIModelConfig.mqh |")
    w("//|                                                                  |")
    w("//|   GENERATED BY mintyai/train.py - DO NOT EDIT BY HAND.           |")
    w("//|   Retraining is how a new model reaches the EA.                  |")
    w("//|                                                                  |")
    w(f"//|   Built  : {shared['built_at']}")
    w(f"//|   Models : {n} network(s), one per symbol, x {len(risk_pcts)} risk levels")
    w(f"//|   Symbols: {', '.join(e['symbol'] for e in entries)}")
    w("//+------------------------------------------------------------------+")
    w("#ifndef MINTYAI_MODEL_CONFIG_MQH")
    w("#define MINTYAI_MODEL_CONFIG_MQH")
    w("")
    w("//--- shared by every model in this build")
    w(f"#define MINTYAI_MODEL_FEATURE_VERSION {shared['feature_version']}")
    w(f"#define MINTYAI_MODEL_FEATURE_COUNT   {shared['feature_count']}")
    w(f"#define MINTYAI_MODEL_SIGNATURE       {shared['signature']}")
    w(f"#define MINTYAI_MODEL_WINDOW          {shared['window']}")
    w(f'#define MINTYAI_MODEL_TIMEFRAME       "{shared["timeframe"]}"')
    w("")
    w("//--- The training bars were stamped in the training SERVER's own local")
    w("//--- clock, not in UTC - see MintyAIExporter.mqh, which writes")
    w("//--- rates[].time unconverted. That is deliberate: the server clock")
    w("//--- moves with daylight saving and so do the sessions the model keys")
    w("//--- off, so hour-of-day stays aligned all year without conversion.")
    w("//---")
    w("//--- This is that server's UTC offset AT EXPORT TIME, a single reading,")
    w("//--- recorded so a broker in a genuinely different timezone can be")
    w("//--- corrected for. It is NOT a year-round constant: on a DST-observing")
    w("//--- server it reads an hour differently in winter, which is why the EA")
    w("//--- ignores gaps of an hour or less rather than 'correcting' a clock")
    w("//--- against itself.")
    w(f"#define MINTYAI_MODEL_UTC_OFFSET_H    {shared.get('utc_offset_h', 0)}")
    w("")
    w("//--- the trade definition the labels were built from. The EA's stop,")
    w("//--- target and time exit MUST mirror these or the probabilities")
    w("//--- describe a trade you are not actually taking.")
    w(f"#define MINTYAI_MODEL_BARRIER_ATR     {shared['barrier_atr']}")
    w(f"#define MINTYAI_MODEL_HORIZON_BARS    {shared['horizon']}")
    w(f"#define MINTYAI_MODEL_SLIPPAGE_POINTS {shared['slippage_points']}")
    w("")
    w(f"#define MINTYAI_MODEL_COUNT           {n}")
    w(f"#define MINTYAI_RISK_COUNT            {len(risk_pcts)}")
    w("")
    w("//--- One network PER SYMBOL. Pooling every instrument into a shared")
    w("//--- model was tried and measured worse: the shared function ends up")
    w("//--- a compromise that prices none of them sharply, and its")
    w("//--- probabilities collapse toward the base rate.")
    w("//---")
    w("//--- Every graph travels inside the .ex5: a strategy tester agent gets")
    w("//--- its own empty MQL5\\Files sandbox, so loading by path fails there")
    w("//--- with error 5019 however plainly the file sits in the data folder.")
    for i, e in enumerate(entries):
        w(f'#resource "\\\\Files\\\\MintyAI\\\\{e["file"]}" as uchar ExtMintyAIModel{i}[]')
    w("")
    w("//--- The models this build carries. The EA's Model input IS this enum,")
    w("//--- so the dropdown cannot offer a model that does not exist, and the")
    w("//--- value doubles as the index into MintyAIModels[].")
    w("//---")
    w("//--- AUTO is declared first so it is what the terminal shows by default.")
    w("//--- It works the model out from the chart symbol, which handles every")
    w("//--- broker that merely decorates the standard name - EURUSD.ecn, and")
    w("//--- so on. Naming a model outright is for the ones that rename the")
    w("//--- instrument altogether: a chart called GOLD carries nothing an")
    w("//--- algorithm could match to XAUUSD, so the user says which it is.")
    w("enum ENUM_MINTYAI_SYMBOL")
    w("  {")
    w(f"   {'MINTYAI_SYM_AUTO':<26} = -1,   // Auto - match the chart symbol")
    for i, e in enumerate(entries):
        comma = "," if i < n - 1 else ""
        w(f"   {mql_ident(e['symbol']):<26} = {i}{comma}   // {e['symbol']}")
    w("  };")
    w("")
    w(f"#define MINTYAI_DEFAULT_SYMBOL    {mql_ident(entries[0]['symbol'])}")
    w("")
    w("//--- Risk presets. The weights behind each are the same; the stake and")
    w("//--- the halts that follow from it are not.")
    w("enum ENUM_MINTYAI_RISK")
    w("  {")
    same = all(e.get("risk_pcts") == entries[0].get("risk_pcts") for e in entries)
    for i, name in enumerate(RISK_NAMES):
        comma = "," if i < len(RISK_NAMES) - 1 else ""
        label = name.capitalize()
        if same and entries[0].get("risk_pcts"):
            note = f"up to {entries[0]['risk_pcts'][i]:g}% of equity per trade"
        else:
            note = "cap is per symbol - see the table below"
        w(f"   MINTYAI_RISK_{name:<6} = {i}{comma}   // {label} - {note}")
    w("  };")
    w("")
    w("struct MintyAIRiskCfg")
    w("  {")
    w("   double            max_risk_pct;     // cap per trade, % of equity")
    w("   double            daily_loss_pct;   // daily halt")
    w("   double            max_dd_pct;       // total drawdown halt")
    w("   double            sim_dd_pct;       // drawdown this cap produced on validation")
    w("  };")
    w("")
    w("struct MintyAIModelCfg")
    w("  {")
    w("   string            symbol;")
    w("   double            min_prob;         // entry gate, chosen on validation")
    w("   double            edge_mult;        // required edge, in units of cost")
    w("   double            kelly_scale;")
    w("   uint              hour_mask;        // 24-bit mask of tradable hours")
    w("   int               wf_trades;        // walk-forward, out-of-sample")
    w("   double            wf_total_r;")
    w("   double            wf_tstat;")
    w(f"   MintyAIRiskCfg    risk[{len(risk_pcts)}];")
    w("  };")
    w("")
    w("MintyAIModelCfg MintyAIModels[MINTYAI_MODEL_COUNT];")
    w("")
    w("//+------------------------------------------------------------------+")
    w("//| Fill the table. Call once from OnInit.                           |")
    w("//+------------------------------------------------------------------+")
    w("void MintyAIInitModels()")
    w("  {")
    for i, e in enumerate(entries):
        hold = e.get("holdout", {})
        w(f"//--- {e['symbol']}: walk-forward t={e['wf_tstat']}, "
          f"holdout t={round(hold.get('tstat', 0.0), 2)}")
        w(f'   MintyAIModels[{i}].symbol      = "{e["symbol"]}";')
        w(f"   MintyAIModels[{i}].min_prob    = {e['min_prob']};")
        w(f"   MintyAIModels[{i}].edge_mult   = {e['edge_mult']};")
        w(f"   MintyAIModels[{i}].kelly_scale = {e['kelly_scale']};")
        w(f"   MintyAIModels[{i}].hour_mask   = 0x{e['hour_mask']:06X};")
        w(f"   MintyAIModels[{i}].wf_trades   = {e['wf_trades']};")
        w(f"   MintyAIModels[{i}].wf_total_r  = {e['wf_total_r']};")
        w(f"   MintyAIModels[{i}].wf_tstat    = {e['wf_tstat']};")
        for k, lim in enumerate(e["risk"]):
            w(f"   MintyAIModels[{i}].risk[{k}].max_risk_pct   = {lim['max_risk_pct']};")
            w(f"   MintyAIModels[{i}].risk[{k}].daily_loss_pct = {lim['daily_loss_pct']};")
            w(f"   MintyAIModels[{i}].risk[{k}].max_dd_pct     = {lim['max_dd_pct']};")
            w(f"   MintyAIModels[{i}].risk[{k}].sim_dd_pct     = {lim['sim_dd_pct']};")
        w("")
    w("  }")
    w("")
    w("//+------------------------------------------------------------------+")
    w("//| Create the ONNX session for a symbol's own network.               |")
    w("//+------------------------------------------------------------------+")
    w("long MintyAICreateOnnx(const int idx)")
    w("  {")
    w("   switch(idx)")
    w("     {")
    for i in range(n):
        w(f"      case {i}: return OnnxCreateFromBuffer(ExtMintyAIModel{i}, ONNX_DEFAULT);")
    w("     }")
    w("   return INVALID_HANDLE;")
    w("  }")
    w("")
    w("//+------------------------------------------------------------------+")
    w("//| Index of a symbol's model, or -1.                                |")
    w("//+------------------------------------------------------------------+")
    w("int MintyAIFindModel(const string symbol)")
    w("  {")
    w("   for(int i = 0; i < MINTYAI_MODEL_COUNT; i++)")
    w("      if(MintyAIModels[i].symbol == symbol)")
    w("         return i;")
    w("   return -1;")
    w("  }")
    w("")
    w("//--- for error messages: every symbol this build can trade")
    w("string MintyAISymbolList()")
    w("  {")
    w("   string s = \"\";")
    w("   for(int i = 0; i < MINTYAI_MODEL_COUNT; i++)")
    w("      s += (i == 0 ? \"\" : \", \") + MintyAIModels[i].symbol;")
    w("   return s;")
    w("  }")
    w("")
    w("#endif // MINTYAI_MODEL_CONFIG_MQH")
    return "\n".join(out) + "\n"


def read_export_meta(bars_path: str) -> dict:
    """The key,value sidecar MintyAIExport.mq5 writes beside a bars file."""
    if not bars_path or not bars_path.endswith("_bars.csv"):
        return {}
    candidates = [bars_path[: -len("_bars.csv")] + "_meta.csv"]
    # the export folder may have moved since that symbol was trained
    candidates.append(os.path.join(FILES_DIR, os.path.basename(candidates[0])))

    for path in candidates:
        if not os.path.exists(path):
            continue
        meta = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    meta[parts[0]] = parts[1]
        return meta
    return {}


def entry_from_report(report_path: str) -> dict | None:
    """Rebuild a config row for a symbol some earlier run trained.

    finalise_export() writes the report in the same breath as the .onnx it
    describes, so the two are always a matched pair. That is what makes
    carrying a symbol forward safe without reloading any weights.
    """
    base = os.path.basename(report_path)[: -len("_report.json")]
    if not os.path.exists(os.path.join(FILES_DIR, base + ".onnx")):
        return None
    try:
        with open(report_path, encoding="utf-8") as fh:
            r = json.load(fh)
    except (OSError, ValueError):
        return None

    a = r.get("args") or {}
    levels = r.get("risk_levels") or {}
    rows = [levels.get(name) for name in RISK_NAMES]
    if not all(rows) or "hour_mask" not in r or "min_prob" not in r:
        return None

    wf = r.get("walk_forward") or {}
    meta = read_export_meta(a.get("bars", ""))

    caps = a.get("risk_pcts")
    if isinstance(caps, str):
        try:
            caps = [float(x) for x in caps.split(",")]
        except ValueError:
            caps = None

    return {
        "symbol": r.get("symbol", ""),
        "timeframe": r.get("timeframe", "H1"),
        "utc_offset_h": int(float(meta.get("server_utc_offset_h", 0) or 0)),
        "file": base + ".onnx",
        "min_prob": r["min_prob"],
        "edge_mult": r["edge_mult"],
        "kelly_scale": a.get("kelly_scale", 0.5),
        "hour_mask": int(r["hour_mask"]),
        "risk_pcts": caps,
        "wf_trades": wf.get("trades", 0),
        "wf_total_r": round(float(wf.get("total_r", 0.0)), 2),
        "wf_tstat": round(float(wf.get("tstat", 0.0)), 3),
        "holdout": r.get("holdout", {}),
        "risk": rows,
        # checked against the build, then dropped
        "_signature": int(meta["feature_signature"]) if "feature_signature" in meta else None,
        "_window": a.get("window"),
        "_horizon": a.get("horizon"),
        "_barrier_atr": a.get("barrier_atr"),
    }


def carry_forward(shared: dict, entries: list) -> list:
    """Re-add every symbol trained earlier that this run did not touch.

    Training one symbol used to rewrite the whole table, so a single-symbol
    run shipped an EA carrying exactly one model and every other symbol
    silently vanished from the product. Now each run contributes its own
    symbol and inherits the rest from the reports on disk.

    A symbol is only carried if it still describes the SAME bet: same
    feature signature, timeframe, window, horizon and barrier. Those are
    compiled into the EA once, for every model at once, so mixing in a
    model fitted to different ones would have it price a trade the EA is
    not placing - with nothing at runtime able to tell.
    """
    fresh = {e["symbol"] for e in entries}
    merged = list(entries)
    carried, skipped = [], []

    for path in sorted(glob.glob(os.path.join(FILES_DIR, "mintyai_*_report.json"))):
        if "_pooled_" in os.path.basename(path):
            continue
        e = entry_from_report(path)
        if not e or not e["symbol"] or e["symbol"] in fresh:
            continue

        why = None
        if e["_signature"] is not None and e["_signature"] != shared["signature"]:
            why = f"feature signature {e['_signature']} != {shared['signature']}"
        elif e["timeframe"] != shared["timeframe"]:
            why = f"{e['timeframe']}, this build is {shared['timeframe']}"
        elif e["_window"] is not None and int(e["_window"]) != int(shared["window"]):
            why = f"window {e['_window']} != {shared['window']}"
        elif e["_horizon"] is not None and int(e["_horizon"]) != int(shared["horizon"]):
            why = f"horizon {e['_horizon']} != {shared['horizon']}"
        elif (e["_barrier_atr"] is not None
              and float(e["_barrier_atr"]) != float(shared["barrier_atr"])):
            why = f"barrier {e['_barrier_atr']} != {shared['barrier_atr']}"

        if why:
            skipped.append(f"{e['symbol']} ({why})")
            continue

        merged.append({k: v for k, v in e.items() if not k.startswith("_")})
        carried.append(e["symbol"])

    merged.sort(key=lambda e: e["symbol"])

    if carried:
        print(f"[mintyai] carried {len(carried)} earlier symbol(s) forward: "
              + ", ".join(carried))
    if skipped:
        print("[mintyai] NOT carried forward, they no longer match this build: "
              + "; ".join(skipped))
    return merged


def write_config(shared: dict, entries: list, merge: bool = True) -> None:
    if merge:
        entries = carry_forward(shared, entries)
    os.makedirs(INCLUDE_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_config(shared, entries))
    print(f"[mintyai] wrote {CONFIG_PATH}  ({len(entries)} model(s))")


# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    dep = load_deploy_defaults()
    p = argparse.ArgumentParser(description="Train the MintyAI price-direction model")
    p.add_argument("--bars", required=True, help="path to *_bars.csv from MintyAIExport.mq5")
    p.add_argument("--window", type=int, default=96, help="input window in bars")
    p.add_argument("--horizon", type=int, default=24, help="max bars a trade stays open")
    # 3 ATR is not arbitrary. The gate needs p_up - p_down > cost / barrier,
    # so a 1 ATR barrier on a typical spread demands a ~0.29 directional
    # edge - far beyond anything real. Widening the barrier lowers that
    # hurdle roughly proportionally, and at 3 ATR / 24 bars the three
    # outcome classes also come out reasonably balanced.
    p.add_argument("--barrier-atr", type=float, default=3.0, help="stop/target in ATR(14) units")
    p.add_argument("--commission-per-lot", type=float, default=0.0)
    p.add_argument("--slippage-points", type=float, default=2.0)
    p.add_argument("--spread-points", type=float, default=None,
                   help="override the recorded spread (points, round trip)")

    p.add_argument("--channels", type=int, default=64,
                   help="width of the network - conv parameters scale with its "
                        "square, so 128 is roughly 4x the size of 64")
    p.add_argument("--dilations", default="1,2,4,8,16",
                   help="dilated causal blocks, one per value. Depth and "
                        "receptive field: with kernel 3 the stack sees "
                        "1 + 2*sum(dilations) bars, so 1,2,4,8,16 reaches 63 - "
                        "less than the default 96-bar window. Add 32 to reach "
                        "127 and cover it.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--risk-weight", type=float, default=0.3,
                   help="weight of the outcome-head NLL against the classifier loss")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpu", action="store_true",
                   help="train on the CPU even if a CUDA device is present. "
                        "Expect it to be far slower; bfloat16 autocast is a "
                        "CUDA path, so it is switched off automatically.")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-every", type=int, default=5)

    p.add_argument("--folds", type=int, default=5, help="walk-forward folds (0 to skip)")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--reverse-split", action="store_true",
                   help="train on the MOST RECENT bars and hold out the oldest, "
                        "with the walk-forward walking backwards too. Fits the "
                        "current regime; the result is not a forecast, because "
                        "the training data postdates every trade it is scored on.")
    p.add_argument("--es-min-prob", type=float, default=0.40)
    p.add_argument("--es-edge-mult", type=float, default=0.5)

    #--- deployment settings: defaults come from mintyai_deploy.json if
    #--- present, so a retrain cannot quietly reset them.
    p.add_argument("--risk-pcts", type=str,
                   default=",".join(str(x) for x in dep.get("risk_pcts", DEFAULT_RISK_PCTS)),
                   help="the three per-trade caps behind Low,Medium,High (percent of equity)")
    p.add_argument("--kelly-scale", type=float, default=dep.get("kelly_scale", KELLY_SCALE),
                   help="fraction of full Kelly to stake (0.5 = half Kelly)")
    p.add_argument("--risk-from-kelly", dest="risk_from_kelly",
                   action="store_true",
                   default=bool(dep.get("risk_from_kelly", True)),
                   help="derive the three caps from the measured f* instead of "
                        "using fixed --risk-pcts (default on)")
    p.add_argument("--fixed-risk-pcts", dest="risk_from_kelly",
                   action="store_false",
                   help="use --risk-pcts verbatim, no derivation")
    p.add_argument("--kelly-multipliers",
                   default=",".join(str(x) for x in dep.get(
                       "kelly_multipliers", [0.5, 1.0, 2.0])),
                   help="Low,Medium,High as multiples of f* (default 0.5,1,2)")

    p.add_argument("--search-attempts", type=int, default=1,
                   help="retrain with fresh seeds until walk-forward clears "
                        "--target-tstat, up to this many times. Above 1 this "
                        "SELECTS on the walk-forward, which inflates it - the "
                        "holdout t becomes the number to trust.")
    p.add_argument("--target-tstat", type=float, default=2.0,
                   help="walk-forward t a search is trying to reach")
    p.add_argument("--out-name", default=None, help="model basename (default from symbol/TF)")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--no-merge", action="store_true",
                   help="write a config holding ONLY the symbol trained in this "
                        "run. By default every other symbol already trained is "
                        "carried forward, so training one does not drop the rest "
                        "out of the EA.")
    return p


def parse_dilations(spec) -> tuple:
    vals = tuple(int(x) for x in str(spec).split(",") if str(x).strip() != "")
    if not vals:
        raise SystemExit(f"--dilations needs at least one value, got '{spec}'")
    return vals


def receptive_field(dilations, kernel: int = 3) -> int:
    return 1 + (kernel - 1) * sum(dilations)


def parse_multipliers(spec) -> list:
    vals = [float(x) for x in str(spec).split(",") if str(x).strip() != ""]
    if len(vals) != 3:
        raise SystemExit(
            f"--kelly-multipliers needs exactly three values, got '{spec}'")
    return vals


def parse_risk_pcts(spec: str) -> list:
    vals = [float(x) for x in str(spec).split(",") if str(x).strip() != ""]
    if len(vals) != 3:
        raise SystemExit(
            f"--risk-pcts needs exactly three values (Low,Medium,High), got '{spec}'"
        )
    return vals


# --------------------------------------------------------------------------
# one symbol, start to finish
# --------------------------------------------------------------------------
def finalise_export(model, ds, args, device, base, report, val_idx, train_idx):
    """Write the ONNX graph, weights and report for a chosen model."""
    onnx_path = os.path.join(FILES_DIR, base + ".onnx")
    export_onnx(model, onnx_path, args.window, device)
    worst = verify_onnx(model, onnx_path, ds,
                        val_idx if len(val_idx) else train_idx, args.window)
    print(f"[mintyai] ONNX exported -> {os.path.basename(onnx_path)}  "
          f"(max abs diff vs PyTorch {worst:.2e})")
    if worst > 1e-3:
        raise SystemExit(
            "ONNX Runtime disagrees with PyTorch by more than 1e-3. "
            "The EA would trade a different model than the one evaluated above."
        )
    torch.save({"state_dict": model.state_dict(), "args": vars(args)},
               os.path.join(FILES_DIR, base + ".pt"))
    with open(os.path.join(FILES_DIR, base + "_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def strip_internals(entry: dict) -> dict:
    """Drop the objects a search needs but the config generator must not see."""
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def run_pipeline(args, device: str, risk_pcts: list, export: bool = True) -> dict:
    """Train one symbol and return the row the config table needs."""
    bars = D.load_bars(args.bars)
    print(f"[mintyai] loaded {len(bars)} bars from {os.path.basename(args.bars)}")

    sig_meta = bars.meta.get("feature_signature")
    if sig_meta is not None and int(sig_meta) != F.feature_signature():
        raise SystemExit(
            f"feature signature mismatch: export says {sig_meta}, "
            f"features.py computes {F.feature_signature()}. "
            "Re-run MintyAIExport.mq5 after syncing the feature engines."
        )

    ds = D.build_dataset(
        bars,
        window=args.window,
        horizon=args.horizon,
        barrier_atr=args.barrier_atr,
        commission_per_lot=args.commission_per_lot,
        slippage_points=args.slippage_points,
        spread_override_points=args.spread_points,
    )

    counts = np.bincount(ds.labels[ds.eligible], minlength=3)
    total = counts.sum()
    print(
        f"[mintyai] {total} usable samples  "
        f"down {counts[0] / total:.1%}  flat {counts[1] / total:.1%}  up {counts[2] / total:.1%}"
    )
    if total < 20000:
        print("[mintyai] WARNING fewer than 20k samples; expect an unreliable estimate")

    feats_gpu = torch.as_tensor(ds.feats, dtype=torch.float32, device=device)
    embargo = args.window + args.horizon

    symbol = ds.meta.get("symbol", "?")
    timeframe = ds.meta.get("timeframe", "?")
    report = {
        "symbol": symbol, "timeframe": timeframe,
        "bars": len(bars), "samples": int(total), "args": vars(args),
    }

    # ---------------- walk-forward: the honest out-of-sample number ----
    wf = {"trades": 0, "total_r": 0.0, "tstat": 0.0}
    if args.folds > 0:
        print(f"\n[mintyai] ==== walk-forward evaluation ({args.folds} folds) ====")
        all_r, fold_rows = [], []
        for k, tr, te in D.walk_forward_folds(ds, args.folds, embargo,
                                              reverse=args.reverse_split):
            cut = int(len(tr) * 0.85)
            tr_in, tr_val = tr[:cut], tr[cut:]
            keep = ds.exit_idx[tr_in] < (tr_val[0] - embargo)
            tr_in = tr_in[keep]

            t0 = time.time()
            model, _ = train_model(ds, feats_gpu, tr_in, tr_val, args, device,
                                   log_prefix=f"  [fold {k}] ")
            model.set_temperature(calibrate_temperature(model, feats_gpu, ds, tr_val, args.window))

            val_probs = predict_probs(model, feats_gpu, tr_val, args.window, use_amp=args.amp)
            mp, em, _ = search_gate(val_probs, ds, tr_val, args.min_trades)

            te_probs = predict_probs(model, feats_gpu, te, args.window, use_amp=args.amp)
            m = gate_pnl(te_probs, ds, te, mp, em)
            fold_rows.append({"fold": k, "train": len(tr_in), "test": len(te),
                              "min_prob": mp, "edge_mult": em, **m})
            print(
                f"  [fold {k}] OOS trades {m['trades']:5d}  R {m['total_r']:8.1f}  "
                f"mean {m['mean_r']:+.4f}  t {m['tstat']:5.2f}  hit {m['hit_rate']:.3f}  "
                f"({time.time() - t0:.0f}s)"
            )
            all_r.append(m)

        tot_trades = sum(m["trades"] for m in all_r)
        tot_r = sum(m["total_r"] for m in all_r)
        pooled_t = (
            sum(m["tstat"] * math.sqrt(max(m["trades"], 1)) for m in all_r)
            / math.sqrt(max(tot_trades, 1))
            if tot_trades else 0.0
        )
        wf = {"trades": tot_trades, "total_r": tot_r, "tstat": pooled_t}
        report["walk_forward"] = {"folds": fold_rows, **wf}
        print(
            f"[mintyai] walk-forward total: {tot_trades} trades, {tot_r:+.1f} R, "
            f"pooled t {pooled_t:.2f}"
        )
        print("[mintyai] verdict: " + (
            "edge is statistically meaningful" if pooled_t > 2.0
            else "NOT distinguishable from noise - do not trade this live"
        ))

    # ---------------- deployable model --------------------------------
    print("\n[mintyai] ==== training deployable model ====")
    train_idx, val_idx, test_idx = D.purged_split(
        ds, args.train_frac, args.val_frac, embargo, reverse=args.reverse_split)
    print(f"[mintyai] train {len(train_idx)}  val {len(val_idx)}  "
          f"holdout {len(test_idx)}"
          + ("   [REVERSED: training on the newest bars, holdout is the oldest]"
             if args.reverse_split else ""))
    def _span(ix):
        if not len(ix):
            return "empty"
        return (time.strftime("%Y-%m-%d", time.gmtime(int(ds.time[ix[0]]))) + " .. "
                + time.strftime("%Y-%m-%d", time.gmtime(int(ds.time[ix[-1]]))))
    print(f"[mintyai]   train   {_span(train_idx)}")
    print(f"[mintyai]   val     {_span(val_idx)}")
    print(f"[mintyai]   holdout {_span(test_idx)}")

    model, info = train_model(ds, feats_gpu, train_idx, val_idx, args, device)
    dil = parse_dilations(getattr(args, "dilations", "1,2,4,8,16"))
    rf = receptive_field(dil)
    print(f"[mintyai] parameters: {count_parameters(model):,}  "
          f"(channels {args.channels}, {len(dil)} blocks)")
    print(f"[mintyai] receptive field: {rf} bars of the {args.window}-bar window"
          + ("" if rf >= args.window else
             f"  <- the convolutions cannot see the oldest "
             f"{args.window - rf} bars; only the mean-pool does"))

    temp = calibrate_temperature(model, feats_gpu, ds, val_idx, args.window)
    model.set_temperature(temp)
    print(f"[mintyai] calibration temperature: {temp:.4f}")

    val_probs = predict_probs(model, feats_gpu, val_idx, args.window, use_amp=args.amp)
    min_prob, edge_mult, val_m = search_gate(val_probs, ds, val_idx, args.min_trades)
    print(f"[mintyai] gate: min_prob {min_prob:.2f}  edge_mult {edge_mult:.2f}")

    val_risk = predict_risk(model, feats_gpu, val_idx, args.window, use_amp=args.amp)

    # ---- what the three presets should be, for THIS model ------------
    #
    # Derived per symbol rather than shared: the growth-optimal stake is a
    # property of the edge, and the edge differs by instrument. Preference
    # goes to the walk-forward numbers - they are out-of-sample and by far
    # the larger sample. Validation is the fallback, and over-betting is
    # punished much harder than under-betting, so the smaller of the two
    # wins when both exist.
    caps = list(risk_pcts)
    if args.risk_from_kelly:
        f_wf = kelly_star(wf)
        f_val = kelly_star(val_m)
        candidates = [f for f in (f_wf, f_val) if f]
        if candidates:
            f_star = min(candidates)
            caps = kelly_presets(f_star, parse_multipliers(args.kelly_multipliers))
            src = ("walk-forward" if f_wf and f_star == f_wf else "validation")
            print(f"[mintyai] Kelly: f* = {f_star * 100:.2f}% per trade (from "
                  f"{src}); growth turns NEGATIVE beyond {f_star * 200:.2f}%")
            print(f"[mintyai]        presets derived -> " + ", ".join(
                f"{n} {c:g}%" for n, c in zip(RISK_NAMES, caps)))
            if f_wf and f_val:
                print(f"[mintyai]        (walk-forward said {f_wf * 100:.2f}%, "
                      f"validation {f_val * 100:.2f}% - took the smaller)")
        else:
            print("[mintyai] Kelly: no positive edge to size against, so there "
                  "is no growth-optimal stake. Falling back to " + ", ".join(
                      f"{n} {c:g}%" for n, c in zip(RISK_NAMES, caps)) +
                  " - but a stake on a model with no edge only sets the speed "
                  "of the loss.")

    risk_rows = []
    for name, cap in zip(RISK_NAMES, caps):
        lim = learn_risk_limits(val_probs, val_risk, ds, val_idx, min_prob, edge_mult,
                                kelly_scale=args.kelly_scale, force_cap_pct=cap)
        risk_rows.append(lim)
        print(
            f"[mintyai] risk {name:<6} cap {lim['max_risk_pct']:6.2f}%  "
            f"mean {lim['mean_risk_pct']:5.2f}%  validation drawdown "
            f"{lim['sim_dd_pct']:5.1f}%  halts {lim['daily_loss_pct']:.1f}%/"
            f"{lim['max_dd_pct']:.1f}%"
        )
        if lim["sim_dd_pct"] > 50.0:
            print(
                f"[mintyai]   WARNING at {name} this model lost "
                f"{lim['sim_dd_pct']:.0f}% of the account at the worst point of the "
                "VALIDATION period - the stretch it looks best on."
            )

    hour_mask = learn_hour_mask(val_probs, ds, val_idx, min_prob, edge_mult)
    hours_on = [h for h in range(24) if hour_mask & (1 << h)]
    print(f"[mintyai] trading hours ({len(hours_on)}/24): {hours_on}")

    if len(test_idx):
        test_probs = predict_probs(model, feats_gpu, test_idx, args.window, use_amp=args.amp)
        test_m = gate_pnl(test_probs, ds, test_idx, min_prob, edge_mult)
        print(f"[mintyai] untouched holdout: trades {test_m['trades']}  "
              f"R {test_m['total_r']:+.1f}  t {test_m['tstat']:.2f}")
        report["holdout"] = test_m

    report.update({
        "temperature": temp, "min_prob": min_prob, "edge_mult": edge_mult,
        "validation": val_m, "best_epoch": info["best_epoch"],
        "parameters": count_parameters(model),
        "risk_levels": {n: r for n, r in zip(RISK_NAMES, risk_rows)},
        "hour_mask": hour_mask,
    })

    base = args.out_name or f"mintyai_{symbol}_{timeframe}"

    if export:
        finalise_export(model, ds, args, device, base, report, val_idx, train_idx)

    tf_bars = ds.time
    return {
        # kept so a search can export the winner without retraining it
        "_model": model, "_ds": ds, "_report": report,
        "_val_idx": val_idx, "_train_idx": train_idx, "_base": base,
        "_holdout_tstat": round(float(report.get("holdout", {}).get("tstat", 0.0)), 3),
        "_seed": args.seed,
        "symbol": symbol,
        "utc_offset_h": int(float(bars.meta.get("server_utc_offset_h", 0) or 0)),
        "timeframe": timeframe,
        "file": base + ".onnx",
        "n_bars": len(bars),
        "date_from": time.strftime("%Y-%m-%d", time.gmtime(int(tf_bars[0]))),
        "date_to": time.strftime("%Y-%m-%d", time.gmtime(int(tf_bars[-1]))),
        "min_prob": min_prob,
        "edge_mult": edge_mult,
        "kelly_scale": args.kelly_scale,
        "hour_mask": hour_mask,
        "risk_pcts": caps,
        "wf_trades": wf["trades"],
        "wf_total_r": round(wf["total_r"], 2),
        "wf_tstat": round(wf["tstat"], 3),
        "risk": risk_rows,
    }


def search_until_signal(args, device: str, risk_pcts: list,
                        attempts: int, target: float) -> dict:
    """Retrain with fresh seeds until walk-forward clears `target`.

    Read this before trusting the number it produces.

    Retrying until a test passes turns that test from a measurement into a
    selection criterion. Under the null - a model with no edge at all - a
    single walk-forward has roughly a 2.3% chance of showing t >= 2 by luck.
    Draw enough models and one of them will, and the t you keep is then a
    record of how many times you rolled, not of any edge.

    So the search reports two numbers. The walk-forward t is the one it
    selected on, and is inflated by construction. The HOLDOUT t is measured
    on the final slice, which no fold and no selection ever touched, and is
    the one that means something. If the winner clears walk-forward but its
    holdout is flat, the search found noise and said so.
    """
    p_single = 0.0228           # one-sided P(t >= 2) under the null, large df
    tries, best = [], None

    for k in range(attempts):
        a = argparse.Namespace(**vars(args))
        a.seed = args.seed + k * 1009      # arbitrary stride, distinct streams
        print(f"\n[mintyai] ---- search attempt {k + 1}/{attempts} (seed {a.seed}) ----")

        entry = run_pipeline(a, device, risk_pcts, export=False)
        t_wf = entry["wf_tstat"]
        t_ho = entry["_holdout_tstat"]
        tries.append((a.seed, t_wf, t_ho))

        if best is None or t_wf > best["wf_tstat"]:
            best = entry

        print(f"[mintyai] attempt {k + 1}: walk-forward t={t_wf:+.2f}  "
              f"holdout t={t_ho:+.2f}  (best so far {best['wf_tstat']:+.2f})")

        if t_wf >= target:
            print(f"[mintyai] cleared t >= {target} on attempt {k + 1}.")
            break

    n = len(tries)
    p_any = 1.0 - (1.0 - p_single) ** n

    print(f"\n[mintyai] ==== search summary: {n} attempt(s) ====")
    print(f"           {'seed':>8} {'walk-forward t':>15} {'holdout t':>11}")
    for seed, t_wf, t_ho in tries:
        print(f"           {seed:>8} {t_wf:>15.2f} {t_ho:>11.2f}")

    if best["wf_tstat"] >= target:
        print(f"[mintyai] kept seed {best['_seed']}: walk-forward t="
              f"{best['wf_tstat']:+.2f}, holdout t={best['_holdout_tstat']:+.2f}")
        print(f"[mintyai] SELECTION WARNING: over {n} attempts, a model with NO "
              f"edge had a {p_any:.0%} chance of producing t >= {target} by luck "
              "alone.")
        print("[mintyai] The walk-forward t above was selected on and is inflated. "
              "The holdout t was not - judge the model by that one.")
        if best["_holdout_tstat"] < 1.0:
            print("[mintyai] Its holdout is flat, which is what a lucky draw looks "
                  "like. Treat this model as unproven.")
    else:
        print(f"[mintyai] no attempt reached t >= {target}. Keeping the best "
              f"(seed {best['_seed']}, t={best['wf_tstat']:+.2f}).")
        print("[mintyai] Failing to find one in this many tries is itself "
              "informative: the edge is not there to be found by reseeding.")

    finalise_export(best["_model"], best["_ds"], args, device, best["_base"],
                    best["_report"], best["_val_idx"], best["_train_idx"])
    return best


def shared_config(args, risk_pcts: list, timeframe: str,
                  utc_offset_h: int = 0) -> dict:
    return {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_version": F.FEATURE_VERSION,
        "feature_count": F.FEATURE_COUNT,
        "signature": F.feature_signature(),
        "window": args.window,
        "timeframe": timeframe,
        "barrier_atr": args.barrier_atr,
        "horizon": args.horizon,
        "slippage_points": args.slippage_points,
        "risk_pcts": risk_pcts,
        "utc_offset_h": utc_offset_h,
    }


def pick_device(force_cpu: bool = False) -> str:
    if force_cpu:
        print("[mintyai] device: CPU (--cpu). This will be much slower than the "
              "GPU path.")
        return "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[mintyai] WARNING no CUDA device found - training on CPU will be slow")
    else:
        print(f"[mintyai] device: {torch.cuda.get_device_name(0)}")
    return device


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.cpu:
        args.amp = False          # bfloat16 autocast is a CUDA path
    risk_pcts = parse_risk_pcts(args.risk_pcts)

    if os.path.exists(DEPLOY_PATH):
        print(f"[mintyai] deployment settings from {os.path.basename(DEPLOY_PATH)}: "
              f"{load_deploy_defaults()}")
    print(f"[mintyai] risk presets: " + ", ".join(
        f"{n} {p:g}%" for n, p in zip(RISK_NAMES, risk_pcts)))

    device = pick_device(args.cpu)
    if args.search_attempts > 1:
        entry = search_until_signal(args, device, risk_pcts,
                                    args.search_attempts, args.target_tstat)
    else:
        entry = run_pipeline(args, device, risk_pcts)

    if args.skip_export:
        print("[mintyai] --skip-export set; config not written")
        return 0

    write_config(shared_config(args, risk_pcts, entry["timeframe"],
                               entry.get("utc_offset_h", 0)),
                 [strip_internals(entry)], merge=not args.no_merge)
    print("[mintyai] done. Recompile MintyAI.mq5 so it picks up the new config.")
    if args.no_merge:
        print("[mintyai] NOTE --no-merge wrote a SINGLE-symbol config. Every other "
              "symbol has been dropped from the EA until you retrain or run "
              "'python -m mintyai.rebuild'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
