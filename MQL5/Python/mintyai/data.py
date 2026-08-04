"""Dataset construction: loading exported bars, triple-barrier labelling,
window views and leakage-safe chronological splits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features as F

# label encoding
LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2
LABEL_AMBIGUOUS = -1  # both barriers touched in the same bar - unknowable order
LABEL_NO_HORIZON = -2  # not enough future bars


@dataclass
class Bars:
    time: np.ndarray  # epoch seconds, int64
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray
    spread_points: np.ndarray
    meta: dict

    def __len__(self) -> int:
        return self.close.shape[0]


def load_bars(bars_csv: str) -> Bars:
    """Load a *_bars.csv produced by MintyAIExport.mq5."""
    df = pd.read_csv(bars_csv)
    required = {"time", "open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{bars_csv} is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.sort_values("time").drop_duplicates(subset="time", keep="last")

    meta = {}
    meta_csv = bars_csv.replace("_bars.csv", "_meta.csv")
    if os.path.exists(meta_csv):
        mdf = pd.read_csv(meta_csv)
        meta = dict(zip(mdf["key"].astype(str), mdf["value"].astype(str)))

    spread = (
        df["spread"].to_numpy(dtype=np.float64)
        if "spread" in df.columns
        else np.zeros(len(df), dtype=np.float64)
    )

    return Bars(
        time=df["time"].to_numpy(dtype=np.int64),
        open=df["open"].to_numpy(dtype=np.float64),
        high=df["high"].to_numpy(dtype=np.float64),
        low=df["low"].to_numpy(dtype=np.float64),
        close=df["close"].to_numpy(dtype=np.float64),
        tick_volume=df["tick_volume"].to_numpy(dtype=np.float64),
        spread_points=spread,
        meta=meta,
    )


def triple_barrier_labels(
    bars: Bars,
    atr_series: np.ndarray,
    barrier_atr: float,
    horizon: int,
):
    """Label each bar by which ATR-scaled barrier the market touches first.

    The decision is taken at the close of bar i. Barriers sit at
    close[i] +/- barrier_atr * ATR[i] and are checked against the highs and
    lows of bars i+1 .. i+horizon.

    Returns (labels, exit_idx, exit_price, barrier_size).
    Bars where both barriers fall inside the same future bar are marked
    LABEL_AMBIGUOUS and excluded from training: the order of touch is not
    recoverable from OHLC data, and guessing it is how backtests start
    lying.
    """
    n = len(bars)
    close = bars.close
    high = bars.high
    low = bars.low

    barrier = barrier_atr * atr_series
    upper = close + barrier
    lower = close - barrier

    labels = np.full(n, LABEL_FLAT, dtype=np.int64)
    exit_idx = np.full(n, -1, dtype=np.int64)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    resolved = np.zeros(n, dtype=bool)

    idx = np.arange(n)
    for j in range(1, horizon + 1):
        fwd = idx + j
        valid = (fwd < n) & ~resolved
        if not valid.any():
            break
        f = np.where(valid, fwd, 0)  # safe gather; masked out below

        touch_up = valid & (high[f] >= upper)
        touch_dn = valid & (low[f] <= lower)

        both = touch_up & touch_dn
        only_up = touch_up & ~touch_dn
        only_dn = touch_dn & ~touch_up

        labels[both] = LABEL_AMBIGUOUS
        exit_idx[both] = fwd[both]
        exit_price[both] = close[f][both]

        labels[only_up] = LABEL_UP
        exit_idx[only_up] = fwd[only_up]
        exit_price[only_up] = upper[only_up]

        labels[only_dn] = LABEL_DOWN
        exit_idx[only_dn] = fwd[only_dn]
        exit_price[only_dn] = lower[only_dn]

        resolved |= both | only_up | only_dn

    # unresolved within the horizon -> flat, exit at the horizon close
    flat = ~resolved
    horizon_idx = idx + horizon
    has_horizon = horizon_idx < n
    ok_flat = flat & has_horizon
    labels[ok_flat] = LABEL_FLAT
    exit_idx[ok_flat] = horizon_idx[ok_flat]
    exit_price[ok_flat] = close[np.where(ok_flat, horizon_idx, 0)][ok_flat]

    # not enough future data to know the outcome at all
    labels[flat & ~has_horizon] = LABEL_NO_HORIZON

    return labels, exit_idx, exit_price, barrier


@dataclass
class Dataset:
    feats: np.ndarray  # [n, F] float32, chronological
    windows: np.ndarray  # strided view [n-W+1, F, W]
    labels: np.ndarray  # [n] int64
    exit_idx: np.ndarray  # [n] int64, last future bar the label depends on
    exit_price: np.ndarray  # [n] float64, price where the position would close
    barrier: np.ndarray  # [n] float64, ATR-scaled barrier in price units
    close: np.ndarray
    time: np.ndarray
    eligible: np.ndarray  # int64 bar indices usable as training samples
    window: int
    horizon: int
    cost: np.ndarray  # [n] round-trip cost in price units
    meta: dict
    symbol_id: np.ndarray = None  # [n] int64, index into meta["symbols"]

    def batch(self, bar_indices: np.ndarray) -> np.ndarray:
        """Materialise [B, F, W] windows for the given bar indices."""
        return np.ascontiguousarray(self.windows[bar_indices - self.window + 1])

    @property
    def symbols(self) -> list:
        return self.meta.get("symbols", [self.meta.get("symbol", "?")])


def build_dataset(
    bars: Bars,
    window: int,
    horizon: int,
    barrier_atr: float,
    commission_per_lot: float = 0.0,
    slippage_points: float = 0.0,
    spread_override_points: float | None = None,
) -> Dataset:
    feats = F.build_features(
        bars.open, bars.high, bars.low, bars.close, bars.tick_volume, bars.time
    )
    atr_series = F.atr(bars.high, bars.low, bars.close, 14)

    labels, exit_idx, exit_price, barrier = triple_barrier_labels(
        bars, atr_series, barrier_atr, horizon
    )

    n = len(bars)
    point = float(bars.meta.get("point", 0.0) or 0.0)
    if point <= 0.0:
        # fall back to a sane guess from the price scale
        digits = int(float(bars.meta.get("digits", 5) or 5))
        point = 10.0 ** (-digits)

    if spread_override_points is not None:
        # an explicit override is a deliberate choice, including zero
        # (used to isolate whether a strategy has any edge before costs)
        spread_pts = np.full(n, float(spread_override_points))
    else:
        spread_pts = bars.spread_points.copy()
        # an empty spread column means the broker did not record it - do
        # not silently let the backtest trade for free
        if np.nanmedian(spread_pts) <= 0.0:
            spread_pts = np.full(n, 10.0)
            print(
                "[data] WARNING bar spread column is empty; assuming 10 points. "
                "Pass --spread-points to model your real cost."
            )
    cost = (spread_pts + slippage_points) * point

    # strided window view: windows[k] covers bars k .. k+window-1
    if n < window:
        raise ValueError(f"only {n} bars, need at least {window}")
    # zero-copy view - materialising this would cost gigabytes
    windows = np.lib.stride_tricks.sliding_window_view(feats, window, axis=0)

    first = max(F.WARMUP, window - 1)
    eligible = np.arange(first, n, dtype=np.int64)
    ok = (labels[eligible] >= 0) & np.isfinite(barrier[eligible]) & (barrier[eligible] > 0)
    eligible = eligible[ok]

    return Dataset(
        feats=feats,
        windows=windows,
        labels=labels,
        exit_idx=exit_idx,
        exit_price=exit_price,
        barrier=barrier,
        close=bars.close,
        time=bars.time,
        eligible=eligible,
        window=window,
        horizon=horizon,
        cost=cost,
        meta=bars.meta,
    )


def pool_datasets(parts: list, symbols: list) -> Dataset:
    """Lay several symbols end to end into one training set.

    One network trades every symbol, so it has to see every symbol during
    training. The features are all scale-free - returns, ratios, ATR
    multiples - so a bar of EURUSD and a bar of gold are directly
    comparable, which is what makes a shared model possible at all.

    Windows are gathered from a single concatenated feature array. A window
    spanning the seam between two symbols would be nonsense, so eligibility
    is offset per symbol: the first usable bar of a symbol is already at
    least `window` bars in, and those seam-crossing windows are never
    selected.
    """
    if not parts:
        raise ValueError("nothing to pool")

    offsets, total = [], 0
    for p in parts:
        offsets.append(total)
        total += p.feats.shape[0]

    feats = np.concatenate([p.feats for p in parts], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(feats, parts[0].window, axis=0)

    labels = np.concatenate([p.labels for p in parts])
    exit_price = np.concatenate([p.exit_price for p in parts])
    barrier = np.concatenate([p.barrier for p in parts])
    close = np.concatenate([p.close for p in parts])
    time = np.concatenate([p.time for p in parts])
    cost = np.concatenate([p.cost for p in parts])

    # exit_idx points at a bar, so it moves with its symbol's block
    exit_idx = np.concatenate([p.exit_idx + off for p, off in zip(parts, offsets)])
    eligible = np.concatenate([p.eligible + off for p, off in zip(parts, offsets)])

    symbol_id = np.zeros(total, dtype=np.int64)
    for i, (p, off) in enumerate(zip(parts, offsets)):
        symbol_id[off : off + p.feats.shape[0]] = i

    meta = dict(parts[0].meta)
    meta["symbols"] = list(symbols)
    meta["symbol"] = "+".join(symbols)

    return Dataset(
        feats=feats, windows=windows, labels=labels, exit_idx=exit_idx,
        exit_price=exit_price, barrier=barrier, close=close, time=time,
        eligible=np.sort(eligible), window=parts[0].window,
        horizon=parts[0].horizon, cost=cost, meta=meta, symbol_id=symbol_id,
    )


def bar_seconds(ds: Dataset) -> int:
    """Seconds per bar, read off the data rather than assumed."""
    t = ds.time[ds.eligible[:5000]]
    if t.size < 3:
        return 3600
    d = np.diff(np.sort(np.unique(t)))
    d = d[d > 0]
    return int(np.median(d)) if d.size else 3600


def purged_split_time(
    ds: Dataset, train_frac: float, val_frac: float, embargo: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split by CALENDAR TIME rather than by row order.

    With several symbols concatenated, row order is symbol-major: splitting
    on it would train on some symbols and validate on others, which
    measures transfer between instruments rather than performance in the
    future. Splitting on time keeps every symbol in every block and makes
    the boundary mean what it says.
    """
    el = ds.eligible
    t = ds.time[el]
    order = np.argsort(t, kind="stable")
    ts = t[order]

    t_train = ts[int(ts.size * train_frac)] if ts.size else 0
    t_val = ts[min(int(ts.size * (train_frac + val_frac)), ts.size - 1)] if ts.size else 0

    emb = embargo * bar_seconds(ds)
    exit_t = ds.time[np.minimum(ds.exit_idx[el], ds.time.shape[0] - 1)]

    train = el[(t < t_train) & (exit_t < t_train - emb)]
    val = el[(t >= t_train) & (t < t_val) & (exit_t < t_val - emb)]
    test = el[t >= t_val]
    return np.sort(train), np.sort(val), np.sort(test)


def walk_forward_folds_time(ds: Dataset, n_folds: int, embargo: int,
                            min_train_frac: float = 0.3):
    """Anchored walk-forward on calendar time - train on the past, test on
    the block of future immediately after it, for every symbol at once."""
    el = ds.eligible
    t = ds.time[el]
    ts = np.sort(t)
    if ts.size == 0:
        return

    emb = embargo * bar_seconds(ds)
    exit_t = ds.time[np.minimum(ds.exit_idx[el], ds.time.shape[0] - 1)]

    start = int(ts.size * min_train_frac)
    block = (ts.size - start) // n_folds
    if block <= 0:
        raise ValueError("not enough samples for the requested number of folds")

    for k in range(n_folds):
        tr_end_i = start + k * block
        te_end_i = tr_end_i + block if k < n_folds - 1 else ts.size - 1
        t_tr_end = ts[tr_end_i]
        t_te_end = ts[min(te_end_i, ts.size - 1)]

        train = el[(t < t_tr_end) & (exit_t < t_tr_end - emb)]
        test = el[(t >= t_tr_end) & (t < t_te_end)]
        if train.size == 0 or test.size == 0:
            continue
        yield k, np.sort(train), np.sort(test)


def purged_split(
    ds: Dataset, train_frac: float, val_frac: float, embargo: int,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/val/test split with purging and an embargo.

    Forward (default): train on the earliest bars, validate on the middle,
    hold out the most recent. Samples whose label depends on bars inside a
    later split are dropped, plus `embargo` bars either side, so no training
    sample shares future information with a validation or test sample.

    Reverse: train on the MOST RECENT bars and hold out the oldest. The
    purge has to flip with it. Forward, the danger is a training sample
    whose outcome resolves inside validation; reversed, it is a TEST sample
    - older - whose outcome resolves inside the training block, because the
    model has then already seen the bars that settle it.
    """
    el = ds.eligible
    n = el.shape[0]

    if not reverse:
        i_train = int(n * train_frac)
        i_val = int(n * (train_frac + val_frac))
        train, val, test = el[:i_train], el[i_train:i_val], el[i_val:]

        def purge_fwd(block, next_start_bar):
            if block.size == 0:
                return block
            return block[ds.exit_idx[block] < (next_start_bar - embargo)]

        if val.size:
            train = purge_fwd(train, val[0])
        if test.size:
            val = purge_fwd(val, test[0])
        return train, val, test

    #--- reversed: oldest -> holdout, middle -> validation, newest -> train
    test_frac = 1.0 - train_frac - val_frac
    i_test = int(n * test_frac)
    i_val = int(n * (test_frac + val_frac))
    test, val, train = el[:i_test], el[i_test:i_val], el[i_val:]

    def purge_back(block, later_start_bar):
        """Drop samples whose outcome resolves at or after `later_start_bar`."""
        if block.size == 0:
            return block
        return block[ds.exit_idx[block] < (later_start_bar - embargo)]

    if train.size:
        val = purge_back(val, train[0])
    if val.size:
        test = purge_back(test, val[0])
    elif train.size:
        test = purge_back(test, train[0])
    return train, val, test


def walk_forward_folds(
    ds: Dataset, n_folds: int, embargo: int, min_train_frac: float = 0.3,
    reverse: bool = False,
):
    """Anchored walk-forward: expanding train window, next block as test.

    Forward (default): train on the past, test on the future immediately
    after it. This is the only evaluation that reflects how the model will
    actually be used.

    Reverse: anchor at the NEWEST end and walk backwards - train on the most
    recent bars, test on the block just older. It answers a different
    question ("how would a model fitted to today's regime have done back
    then") and it is not a forecast, because the training data postdates
    every trade it is scored on.
    """
    el = ds.eligible
    n = el.shape[0]
    start = int(n * min_train_frac)
    block = (n - start) // n_folds
    if block <= 0:
        raise ValueError("not enough samples for the requested number of folds")

    for k in range(n_folds):
        if not reverse:
            tr_end = start + k * block
            te_end = tr_end + block if k < n_folds - 1 else n
            train, test = el[:tr_end], el[tr_end:te_end]
            if train.size == 0 or test.size == 0:
                continue
            # a training sample must not resolve inside the test block
            yield k, train[ds.exit_idx[train] < (test[0] - embargo)], test
        else:
            tr_start = n - start - k * block
            te_start = tr_start - block if k < n_folds - 1 else 0
            train, test = el[tr_start:], el[te_start:tr_start]
            if train.size == 0 or test.size == 0:
                continue
            # here it is the TEST sample that must not resolve inside train
            yield k, train, test[ds.exit_idx[test] < (train[0] - embargo)]
