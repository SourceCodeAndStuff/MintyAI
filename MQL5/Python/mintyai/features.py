"""Exact Python mirror of Include/MintyAI/MintyAIFeatures.mqh.

Every function here is a literal transcription of its MQL5 counterpart,
including the seeding of the recursive filters and the partial-window
behaviour of the rolling statistics. It is deliberately not idiomatic
pandas: the goal is bit-comparable output, not brevity.

If you change a formula here, change it in MintyAIFeatures.mqh and bump
FEATURE_VERSION in both files.
"""

from __future__ import annotations

import numpy as np

# v1: timestamps are the broker's SERVER time, unconverted, on both sides.
# Reverted from the UTC version to match the 2.0 build. A model trained
# here is only valid on a broker in the same timezone - retrain per broker.
FEATURE_VERSION = 1
FEATURE_COUNT = 24
# Must match MINTYAI_WARMUP in MintyAIFeatures.mqh - see the comment there for
# why it is sized off EMA200's seed decay rather than picked by feel.
WARMUP = 2000

FEATURE_NAMES = [
    "ret_atr",
    "range_atr",
    "body_ratio",
    "upper_wick",
    "lower_wick",
    "gap_atr",
    "vol_rel",
    "rsi14",
    "macd_hist_atr",
    "d_ema20_atr",
    "d_ema50_atr",
    "d_ema200_atr",
    "atr_pct",
    "vol_ratio",
    "bb_pctb",
    "stoch_k",
    "adx14",
    "mom5_atr",
    "mom20_atr",
    "hl_pos50",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
]

EPS = 1.0e-12
CLIP = 10.0


def fnv1a(s: str) -> int:
    """32-bit FNV-1a over UTF-8 bytes. Mirrors MintyAIFnv1a()."""
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def feature_signature() -> int:
    """Mirrors MintyAIFeatureSignature()."""
    s = "v%d" % FEATURE_VERSION
    for name in FEATURE_NAMES:
        s += "|" + name
    return fnv1a(s)


# --------------------------------------------------------------------------
# primitive series helpers (chronological arrays, index 0 = oldest)
# --------------------------------------------------------------------------


def ema(src: np.ndarray, period: int) -> np.ndarray:
    n = src.shape[0]
    dst = np.empty(n, dtype=np.float64)
    if n == 0:
        return dst
    alpha = 2.0 / (period + 1.0)
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = alpha * src[i] + (1.0 - alpha) * dst[i - 1]
    return dst


def rma(src: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing, seeded with the first sample."""
    n = src.shape[0]
    dst = np.empty(n, dtype=np.float64)
    if n == 0:
        return dst
    alpha = 1.0 / period
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = alpha * src[i] + (1.0 - alpha) * dst[i - 1]
    return dst


def sma(src: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean using the same running-sum recursion as MQL5."""
    n = src.shape[0]
    dst = np.empty(n, dtype=np.float64)
    total = 0.0
    for i in range(n):
        total += src[i]
        if i >= period:
            total -= src[i - period]
        cnt = i + 1 if i + 1 < period else period
        dst[i] = total / cnt
    return dst


def stdev(src: np.ndarray, period: int) -> np.ndarray:
    """Population stdev via the same sum/sum-of-squares recursion as MQL5."""
    n = src.shape[0]
    dst = np.empty(n, dtype=np.float64)
    s = 0.0
    s2 = 0.0
    for i in range(n):
        s += src[i]
        s2 += src[i] * src[i]
        if i >= period:
            s -= src[i - period]
            s2 -= src[i - period] * src[i - period]
        cnt = i + 1 if i + 1 < period else period
        mean = s / cnt
        var = s2 / cnt - mean * mean
        dst[i] = np.sqrt(var) if var > 0.0 else 0.0
    return dst


def roll_max(src: np.ndarray, period: int) -> np.ndarray:
    n = src.shape[0]
    dst = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        dst[i] = src[start : i + 1].max()
    return dst


def roll_min(src: np.ndarray, period: int) -> np.ndarray:
    n = src.shape[0]
    dst = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        dst[i] = src[start : i + 1].min()
    return dst


def clean(x: np.ndarray) -> np.ndarray:
    """Mirrors MintyAIClean(): scrub non-finite values, then clamp."""
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, -CLIP, CLIP)


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = h.shape[0]
    tr = np.empty(n, dtype=np.float64)
    if n:
        tr[0] = h[0] - l[0]
    if n > 1:
        a = h[1:] - l[1:]
        b = np.abs(h[1:] - c[:-1])
        d = np.abs(l[1:] - c[:-1])
        tr[1:] = np.maximum(a, np.maximum(b, d))
    return tr


def atr(h, l, c, period=14) -> np.ndarray:
    return rma(true_range(h, l, c), period)


def rsi(c: np.ndarray, period=14) -> np.ndarray:
    n = c.shape[0]
    up = np.zeros(n, dtype=np.float64)
    dn = np.zeros(n, dtype=np.float64)
    if n > 1:
        d = c[1:] - c[:-1]
        up[1:] = np.where(d > 0.0, d, 0.0)
        dn[1:] = np.where(d < 0.0, -d, 0.0)
    rup = rma(up, period)
    rdn = rma(dn, period)
    denom = rup + rdn
    return np.where(denom > EPS, 100.0 * rup / np.maximum(denom, EPS), 50.0)


def adx(h, l, c, period=14) -> np.ndarray:
    n = h.shape[0]
    pdm = np.zeros(n, dtype=np.float64)
    ndm = np.zeros(n, dtype=np.float64)
    tr = true_range(h, l, c)
    if n > 1:
        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]
        pdm[1:] = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        ndm[1:] = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    spdm = rma(pdm, period)
    sndm = rma(ndm, period)
    stro = rma(tr, period)
    denom_tr = np.maximum(stro, EPS)
    pdi = 100.0 * spdm / denom_tr
    ndi = 100.0 * sndm / denom_tr
    s = pdi + ndi
    dx = np.where(s > EPS, 100.0 * np.abs(pdi - ndi) / np.maximum(s, EPS), 0.0)
    return rma(dx, period)


# --------------------------------------------------------------------------
# feature matrix
# --------------------------------------------------------------------------


def build_features(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    vol: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    """Mirrors MintyAIBuildFeatures(). Returns float32 [n, FEATURE_COUNT]."""
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)
    n = c.shape[0]
    if n == 0:
        return np.zeros((0, FEATURE_COUNT), dtype=np.float32)

    a_atr = atr(h, l, c, 14)
    v_rsi = rsi(c, 14)
    v_adx = adx(h, l, c, 14)

    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    macd = ema12 - ema26
    signal = ema(macd, 9)

    vol_sma = sma(vol, 50)

    ret = np.zeros(n, dtype=np.float64)
    if n > 1:
        prev = np.maximum(c[:-1], EPS)
        ret[1:] = np.log(c[1:] / prev)
    ret2 = ret * ret
    rv20 = np.sqrt(sma(ret2, 20))
    rv100 = np.sqrt(sma(ret2, 100))

    sma20 = sma(c, 20)
    std20 = stdev(c, 20)
    hh14 = roll_max(h, 14)
    ll14 = roll_min(l, 14)
    hh50 = roll_max(h, 50)
    ll50 = roll_min(l, 50)

    a = np.maximum(a_atr, EPS)
    rng = h - l
    rden = np.maximum(rng, EPS)
    px = np.maximum(c, EPS)

    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = c[0]
    if n > 1:
        prev_close[1:] = c[:-1]

    c5 = np.empty(n, dtype=np.float64)
    c5[:5] = c[0]
    if n > 5:
        c5[5:] = c[:-5]

    c20 = np.empty(n, dtype=np.float64)
    c20[:20] = c[0]
    if n > 20:
        c20[20:] = c[:-20]

    bb_up = sma20 + 2.0 * std20
    bb_lo = sma20 - 2.0 * std20
    bb_den = np.maximum(bb_up - bb_lo, EPS)
    st_den = np.maximum(hh14 - ll14, EPS)
    ch_den = np.maximum(hh50 - ll50, EPS)

    # calendar features from the raw epoch seconds, identical integer maths
    hod = (ts % 86400).astype(np.float64) / 3600.0
    days = ts // 86400
    dow = ((days + 4) % 7).astype(np.float64)

    v_rel = np.log((vol + 1.0) / (vol_sma + 1.0))

    out = np.empty((n, FEATURE_COUNT), dtype=np.float64)
    out[:, 0] = (c - prev_close) / a
    out[:, 1] = rng / a
    out[:, 2] = (c - o) / rden
    out[:, 3] = (h - np.maximum(o, c)) / rden
    out[:, 4] = (np.minimum(o, c) - l) / rden
    out[:, 5] = (o - prev_close) / a
    out[:, 6] = v_rel
    out[:, 7] = v_rsi / 100.0 - 0.5
    out[:, 8] = (macd - signal) / a
    out[:, 9] = (c - ema20) / a
    out[:, 10] = (c - ema50) / a
    out[:, 11] = (c - ema200) / a
    out[:, 12] = a / px * 100.0
    out[:, 13] = rv20 / (rv100 + EPS)
    out[:, 14] = (c - bb_lo) / bb_den - 0.5
    out[:, 15] = (c - ll14) / st_den - 0.5
    out[:, 16] = v_adx / 100.0
    out[:, 17] = (c - c5) / a
    out[:, 18] = (c - c20) / a
    out[:, 19] = (c - ll50) / ch_den - 0.5
    out[:, 20] = np.sin(2.0 * np.pi * hod / 24.0)
    out[:, 21] = np.cos(2.0 * np.pi * hod / 24.0)
    out[:, 22] = np.sin(2.0 * np.pi * dow / 7.0)
    out[:, 23] = np.cos(2.0 * np.pi * dow / 7.0)

    return clean(out).astype(np.float32)
