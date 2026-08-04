# MintyAI

A MetaTrader 5 Expert Advisor that trades a GPU-trained neural network's
forecast of short-horizon price direction.

## Read this first

**No EA makes "solely profits", and this one does not either.** Anything
that claims otherwise is either curve-fitted, lying, or both. Markets are
close to efficient at the timescales an EA operates on, and the spread is
charged on every single trade whether you are right or wrong.

What MintyAI actually gives you:

- an honest measurement of whether a model has any edge on *your* symbol,
  *after* your real costs, measured out-of-sample
- machinery that refuses to trade when that edge is not there
- risk controls that keep a losing stretch survivable rather than fatal
- an EA with no strategy settings to tune, because a knob on the chart is
  a second opinion competing with the one that was actually validated

The pipeline prints a verdict after training. If it says
`NOT distinguishable from noise`, the correct response is to not trade it.
That verdict is the most valuable output in this repository.

## Layout

| Path | What it is |
|---|---|
| `Experts/MintyAI/MintyAI.mq5` | the EA: inference, expected-value gate, risk, execution |
| `Include/MintyAI/MintyAIFeatures.mqh` | feature engine (the contract) |
| `Include/MintyAI/MintyAIExporter.mqh` | history download + CSV export, shared by both scripts |
| `Include/MintyAI/MintyAIModelConfig.mqh` | **generated** by training; do not hand-edit |
| `Scripts/MintyAI/MintyAIExportAll.mq5` | downloads every Market Watch symbol |
| `Scripts/MintyAI/MintyAIExport.mq5` | the same, for one symbol |
| `Python/mintyai/features.py` | exact mirror of `MintyAIFeatures.mqh` |
| `Python/mintyai/data.py` | triple-barrier labelling, symbol pooling, time-based splits |
| `Python/mintyai/model.py` | dilated causal TCN, class head + outcome head |
| `Python/mintyai/train.py` | GPU training, calibration, ONNX export, config generation |
| `Python/mintyai/train_all.py` | **the entry point** — pools every exported symbol |
| `Python/mintyai/parity.py` | proves the two feature engines agree |
| `Python/mintyai_deploy.json` | risk presets that retraining must not reset |
| `Python/tools/make_synthetic.py` | synthetic data for smoke-testing |

## Setup

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r Python/requirements.txt
```

Verified on Python 3.14, torch 2.11.0+cu128, RTX 4070.

## Workflow

### 1. Export history

Put the instruments you care about in Market Watch, then run
`Scripts/MintyAI/MintyAIExportAll.mq5`. With its `Symbols` input blank it
exports **everything in Market Watch**, downloading history as it goes —
it waits for each series to finish syncing rather than writing whatever
happens to be cached.

Writes to `MQL5/Files/MintyAI/`, per symbol:
`<SYMBOL>_<TF>_bars.csv`, `_meta.csv`, `_features_mql5.csv`.

Two things to get right here:

- **Trim Market Watch first.** Every symbol left in it becomes training
  data and an entry in the EA's model table. Aim for 50k+ bars each, which
  on H1 means about a decade.
- **Symbol names are your broker's.** They are written into the model
  table as-is, which is why the EA needs no prefix/suffix setting to
  recognise them later.

`MintyAIExport.mq5` does the same for a single symbol if you only want one.

### 2. Prove feature parity

```bash
cd Python
python -m mintyai.parity --bars ../Files/MintyAI/EURUSD_H1_bars.csv
```

This must print `PASS` before anything else. A model trained on Python
features but fed MQL5 features at runtime produces confident, meaningless
predictions — and nothing else in the system will warn you. The check
compares all 24 features across every warmed-up bar.

### 3. Train

```bash
cd Python
python -m mintyai.train_all
```

That is the whole command. It reads `Files/MintyAI/`, pools every symbol it
finds, and trains **one network over all of them**. There is no symbol list
to keep in sync with the export.

**Why one network and not one per symbol.** Every feature is scale-free —
returns, ratios, ATR multiples — so a bar of EURUSD and a bar of gold are
directly comparable. Pooling gives the network millions of examples of what
a *setup* looks like instead of a few hundred thousand of what one
instrument looks like, and a shared model cannot overfit to a single
symbol's quirks the way twenty separate ones can. It also means the `.ex5`
holds one 425 KB graph rather than one per pair.

What stays per-symbol is the **calibration**: the entry gate, the tradable
hours and the stake, each measured on that symbol's own slice of
validation. Gold and EURGBP behave nothing alike even when one network
prices both.

Useful knobs:

| Flag | Default | Notes |
|---|---|---|
| `--symbols` | all | pool only these |
| `--skip` | none | exclude some |
| `--dry-run` | off | list what would be trained, then stop |
| `--barrier-atr` | 3.0 | stop *and* target, in ATR(14). See below. |
| `--horizon` | 24 | max bars held; also the model's forecast horizon |
| `--window` | 96 | bars of history fed to the network |
| `--folds` | 5 | walk-forward folds; 0 skips the honest evaluation |
| `--epochs` | 40 | early stopping usually triggers well before this |
| `--spread-points` | recorded | override the exported spread, in points |
| `--risk-pcts` | `2,5,10` | the three caps behind Low / Medium / High |
| `--kelly-scale` | 0.5 | fraction of full Kelly |

Set `--spread-points` only if you distrust what the export recorded.
Optimism there is the single easiest way to produce a backtest that cannot
be traded.

`python -m mintyai.train` still exists for a single symbol, and writes the
same config with one entry in it.

**Why the barrier defaults to 3 ATR.** A trade is only worth taking when
`p_up − p_down > cost / barrier`. On measured data a 1 ATR barrier
demands a ~0.29 directional edge, which no model achieves. Widening the
barrier lowers that hurdle roughly proportionally — at 3 ATR it is around
0.10, and the three outcome classes stay reasonably balanced. Tight
barriers lose to the spread before the model gets a vote.

### 4. Read the verdict

Training prints a pooled walk-forward t-statistic. Rough reading:

- **t > 3** — worth forward-testing on demo
- **2 < t < 3** — weak; do not size it up
- **t < 2** — no measurable edge; do not trade it live

Then check the untouched holdout line. If validation looks strong and the
holdout does not, the gate search overfitted and the walk-forward number
is the one to believe.

### 5. Recompile and attach

Training rewrites `Include/MintyAI/MintyAIModelConfig.mqh`. **Recompile
`MintyAI.mq5`** so the EA picks up the new model, then attach it to a chart
of any symbol it was trained on. The chart *is* the instrument — there is
no symbol to select.

Recompiling is not optional after retraining: the `.onnx` file is embedded
into the `.ex5` at compile time via `#resource`, so an un-recompiled EA
keeps trading the *previous* weights. Embedding is also what makes the EA
work in the strategy tester — each tester agent runs with its own empty
`MQL5\Files` sandbox, so loading the graph by path there fails with error
5019 however plainly the file sits in the terminal's own folder.

Start on a demo account. Then keep it on a demo account until it has
traded enough to mean something — a few dozen trades tells you almost
nothing.

## The EA has no strategy settings

Three inputs exist, and only one of them touches trading at all:

| Group | Input | Purpose |
|---|---|---|
| Trading | `InpRisk` | Low (2%) / Medium (5%) / High (10%) per trade |
| Operation | `InpMagic` | keeps positions separate from other EAs |
| Operation | `InpVerbose` | log every decision |

Even `InpRisk` does not invent a number: it selects between three stakes
that were *measured* during training. Everything else is compiled in from
`MintyAIModelConfig.mqh`, which training regenerates.

There is no symbol setting — the EA trades the chart it is attached to.
There is no broker prefix/suffix setting either: the export writes your
broker's own names into the model table, so the chart symbol already
matches. If a build trained elsewhere meets a broker that spells the pair
differently, `ResolveModel()` finds it anyway and logs that it did, but
only when exactly one trained symbol matches — guessing between two would
trade a real instrument on another instrument's calibration.

The status panel is not a setting either: it draws on a live or demo
chart and in a visual-mode test, and nowhere else. In a plain backtest or
an optimisation pass there is nobody to read it, and `Comment()` on every
tick across millions of them is pure overhead.

| Was an input | Now |
|---|---|
| model file | embedded in the `.ex5` via `#resource` |
| symbol | gone; the EA trades its chart, and refuses if that symbol was never trained |
| broker prefix / suffix | gone; resolved automatically from the trained names |
| timeframe lock | gone; the EA always reads **H1** bars whatever chart it sits on |
| barrier, horizon, time exit | taken from the trade definition the labels were built from |
| risk %, max lot | half-Kelly from the model's outcome head, capped at one of three measured levels |
| max positions | 1 — the model prices one bet from the current bar |
| daily loss, max drawdown | twice the worst the sized validation curve produced |
| cooldown after a loss | gone; one position at a time with a horizon exit already spaces trades |
| max spread | gone; cost is subtracted from EV, so an expensive spread fails the gate on its own |
| session hours | 24-bit mask, learned from per-hour validation results |
| Friday flatten | gone; the horizon counts **bars**, and bars do not advance over the weekend |
| break-even, trailing | gone; see below |

Break-even and trailing stops are not a missing feature. The model was
trained to answer "which barrier gets touched first within N bars", and
the EA places exactly that bet. Moving the stop creates a different trade
whose probabilities were never estimated — it looks prudent and quietly
truncates the winners the edge depends on.

## How the EA decides

Once per closed H1 bar:

1. build 96 bars × 24 features with the same code path used in training
2. run the ONNX graph → `P(down)`, `P(flat)`, `P(up)` and `mu`, `sigma`
3. compute `EV_long = barrier × (P(up) − P(down)) − cost`, and the mirror
   for short, where `cost = (spread + slippage) × point`
4. enter only if `EV > edge_mult × cost` **and** the directional
   probability clears `min_prob`
5. size it: `risk = ½ × (mu − cost_R) / sigma²`, capped
6. stop and target at ±`barrier`, horizon exit after N **bars**

### Backtest only on data the model never saw

Training splits by **calendar time**, not by row order: the earliest 70% is
TRAIN, the next 15% VAL, the last 15% an untouched HOLDOUT. Every run
prints the dates. Splitting on time rather than position matters once
symbols are pooled — a positional split would train on some symbols and
validate on others, measuring transfer between instruments instead of
performance in the future.

A strategy-tester backtest that starts before the holdout is scoring the
model on bars it was fitted to, and it will look excellent no matter how
worthless the model is. Set the tester's From date to the holdout start, or
read the walk-forward number instead.

### The three risk levels

Low, Medium and High are the *same network* evaluated at three stakes — 2%,
5% and 10% of equity per trade. The cap never enters the loss function, so
it cannot change the weights; what changes is the stake and the halts that
follow from it, both measured on validation at that stake.

Those caps live in `Python/mintyai_deploy.json`, which training reads and
never writes, so a retrain cannot quietly reset them:

```json
{ "risk_pcts": [2.0, 5.0, 10.0], "kelly_scale": 0.5 }
```

An explicit `--risk-pcts 1,3,7` still overrides it for one run.

Raw Kelly asks for far more than any of these — on EURUSD H1 it wanted a
*median 9.6% of equity per trade* — so the cap binds on nearly every trade.
That is the point: the cap, not the model, is what decides position size.

Do not read the drawdown figures printed beside each level as forecasts.
They are measured on validation data, the same slice the entry gate was
tuned on, so they are the most flattering estimate available. Compare them
with the walk-forward number before believing them.

Step 5 is why the graph has a second output. Kelly needs edge over
variance, and a distribution over three barrier outcomes says nothing
about the dispersion of the result — so the network predicts `mu` and
`sigma` of the bar's outcome in R units, trained by Gaussian likelihood
against what actually happened. A model exported before that head existed
still runs: the EA then derives both from the class probabilities, using
the same arithmetic.

## Safety rails

These are not decoration; they are what keeps a bad model from being an
account-ending one.

- **Feature signature check** — the EA hashes its compiled-in feature list
  and refuses to start if it disagrees with the trained model
- **Untrained-symbol refusal** — on a chart it has no calibration for, the
  EA prints `NO TRAINING DATA`, lists what it does know, and does not start
- **Softmax sanity check** — refuses to trade on outputs that are not
  probabilities, or on a non-finite `mu`/`sigma`
- **Risk-based sizing** — lots derived from the stop distance; never
  rounds *up* into more risk than the model asked for
- **Margin check** — skips any trade needing more than half the free margin
- **Daily loss halt** and **max drawdown halt**, at levels training measured

The refusal is the important one. The network is symbol-agnostic and would
happily emit confident-looking probabilities for an instrument it never saw
— but there would be no gate, no hour mask and no measured stake for it, so
the EA would be trading on thresholds borrowed from unrelated symbols.
Refusing makes that mistake unreachable.

**Halts resume.** Both the daily-loss and total-drawdown halts lift at the
next trading day, and the drawdown halt re-anchors its peak to current
equity as it does. That re-anchor is what makes resumption possible at all:
a halted EA holds no positions, so its equity cannot move, and a halt
waiting for the account to *recover* would wait forever.

## Halting for good measure

Halts close every open position and block new ones. They are account-level,
so with several instances running on different charts each sees the same
equity curve and each will halt on it.

## Smoke-testing without a terminal

```bash
cd Python
python tools/make_synthetic.py --bars 60000 --phi 0.35 --out ../Files/MintyAI/SIGNAL_H1 --symbol SIGNAL
python -m mintyai.train_all --symbols SIGNAL \
    --spread-points 0 --barrier-atr 1.5 --horizon 6 --epochs 25 --folds 3
```

The synthetic series has a deliberate return autocorrelation, so with
costs switched off a working pipeline finds a large, consistent edge
(this build: **+3228 R over 25k trades, pooled t = 22.4**, and t = 8.9 on
the untouched holdout). Re-run it with realistic costs and the edge
disappears — which is the honest result, and a good illustration of what
the spread does to a real strategy.

Use this to verify the plumbing after changing anything. Never as
evidence about a real market.

## Everything runs on UTC

Four of the 24 features are sin/cos of hour-of-day and day-of-week, and the
learned trading-hour mask is a clock too. `MqlRates.time` is in the
**broker's server timezone**, which varies by broker — GMT+2, GMT+3, GMT-5.
Left alone, a model trained on one server reads every session two or three
hours out on another, silently, because nothing else in the pipeline can
see a timezone.

So timestamps are converted once at each boundary:

| Boundary | What happens |
|---|---|
| `MintyAIExportAll.mq5` | writes bar times, and the parity dump, in UTC; records the server offset in `_meta.csv` |
| `MintyAIFeaturesFromSymbol()` | subtracts the live server offset before building features |
| `CurrentHour()` | returns the UTC hour, which is what the mask was learned in |

The offset comes from `TimeCurrent() - TimeGMT()`, rounded to the hour and
rejected if it falls outside `[-12, +14]`. It is cached — it is needed for
every bar of every inference — and printed at startup:

```
MintyAI clock: server is UTC+3, so hour 14 here is 11 UTC - the
trading-hour mask and the time features both run on UTC.
```

Check that line against your broker once. If it says UTC+0 on a broker you
know runs GMT+2, the terminal had no server time yet when the EA started;
reload it.

This changed the feature contract, so `FEATURE_VERSION` went to **2** and
the signature to **2657182663**. Models trained before it are rejected at
startup rather than silently mis-read — re-export and retrain.

## Changing features

1. edit `MintyAIFeatures.mqh` **and** `features.py` together
2. bump `MINTYAI_FEATURE_VERSION` / `FEATURE_VERSION` in both
3. re-export, re-run `mintyai.parity` until it passes
4. retrain — old models are now incompatible, and the EA will say so

## Known limits

- ONNX inference in MetaTrader runs on CPU. Only training uses the GPU.
  Inference is once per bar and takes under a millisecond, so this does
  not matter.
- Backtests use MetaTrader's modelled spread, which is usually kinder than
  live fills. Slippage and commission are your inputs to get right.
- The model has no knowledge of scheduled news. A calendar filter is a
  reasonable addition and is not implemented here.
- One network covers every trained symbol, but only trained symbols. It
  refuses on anything else rather than extrapolating.
- The EA is hard-wired to H1. Training on another timeframe produces a
  model whose config says so and whose probabilities the EA will still
  feed with H1 bars — it warns at startup, but retraining on H1 is the
  fix, not ignoring it.
- Pooling assumes the symbols share a data-generating process well enough
  for one model to price them all. That is plausible across FX majors and
  metals; it is an assumption, not a proof, and the per-symbol walk-forward
  columns are where it gets tested.
- One instance trades one chart. Several symbols means several charts, and
  the account-level halts are then shared between them.
- The learned hour mask, risk cap and halt levels come from **validation**
  data, not the walk-forward folds. They are honest measurements of one
  period, not out-of-sample guarantees.
