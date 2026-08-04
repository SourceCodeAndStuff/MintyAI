# MintyAI — update log

## 3.3

**New: pick the model yourself, for brokers that rename instruments.**

Some brokers do not decorate the standard name, they replace it — gold is
listed as `GOLD` rather than `XAUUSD`. No amount of automatic matching will
ever connect those two: they share no letters, and the only thing that knows
`GOLD` is gold is you.

So there is now a **Model** setting. It lists every instrument the EA
carries, and defaults to **Auto**:

- **Auto** — reads the model off the chart symbol, as before. This already
  handles brokers that decorate the standard name: `EURUSD.r`,
  `EURUSD.ecn`, `mEURUSD`, `eurusd.ecn` all resolve on their own.
- **A named instrument** — trades the chart you attached it to using that
  instrument's model. Attach to `GOLD`, choose `XAUUSD`, done.

The list is generated from the models actually built into the EA, so it
cannot offer one that is not there. When you pick a model by hand the panel
shows it, marked `selected`, for as long as it runs.

Nothing checks that the chart and the model are the same instrument — that
part is your judgement. Point it at the wrong one and it will trade the
wrong one.

**Removed: the symbol prefix and suffix settings.** They only ever covered
decorated names, which Auto already handles, and they could not express a
rename at all. The Model setting does both jobs, so they have gone. If you
had them filled in, clear them — they no longer exist, and Auto will
recognise your broker's spelling on its own.

### Known behaviour: restart MetaTrader between backtests

On a machine with a dedicated graphics card, MetaTrader compiles the model
for your GPU on the first run and caches the result in the tester's own
working folder. Later runs reuse that cache rather than rebuilding it, and
the two do not always produce identical numbers. The difference is far too
small to see, but it is enough to flip a borderline trade, so two tests of
the same period can disagree.

Closing MetaTrader 5 and opening it again clears the cache. Do it between
runs, and particularly when switching instruments. If a second run of a
test looks unlike the first, this is why — re-run it from a fresh start
before reading anything into it.

This is MetaTrader's model cache rather than anything the EA controls, and
it affects comparisons between runs, not the models themselves.

---

## 3.2

**All 37 instruments are back — metals and crypto rejoin the 28 forex pairs
— and brokers that decorate their symbol names are now handled explicitly
rather than by guesswork.**

### Metals and crypto restored

3.1 shipped forex only, to bring the file size down. 3.2 puts all nine back,
each with the dedicated model it always had:

**Metals (3)** XAUUSD · XAGUSD · XAUEUR

**Crypto (6)** BTCUSD · ETHUSD · SOLUSD · XRPUSD · ADAUSD · DOGEUSD

| | 3.1 | 3.2 |
|---|---|---|
| Instruments | 28 | 37 |
| Models built in | 28 | 37 |
| Download size | 50.1 MB | 66.2 MB |

**The forex models are untouched.** Every instrument has its own network and
nothing is shared between them — no pooled weights, no common scaling, no
jointly-set thresholds — so adding nine instruments back cannot change the
other 28, and did not. Same models, same entry filters, same trading hours,
same position sizing and safety limits as 3.1. If you are running 3.1 on a
forex pair, 3.2 trades it identically and your backtests still stand.

### New: symbol prefix and suffix

Two settings, both empty by default:

| Setting | Example |
|---|---|
| **Symbol prefix** | `m` for `mEURUSD` |
| **Symbol suffix** | `.ecn` for `EURUSD.ecn`, `.r` for `EURUSD.r` |

Left empty, the EA works your broker's naming out for itself, as before.

Filled in, they stop being a guess and become an instruction: the chart name
must be exactly your prefix + one of the 37 names + your suffix, and
anything else is treated as an instrument the EA has no model for. That is
the point of the setting. On brokers where two instruments differ only by
their decoration, working it out by similarity could put the calibration
measured for one onto the other; being told outright cannot. With `.ecn`
set, `EURUSD.ecn` trades and `EURUSD.r` is refused, by name, in the log.

Upper and lower case both work, and a chart whose name already matches a
model exactly is always accepted regardless of what you have configured.

### Automatic naming detection improved

When the prefix and suffix are left empty, matching now ignores case and
punctuation, so spellings that previously went unrecognised are picked up:
`eurusd.ecn`, `EUR/USD`, `mEURUSD_i`. Where a name genuinely could be two
different instruments the EA still refuses to guess — and now tells you to
set the prefix/suffix instead.

### Charts with no model of their own

Unchanged in principle: the EA still loads, borrows the EURUSD model, and
says so in the log and on the panel. It is now more robust about which model
it borrows, so it cannot end up loaded but idle.

---

## 3.1

Forex-only release, cutting the download size. Also in this version:

- **Fixed:** orders were being rejected outright with *Volume limit reached*
  on brokers that cap the total volume held in one direction on a symbol.
  Position size is now capped to what the broker will actually accept, and
  the log says when that happens. It can only ever lower the size, never
  raise it.
- **Fixed:** a chart the EA had no model for stopped it loading at all.
  It now loads, borrows the EURUSD model and says so.

## 3.0

First release with a separate model per instrument, replacing the single
shared one.

- Each instrument carries its own entry filter, learned trading hours,
  position sizing and safety limits, measured on its own history.
- A fixed bet: stop and target set from current volatility, plus a 24-hour
  time exit — the same trade the model was scored on. No trailing stop, no
  break-even, no averaging down.
- Three risk levels derived from each instrument's own measured results.
- Entry cost checked against your broker's live spread, so there is no
  max-spread setting to get wrong.
- Daily loss and drawdown limits that close everything, then lift at the
  next trading day.
- Server timezone handled automatically, so trading hours line up on any
  broker.
- Three settings, only one of which touches trading.
