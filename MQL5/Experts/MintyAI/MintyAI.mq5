//+------------------------------------------------------------------+
//|                                                      MintyAI.mq5 |
//|                                                                  |
//|   Trades GPU-trained ONNX price-direction models.                |
//|                                                                  |
//|   There are no strategy settings. Every number this EA trades on |
//|   was measured during training and compiled in: the barrier and  |
//|   horizon that define the bet, the entry gate, the position size,|
//|   the equity halts and the trading hours. A knob here would be a |
//|   second, unvalidated opinion competing with the one the model   |
//|   was fitted to - so there isn't one.                            |
//|                                                                  |
//|   You choose two things: which symbols to trade, and how much to |
//|   stake. Each symbol runs its own model, on its own bar clock,   |
//|   independently of the others.                                   |
//|                                                                  |
//|   Every graph has two outputs:                                   |
//|     probs [1,3]  P(down), P(flat), P(up) - calibrated            |
//|     risk  [1,2]  mu, log-sigma of the outcome in R units         |
//|   The second is what sizing needs: Kelly divides edge by         |
//|   variance, and a class distribution cannot express variance.    |
//|                                                                  |
//|   Setup:                                                         |
//|     1. Scripts\MintyAI\MintyAIExportAll.mq5 - download history   |
//|     2. python -m mintyai.parity             - feature parity     |
//|     3. python -m mintyai.train_all          - train every symbol |
//|     4. recompile this EA                    - embeds the models  |
//+------------------------------------------------------------------+
#property copyright "MintyAI"
#property version   "3.30"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>
#include <MintyAI/MintyAIFeatures.mqh>
#include <MintyAI/MintyAIModelConfig.mqh>

//--- Every model this build carries is declared and embedded inside
//--- MintyAIModelConfig.mqh, which training regenerates. Adding a symbol
//--- means retraining and recompiling - nothing here changes.

//+------------------------------------------------------------------+
//| Fixed operating parameters                                       |
//+------------------------------------------------------------------+
//--- The EA works on H1 whatever the chart says, so attaching it to a
//--- five-minute chart cannot quietly feed a model five-minute bars.
//--- Symbols are handled the same way: they come from the settings and
//--- the model table, not from the chart.
#define MINTYAI_TF            PERIOD_H1

//--- One bet at a time PER SYMBOL. A model prices a single barrier trade
//--- from the current bar; a second concurrent position in the same
//--- instrument is a bet it never priced.
#define MINTYAI_MAX_POSITIONS 1

//--- hard ceiling on how many symbols one instance will run
#define MINTYAI_MAX_SLOTS     32

//--- The model used on a chart this build was not trained for. The
//--- features are all scale-free - returns, ratios, volatility multiples -
//--- so the network reads any instrument without complaint, and EURUSD is
//--- the deepest, cleanest history of the set to have learned from.
//---
//--- What does NOT carry across is everything measured ALONGSIDE the
//--- weights: the entry gate, the hour mask, the stake and the halt levels
//--- were fitted to EURUSD's spread, session and volatility. On another
//--- instrument they are an assumption, not a measurement, which is why
//--- this is announced loudly rather than quietly relied on.
#define MINTYAI_FALLBACK_SYM  "EURUSD"

//--- Ceilings on the portfolio-widened halts. However many symbols run,
//--- losing this much in a day, or this much from peak, is not a bad
//--- stretch to be sat through - it is a broken model.
#define MINTYAI_HALT_CEIL_DAILY  50.0
#define MINTYAI_HALT_CEIL_TOTAL  90.0

//+------------------------------------------------------------------+
//| Light theme palette                                              |
//|                                                                  |
//| Text sits on its own filled card rather than directly on the     |
//| chart, because Comment() text inherits the chart foreground and  |
//| becomes unreadable the moment a candle passes behind it.         |
//+------------------------------------------------------------------+
#define UI_PREFIX      "MintyAI_"

#define UI_BG          C'252,252,253'   // chart background
#define UI_FG          C'60,66,78'      // axes, scales
#define UI_GRID        C'229,233,239'
#define UI_BULL_LINE   C'26,140,94'
#define UI_BEAR_LINE   C'198,62,62'
#define UI_BULL_FILL   C'233,247,240'
#define UI_BEAR_FILL   C'253,236,236'
#define UI_NEUTRAL     C'146,155,170'

#define UI_CARD        C'255,255,255'   // panel body
#define UI_BORDER      C'216,221,230'
#define UI_HEADER      C'23,110,84'     // minty header strip
#define UI_HEADER_TX   C'255,255,255'
#define UI_KEY_TX      C'112,120,134'   // row labels
#define UI_VAL_TX      C'26,31,40'      // row values
#define UI_WARN        C'193,124,18'

//--- panel geometry
#define UI_X           12
#define UI_Y           20
#define UI_W           320
#define UI_PAD         10
#define UI_ROW_H       17
#define UI_HEADER_H    28
#define UI_MAX_ROWS    48

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "=== Trading ==="
//--- The EA trades the chart it is attached to, and nothing else. Attach it
//--- to EURUSD and it trades EURUSD. There is no symbol setting because the
//--- chart already is one.
//--- Which of the three stakes the model was measured at. Low is the
//--- conservative one; High risks ten times as much per trade and draws
//--- down accordingly.
input ENUM_MINTYAI_RISK InpRisk = MINTYAI_RISK_MEDIUM;            // Risk level

//--- Which model to run. Auto reads the chart symbol and finds the model
//--- itself, which covers every broker that merely decorates the standard
//--- name - EURUSD.r, EURUSD.ecn, mEURUSD.
//---
//--- Naming one outright is for brokers that rename the instrument instead
//--- of decorating it. A chart called GOLD carries nothing an algorithm
//--- could match to XAUUSD, and no amount of pattern matching will find it;
//--- the only thing that knows GOLD is XAUUSD is you. Pick XAUUSD here and
//--- it trades the GOLD chart on the XAUUSD model.
//---
//--- The dropdown is generated from the models actually compiled in, so it
//--- cannot offer one that does not exist.
input ENUM_MINTYAI_SYMBOL InpModel = MINTYAI_SYM_AUTO;            // Model

input group "=== Operation ==="
//--- Deliberately outside 20260700-20260799: the separate Q-learning EA
//--- Experts\Hades.mq5 sweeps that range as its generation counter, and
//--- two EAs sharing a magic number will close each other's positions.
input long   InpMagic        = 51230001;   // Magic number
input bool   InpVerbose      = true;       // Log every decision

//+------------------------------------------------------------------+
//| Per-symbol state                                                 |
//|                                                                  |
//| Flat rather than nesting the config structs: this is the hot     |
//| path, and every field here is read on each tick.                 |
//+------------------------------------------------------------------+
struct MintyAISlot
  {
   string            symbol;        // as the broker spells it
   string            plain;         // as the model table spells it
   int               modelIdx;
   long              onnx;
   bool              hasRiskHead;
   datetime          lastBar;
   //--- this bar's inference, and whether it has been acted on. The entry
   //--- test reruns every tick, so it needs to know what is still open.
   bool              ready;      // model output for lastBar is valid
   bool              entered;    // already took this bar's trade
   bool              logged;     // already explained this bar's rejection
   //--- copied from the model row
   double            minProb;
   double            edgeMult;
   double            kellyScale;
   uint              hourMask;
   double            maxRiskPct;
   double            wfTstat;
   //--- last inference, for the panel
   double            pDown, pFlat, pUp;
   double            mu, sigma;
   double            ev, risk, atr;
  };

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CTrade         g_trade;
CPositionInfo  g_pos;

MintyAISlot g_slots[];
int         g_slotCount   = 0;

bool        g_showPanel   = true;

//--- Loaded, but with no model for this chart. See Idle().
bool        g_idle        = false;
string      g_idleWhy     = "";

//--- the model was named in the settings rather than found from the chart
bool        g_chosen      = false;

//--- first inference of the run gets fingerprinted, so two runs that should
//--- be identical can be held against each other
bool        g_probed      = false;

//--- trading an untrained symbol on the fallback model
bool        g_fallback    = false;

double      g_dayStartEq  = 0.0;
int         g_dayStamp    = -1;
double      g_peakEquity  = 0.0;
//--- Which halt we are in. Tracked as a kind rather than by comparing the
//--- reason string, because that comparison is what previously let a daily
//--- halt escalate into a drawdown halt that nothing could lift.
#define HALT_NONE      0
#define HALT_DAILY     1
#define HALT_DRAWDOWN  2

bool        g_halted      = false;
int         g_haltKind    = HALT_NONE;
string      g_haltReason  = "";

//--- Account-level halts. Each model carries its own, measured at the
//--- chosen stake; running several at once we take the tightest, because
//--- the account is shared and the loosest would let one symbol spend
//--- another's budget.
double      g_dailyLossPct = 0.0;
double      g_maxDdPct     = 0.0;

//+------------------------------------------------------------------+
int OnInit()
  {
//--- 1) the feature engine compiled into this EA must be the one the
//---    models were trained on. Everything else is downstream of this.
   if(MintyAIFeatureSignature() != (uint)MINTYAI_MODEL_SIGNATURE)
     {
      PrintFormat("MintyAI: FEATURE SIGNATURE MISMATCH. EA=%u models=%u. "
                  "The models and the feature engine in this build do not "
                  "belong together - reinstall it.",
                  MintyAIFeatureSignature(), (uint)MINTYAI_MODEL_SIGNATURE);
      return INIT_FAILED;
     }
   if(MINTYAI_FEATURE_COUNT != MINTYAI_MODEL_FEATURE_COUNT)
     {
      Print("MintyAI: feature count mismatch");
      return INIT_FAILED;
     }

   MintyAIInitModels();

//--- 1b) line this broker's clock up with the one the model was trained on.
//---
//---     The hour-of-day features and the trading-hour mask are both in the
//---     TRAINING broker's server time. On that same broker the shift is
//---     zero and nothing happens - which is the normal case, and exactly
//---     what 2.0 did. On a broker in another timezone the difference is
//---     added to every bar, so the model still sees the hour it was taught.
//---
//---     Detection is only trusted on a live chart: in the tester TimeGMT()
//---     is modelled from the LOCAL PC's timezone, not the server's, so it
//---     would report this machine's offset and shift everything wrongly.
//---     A backtest runs on the broker that produced the data, so assuming
//---     no shift there is both safe and correct.
   long trainOffset = (long)MINTYAI_MODEL_UTC_OFFSET_H * 3600;
   long liveOffset  = trainOffset;
   string how       = "tester - assuming the reference broker";

   if(!MQLInfoInteger(MQL_TESTER))
     {
      long det = MintyAIDetectUtcOffset();
      if(det != MINTYAI_UTC_UNKNOWN)
        {
         liveOffset = det;
         how = "detected";
        }
      else
         how = "GMT unreadable - assuming the reference broker";
     }

//---     A ONE-HOUR gap is daylight saving, not a different timezone, and
//---     must not be corrected. The training bars were stamped in the
//---     training server's own local clock - which moves with DST, and so
//---     do the London and New York sessions the model keys off, so the two
//---     stay aligned by themselves all year. The recorded reference is a
//---     single reading taken whenever the export happened to run (summer,
//---     here), while the live reading moves. Subtracting one from the
//---     other would put the model an hour out every winter on the very
//---     broker it was trained on. Only a gap too large to be DST is a real
//---     timezone difference worth correcting.
   long rawDiff = trainOffset - liveOffset;
   long shift   = (rawDiff >= -3600 && rawDiff <= 3600) ? 0 : rawDiff;

   MintyAISetTimeShift(shift);

   if(shift == 0 && rawDiff == 0)
      PrintFormat("MintyAI clock: this server is UTC%+d (%s), same as the "
                  "model's reference server. Bar times used as-is.",
                  (int)(liveOffset / 3600), how);
   else if(shift == 0)
      PrintFormat("MintyAI clock: this server is UTC%+d (%s) against the "
                  "model's reference UTC%+d. That hour is daylight saving, "
                  "not a different timezone - the model reads this server's "
                  "own clock, which moves with the sessions. Bar times used "
                  "as-is.",
                  (int)(liveOffset / 3600), how,
                  (int)MINTYAI_MODEL_UTC_OFFSET_H);
   else
      PrintFormat("MintyAI clock: this server is UTC%+d (%s) but the model "
                  "expects UTC%+d - shifting bar times by %+d hour(s) so the "
                  "session features line up.",
                  (int)(liveOffset / 3600), how,
                  (int)MINTYAI_MODEL_UTC_OFFSET_H,
                  (int)(MintyAITimeShift() / 3600));

//--- 2) which model, on the chart the EA is attached to
   int idx = -1;

   if(InpModel != MINTYAI_SYM_AUTO)
     {
      //--- Named outright. The enum is generated from the models compiled in,
      //--- so the only way this is out of range is a settings file saved
      //--- against a build carrying different models.
      idx = (int)InpModel;
      if(idx < 0 || idx >= MINTYAI_MODEL_COUNT)
        {
         PrintFormat("MintyAI: the Model setting refers to model %d, but this "
                     "build carries %d. Re-pick it from the dropdown.",
                     idx, MINTYAI_MODEL_COUNT);
         return Idle("the Model setting does not match this build");
        }

      g_chosen = true;
      PrintFormat("MintyAI: trading the %s chart on the %s model, because you "
                  "chose it. Nothing here checks that they are the same "
                  "instrument - that part is your call.",
                  _Symbol, MintyAIModels[idx].symbol);
     }
   else
      idx = ResolveModel(_Symbol);

   if(idx < 0)
     {
      //--- No model for this chart, so borrow one rather than sitting out.
      //--- It is a real network reading real features, but every threshold
      //--- around it belongs to another instrument - so say so, twice, and
      //--- put it on the panel too.
      //---
      //--- The preferred stand-in is the reference symbol. A build that does
      //--- not carry it takes the first model it does carry: idling because
      //--- one particular name is absent would leave the EA dead on every
      //--- chart, which is the failure this whole path exists to avoid.
      idx = MintyAIFindModel(MINTYAI_FALLBACK_SYM);
      if(idx < 0 && MINTYAI_MODEL_COUNT > 0)
         idx = 0;
      if(idx < 0)
         return Idle("this build carries no models at all");

      string via = MintyAIModels[idx].symbol;
      g_fallback = true;
      Print("MintyAI: ============ NO MODEL FOR THIS SYMBOL ============");
      PrintFormat("MintyAI: '%s' is not one of the symbols this EA carries a "
                  "model for. Falling back to the %s model so it still runs - "
                  "but the entry gate, trading hours, stake and halt levels "
                  "around it were all measured on %s, not on %s. Nothing here "
                  "is calibrated to %s; treat it as a look, not a strategy.",
                  _Symbol, via, via, _Symbol, _Symbol);
      PrintFormat("MintyAI: the symbols with their own model are: %s",
                  MintyAISymbolList());
      PrintFormat("MintyAI: if your broker just calls one of those something "
                  "else - GOLD for XAUUSD, say - pick it in the Model setting "
                  "and %s will trade on its own calibration instead.", _Symbol);
     }

   ArrayResize(g_slots, 1);
   g_slotCount = 0;
   if(!SetupSlot(_Symbol, idx))
      return Idle("cannot run on " + _Symbol + " - see the log");
   g_slotCount = 1;

//--- 3) account-level halts, as measured for this model at this stake
   MintyAIRiskCfg r = MintyAIModels[idx].risk[(int)InpRisk];
   g_dailyLossPct = MathMin(r.daily_loss_pct, MINTYAI_HALT_CEIL_DAILY);
   g_maxDdPct     = MathMin(r.max_dd_pct,     MINTYAI_HALT_CEIL_TOTAL);

//--- 4) the panel and chart theme are drawn wherever someone can see
//---    them: a live or demo chart, and a visual-mode test. In a plain
//---    backtest or an optimisation pass nobody is watching.
   g_showPanel = (!MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE));
   if(g_showPanel)
      ApplyLightTheme();

//--- 5) trading environment
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints((int)MathMax(MINTYAI_MODEL_SLIPPAGE_POINTS, 1.0));
   g_trade.SetAsyncMode(false);

   g_dayStartEq = AccountInfoDouble(ACCOUNT_EQUITY);
   g_peakEquity = g_dayStartEq;
   g_dayStamp   = DayStamp(TimeCurrent());

//--- Which process this actually ran in. Consecutive tester runs are handed
//--- to local agents in rotation, so a fault that appears on every second
//--- run is usually one agent, not one run - and the agent keeps whatever
//--- it cached until the terminal restarts it. The path ends in the agent's
//--- name, so two runs can be told apart after the fact.
   PrintFormat("MintyAI env: %s | build %d | %s",
               TerminalInfoString(TERMINAL_DATA_PATH),
               (int)TerminalInfoInteger(TERMINAL_BUILD),
               MQLInfoInteger(MQL_TESTER) ? "tester" : "live chart");

   PrintFormat("MintyAI ready. %d symbol(s) on %s at %s risk | window %d | "
               "barrier %.2f ATR | horizon %d bars",
               g_slotCount, MINTYAI_MODEL_TIMEFRAME, RiskName(InpRisk),
               MINTYAI_MODEL_WINDOW, MINTYAI_MODEL_BARRIER_ATR,
               MINTYAI_MODEL_HORIZON_BARS);
   if(g_fallback)
      PrintFormat("MintyAI: running %s on the %s model. The numbers below were "
                  "measured on %s and are borrowed, not measured here.",
                  _Symbol, g_slots[0].plain, g_slots[0].plain);
   PrintFormat("MintyAI halts: %.2f%% daily, %.2f%% total (measured for this "
               "model at this stake)", g_dailyLossPct, g_maxDdPct);

   int weak = 0;
   for(int i = 0; i < g_slotCount; i++)
     {
      PrintFormat("MintyAI   %-12s gate p>=%.2f edge>=%.2fx | stake <=%.2f%% | "
                  "hours 0x%06X | walk-forward t=%.2f",
                  g_slots[i].symbol, g_slots[i].minProb, g_slots[i].edgeMult,
                  g_slots[i].maxRiskPct, g_slots[i].hourMask, g_slots[i].wfTstat);
      if(g_slots[i].wfTstat < 2.0)
         weak++;
     }
   if(weak > 0)
      PrintFormat("MintyAI: NOTE %d of %d models had no statistically significant "
                  "out-of-sample edge (t < 2). Treat those as an experiment.",
                  weak, g_slotCount);

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Prepare one symbol. Returns false if it cannot run - a bad symbol |
//| in a list must not stop the others from trading.                  |
//+------------------------------------------------------------------+
bool SetupSlot(const string broker, const int idx)
  {
   int s = g_slotCount;

   if(!SymbolSelect(broker, true))
     {
      PrintFormat("MintyAI: '%s' could not be selected. Refusing to start.",
                  broker);
      return false;
     }

   int have = Bars(broker, MINTYAI_TF);
   int need = MINTYAI_MODEL_WINDOW + MINTYAI_WARMUP + 10;
   if(have < need)
     {
      PrintFormat("MintyAI: skipping %s - only %d %s bars, need %d. Let the "
                  "terminal finish downloading its history.",
                  broker, have, PeriodTag(MINTYAI_TF), need);
      return false;
     }

//--- Everything the sizing and the features depend on, printed once, so a
//--- run that came out wrong can be held against one that came out right.
//--- The tester hands over the previous symbol's contract data often enough
//--- that "which symbol was this actually sized as" is a question worth
//--- being able to answer after the fact.
   int    digits = (int)SymbolInfoInteger(broker, SYMBOL_DIGITS);
   double refDist = 100 * SymbolInfoDouble(broker, SYMBOL_POINT);
   PrintFormat("MintyAI %s: contract - digits %d, point %.*f, tick %.*f @ %.4f, "
               "lots %.2f-%.2f step %.2f, spread %d pts, 100pt/lot = %.2f %s",
               broker, digits, digits, SymbolInfoDouble(broker, SYMBOL_POINT),
               digits, SymbolInfoDouble(broker, SYMBOL_TRADE_TICK_SIZE),
               SymbolInfoDouble(broker, SYMBOL_TRADE_TICK_VALUE),
               SymbolInfoDouble(broker, SYMBOL_VOLUME_MIN),
               SymbolInfoDouble(broker, SYMBOL_VOLUME_MAX),
               SymbolInfoDouble(broker, SYMBOL_VOLUME_STEP),
               (int)SymbolInfoInteger(broker, SYMBOL_SPREAD),
               LossPerLot(broker, refDist),
               AccountInfoString(ACCOUNT_CURRENCY));
   PrintFormat("MintyAI %s: history - %d %s bars, %s .. %s",
               broker, have, PeriodTag(MINTYAI_TF),
               TimeToString(iTime(broker, MINTYAI_TF, have - 1),
                            TIME_DATE | TIME_MINUTES),
               TimeToString(iTime(broker, MINTYAI_TF, 0),
                            TIME_DATE | TIME_MINUTES));

   long onnx = MintyAICreateOnnx(idx);
   if(onnx == INVALID_HANDLE)
     {
      PrintFormat("MintyAI: skipping %s - the embedded graph would not load "
                  "(error %d). This build is damaged; reinstall it.",
                  broker, GetLastError());
      return false;
     }

   const long inShape[] = {1, MINTYAI_MODEL_WINDOW, MINTYAI_FEATURE_COUNT};
   if(!OnnxSetInputShape(onnx, 0, inShape))
     {
      PrintFormat("MintyAI: skipping %s - OnnxSetInputShape failed, error %d",
                  broker, GetLastError());
      OnnxRelease(onnx);
      return false;
     }
   const long probShape[] = {1, 3};
   if(!OnnxSetOutputShape(onnx, 0, probShape))
      PrintFormat("MintyAI: %s OnnxSetOutputShape(probs) failed (error %d) - "
                  "continuing, the graph may declare it already.",
                  broker, GetLastError());

//--- A model exported before the outcome head exists still runs; sizing
//--- then falls back to what the class probabilities alone imply.
   bool hasRisk = (OnnxGetOutputCount(onnx) >= 2);
   if(hasRisk)
     {
      const long riskShape[] = {1, 2};
      if(!OnnxSetOutputShape(onnx, 1, riskShape))
         PrintFormat("MintyAI: %s OnnxSetOutputShape(risk) failed (error %d)",
                     broker, GetLastError());
     }
   else
      PrintFormat("MintyAI: %s has no outcome head - sizing from the class "
                  "probabilities instead.", broker);

   MintyAIRiskCfg risk = MintyAIModels[idx].risk[(int)InpRisk];

   g_slots[s].symbol      = broker;
   g_slots[s].plain       = MintyAIModels[idx].symbol;
   g_slots[s].modelIdx    = idx;
   g_slots[s].onnx        = onnx;
   g_slots[s].hasRiskHead = hasRisk;
   g_slots[s].lastBar     = iTime(broker, MINTYAI_TF, 0);
   g_slots[s].minProb     = MintyAIModels[idx].min_prob;
   g_slots[s].edgeMult    = MintyAIModels[idx].edge_mult;
   g_slots[s].kellyScale  = MintyAIModels[idx].kelly_scale;
   g_slots[s].hourMask    = MintyAIModels[idx].hour_mask;
   g_slots[s].maxRiskPct  = risk.max_risk_pct;
   g_slots[s].wfTstat     = MintyAIModels[idx].wf_tstat;
   g_slots[s].ready = false; g_slots[s].entered = false; g_slots[s].logged = false;
   g_slots[s].pDown = 0.0; g_slots[s].pFlat = 0.0; g_slots[s].pUp = 0.0;
   g_slots[s].mu = 0.0; g_slots[s].sigma = 0.0;
   g_slots[s].ev = 0.0; g_slots[s].risk = 0.0; g_slots[s].atr = 0.0;
   return true;
  }

//+------------------------------------------------------------------+
//| Load, but with nothing to trade.                                 |
//|                                                                  |
//| Having no model for the chart is a reason not to trade, not a    |
//| reason to refuse to load. INIT_FAILED aborts a strategy tester   |
//| run outright - so a single untrained symbol takes the whole pass |
//| down with it, which is what the Market validator sees when it    |
//| picks a symbol this build was not trained on - and on a live     |
//| chart it detaches the EA before anyone reads why.                |
//|                                                                  |
//| Idling says exactly the same thing, leaves the explanation in    |
//| the log and on the chart, and places no orders.                  |
//+------------------------------------------------------------------+
int Idle(const string why)
  {
   g_idle      = true;
   g_idleWhy   = why;
   g_slotCount = 0;

   g_showPanel = (!MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE));
   if(g_showPanel)
      ApplyLightTheme();

   PrintFormat("MintyAI: IDLE - %s. The EA is loaded but will not place an "
               "order on this chart.", why);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   for(int i = 0; i < g_slotCount; i++)
      if(g_slots[i].onnx != INVALID_HANDLE)
        {
         OnnxRelease(g_slots[i].onnx);
         g_slots[i].onnx = INVALID_HANDLE;
        }
   ObjectsDeleteAll(0, UI_PREFIX);
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
//--- nothing to price, nothing to manage, nothing to halt
   if(g_idle)
     {
      if(g_showPanel)
         DrawPanel();
      return;
     }

   UpdateEquityGuards();

//--- the horizon exit is checked every tick so it cannot overshoot
   ManageOpenPositions();

   if(g_showPanel)
      DrawPanel();

//--- Each symbol has its own bar clock. Note that OnTick only fires on
//--- the CHART symbol's ticks, so a quiet host chart wakes the EA less
//--- often than the symbols it trades would on their own.
   for(int i = 0; i < g_slotCount; i++)
     {
      datetime bar = iTime(g_slots[i].symbol, MINTYAI_TF, 0);
      if(bar == 0 || bar == g_slots[i].lastBar)
         continue;
      g_slots[i].lastBar = bar;

      //--- One decision per closed bar, which is what training measured:
      //--- entry at the bar's close, priced with that bar's spread. Testing
      //--- again later in the hour would enter at a price and a cost the
      //--- walk-forward never scored.
      EvaluateBar(i);
      TryEnter(i);
     }
  }

//+------------------------------------------------------------------+
//| Once per closed bar: rebuild features and run the model.         |
//|                                                                  |
//| Everything here depends only on closed bars, so it cannot change |
//| again until the next one - which is why it is not in TryEnter.   |
//+------------------------------------------------------------------+
void EvaluateBar(const int s)
  {
   string sym = g_slots[s].symbol;

   g_slots[s].ready   = false;
   g_slots[s].entered = false;
   g_slots[s].logged  = false;

   float feats[];
   int rows = MintyAIFeaturesFromSymbol(sym, MINTYAI_TF, 1, MINTYAI_MODEL_WINDOW, feats);
   if(rows != MINTYAI_MODEL_WINDOW)
     {
      PrintFormat("MintyAI %s: feature build failed (insufficient history?)", sym);
      return;
     }

   double probs[3];
   double mu = 0.0, sigma = 0.0;
   if(!RunModel(s, feats, probs, mu, sigma))
      return;

   double atr = CurrentAtr(sym);
   if(atr <= 0.0)
     {
      PrintFormat("MintyAI %s: ATR unavailable", sym);
      return;
     }

   g_slots[s].pDown = probs[0];
   g_slots[s].pFlat = probs[1];
   g_slots[s].pUp   = probs[2];
   g_slots[s].mu    = mu;
   g_slots[s].sigma = sigma;
   g_slots[s].atr   = atr;
   g_slots[s].ready = true;

//--- Fingerprint the first inference of the run. Two runs over the same
//--- dates must produce identical numbers here; where they first diverge
//--- says which half is at fault. A different checksum means the model was
//--- fed different bars, so the fault is in the data the terminal served.
//--- The same checksum with different probabilities means the same bars
//--- produced different arithmetic, so the fault is the graph or its
//--- session. Identical on both and the difference is downstream, in
//--- sizing or execution.
   if(!g_probed)
     {
      g_probed = true;
      double sum = 0.0, sq = 0.0;
      int    n   = ArraySize(feats);
      for(int k = 0; k < n; k++)
        {
         sum += (double)feats[k];
         sq  += (double)feats[k] * (double)feats[k];
        }
      PrintFormat("MintyAI %s: FIRST BAR %s | features n=%d sum=%.6f sq=%.6f "
                  "[0]=%.6f [%d]=%.6f | p=%.6f/%.6f/%.6f mu=%.6f sigma=%.6f | "
                  "ATR=%.8f spread=%d",
                  sym, TimeToString(g_slots[s].lastBar, TIME_DATE | TIME_MINUTES),
                  n, sum, sq, feats[0], n - 1, feats[n - 1],
                  probs[0], probs[1], probs[2], mu, sigma, atr,
                  (int)SymbolInfoInteger(sym, SYMBOL_SPREAD));
     }

   if(InpVerbose)
      PrintFormat("MintyAI %s: bar %s  p(dn/flat/up)=%.3f/%.3f/%.3f  mu=%.4f "
                  "sigma=%.4f  ATR=%.5f",
                  sym, TimeToString(g_slots[s].lastBar, TIME_DATE | TIME_MINUTES),
                  probs[0], probs[1], probs[2], mu, sigma, atr);
  }

//+------------------------------------------------------------------+
//| Every tick: is the trade worth taking at the CURRENT cost?       |
//+------------------------------------------------------------------+
void TryEnter(const int s)
  {
   if(g_halted || !g_slots[s].ready || g_slots[s].entered)
      return;

   string sym = g_slots[s].symbol;

   if(CountPositions(sym) >= MINTYAI_MAX_POSITIONS)
      return;
   if(!HourAllowed(s))
     {
      if(InpVerbose && !g_slots[s].logged)
        {
         g_slots[s].logged = true;
         PrintFormat("MintyAI %s: skip, hour %d is switched off by the model",
                     sym, CurrentHour());
        }
      return;
     }

   double atr     = g_slots[s].atr;
   double barrier = MINTYAI_MODEL_BARRIER_ATR * atr;
   double point   = SymbolInfoDouble(sym, SYMBOL_POINT);
   long   spread  = SymbolInfoInteger(sym, SYMBOL_SPREAD);

//--- No spread ceiling: the cost is subtracted from the expected value
//--- below, so a spread wide enough to matter fails the gate on its own.
//--- A separate cap would be a second threshold to get wrong.
   double cost    = ((double)spread + MINTYAI_MODEL_SLIPPAGE_POINTS) * point;
   double costR   = (barrier > 0.0 ? cost / barrier : 1.0);

   double evLong  = barrier * (g_slots[s].pUp - g_slots[s].pDown) - cost;
   double evShort = barrier * (g_slots[s].pDown - g_slots[s].pUp) - cost;
   double thresh  = g_slots[s].edgeMult * cost;

   bool goLong  = (evLong  > thresh) && (g_slots[s].pUp   >= g_slots[s].minProb);
   bool goShort = (evShort > thresh) && (g_slots[s].pDown >= g_slots[s].minProb);
   g_slots[s].ev = MathMax(evLong, evShort);

   if(!goLong && !goShort)
     {
      g_slots[s].risk = 0.0;
      //--- Logged once per bar, not once per tick. The reason can change
      //--- within the bar as the spread moves, and that is the point.
      if(InpVerbose && !g_slots[s].logged)
        {
         g_slots[s].logged = true;
         PrintFormat("MintyAI %s: cost=%.5f  EV L/S=%.6f/%.6f  thresh=%.6f  -> "
                     "no trade yet; will re-check as the spread moves",
                     sym, cost, evLong, evShort, thresh);
        }
      return;
     }

//--- size the bet from the model's own view of its outcome
   double frac = RiskFraction(s, goLong, costR);
   g_slots[s].risk = frac;
   if(frac <= 0.0)
     {
      if(InpVerbose && !g_slots[s].logged)
        {
         g_slots[s].logged = true;
         PrintFormat("MintyAI %s: %s passes the gate but Kelly asks for no stake "
                     "(mu=%.4f sigma=%.4f costR=%.4f)",
                     sym, goLong ? "LONG" : "SHORT",
                     g_slots[s].mu, g_slots[s].sigma, costR);
        }
      return;
     }

//--- One entry per bar. The model priced one bet from this bar; taking a
//--- second after the first is stopped out would be doubling down on a
//--- signal that has already been tested and lost.
   g_slots[s].entered = true;

   if(InpVerbose)
      PrintFormat("MintyAI %s: cost=%.5f  EV=%.6f  risk=%.3f%%  -> %s",
                  sym, cost, g_slots[s].ev, frac * 100.0,
                  goLong ? "LONG" : "SHORT");

   OpenTrade(s, goLong, barrier, frac);
  }

//+------------------------------------------------------------------+
//| Position size, straight from the model.                          |
//|                                                                  |
//| Kelly for a bet returning r per unit staked is E[r]/Var[r]. The  |
//| outcome head gives both: mu is the expected long-side result in  |
//| R units, sigma its spread. Halving it (kellyScale) costs a       |
//| quarter of the growth rate and roughly halves the drawdown,      |
//| which matters because mu is an estimate, not a fact.             |
//+------------------------------------------------------------------+
double RiskFraction(const int s, bool isLong, double costR)
  {
   double mu    = g_slots[s].mu;
   double sigma = g_slots[s].sigma;

   double edge = (isLong ? mu : -mu) - costR;
   if(edge <= 0.0 || sigma <= 1e-6)
      return 0.0;

   double f = edge / (sigma * sigma) * g_slots[s].kellyScale;
   return MathMin(f, g_slots[s].maxRiskPct / 100.0);
  }

//+------------------------------------------------------------------+
//| Inference                                                        |
//+------------------------------------------------------------------+
bool RunModel(const int s, const float &feats[], double &probs[],
              double &mu, double &sigma)
  {
   static float probBuf[];
   static float riskBuf[];
   if(ArraySize(probBuf) != 3)
      ArrayResize(probBuf, 3);
   if(ArraySize(riskBuf) != 2)
      ArrayResize(riskBuf, 2);

   ResetLastError();
   bool ok = g_slots[s].hasRiskHead
             ? OnnxRun(g_slots[s].onnx, ONNX_NO_CONVERSION, feats, probBuf, riskBuf)
             : OnnxRun(g_slots[s].onnx, ONNX_NO_CONVERSION, feats, probBuf);
   if(!ok)
     {
      PrintFormat("MintyAI %s: OnnxRun failed, error %d",
                  g_slots[s].symbol, GetLastError());
      return false;
     }

   double sum = 0.0;
   for(int i = 0; i < 3; i++)
     {
      double v = (double)probBuf[i];
      if(!MathIsValidNumber(v) || v < 0.0)
        {
         PrintFormat("MintyAI %s: model returned a non-probability - refusing "
                     "to trade on it", g_slots[s].symbol);
         return false;
        }
      probs[i] = v;
      sum += v;
     }
//--- the graph ends in softmax; if this is not ~1 something is wrong
   if(sum < 0.99 || sum > 1.01)
     {
      PrintFormat("MintyAI %s: output does not sum to 1 (%.6f) - refusing to trade",
                  g_slots[s].symbol, sum);
      return false;
     }
   for(int i = 0; i < 3; i++)
      probs[i] /= sum;

   if(g_slots[s].hasRiskHead)
     {
      mu    = (double)riskBuf[0];
      sigma = MathExp((double)riskBuf[1]);
      if(!MathIsValidNumber(mu) || !MathIsValidNumber(sigma) || sigma <= 0.0)
        {
         PrintFormat("MintyAI %s: outcome head returned a non-number - refusing "
                     "to trade", g_slots[s].symbol);
         return false;
        }
      return true;
     }

//--- Fallback for a graph without the outcome head. Treat the trade as a
//--- +-1R bet with P(flat) landing at zero: mu and sigma then follow from
//--- the class probabilities alone. Less informed than a learned sigma,
//--- but it is the same arithmetic and needs no settings.
   mu = probs[2] - probs[0];
   double var = (probs[0] + probs[2]) - mu * mu;
   sigma = MathSqrt(MathMax(var, 1e-6));
   return true;
  }

//+------------------------------------------------------------------+
//| ATR(14) computed with the SAME code path as training             |
//+------------------------------------------------------------------+
double CurrentAtr(const string sym)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int got = CopyRates(sym, MINTYAI_TF, 1, MINTYAI_WARMUP, rates);
   if(got < 50)
      return 0.0;

   double h[], l[], c[], atr[];
   ArrayResize(h, got);
   ArrayResize(l, got);
   ArrayResize(c, got);
   for(int i = 0; i < got; i++)
     {
      h[i] = rates[i].high;
      l[i] = rates[i].low;
      c[i] = rates[i].close;
     }
   MintyAIAtr(h, l, c, got, 14, atr);
   return atr[got - 1];
  }

//+------------------------------------------------------------------+
//| Entry                                                            |
//+------------------------------------------------------------------+
void OpenTrade(const int s, bool isLong, double barrier, double riskFraction)
  {
   string sym = g_slots[s].symbol;

   double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
   double bid = SymbolInfoDouble(sym, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return;

   int    digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   double price  = isLong ? ask : bid;
   double sl     = isLong ? price - barrier : price + barrier;
   double tp     = isLong ? price + barrier : price - barrier;

   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   if(!RespectsStopLevel(sym, price, sl) || !RespectsStopLevel(sym, price, tp))
     {
      if(InpVerbose)
         PrintFormat("MintyAI %s: skip, barrier is inside the broker's minimum "
                     "stop distance", sym);
      return;
     }

   double lot = CalcLot(sym, isLong, barrier, riskFraction);
   if(lot <= 0.0)
      return;

   g_trade.SetTypeFillingBySymbol(sym);
   bool ok = isLong ? g_trade.Buy(lot, sym, 0.0, sl, tp, "MintyAI")
                    : g_trade.Sell(lot, sym, 0.0, sl, tp, "MintyAI");
   if(!ok)
     {
      PrintFormat("MintyAI %s: order failed. retcode=%d (%s)",
                  sym, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
      return;
     }

   PrintFormat("MintyAI %s: %s %.2f lots @ %.*f  SL %.*f  TP %.*f  "
               "(p=%.3f, risk %.3f%%)",
               sym, isLong ? "BUY" : "SELL", lot, digits, g_trade.ResultPrice(),
               digits, sl, digits, tp,
               isLong ? g_slots[s].pUp : g_slots[s].pDown, riskFraction * 100.0);
  }

//+------------------------------------------------------------------+
//| Volume already committed to one side of one symbol.               |
//|                                                                   |
//| Magic is deliberately ignored: SYMBOL_VOLUME_LIMIT is the         |
//| broker's ceiling on the whole account's exposure to that side of  |
//| that symbol, so another EA's position or a manual one fills the   |
//| same bucket and has to be counted.                                |
//+------------------------------------------------------------------+
double SideVolume(const string sym, const bool isLong)
  {
   double v = 0.0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != sym)
         continue;
      if((g_pos.PositionType() == POSITION_TYPE_BUY) == isLong)
         v += g_pos.Volume();
     }

//--- pending orders count toward the same limit. This EA places none, but
//--- it is not the only thing that can be running on the account.
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderGetTicket(i) == 0 || OrderGetString(ORDER_SYMBOL) != sym)
         continue;
      bool buySide;
      switch((ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE))
        {
         case ORDER_TYPE_BUY:
         case ORDER_TYPE_BUY_LIMIT:
         case ORDER_TYPE_BUY_STOP:
         case ORDER_TYPE_BUY_STOP_LIMIT: buySide = true;  break;
         default:                        buySide = false; break;
        }
      if(buySide == isLong)
         v += OrderGetDouble(ORDER_VOLUME_CURRENT);
     }

   return v;
  }

//+------------------------------------------------------------------+
//| What one lot loses if price travels `distance` against you.      |
//|                                                                  |
//| OrderCalcProfit is asked first: it is the terminal's own          |
//| valuation for THIS symbol, converted into the account currency,   |
//| which is the thing SYMBOL_TRADE_TICK_VALUE only approximates. The |
//| tick arithmetic stays as a fallback for when the call fails.      |
//|                                                                  |
//| They are then compared, because a large disagreement means the    |
//| contract data is not describing this instrument - which is what   |
//| the strategy tester serves up on the first run after the symbol   |
//| is changed, still holding the previous symbol's figures. Entries  |
//| are unaffected by that, so nothing looks wrong; only the stake is |
//| computed from another instrument's tick value, and the run comes  |
//| out bad for no visible reason. Now it says so.                    |
//+------------------------------------------------------------------+
double LossPerLot(const string sym, const double distance)
  {
   if(distance <= 0.0)
      return 0.0;

   double tickValue = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double byTicks   = (tickSize > 0.0 && tickValue > 0.0)
                      ? (distance / tickSize) * tickValue
                      : 0.0;

   double byCalc = 0.0;
   double price  = SymbolInfoDouble(sym, SYMBOL_ASK);
   double profit = 0.0;
   if(price > 0.0 &&
      OrderCalcProfit(ORDER_TYPE_BUY, sym, 1.0, price, price - distance, profit)
      && profit < 0.0)
      byCalc = -profit;

   if(byCalc <= 0.0 && byTicks <= 0.0)
     {
      PrintFormat("MintyAI %s: cannot size a position - the broker reports "
                  "neither usable tick data nor a profit calculation", sym);
      return 0.0;
     }
   if(byCalc <= 0.0)
      return byTicks;

   if(byTicks > 0.0)
     {
      double ratio = byCalc / byTicks;
      if(ratio < 0.9 || ratio > 1.1)
         PrintFormat("MintyAI %s: SIZING INPUTS DISAGREE - the contract data "
                     "says %.2f per lot over %.5f, the terminal's own "
                     "calculation says %.2f. Using %.2f. In the tester this "
                     "means the terminal is still holding another symbol's "
                     "contract specification: restart it before trusting this "
                     "run.", sym, byTicks, distance, byCalc, byCalc);
     }
   return byCalc;
  }

//+------------------------------------------------------------------+
//| Lots from the stop distance and the model's risk fraction.       |
//| Never rounds up: an unfundable trade is skipped, not shrunk into |
//| a bigger risk than the model asked for.                          |
//+------------------------------------------------------------------+
double CalcLot(const string sym, const bool isLong, double stopDistance,
               double riskFraction)
  {
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskCash = equity * riskFraction;

   double lossPerLot = LossPerLot(sym, stopDistance);
   if(lossPerLot <= 0.0)
      return 0.0;

   double lot     = riskCash / lossPerLot;
   double desired = lot;   // before any rounding, for the log lines below

   double minLot  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   if(stepLot <= 0.0)
      stepLot = 0.01;

//--- Two different ceilings, and both have to be respected.
//---
//--- SYMBOL_VOLUME_MAX caps a SINGLE order. SYMBOL_VOLUME_LIMIT caps the
//--- AGGREGATE volume held in one direction on this symbol - every position
//--- and pending order on it, whoever opened them. Sizing that respected
//--- only the first is what the server rejected with "Volume limit
//--- reached": the order never reached the book and the signal was lost
//--- outright, which is strictly worse than taking the largest size the
//--- broker will actually accept.
//---
//--- Clamping here can only ever REDUCE the stake, never raise it, which is
//--- the same direction as the existing max-lot clamp. The model's number
//--- stays the ceiling; the broker's is applied under it.
//--- A limit of 0 means the broker sets no per-side ceiling at all, which is
//--- the common case - not a ceiling of zero.
   double cap   = maxLot;
   double limit = SymbolInfoDouble(sym, SYMBOL_VOLUME_LIMIT);
   double used  = 0.0;
   if(limit > 0.0)
     {
      used = SideVolume(sym, isLong);
      cap  = MathMin(cap, limit - used);
     }

   if(cap < minLot)
     {
      PrintFormat("MintyAI %s: skip, no room on the %s side - the broker caps "
                  "this symbol at %.2f lots and %.2f is already open",
                  sym, isLong ? "buy" : "sell", limit, used);
      return 0.0;
     }

   bool capped = (lot > cap);
   if(capped)
      lot = cap;

   lot = MathFloor(lot / stepLot) * stepLot;
   lot = NormalizeDouble(lot, 2);

//--- Only when a ceiling actually bound. A sub-minimum size rounding down to
//--- zero is the risk budget being too small, not the broker capping
//--- anything, and reporting it as a cap points the reader at the wrong
//--- number - the line below says what really happened.
   if(capped)
      PrintFormat("MintyAI %s: %.2f lots trimmed to %.2f by the broker's volume "
                  "ceiling (%.2f per order, %.2f per side)",
                  sym, desired, lot, maxLot, limit);

   if(lot < minLot)
     {
      if(InpVerbose)
         PrintFormat("MintyAI %s: skip, risk budget %.2f allows %.4f lots but the "
                     "minimum is %.2f", sym, riskCash, desired, minLot);
      return 0.0;
     }

//--- refuse if the margin is not comfortably there. With several symbols
//--- running this also stops the first one to fire from eating the margin
//--- the others need.
   double margin = 0.0;
   double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
   if(OrderCalcMargin(ORDER_TYPE_BUY, sym, lot, ask, margin))
     {
      double free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
      if(margin > free * 0.5)
        {
         PrintFormat("MintyAI %s: skip, margin %.2f exceeds half of free margin "
                     "%.2f", sym, margin, free);
         return 0.0;
        }
     }
   return lot;
  }

//+------------------------------------------------------------------+
//| Open positions: the horizon exit, and nothing else.              |
//|                                                                  |
//| No break-even, no trailing. A model priced one specific bet - a  |
//| fixed stop, a fixed target, a fixed number of bars. Moving the   |
//| stop turns it into a different bet whose probabilities were      |
//| never estimated, and it is exactly the kind of tinkering that    |
//| looks prudent while quietly cutting the winners.                 |
//+------------------------------------------------------------------+
void ManageOpenPositions()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i))
         continue;
      if(g_pos.Magic() != InpMagic)
         continue;
      if(SlotOf(g_pos.Symbol()) < 0)
         continue;

      //--- Counted in BARS, not seconds. The horizon the model was trained
      //--- on is a bar count, and bars do not advance over the weekend -
      //--- which is also why this EA needs no weekend handling.
      int bars = BarsSince(g_pos.Symbol(), g_pos.Time());
      if(bars >= MINTYAI_MODEL_HORIZON_BARS)
        {
         ulong  ticket = g_pos.Ticket();
         string sym    = g_pos.Symbol();
         if(g_trade.PositionClose(ticket))
            PrintFormat("MintyAI %s: horizon exit after %d bars (ticket %I64u)",
                        sym, bars, ticket);
        }
     }
  }

//+------------------------------------------------------------------+
//| Account-level guards, at the levels training measured            |
//+------------------------------------------------------------------+
void UpdateEquityGuards()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   int stamp = DayStamp(TimeCurrent());
   if(stamp != g_dayStamp)
     {
      g_dayStamp   = stamp;
      g_dayStartEq = equity;

      //--- Both halts lift on the day boundary.
      //---
      //--- The drawdown halt MUST re-anchor its peak as it lifts. A halted
      //--- EA holds no positions, so its equity cannot move: measuring the
      //--- next drawdown from the old peak would re-trip the halt on the
      //--- very next tick and it would never trade again. That is what "it
      //--- stops trading altogether" was.
      if(g_halted)
        {
         if(g_haltKind == HALT_DRAWDOWN)
           {
            g_peakEquity = equity;
            PrintFormat("MintyAI: new trading day - drawdown halt lifted, peak "
                        "re-anchored to %.2f. The next %.2f%% is measured from "
                        "here.", equity, g_maxDdPct);
           }
         else
            Print("MintyAI: new trading day - daily loss halt lifted.");

         g_halted     = false;
         g_haltKind   = HALT_NONE;
         g_haltReason = "";
        }
     }

   if(equity > g_peakEquity)
      g_peakEquity = equity;

   if(g_halted)
      return;

   if(g_dailyLossPct > 0.0 && g_dayStartEq > 0.0)
     {
      double lossPct = (g_dayStartEq - equity) / g_dayStartEq * 100.0;
      if(lossPct >= g_dailyLossPct)
        {
         Halt(HALT_DAILY, "daily loss limit");
         CloseAll("daily loss limit");
         return;
        }
     }

   if(g_maxDdPct > 0.0 && g_peakEquity > 0.0)
     {
      double ddPct = (g_peakEquity - equity) / g_peakEquity * 100.0;
      if(ddPct >= g_maxDdPct)
        {
         Halt(HALT_DRAWDOWN, "max drawdown");
         CloseAll("max drawdown");
        }
     }
  }

void Halt(const int kind, string reason)
  {
   g_halted     = true;
   g_haltKind   = kind;
   g_haltReason = reason;
   PrintFormat("MintyAI: HALTED (%s). No new positions until the next trading "
               "day.", reason);
  }

void CloseAll(string why)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i))
         continue;
      if(g_pos.Magic() != InpMagic || SlotOf(g_pos.Symbol()) < 0)
         continue;
      string sym = g_pos.Symbol();
      ulong  tk  = g_pos.Ticket();
      if(g_trade.PositionClose(tk))
         PrintFormat("MintyAI %s: closed %I64u (%s)", sym, tk, why);
     }
  }

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
//--- which running slot a broker symbol belongs to, or -1
int SlotOf(const string sym)
  {
   for(int i = 0; i < g_slotCount; i++)
      if(g_slots[i].symbol == sym)
         return i;
   return -1;
  }

int CountPositions(const string sym)
  {
   int n = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i))
         continue;
      if(g_pos.Symbol() == sym && g_pos.Magic() == InpMagic)
         n++;
     }
   return n;
  }

int TotalPositions()
  {
   int n = 0;
   for(int i = 0; i < g_slotCount; i++)
      n += CountPositions(g_slots[i].symbol);
   return n;
  }

//--- closed bars between `when` and now, on the model's timeframe
int BarsSince(const string sym, datetime when)
  {
   int shift = iBarShift(sym, MINTYAI_TF, when, false);
   return (shift < 0 ? 0 : shift);
  }

bool RespectsStopLevel(const string sym, double price, double level)
  {
   long stopLevel = SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);
   if(stopLevel <= 0)
      return true;
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   return MathAbs(price - level) >= stopLevel * point;
  }

//--- The hour as the TRAINING broker's clock would show it. Same shift the
//--- features get, so the mask and the model always agree on what hour it
//--- is - and on the training broker the shift is zero.
int CurrentHour()
  {
   MqlDateTime t;
   TimeToStruct((datetime)((long)TimeCurrent() + MintyAITimeShift()), t);
   return t.hour;
  }

//--- the hours this symbol's model made money in during validation
bool HourAllowed(const int s)
  {
   return ((g_slots[s].hourMask >> CurrentHour()) & 1) != 0;
  }

int DayStamp(datetime t)
  {
   MqlDateTime s;
   TimeToStruct(t, s);
   return s.year * 10000 + s.mon * 100 + s.day;
  }

string PeriodTag(ENUM_TIMEFRAMES tf)
  {
   string s = EnumToString(tf);
   StringReplace(s, "PERIOD_", "");
   return s;
  }

string RiskName(ENUM_MINTYAI_RISK r)
  {
   switch(r)
     {
      case MINTYAI_RISK_LOW:    return "Low";
      case MINTYAI_RISK_MEDIUM: return "Medium";
      case MINTYAI_RISK_HIGH:   return "High";
     }
   return "?";
  }

//+------------------------------------------------------------------+
//| A symbol name reduced to what brokers cannot decorate: letters   |
//| and digits, upper-cased. EURUSD.r, eurusd.ecn, EUR/USD and       |
//| mEURUSD_i all collapse onto something containing EURUSD.         |
//+------------------------------------------------------------------+
string NormalizeSymbol(const string s)
  {
   string up = s;
   StringToUpper(up);

   string out = "";
   int    n   = StringLen(up);
   for(int i = 0; i < n; i++)
     {
      ushort c = StringGetCharacter(up, i);
      if((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
         out += ShortToString(c);
     }
   return out;
  }

//+------------------------------------------------------------------+
//| Which model covers the chart's symbol, or -1.                    |
//|                                                                  |
//| Normally an exact match: the model table holds the standard      |
//| names, and most brokers use them.                                |
//|                                                                  |
//| Otherwise compare on letters and digits alone, upper-cased, so   |
//| the decorations brokers add come out in the wash: EURUSD.r,      |
//| eurusd.ecn, mEURUSD and EUR/USD all resolve. Two matches is a    |
//| guess, and a guess here would trade one instrument on another's  |
//| calibration - so it refuses and leaves it to the Model setting.  |
//|                                                                  |
//| What this CANNOT do is match a renamed instrument. GOLD shares   |
//| no letters with XAUUSD; nothing here will ever connect the two,  |
//| and no amount of cleverness would - that is what naming the      |
//| model in the settings is for.                                    |
//+------------------------------------------------------------------+
int ResolveModel(const string chartSym)
  {
   int exact = MintyAIFindModel(chartSym);
   if(exact >= 0)
      return exact;

   string want = NormalizeSymbol(chartSym);

   int found = -1, hits = 0;
   for(int i = 0; i < MINTYAI_MODEL_COUNT; i++)
     {
      string m = NormalizeSymbol(MintyAIModels[i].symbol);
      if(StringLen(m) >= 3 && StringFind(want, m) >= 0)
        {
         found = i;
         hits++;
        }
     }

   if(hits == 1)
     {
      PrintFormat("MintyAI: chart symbol '%s' matched the '%s' model - your "
                  "broker spells that instrument differently. Trading '%s'.",
                  chartSym, MintyAIModels[found].symbol, chartSym);
      return found;
     }
   if(hits > 1)
      PrintFormat("MintyAI: '%s' matches %d models - too ambiguous to pick one "
                  "safely. Name the one you want in the Model setting and it "
                  "will stop guessing.", chartSym, hits);
   return -1;
  }

//+------------------------------------------------------------------+
//| Chart theme                                                      |
//+------------------------------------------------------------------+
void ApplyLightTheme()
  {
   ChartSetInteger(0, CHART_MODE,               CHART_CANDLES);
   ChartSetInteger(0, CHART_COLOR_BACKGROUND,   UI_BG);
   ChartSetInteger(0, CHART_COLOR_FOREGROUND,   UI_FG);
   ChartSetInteger(0, CHART_COLOR_GRID,         UI_GRID);
   ChartSetInteger(0, CHART_COLOR_CHART_UP,     UI_BULL_LINE);
   ChartSetInteger(0, CHART_COLOR_CHART_DOWN,   UI_BEAR_LINE);
   ChartSetInteger(0, CHART_COLOR_CHART_LINE,   UI_NEUTRAL);
   ChartSetInteger(0, CHART_COLOR_CANDLE_BULL,  UI_BULL_FILL);
   ChartSetInteger(0, CHART_COLOR_CANDLE_BEAR,  UI_BEAR_FILL);
   ChartSetInteger(0, CHART_COLOR_VOLUME,       UI_NEUTRAL);
   ChartSetInteger(0, CHART_COLOR_BID,          UI_NEUTRAL);
   ChartSetInteger(0, CHART_COLOR_ASK,          UI_NEUTRAL);
   ChartSetInteger(0, CHART_COLOR_LAST,         UI_FG);
   ChartSetInteger(0, CHART_COLOR_STOP_LEVEL,   UI_BEAR_LINE);
   ChartSetInteger(0, CHART_SHOW_GRID,          true);
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
//| Panel primitives                                                 |
//+------------------------------------------------------------------+
void UiRect(string tag, int x, int y, int w, int h, color bg, color border)
  {
   string name = UI_PREFIX + tag;
   if(ObjectFind(0, name) < 0)
     {
      ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER,      CORNER_LEFT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, name, OBJPROP_BACK,        false);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE,  false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN,      true);
     }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE,     (w < 1 ? 1 : w));
   ObjectSetInteger(0, name, OBJPROP_YSIZE,     (h < 1 ? 1 : h));
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR,   bg);
   ObjectSetInteger(0, name, OBJPROP_COLOR,     border);
  }

void UiText(string tag, int x, int y, string text, color clr, int size,
            string font, ENUM_ANCHOR_POINT anchor)
  {
   string name = UI_PREFIX + tag;
   if(ObjectFind(0, name) < 0)
     {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER,     CORNER_LEFT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
     }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR,    anchor);
   ObjectSetInteger(0, name, OBJPROP_COLOR,     clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,  size);
   ObjectSetString(0, name, OBJPROP_FONT,       font);
   ObjectSetString(0, name, OBJPROP_TEXT,       text);
  }

//--- one label/value line: dim key on the left, value right-aligned
void UiRow(int idx, int &y, string key, string value, color valueClr)
  {
   string tag = "r" + IntegerToString(idx);
   UiText(tag + "k", UI_X + UI_PAD, y, key, UI_KEY_TX, 8,
          "Segoe UI", ANCHOR_LEFT_UPPER);
   UiText(tag + "v", UI_X + UI_W - UI_PAD, y, value, valueClr, 8,
          "Consolas", ANCHOR_RIGHT_UPPER);
   y += UI_ROW_H;
  }

//+------------------------------------------------------------------+
//| Status panel                                                     |
//+------------------------------------------------------------------+
void DrawPanel()
  {
//--- Objects are rewritten, not recreated, but this still runs on every
//--- tick. A few frames a second is plenty for a human and keeps visual
//--- backtests from crawling.
   static ulong lastDraw = 0;
   ulong now = GetTickCount64();
   if(now - lastDraw < 250)
      return;
   lastDraw = now;

//--- there is no slot to report on, so report why instead
   if(g_idle)
     {
      DrawIdlePanel();
      return;
     }

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double ddPct  = (g_peakEquity > 0.0
                    ? (g_peakEquity - equity) / g_peakEquity * 100.0 : 0.0);
   int    open   = TotalPositions();

   string state    = g_halted ? "HALTED" : (open > 0 ? "IN POSITION" : "WAITING");
   color  stateClr = g_halted ? UI_BEAR_LINE : (open > 0 ? UI_HEADER : UI_KEY_TX);

//--- the model gets its own row whenever it is not simply the chart's
   bool showModel = (g_fallback || g_chosen);
   int rows   = 8 + (g_halted ? 1 : 0) + (showModel ? 1 : 0);
   int barsY  = UI_Y + UI_HEADER_H + UI_PAD;
   int rowsY  = barsY + 20;
   int height = UI_HEADER_H + UI_PAD + 20 + rows * UI_ROW_H + UI_PAD;

   UiRect("card", UI_X, UI_Y, UI_W, height, UI_CARD, UI_BORDER);
   UiRect("hdr",  UI_X, UI_Y, UI_W, UI_HEADER_H, UI_HEADER, UI_HEADER);
   UiText("title", UI_X + UI_PAD, UI_Y + 8,
          "MINTYAI   " + g_slots[0].symbol + " " + PeriodTag(MINTYAI_TF),
          UI_HEADER_TX, 9, "Segoe UI Semibold", ANCHOR_LEFT_UPPER);
   UiText("state", UI_X + UI_W - UI_PAD, UI_Y + 9,
          RiskName(InpRisk) + " risk", UI_HEADER_TX, 8, "Segoe UI",
          ANCHOR_RIGHT_UPPER);

//--- probability bar: down | flat | up, widths in proportion
   int barW = UI_W - 2 * UI_PAD;
   int wD   = (int)MathRound(barW * g_slots[0].pDown);
   int wF   = (int)MathRound(barW * g_slots[0].pFlat);
   int wU   = barW - wD - wF;
   UiRect("barD", UI_X + UI_PAD,           barsY, wD, 7, UI_BEAR_LINE, UI_BEAR_LINE);
   UiRect("barF", UI_X + UI_PAD + wD,      barsY, wF, 7, UI_NEUTRAL,   UI_NEUTRAL);
   UiRect("barU", UI_X + UI_PAD + wD + wF, barsY, wU, 7, UI_BULL_LINE, UI_BULL_LINE);

   int y = rowsY;
   int i = 0;

   UiRow(i++, y, "state", state, stateClr);
   if(showModel)
      UiRow(i++, y, "model",
            g_slots[0].plain + (g_chosen ? " - selected" : " - borrowed"),
            g_chosen ? UI_VAL_TX : UI_WARN);
   UiRow(i++, y, "p  down / flat / up",
         StringFormat("%.2f  %.2f  %.2f", g_slots[0].pDown, g_slots[0].pFlat,
                      g_slots[0].pUp), UI_VAL_TX);
   UiRow(i++, y, g_slots[0].hasRiskHead ? "outcome  mu / sigma"
                                        : "outcome  (from probs)",
         StringFormat("%+.3f  %.3f", g_slots[0].mu, g_slots[0].sigma), UI_VAL_TX);
   UiRow(i++, y, "best EV", StringFormat("%.6f", g_slots[0].ev),
         g_slots[0].ev > 0.0 ? UI_BULL_LINE : UI_KEY_TX);
   UiRow(i++, y, "next risk",
         StringFormat("%.3f%%  of %.2f%%", g_slots[0].risk * 100.0,
                      g_slots[0].maxRiskPct),
         g_slots[0].risk > 0.0 ? UI_VAL_TX : UI_KEY_TX);
   UiRow(i++, y, "ATR(14)", StringFormat("%.5f", g_slots[0].atr), UI_VAL_TX);
   UiRow(i++, y, "spread",
         StringFormat("%d pts",
                      (int)SymbolInfoInteger(g_slots[0].symbol, SYMBOL_SPREAD)),
         UI_VAL_TX);
   UiRow(i++, y, "equity", StringFormat("%.2f", equity), UI_VAL_TX);
   UiRow(i++, y, "drawdown",
         StringFormat("%.2f%%  of %.2f%%", ddPct, g_maxDdPct),
         ddPct >= g_maxDdPct * 0.5 ? UI_WARN : UI_VAL_TX);
   if(g_halted)
      UiRow(i++, y, "halt reason", g_haltReason, UI_BEAR_LINE);

   DropRowsFrom(i);
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
//| The row count changes when a halt lifts or a slot drops. Without |
//| this the orphaned labels keep drawing below the card, which      |
//| reads as live.                                                   |
//+------------------------------------------------------------------+
void DropRowsFrom(const int first)
  {
   for(int drop = first; drop < UI_MAX_ROWS; drop++)
     {
      string tag = UI_PREFIX + "r" + IntegerToString(drop);
      if(ObjectFind(0, tag + "k") >= 0)
         ObjectDelete(0, tag + "k");
      if(ObjectFind(0, tag + "v") >= 0)
         ObjectDelete(0, tag + "v");
     }
  }

//+------------------------------------------------------------------+
//| Loaded with no model for this chart. Says so, rather than        |
//| leaving a blank panel that looks like a live one waiting.        |
//+------------------------------------------------------------------+
void DrawIdlePanel()
  {
   int height = UI_HEADER_H + UI_PAD + 2 * UI_ROW_H + UI_PAD;

   UiRect("card", UI_X, UI_Y, UI_W, height, UI_CARD, UI_BORDER);
   UiRect("hdr",  UI_X, UI_Y, UI_W, UI_HEADER_H, UI_HEADER, UI_HEADER);
   UiText("title", UI_X + UI_PAD, UI_Y + 8, "MINTYAI   " + _Symbol,
          UI_HEADER_TX, 9, "Segoe UI Semibold", ANCHOR_LEFT_UPPER);
   UiText("state", UI_X + UI_W - UI_PAD, UI_Y + 9, "IDLE",
          UI_HEADER_TX, 8, "Segoe UI", ANCHOR_RIGHT_UPPER);

   int y = UI_Y + UI_HEADER_H + UI_PAD;
   UiRow(0, y, "not trading", g_idleWhy, UI_WARN);
   UiRow(1, y, "supported symbols", "listed in the Experts log", UI_KEY_TX);

   DropRowsFrom(2);
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
