//+------------------------------------------------------------------+
//|                                               MintyAIFeatures.mqh  |
//|   Feature engine shared by the MintyAI EA and the Python trainer.  |
//|                                                                  |
//|   EVERY formula here is mirrored exactly in python/mintyai/         |
//|   features.py. If you change one side you MUST change the other  |
//|   and bump MINTYAI_FEATURE_VERSION, otherwise the model will be     |
//|   fed inputs it was never trained on and the EA will refuse to    |
//|   start (signature mismatch).                                     |
//+------------------------------------------------------------------+
#property strict

//--- v1: hour-of-day and day-of-week come from the SERVER clock of the
//--- broker the model was trained on, both here and in the exported
//--- training data. No conversion to UTC - that was v2, reverted.
//---
//--- On the training broker this is exactly the 2.0 behaviour: raw server
//--- time, no timezone plumbing, the two sides agreeing by construction.
//--- On a broker in a different zone the EA shifts live bars by the
//--- difference between the two servers, so the model still sees the hour
//--- its training data would have shown. See MintyAISetTimeShift().
#define MINTYAI_FEATURE_VERSION 1
#define MINTYAI_FEATURE_COUNT   24

//--- Number of leading bars whose features are unreliable, and the amount
//--- of history the EA must load before every inference.
//---
//--- Sized by the slowest recursive filter, EMA200 (alpha = 2/201). Its
//--- seed decays as (1-alpha)^n, so after 600 bars ~0.25% of the seed
//--- survives - enough to shift d_ema200_atr by ~0.07 against a model
//--- trained on full history. At 2000 bars the residual is ~2e-9.
//--- The parity check cannot catch this, because both sides of it see
//--- the same full series; only this constant prevents it.
#define MINTYAI_WARMUP          2000

//+------------------------------------------------------------------+
//| Server time -> UTC                                               |
//|                                                                  |
//| Brokers run their servers on whatever timezone they like, and    |
//| MqlRates.time is in it. Four of the features below are hour-of-  |
//| day and day-of-week, so unless every terminal agrees on what     |
//| "hour" means, the same model reads a different session on every  |
//| broker. Everything is therefore converted to UTC first: training |
//| data at export, live bars here.                                  |
//|                                                                  |
//| Cached because it is called for every bar of every inference.    |
//+------------------------------------------------------------------+
#define MINTYAI_UTC_UNKNOWN  LONG_MIN

long g_mintyaiUtcOffset = MINTYAI_UTC_UNKNOWN;

//--- Ask the terminal. Returns MINTYAI_UTC_UNKNOWN when the answer is not
//--- believable, which is the normal case in the strategy tester: there
//--- TimeCurrent() is a modelled historical time while TimeGMT() is not,
//--- so their difference is years rather than hours. Silently treating
//--- that as zero is what shifted backtest features by the broker's whole
//--- timezone against the UTC-stamped training data.
long MintyAIDetectUtcOffset()
  {
   long hrs = (long)MathRound(((double)TimeCurrent() - (double)TimeGMT()) / 3600.0);
   if(hrs < -12 || hrs > 14)
      return MINTYAI_UTC_UNKNOWN;
   return hrs * 3600;
  }

//--- Set explicitly at startup, so the caller can fall back to the offset
//--- recorded when the model was trained rather than guessing.
void MintyAISetUtcOffset(const long secs)
  {
   g_mintyaiUtcOffset = secs;
  }

long MintyAIServerUtcOffset()
  {
   if(g_mintyaiUtcOffset != MINTYAI_UTC_UNKNOWN)
      return g_mintyaiUtcOffset;

   long det = MintyAIDetectUtcOffset();
   g_mintyaiUtcOffset = (det == MINTYAI_UTC_UNKNOWN ? 0 : det);
   return g_mintyaiUtcOffset;
  }

//+------------------------------------------------------------------+
//| Seconds to add to this broker's bar times so they read as the     |
//| TRAINING broker's server clock did.                               |
//|                                                                   |
//| Zero on the broker the model was trained on, which is the normal  |
//| case and reproduces the 2.0 behaviour exactly. Elsewhere it is    |
//| (training offset - this broker's offset): a model trained on a    |
//| UTC+3 server, run on a UTC+2 one, gets +1h so that 09:00 here is  |
//| presented as the 10:00 the model was taught to recognise.         |
//+------------------------------------------------------------------+
long g_mintyaiTimeShift = 0;

void MintyAISetTimeShift(const long secs)
  {
   g_mintyaiTimeShift = secs;
  }

long MintyAITimeShift()
  {
   return g_mintyaiTimeShift;
  }

//--- feature order. Do not reorder without bumping the version.
const string MintyAIFeatureNames[MINTYAI_FEATURE_COUNT] =
  {
   "ret_atr",       // 0  bar return in ATR units
   "range_atr",     // 1  (high-low) / ATR
   "body_ratio",    // 2  (close-open) / range
   "upper_wick",    // 3  upper shadow fraction of range
   "lower_wick",    // 4  lower shadow fraction of range
   "gap_atr",       // 5  (open - prev close) / ATR
   "vol_rel",       // 6  log(volume / SMA50(volume))
   "rsi14",         // 7  Wilder RSI(14)/100 - 0.5
   "macd_hist_atr", // 8  MACD(12,26,9) histogram / ATR
   "d_ema20_atr",   // 9  (close - EMA20) / ATR
   "d_ema50_atr",   // 10 (close - EMA50) / ATR
   "d_ema200_atr",  // 11 (close - EMA200) / ATR
   "atr_pct",       // 12 ATR / close * 100
   "vol_ratio",     // 13 realised vol 20 / realised vol 100
   "bb_pctb",       // 14 Bollinger %B(20,2) - 0.5
   "stoch_k",       // 15 Stochastic %K(14)/100 - 0.5
   "adx14",         // 16 Wilder ADX(14)/100
   "mom5_atr",      // 17 5-bar momentum / ATR
   "mom20_atr",     // 18 20-bar momentum / ATR
   "hl_pos50",      // 19 position in 50-bar high/low channel - 0.5
   "sin_hour",      // 20 intraday seasonality
   "cos_hour",      // 21
   "sin_dow",       // 22 weekly seasonality
   "cos_dow"        // 23
  };

#define MINTYAI_EPS  1.0e-12
#define MINTYAI_CLIP 10.0

//+------------------------------------------------------------------+
//| FNV-1a 32-bit hash - identical implementation in Python.         |
//| Used to prove the EA and the trained model share a feature set.  |
//+------------------------------------------------------------------+
uint MintyAIFnv1a(const string s)
  {
   uint hash = 2166136261;
   uchar bytes[];
   int n = StringToCharArray(s, bytes, 0, WHOLE_ARRAY, CP_UTF8);
//--- StringToCharArray appends a terminating zero; exclude it
   if(n > 0 && bytes[n - 1] == 0)
      n--;
   for(int i = 0; i < n; i++)
     {
      hash ^= (uint)bytes[i];
      hash *= 16777619;
     }
   return hash;
  }

//+------------------------------------------------------------------+
//| Signature over feature version + names. Must match the value the |
//| trainer baked into MintyAIModelConfig.mqh.                         |
//+------------------------------------------------------------------+
uint MintyAIFeatureSignature()
  {
   string s = "v" + IntegerToString(MINTYAI_FEATURE_VERSION);
   for(int i = 0; i < MINTYAI_FEATURE_COUNT; i++)
      s += "|" + MintyAIFeatureNames[i];
   return MintyAIFnv1a(s);
  }

//+------------------------------------------------------------------+
//| Primitive series helpers. All arrays are CHRONOLOGICAL:          |
//| index 0 = oldest bar. Python uses the same convention.           |
//+------------------------------------------------------------------+

//--- exponential moving average, seeded with the first sample
void MintyAIEma(const double &src[], int n, int period, double &dst[])
  {
   ArrayResize(dst, n);
   if(n <= 0)
      return;
   double alpha = 2.0 / (period + 1.0);
   dst[0] = src[0];
   for(int i = 1; i < n; i++)
      dst[i] = alpha * src[i] + (1.0 - alpha) * dst[i - 1];
  }

//--- Wilder's smoothing (RMA), seeded with the first sample
void MintyAIRma(const double &src[], int n, int period, double &dst[])
  {
   ArrayResize(dst, n);
   if(n <= 0)
      return;
   double alpha = 1.0 / period;
   dst[0] = src[0];
   for(int i = 1; i < n; i++)
      dst[i] = alpha * src[i] + (1.0 - alpha) * dst[i - 1];
  }

//--- simple moving average; the first (period-1) values use a partial window
void MintyAISma(const double &src[], int n, int period, double &dst[])
  {
   ArrayResize(dst, n);
   double sum = 0.0;
   for(int i = 0; i < n; i++)
     {
      sum += src[i];
      if(i >= period)
         sum -= src[i - period];
      int cnt = (i + 1 < period ? i + 1 : period);
      dst[i] = sum / cnt;
     }
  }

//--- rolling standard deviation (population), partial window at the start
void MintyAIStdev(const double &src[], int n, int period, double &dst[])
  {
   ArrayResize(dst, n);
   double sum = 0.0, sum2 = 0.0;
   for(int i = 0; i < n; i++)
     {
      sum  += src[i];
      sum2 += src[i] * src[i];
      if(i >= period)
        {
         sum  -= src[i - period];
         sum2 -= src[i - period] * src[i - period];
        }
      int cnt = (i + 1 < period ? i + 1 : period);
      double mean = sum / cnt;
      double var  = sum2 / cnt - mean * mean;
      dst[i] = (var > 0.0 ? MathSqrt(var) : 0.0);
     }
  }

//--- rolling max / min over `period` bars (partial window at the start)
void MintyAIRollMax(const double &src[], int n, int period, double &dst[])
  {
   ArrayResize(dst, n);
   for(int i = 0; i < n; i++)
     {
      int start = i - period + 1;
      if(start < 0)
         start = 0;
      double m = src[start];
      for(int j = start + 1; j <= i; j++)
         if(src[j] > m)
            m = src[j];
      dst[i] = m;
     }
  }

void MintyAIRollMin(const double &src[], int n, int period, double &dst[])
  {
   ArrayResize(dst, n);
   for(int i = 0; i < n; i++)
     {
      int start = i - period + 1;
      if(start < 0)
         start = 0;
      double m = src[start];
      for(int j = start + 1; j <= i; j++)
         if(src[j] < m)
            m = src[j];
      dst[i] = m;
     }
  }

//--- clamp + scrub non-finite values
double MintyAIClean(double x)
  {
   if(!MathIsValidNumber(x))
      return 0.0;
   if(x >  MINTYAI_CLIP)
      return  MINTYAI_CLIP;
   if(x < -MINTYAI_CLIP)
      return -MINTYAI_CLIP;
   return x;
  }

//+------------------------------------------------------------------+
//| True Range and ATR (Wilder, period 14).                          |
//+------------------------------------------------------------------+
void MintyAIAtr(const double &h[], const double &l[], const double &c[], int n,
              int period, double &atr[])
  {
   double tr[];
   ArrayResize(tr, n);
   if(n > 0)
      tr[0] = h[0] - l[0];
   for(int i = 1; i < n; i++)
     {
      double a = h[i] - l[i];
      double b = MathAbs(h[i] - c[i - 1]);
      double d = MathAbs(l[i] - c[i - 1]);
      tr[i] = MathMax(a, MathMax(b, d));
     }
   MintyAIRma(tr, n, period, atr);
  }

//+------------------------------------------------------------------+
//| Wilder RSI.                                                      |
//+------------------------------------------------------------------+
void MintyAIRsi(const double &c[], int n, int period, double &rsi[])
  {
   double up[], dn[], rup[], rdn[];
   ArrayResize(up, n);
   ArrayResize(dn, n);
   if(n > 0)
     {
      up[0] = 0.0;
      dn[0] = 0.0;
     }
   for(int i = 1; i < n; i++)
     {
      double d = c[i] - c[i - 1];
      up[i] = (d > 0.0 ?  d : 0.0);
      dn[i] = (d < 0.0 ? -d : 0.0);
     }
   MintyAIRma(up, n, period, rup);
   MintyAIRma(dn, n, period, rdn);
   ArrayResize(rsi, n);
   for(int i = 0; i < n; i++)
     {
      double denom = rup[i] + rdn[i];
      rsi[i] = (denom > MINTYAI_EPS ? 100.0 * rup[i] / denom : 50.0);
     }
  }

//+------------------------------------------------------------------+
//| Wilder ADX.                                                      |
//+------------------------------------------------------------------+
void MintyAIAdx(const double &h[], const double &l[], const double &c[], int n,
              int period, double &adx[])
  {
   double pdm[], ndm[], tr[];
   ArrayResize(pdm, n);
   ArrayResize(ndm, n);
   ArrayResize(tr,  n);
   if(n > 0)
     {
      pdm[0] = 0.0;
      ndm[0] = 0.0;
      tr[0]  = h[0] - l[0];
     }
   for(int i = 1; i < n; i++)
     {
      double upMove   = h[i] - h[i - 1];
      double downMove = l[i - 1] - l[i];
      pdm[i] = (upMove > downMove && upMove > 0.0 ? upMove : 0.0);
      ndm[i] = (downMove > upMove && downMove > 0.0 ? downMove : 0.0);
      double a = h[i] - l[i];
      double b = MathAbs(h[i] - c[i - 1]);
      double d = MathAbs(l[i] - c[i - 1]);
      tr[i] = MathMax(a, MathMax(b, d));
     }
   double spdm[], sndm[], str[], dx[];
   MintyAIRma(pdm, n, period, spdm);
   MintyAIRma(ndm, n, period, sndm);
   MintyAIRma(tr,  n, period, str);
   ArrayResize(dx, n);
   for(int i = 0; i < n; i++)
     {
      double denomTr = (str[i] > MINTYAI_EPS ? str[i] : MINTYAI_EPS);
      double pdi = 100.0 * spdm[i] / denomTr;
      double ndi = 100.0 * sndm[i] / denomTr;
      double s   = pdi + ndi;
      dx[i] = (s > MINTYAI_EPS ? 100.0 * MathAbs(pdi - ndi) / s : 0.0);
     }
   MintyAIRma(dx, n, period, adx);
  }

//+------------------------------------------------------------------+
//| Build the full feature matrix.                                   |
//|                                                                  |
//| Inputs are chronological arrays of length n (index 0 = oldest).  |
//| `out` is filled row-major: out[i*MINTYAI_FEATURE_COUNT + f].       |
//| Rows < MINTYAI_WARMUP are computed but must not be trusted.        |
//+------------------------------------------------------------------+
bool MintyAIBuildFeatures(const double &o[], const double &h[], const double &l[],
                        const double &c[], const double &vol[], const long &ts[],
                        int n, float &out[])
  {
   if(n <= 0)
      return false;

   double atr[], rsi[], adx[];
   MintyAIAtr(h, l, c, n, 14, atr);
   MintyAIRsi(c, n, 14, rsi);
   MintyAIAdx(h, l, c, n, 14, adx);

   double ema12[], ema26[], ema20[], ema50[], ema200[], macd[], signal[];
   MintyAIEma(c, n, 12,  ema12);
   MintyAIEma(c, n, 26,  ema26);
   MintyAIEma(c, n, 20,  ema20);
   MintyAIEma(c, n, 50,  ema50);
   MintyAIEma(c, n, 200, ema200);
   ArrayResize(macd, n);
   for(int i = 0; i < n; i++)
      macd[i] = ema12[i] - ema26[i];
   MintyAIEma(macd, n, 9, signal);

   double volSma[];
   MintyAISma(vol, n, 50, volSma);

//--- log returns and realised volatility
   double ret[], ret2[], rv20v[], rv100v[];
   ArrayResize(ret,  n);
   ArrayResize(ret2, n);
   ret[0]  = 0.0;
   ret2[0] = 0.0;
   for(int i = 1; i < n; i++)
     {
      double prev = (c[i - 1] > MINTYAI_EPS ? c[i - 1] : MINTYAI_EPS);
      ret[i]  = MathLog(c[i] / prev);
      ret2[i] = ret[i] * ret[i];
     }
   MintyAISma(ret2, n, 20,  rv20v);
   MintyAISma(ret2, n, 100, rv100v);

   double sma20[], std20[], hh14[], ll14[], hh50[], ll50[];
   MintyAISma(c, n, 20, sma20);
   MintyAIStdev(c, n, 20, std20);
   MintyAIRollMax(h, n, 14, hh14);
   MintyAIRollMin(l, n, 14, ll14);
   MintyAIRollMax(h, n, 50, hh50);
   MintyAIRollMin(l, n, 50, ll50);

   ArrayResize(out, n * MINTYAI_FEATURE_COUNT);

   for(int i = 0; i < n; i++)
     {
      double a     = (atr[i] > MINTYAI_EPS ? atr[i] : MINTYAI_EPS);
      double range = h[i] - l[i];
      double rden  = (range > MINTYAI_EPS ? range : MINTYAI_EPS);
      double px    = (c[i] > MINTYAI_EPS ? c[i] : MINTYAI_EPS);

      double prevClose = (i > 0 ? c[i - 1] : c[i]);
      double c5        = (i >= 5  ? c[i - 5]  : c[0]);
      double c20       = (i >= 20 ? c[i - 20] : c[0]);

      double rv20  = MathSqrt(rv20v[i]);
      double rv100 = MathSqrt(rv100v[i]);

      double bbUp  = sma20[i] + 2.0 * std20[i];
      double bbLo  = sma20[i] - 2.0 * std20[i];
      double bbDen = bbUp - bbLo;
      bbDen = (bbDen > MINTYAI_EPS ? bbDen : MINTYAI_EPS);

      double stDen = hh14[i] - ll14[i];
      stDen = (stDen > MINTYAI_EPS ? stDen : MINTYAI_EPS);
      double chDen = hh50[i] - ll50[i];
      chDen = (chDen > MINTYAI_EPS ? chDen : MINTYAI_EPS);

      //--- calendar features from the raw epoch timestamp, so that MQL5 and
      //--- Python derive them with identical integer arithmetic
      long   sec  = ts[i];
      double hod  = (double)(sec % 86400) / 3600.0;           // hour of day
      long   days = sec / 86400;
      double dow  = (double)((days + 4) % 7);                 // 0 = Sunday

      double vRel = MathLog((vol[i] + 1.0) / (volSma[i] + 1.0));

      int b = i * MINTYAI_FEATURE_COUNT;
      out[b + 0]  = (float)MintyAIClean((c[i] - prevClose) / a);
      out[b + 1]  = (float)MintyAIClean(range / a);
      out[b + 2]  = (float)MintyAIClean((c[i] - o[i]) / rden);
      out[b + 3]  = (float)MintyAIClean((h[i] - MathMax(o[i], c[i])) / rden);
      out[b + 4]  = (float)MintyAIClean((MathMin(o[i], c[i]) - l[i]) / rden);
      out[b + 5]  = (float)MintyAIClean((o[i] - prevClose) / a);
      out[b + 6]  = (float)MintyAIClean(vRel);
      out[b + 7]  = (float)MintyAIClean(rsi[i] / 100.0 - 0.5);
      out[b + 8]  = (float)MintyAIClean((macd[i] - signal[i]) / a);
      out[b + 9]  = (float)MintyAIClean((c[i] - ema20[i]) / a);
      out[b + 10] = (float)MintyAIClean((c[i] - ema50[i]) / a);
      out[b + 11] = (float)MintyAIClean((c[i] - ema200[i]) / a);
      out[b + 12] = (float)MintyAIClean(a / px * 100.0);
      out[b + 13] = (float)MintyAIClean(rv20 / (rv100 + MINTYAI_EPS));
      out[b + 14] = (float)MintyAIClean((c[i] - bbLo) / bbDen - 0.5);
      out[b + 15] = (float)MintyAIClean((c[i] - ll14[i]) / stDen - 0.5);
      out[b + 16] = (float)MintyAIClean(adx[i] / 100.0);
      out[b + 17] = (float)MintyAIClean((c[i] - c5)  / a);
      out[b + 18] = (float)MintyAIClean((c[i] - c20) / a);
      out[b + 19] = (float)MintyAIClean((c[i] - ll50[i]) / chDen - 0.5);
      out[b + 20] = (float)MintyAIClean(MathSin(2.0 * M_PI * hod / 24.0));
      out[b + 21] = (float)MintyAIClean(MathCos(2.0 * M_PI * hod / 24.0));
      out[b + 22] = (float)MintyAIClean(MathSin(2.0 * M_PI * dow / 7.0));
      out[b + 23] = (float)MintyAIClean(MathCos(2.0 * M_PI * dow / 7.0));
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Convenience wrapper: pull `count` closed bars ending at shift     |
//| `endShift` (1 = last closed bar) and build features for them.     |
//| Returns the number of rows written, or -1 on failure.             |
//+------------------------------------------------------------------+
int MintyAIFeaturesFromSymbol(const string symbol, ENUM_TIMEFRAMES tf,
                            int endShift, int count, float &out[])
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int need = count + MINTYAI_WARMUP;
   int got  = CopyRates(symbol, tf, endShift, need, rates);
   if(got < need)
      return -1;

   double o[], h[], l[], c[], v[];
   long   ts[];
   ArrayResize(o,  got);
   ArrayResize(h,  got);
   ArrayResize(l,  got);
   ArrayResize(c,  got);
   ArrayResize(v,  got);
   ArrayResize(ts, got);
   for(int i = 0; i < got; i++)
     {
      o[i]  = rates[i].open;
      h[i]  = rates[i].high;
      l[i]  = rates[i].low;
      c[i]  = rates[i].close;
      v[i]  = (double)rates[i].tick_volume;
      //--- Server time, as the exporter wrote it into the training data,
      //--- plus a shift that is zero on the training broker and corrects
      //--- for the difference on any other.
      ts[i] = (long)rates[i].time + g_mintyaiTimeShift;
     }

   float all[];
   if(!MintyAIBuildFeatures(o, h, l, c, v, ts, got, all))
      return -1;

//--- keep only the trailing `count` rows (the warmed-up tail)
   ArrayResize(out, count * MINTYAI_FEATURE_COUNT);
   int offset = (got - count) * MINTYAI_FEATURE_COUNT;
   for(int i = 0; i < count * MINTYAI_FEATURE_COUNT; i++)
      out[i] = all[offset + i];
   return count;
  }
//+------------------------------------------------------------------+
