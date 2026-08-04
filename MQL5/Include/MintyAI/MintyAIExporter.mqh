//+------------------------------------------------------------------+
//|                                              MintyAIExporter.mqh  |
//|                                                                   |
//|   Downloads history for a symbol and writes the three files the   |
//|   trainer needs:                                                  |
//|                                                                   |
//|     <SYMBOL>_<TF>_bars.csv          raw OHLCV + spread            |
//|     <SYMBOL>_<TF>_meta.csv          tick size/value, costs        |
//|     <SYMBOL>_<TF>_features_mql5.csv feature dump for parity       |
//|                                                                   |
//|   Shared by MintyAIExport.mq5 (one symbol) and                    |
//|   MintyAIExportAll.mq5 (every symbol you list).                   |
//+------------------------------------------------------------------+
#ifndef MINTYAI_EXPORTER_MQH
#define MINTYAI_EXPORTER_MQH

#include <MintyAI/MintyAIFeatures.mqh>

//+------------------------------------------------------------------+
//| Filename-safe timeframe tag.                                     |
//+------------------------------------------------------------------+
string MintyAIPeriodTag(ENUM_TIMEFRAMES tf)
  {
   string s = EnumToString(tf);
   StringReplace(s, "PERIOD_", "");
   return s;
  }

//+------------------------------------------------------------------+
//| Ask the server for history and wait for it to arrive.            |
//|                                                                  |
//| CopyRates kicks off the download but returns whatever is cached  |
//| right now, so a first call on a fresh symbol usually comes back  |
//| nearly empty. Without this wait an unattended run would export   |
//| a few hundred bars per symbol and look like it had worked.       |
//+------------------------------------------------------------------+
bool MintyAIWaitForHistory(string symbol, ENUM_TIMEFRAMES tf, int timeoutSec = 120)
  {
   uint deadline = GetTickCount() + (uint)timeoutSec * 1000;
   long prevBars = -1;

   while(GetTickCount() < deadline)
     {
      if(IsStopped())
         return false;

      //--- touching the series is what triggers the download
      datetime dummy[];
      CopyTime(symbol, tf, 0, 1, dummy);

      bool synced   = (bool)SeriesInfoInteger(symbol, tf, SERIES_SYNCHRONIZED);
      long barCount = SeriesInfoInteger(symbol, tf, SERIES_BARS_COUNT);

      if(synced && barCount > 0 && barCount == prevBars)
         return true;

      prevBars = barCount;
      Sleep(500);
     }

   PrintFormat("MintyAI export: %s %s did not finish syncing within %ds - "
               "exporting what is cached.", symbol, MintyAIPeriodTag(tf), timeoutSec);
   return true;
  }

//+------------------------------------------------------------------+
//| Export one symbol. Returns true if the bars file was written.    |
//+------------------------------------------------------------------+
bool MintyAIExportSymbol(string symbol,
                         ENUM_TIMEFRAMES tf,
                         datetime from,
                         datetime to,
                         bool dumpFeatures,
                         int parityRows)
  {
   if(!SymbolSelect(symbol, true))
     {
      PrintFormat("MintyAI export: cannot select '%s' - check the name and your "
                  "broker's symbol list.", symbol);
      return false;
     }

   MintyAIWaitForHistory(symbol, tf);

   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   datetime end = (to > TimeCurrent() ? TimeCurrent() : to);

   int got = CopyRates(symbol, tf, from, end, rates);
   if(got <= 0)
     {
      PrintFormat("MintyAI export: %s CopyRates failed (%d). Open its chart once "
                  "so the terminal downloads history, then retry.",
                  symbol, GetLastError());
      return false;
     }
   PrintFormat("MintyAI export: %s %s - %d bars", symbol, MintyAIPeriodTag(tf), got);

   if(got < MINTYAI_WARMUP + 1000)
      PrintFormat("MintyAI export: WARNING %s has only %d bars - far too little to "
                  "train on. Aim for 50k+.", symbol, got);

//--- Timestamps are written in SERVER time, unconverted, and the EA reads
//--- server time too - so the hour-of-day features agree by construction.
//--- Recorded below for reference: a model trained here is only valid on a
//--- broker whose server sits in the same timezone.
   long utcOffset = MintyAIDetectUtcOffset();
   if(utcOffset == MINTYAI_UTC_UNKNOWN)
      utcOffset = 0;
   PrintFormat("MintyAI export: server is UTC%+d; timestamps written in "
               "SERVER time.", (int)(utcOffset / 3600));

   string base = "MintyAI\\" + symbol + "_" + MintyAIPeriodTag(tf);

//--- 1) raw bars ---------------------------------------------------
   string barsFile = base + "_bars.csv";
   int fh = FileOpen(barsFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
     {
      PrintFormat("MintyAI export: cannot open %s err=%d", barsFile, GetLastError());
      return false;
     }
   FileWrite(fh, "time", "open", "high", "low", "close", "tick_volume",
             "real_volume", "spread");
   for(int i = 0; i < got; i++)
     {
      FileWrite(fh,
                (long)rates[i].time,
                DoubleToString(rates[i].open,  8),
                DoubleToString(rates[i].high,  8),
                DoubleToString(rates[i].low,   8),
                DoubleToString(rates[i].close, 8),
                (long)rates[i].tick_volume,
                (long)rates[i].real_volume,
                (int)rates[i].spread);
     }
   FileClose(fh);
   PrintFormat("MintyAI export: wrote MQL5\\Files\\%s", barsFile);

//--- 2) symbol metadata the trainer needs to model costs ------------
   string metaFile = base + "_meta.csv";
   fh = FileOpen(metaFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(fh != INVALID_HANDLE)
     {
      FileWrite(fh, "key", "value");
      FileWrite(fh, "symbol",         symbol);
      FileWrite(fh, "timeframe",      MintyAIPeriodTag(tf));
      FileWrite(fh, "digits",         (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));
      FileWrite(fh, "point",          DoubleToString(SymbolInfoDouble(symbol, SYMBOL_POINT), 10));
      FileWrite(fh, "tick_size",      DoubleToString(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE), 10));
      FileWrite(fh, "tick_value",     DoubleToString(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE), 10));
      FileWrite(fh, "volume_min",     DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN), 4));
      FileWrite(fh, "volume_step",    DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP), 4));
      FileWrite(fh, "spread_current", (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD));
      //--- reference only: the timestamps above are raw server time
      FileWrite(fh, "server_utc_offset_h", (int)(utcOffset / 3600));
      FileWrite(fh, "time_basis",     "SERVER");
      FileWrite(fh, "feature_version", MINTYAI_FEATURE_VERSION);
      FileWrite(fh, "feature_count",   MINTYAI_FEATURE_COUNT);
      FileWrite(fh, "feature_signature", (long)MintyAIFeatureSignature());
      FileClose(fh);
      PrintFormat("MintyAI export: wrote MQL5\\Files\\%s", metaFile);
     }

//--- 3) feature parity dump ----------------------------------------
   if(dumpFeatures)
     {
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
         ts[i] = (long)rates[i].time;   // server time, as in the bars file
        }

      float feats[];
      if(!MintyAIBuildFeatures(o, h, l, c, v, ts, got, feats))
        {
         PrintFormat("MintyAI export: %s feature build failed", symbol);
         return true;   // bars are written, which is the part training needs
        }

      int rows  = (parityRows <= 0 ? got : MathMin(parityRows, got));
      int start = got - rows;

      string featFile = base + "_features_mql5.csv";
      fh = FileOpen(featFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
      if(fh == INVALID_HANDLE)
        {
         PrintFormat("MintyAI export: cannot open %s err=%d", featFile, GetLastError());
         return true;
        }
      string header = "time";
      for(int f = 0; f < MINTYAI_FEATURE_COUNT; f++)
         header += "," + MintyAIFeatureNames[f];
      FileWriteString(fh, header + "\r\n");

      for(int i = start; i < got; i++)
        {
         string line = IntegerToString(ts[i]);
         int b = i * MINTYAI_FEATURE_COUNT;
         for(int f = 0; f < MINTYAI_FEATURE_COUNT; f++)
            line += "," + DoubleToString((double)feats[b + f], 8);
         FileWriteString(fh, line + "\r\n");
        }
      FileClose(fh);
      PrintFormat("MintyAI export: wrote MQL5\\Files\\%s (%d rows, signature %u)",
                  featFile, rows, MintyAIFeatureSignature());
     }

   return true;
  }

//+------------------------------------------------------------------+
//| Split "EURUSD, GBPUSD ,USDJPY" into a clean symbol array.        |
//+------------------------------------------------------------------+
int MintyAIParseSymbolList(string list, string &out[])
  {
   ArrayResize(out, 0);
   string parts[];
   int n = StringSplit(list, ',', parts);
   for(int i = 0; i < n; i++)
     {
      string s = parts[i];
      StringTrimLeft(s);
      StringTrimRight(s);
      if(s == "")
         continue;
      int k = ArraySize(out);
      ArrayResize(out, k + 1);
      out[k] = s;
     }
   return ArraySize(out);
  }

#endif // MINTYAI_EXPORTER_MQH
