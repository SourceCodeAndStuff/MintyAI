//+------------------------------------------------------------------+
//|                                             MintyAIExportAll.mq5 |
//|                                                                  |
//|   Downloads and exports training data for every symbol in Market |
//|   Watch, one after another. Curate Market Watch, run this once,  |
//|   then train:                                                    |
//|                                                                  |
//|       python -m mintyai.train_all                                |
//|                                                                  |
//|   The trainer discovers whatever this wrote - it does not need   |
//|   to be told which symbols you chose.                            |
//|                                                                  |
//|   Output lands in MQL5\Files\MintyAI\.                           |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

#include <MintyAI/MintyAIExporter.mqh>

//--- Blank means every symbol currently in Market Watch, which is the
//--- normal way to use this: curate the list there, run this once.
input string          InpSymbols      = "";                 // Symbols ("" = everything in Market Watch)
input ENUM_TIMEFRAMES InpTimeframe    = PERIOD_H1;          // Timeframe
input datetime        InpFrom         = D'2010.01.01';      // Export from
input datetime        InpTo           = D'2100.01.01';      // Export to (future = now)
input bool            InpDumpFeatures = true;               // Also dump MQL5 features (parity check)
input int             InpParityRows   = 5000;               // Feature rows to dump (0 = all)

//+------------------------------------------------------------------+
void OnStart()
  {
   string symbols[];
   int n = 0;

   string spec = InpSymbols;
   StringTrimLeft(spec);
   StringTrimRight(spec);

   if(spec == "")
     {
      int total = SymbolsTotal(true);      // true = Market Watch only
      ArrayResize(symbols, total);
      for(int i = 0; i < total; i++)
         symbols[i] = SymbolName(i, true);
      n = total;
      PrintFormat("MintyAI export: %d symbols in Market Watch", n);
      if(n == 0)
         Print("MintyAI export: Market Watch is empty - right-click it and "
               "'Show All', or list symbols in the input.");
     }
   else
      n = MintyAIParseSymbolList(spec, symbols);

   if(n <= 0)
     {
      Print("MintyAI export: no symbols to export.");
      return;
     }

   PrintFormat("MintyAI export: %d symbols on %s. Downloading history can take "
               "a few minutes each the first time - leave the terminal open.",
               n, MintyAIPeriodTag(InpTimeframe));

   int ok = 0;
   string failed = "";

   for(int i = 0; i < n; i++)
     {
      if(IsStopped())
        {
         Print("MintyAI export: stopped by user.");
         break;
        }

      PrintFormat("MintyAI export: [%d/%d] %s", i + 1, n, symbols[i]);

      if(MintyAIExportSymbol(symbols[i], InpTimeframe, InpFrom, InpTo,
                             InpDumpFeatures, InpParityRows))
         ok++;
      else
         failed += (failed == "" ? "" : ", ") + symbols[i];
     }

   PrintFormat("MintyAI export: finished - %d of %d symbols exported.", ok, n);
   if(failed != "")
      PrintFormat("MintyAI export: failed for %s. These are usually names your "
                  "broker spells differently, e.g. a suffix like .pro", failed);
   if(ok > 0)
      Print("MintyAI export: next step ->  python -m mintyai.train_all");
  }
//+------------------------------------------------------------------+
