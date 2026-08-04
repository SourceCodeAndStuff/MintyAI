//+------------------------------------------------------------------+
//|                                                MintyAIExport.mq5 |
//|                                                                  |
//|   Downloads and exports training data for ONE symbol.            |
//|   For every symbol at once, use MintyAIExportAll.mq5.            |
//|                                                                  |
//|   Output lands in MQL5\Files\MintyAI\.                           |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

#include <MintyAI/MintyAIExporter.mqh>

input string          InpSymbol       = "";                // Symbol ("" = chart symbol)
input ENUM_TIMEFRAMES InpTimeframe    = PERIOD_H1;         // Timeframe
input datetime        InpFrom         = D'2010.01.01';     // Export from
input datetime        InpTo           = D'2100.01.01';     // Export to (future = now)
input bool            InpDumpFeatures = true;              // Also dump MQL5 features (parity check)
input int             InpParityRows   = 5000;              // Feature rows to dump (0 = all)

//+------------------------------------------------------------------+
void OnStart()
  {
   string symbol = (InpSymbol == "" ? _Symbol : InpSymbol);
   StringTrimLeft(symbol);
   StringTrimRight(symbol);

   if(MintyAIExportSymbol(symbol, InpTimeframe, InpFrom, InpTo,
                          InpDumpFeatures, InpParityRows))
      Print("MintyAI export: done. Now run  python -m mintyai.train_all");
   else
      Print("MintyAI export: FAILED for ", symbol);
  }
//+------------------------------------------------------------------+
