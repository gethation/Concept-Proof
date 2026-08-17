# archive/

Closed studies. Kept because the reasoning and the measurement code are worth
re-reading, not because anything here is on the live path — nothing in
`make_reports.py` reaches into this directory.

| directory | what it was | outcome |
|---|---|---|
| `us_index/` | TAIFEX UDF/SPF/UNF/SXF against US index futures via IBKR | Measured **no**: ~5.5%/yr even with a zero-cost book; the apparent edge was bid-ask bounce. |
| `qff_tsm_15m/` | The 15m QFF/TSM spread builder (OKX legs aggregated 1m→15m onto the QFF 15m grid) | Superseded by the 1m pipeline. Not folded into `features/spread.py`: it also emits session-aligned leg OHLCV files, a different output contract from the other three spread scripts. |
| `old_reports/` | Per-topic HTML generators (backtest, indicator, hold-profit) | Superseded by `report/theme.py` + `report/pair.py`, which share one visual language and recompute every figure at run time. |
| `ml/` | XGBoost on 15m spread change | Kept for the feature-importance result; not part of any live decision. |
| `studies/` | Kalman dynamic hedge, margin-transfer simulation, 5m seed builder, factor downloads, tvdatafeed probes | One-off answers to one-off questions. `margin_transfers.py` backs `docs/margin_management_analysis.md`. |

## Running these again

**Their default input paths point at files that no longer exist.** Two reasons:

- Bar files moved and lost their vintage suffixes — see `lib/paths.py` for where
  each series lives now.
- Their spread/z-score inputs were derived files, and derived files are no
  longer kept on disk (`data/features/` is rebuildable and gitignored). Rebuild
  what a script needs with `features/spread.py` and `features/zscore.py` first,
  then pass the path explicitly.

The outputs these produced are still here: `data/runs/<tag>/summary.json` and
`trades.csv` for every historical run, plus `data/runs/_probes/` and
`data/runs/_grids/`. Equity curves were not kept.
