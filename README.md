# TAIFEX / US Pairs Trading Research

Statistical arbitrage between TAIFEX single-stock futures and their US-listed
counterparts. The same company is priced in two markets; the gap between those
prices oscillates around a mean, and the strategy takes an offsetting position
in both legs when the gap widens, then waits for it to close.

Two active pairs:

| Pair | Taiwan leg | US leg | FX |
|---|---|---|---|
| **CCF / UMC** | TAIFEX CCF (UMC stock futures, 2,000 shares/contract) | NYSE:UMC ADR (1 ADR = 5 shares) | FX_IDC USDTWD |
| **QFF / TSM** | TAIFEX QFF (TSMC stock futures, 100 shares/contract) | Binance TSMUSDT perpetual | BitoPro USDT/TWD |

All timestamps are Taipei time (`+08:00`).

## The strategy in brief

The spread is defined on a mid-price percentage scale, so one spread unit is
roughly 1% of leg notional:

```text
fair   = us_leg_close × fx / 5
spread = (fair − tw_leg_close) / (fair + tw_leg_close) × 200
```

A rolling z-score measures how far the spread has been pulled: at `z > entry_z`
short the US leg and long the Taiwan leg, close when `z` comes back inside
`exit_z`, and the mirror image for the other direction. Fills happen at the next
bar's open.

The design decision that matters most is that **signals are not scored at mid**.
The engine builds one executable spread per direction — short uses the US bid
against the Taiwan ask, long the reverse — and tests both entry and exit against
the side the order would actually have to cross, charging the same displacement
as a crossing cost. Without that correction the optimiser walks straight to the
high-frequency corner of the grid, where the apparent profit is bid-ask bounce
rather than edge.

The hard floor for telling real edge from noise is the **tick**. A bid-ask
spread can never be narrower than one tick, so a trade earning less than one
tick per contract cannot be distinguished from the randomness of whether a fill
printed on the bid or the ask. The reports screen at a median edge of ≥ 2 ticks.

## Backtest results

Best configuration for each pair, on 2M TWD capital with 1M notional per leg.
**These figures are recomputed by `make_reports.py` on every run; below is the
result on data through 2026-08-15.**

| | CCF / UMC | QFF / TSM |
|---|---|---|
| Period | 2026-06-09 → 08-15 (46 sessions) | 2026-07-06 → 08-15 (29 sessions) |
| Configuration | w2500 / entry 1.0 / exit 0.25 | w1560 / entry 2.0 / exit 0 |
| Trades | 32 | 21 |
| Total return | **+88.0%** annual | **27.2%** annual |
| Sharpe | 8.75 | 3.63 |
| Max drawdown | −1.97% | −1.59% |
| Return / drawdown | 8.1x | 1.9x |
| Median edge per contract | 2.21 ticks (6% inside one tick) | 8.08 ticks (14% inside one tick) |
| Round-trip cost | 54.9 bps (78.9% crossing) | 32.8 bps (46.0% crossing) |

QFF/TSM covers a shorter period because **QFF's tick size dropped from 5 TWD to
1 TWD on 2026-07-05**, cutting its crossing cost by four fifths. A single
displacement parameter cannot span that break, so only the post-change segment
is scored.



## Usage

### Environment

Python 3.10+ (developed on 3.12.13). Everything runs in the `Quant` conda
environment:

```bash
conda create -n Quant python=3.12
conda run -n Quant pip install -r requirements.txt
```

`pandas` and `numpy` are pinned exactly, because spread and z-score outputs are
verified byte-identical against a stored SHA256 baseline and a minor release of
either is enough to move them. Bump them deliberately and re-run the baseline.
`requirements-archive.txt` adds `xgboost` and `ib_async` for the closed studies
in `scripts/archive/`; nothing reachable from `make_reports.py` needs them.

### Refresh the data

Every step append-merges, so re-running is safe:

```bash
python scripts/ingest/taifex_1m.py --product CCF
python scripts/ingest/taifex_1m.py --product QFF
python scripts/ingest/tv_umc.py
python scripts/ingest/tv_ccf_umc.py
python scripts/ingest/ccxt_ohlcv.py --feed binance_tsmusdtp
python scripts/ingest/ccxt_ohlcv.py --feed bitopro_usdttwd
```

### Build the spreads

```bash
python scripts/features/spread.py --pair ccf_umc --interval 1m --weekend-policy none
python scripts/features/spread.py --pair qff_tsm --interval 1m
```

### Rebuild every report

```bash
python scripts/make_reports.py
```

Takes about 30 minutes — QFF/TSM alone runs a full grid over two tick regimes ×
five windows. Use `--only ccf_umc` for a single report. Output lands in
`reports/`.

On Windows PowerShell:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n Quant python scripts/make_reports.py
```

### Run a single backtest

Compute the z-score, then run the engine:

```bash
python scripts/features/zscore.py --spread-path data/features/qff_tsm/spread_1m.csv --out data/features/qff_tsm/zscore_1m.csv --window 1560
```

```bash
python scripts/backtest/engine.py --input data/features/qff_tsm/zscore_1m.csv --entry-z 2.0 --exit-z 0.0 --executable-displacement 0.0755
```

Output lands in `data/runs/scratch/`. Point `--equity-out` / `--trades-out` /
`--summary-out` at `data/runs/<tag>/` for anything worth keeping.

Three things the CLI does not do for you, all of which the reports handle:

- **The engine's defaults are QFF/TSM-shaped** — contract multiplier 100, US leg
  5 bps, and the QFF/TSM bar files for fill prices. Running CCF/UMC through it
  needs `--qff-contract-multiplier 2000 --tsm-fee-bps 2.5` plus `--qff-ohlcv` /
  `--tsm-ohlcv` / `--usdttwd-ohlcv` pointing at that pair's legs. Getting this
  wrong produces a quietly wrong answer rather than an error. The per-pair
  values live in the `PAIRS` table in `report/pair.py`.
- `--executable-displacement` is each pair's measured book displacement.
  **Omitting it prices at mid and overstates performance.**
- The reports seed the rolling window (`--seed-spread-path`) so a long window
  does not burn the sample on warmup. `report/pair.py` trims the seed to before
  the sample start; the CLI does not, and rejects an overlapping seed.

## Repository layout

```
scripts/
  lib/        shared library: time, bar I/O, FX, sessions, pair config, paths
  ingest/     downloads → data/bars/
  features/   spread / z-score / entry gates → data/features/
  backtest/   engine, grid search, OU thresholds
  report/     HTML report generators
  archive/    closed studies (see scripts/archive/README.md)
  make_reports.py

data/
  raw/        exchange downloads, as received
  bars/       canonical bar series, one file per symbol+interval
  features/   spread / z-score (rebuildable, gitignored)
  runs/       one directory per backtest: summary.json + trades.csv
reports/      generated HTML
```

Three rules keep this from decaying again:

1. **Bar filenames carry no date or vintage suffix.** Suffixes like `_0812`,
   `_cumulative` and `_latest` caused a real incident: the download step wrote
   the undated default while the spread step read the dated file, so running the
   documented refresh updated a file nothing else read — and 10,080 minutes of
   early history were permanently lost in the process. Use `--start` / `--end`
   for a historical slice.
2. **One directory per run, parameters travelling with results.** A filename
   cannot encode twenty-odd parameters; `summary.json`'s `parameters` block can.
3. **Paths are defined once**, in `lib/paths.py`.

## Further reading

- [docs/methodology.md](docs/methodology.md) — session alignment, FX splicing,
  cost model, position sizing, known differences between the two code paths
- [docs/ccf_umc_weekend_policy.md](docs/ccf_umc_weekend_policy.md) — weekend
  rules and the Monday unhedged window
- [docs/margin_management_analysis.md](docs/margin_management_analysis.md) —studies and what they concluded
