# TAIFEX / US Pairs Trading Research

Statistical arbitrage between TAIFEX single-stock futures and their US-listed
counterparts. The same company is priced in two markets; the gap between those
prices oscillates around a mean, and the strategy takes an offsetting position
in both legs when the gap widens, then waits for it to close.

Two active pairs:

| Pair | Taiwan leg | US leg | FX |
|---|---|---|---|
| **CCF / UMC** | TAIFEX CCF (UMC stock futures, 2,000 shares/contract) | NYSE:UMC ADR (1 ADR = 5 shares) | FX_IDC USDTWD |
| **QFF / TSM** | TAIFEX QFF (TSMC stock futures, 100 shares/contract) | OKX TSM-USDT perpetual | BitoPro USDT/TWD |

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

Both columns are `make_reports.py` on data through 2026-08-26, ranked by
**linearly annualised return**.

**Neither headline passes the quality screen.** Ranking by return and screening
for verifiability are different questions, and on this data they disagree: the
highest-annualised cell in each grid takes more trades, each one thinner, and a
large share of them earn less than one tick — which is to say their profit is
inside the noise of whether a fill printed on the bid or the ask. The reports
label every cell with its screen verdict so the two can be read apart.

| | CCF / UMC | QFF / TSM |
|---|---|---|
| Period | 2026-01-06 → 08-26 (149 sessions) | 2026-04-01 → 08-26 (98 sessions) |
| Configuration | w1560 / entry 1.0 / exit 0.5 | w2500 / entry 1.5 / exit 0 |
| Displacement | 0.2317 (measured), 1/price scaled from 121.50 | 0.1589 pre-2026-07-05, 0.0755 after |
| US-leg fee | IBKR per-ADR (5.39 bps realised median) | 5.0 bps flat (measured on Binance, leg is OKX) |
| Trades | 73 | 70 |
| Total return | 27.89% over 231 days | 11.99% over 148 days |
| Linear annualised | **44.0%** | **29.6%** |
| Sharpe | 4.08 | 3.55 |
| Max drawdown | −3.47% | −3.22% |
| Median edge per contract | 1.91 ticks | 2.39 ticks |
| Inside one tick | 34.2% | 30.0% |
| Round-trip cost | 71.4 bps (77.9% crossing) | 46.1 bps (60.6% crossing) |
| **Passes screen** | **no** | **no** |
| Cells passing screen | 7 of 140 | **0 of 140** |

The most conservative alternative — highest annualised return *among* cells that
pass all four thresholds — is **w2500 / entry 2.0 / exit 0** for CCF/UMC: 26.9%
annualised, 28 trades, 14.3% inside one tick. QFF/TSM has no such cell at any
setting; raising entry_z far enough to clear the sub-tick share collapses its
trade count below the 15-trade minimum, which is the finding rather than a
tuning problem.

Linear annualisation multiplies a 231-day (CCF/UMC) or 148-day (QFF/TSM) sample
by 1.6x and 2.5x respectively. It is an extrapolation, not a measurement; the
raw period return is the number that actually happened.

Fills are priced at the **open of the next tradable bar**. Both reports
previously filled at that bar's *close* while their own metadata and prose said
open — the spread files carry no open columns, and the engine fell back to the
close without saying so. On the configurations above the correction is worth
+0.25 pp (CCF/UMC) and +1.03 pp (QFF/TSM) of total return.

Rolling windows are also no longer allowed to span a hole in the data. QFF's 1m
series is missing 2026-07-01..07-06 outright, and CCF/UMC spans Lunar New Year
and Qingming; a window reaching across any of them was splicing two regimes into
one mean. Withholding those windows costs CCF/UMC 3 trades and 1.9 pp.

**Quote the displacement with the result, or the result is not reproducible.**
Earlier versions of this table reported far higher annualised returns (88.0%,
then 68.2%) on far shorter samples. None of that difference was the market: it
was the displacement constant changing (0.2151 inferred, then 0.2317 measured),
the sample lengthening from 52 sessions to 149, and the fill and gap corrections
above. A larger displacement moves every z threshold by
`displacement / spread_std` and charges more per crossing, so it changes trade
count, cost and return together — which is why it has a row in the table rather
than living only in the methodology. Annualising a two-month sample linearly is
also what produced the largest of those headline numbers; the current table
quotes the raw period return first for that reason.

The US-leg fee row is there for the same reason. CCF/UMC is priced with IBKR's
real schedule -- per ADR, minimum and cap, SEC and FINRA on sells only -- rather
than a flat bps figure, and which of the two is more conservative depends on the
ADR price: they cross near $23.2. Over this sample UMC ran $16.74 to $28.87 and
the two models very nearly cancel (-1.3% on the US leg). Over August alone, with
UMC at $19, the per-ADR model charges 12.9% more -- which matches the +13% the
live fills actually paid. See `scripts/lib/ibkr_fees.py`.

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
python scripts/ingest/qff_tsm_15m.py          # OKX perp leg, the one qff_tsm reads
python scripts/ingest/ccxt_ohlcv.py --feed bitopro_usdttwd
```

Two more that are not part of the routine refresh:

```bash
python scripts/ingest/lux_quotes.py                    # book mid, from Project Lux
python scripts/ingest/ibkr_umc.py --duration-weeks 52  # deeper UMC history
```

`lux_quotes.py` turns Project Lux's stored quote stream into mid-price minute
bars under `data/bars/lux/`. It exists because every price here is a LAST TRADED
price -- TAIFEX publishes time-and-sales with no quote columns at all -- while
the live system scores on the book mid, and for a one-tick-wide CCF book those
differ by exactly half a tick at random sign. Measured against 3,807 shared
minutes, rebuilding the spread on mid cuts its deviation from what live actually
saw by 77% (sigma 0.238 to 0.055) and lifts native-bar coverage from 64% to
100%, because a book quotes every second even when nothing trades. **Its
coverage starts 2026-08-07** and no source can reach back further, so it is a
calibration series until the live store has accumulated a window plus a sample.

`ibkr_umc.py` deepens the UMC leg past tvdatafeed's rolling 19 days. It refuses
to run while a Project Lux live session is trading, because IBKR meters
historical requests per ACCOUNT and a backfill would throttle the live run's
market data.

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
