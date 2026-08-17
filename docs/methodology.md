# Methodology

Supplement to `README.md`. This is the "why it is computed this way" part — how
the index is built, how FX is spliced, how costs are charged, how position size
is derived, and which differences between the two CCF/UMC code paths are
deliberate.

## Spread and z-score

The spread is one formula for both pairs:

```text
fair   = us_leg_close × usdtwd / share_ratio      (share_ratio = 5)
spread = (fair − tw_leg_close) / (fair + tw_leg_close) × 200
```

A mid-price percentage scale, so one spread unit is roughly 1% of leg notional.

```text
spread_mean = rolling_mean(spread, window)
spread_std  = rolling_std(spread, window, ddof=0)
spread_zscore = (spread − spread_mean) / spread_std
```

The first `window − 1` rows are warmup and carry `zscore_valid = False`. The
CCF/UMC reports pre-fill the rolling window from a seed file
(`--seed-spread-path`) so every window trades the same period; otherwise a long
window burns most of the sample on warmup.

## Two index strategies

The difference between the pairs is not the formula but the index it is
evaluated on. `scripts/lib/sessions.py` holds two strategies.

### `taifex_grid` (QFF / TSM)

The index is synthesised from the TAIFEX session clock — day 08:45–13:45, night
17:25–05:00 the next day — rather than from the bars that happen to exist. A
minute with no trade still gets a row, is forward-filled, and is flagged with
`qff_was_filled`, instead of silently vanishing from the index.

The other two legs trade 24/7 (Binance perpetual, BitoPro), so a missing minute
means broken data rather than a closed market, and `assert_external_complete`
raises rather than interpolating.

### `us_rth` (CCF / UMC)

The index is the US leg's own RTH bars, with the TAIFEX leg aligned as-of — the
last TAIFEX bar starting at or before each index bar.

Both legs are exchange-traded, and NYSE RTH is the only window in which the pair
can actually be traded together. TAIFEX trades far longer, so TAIFEX is the leg
that bends. On the 5m grid this is an exact match; on the 15m grid TAIFEX night
bars sit on a grid 5 minutes earlier than UMC's, making it a true as-of match
whose close is known 5 minutes before the UMC bar closes — no lookahead either
way.

Whole sessions in which the TAIFEX leg barely traded during US hours are
dropped (`--min-tw-bars-per-session`). This is what caught the Taiwan holidays
on 2/27, 4/6, 5/1 and 7/10.

## FX splicing

QFF/TSM uses the BitoPro USDT/TWD 1m series directly as a leg.

CCF/UMC uses FX_IDC USDTWD, but that feed has multi-hour outages at **every**
interval, so the usable series is a splice: the finest interval wins on a shared
bar start, coarser intervals fill the holes (5m > 15m > 1h).

The detail that matters: alignment keys on **known-time**, meaning a bar's close
counts as known only once the bar has ended — not on the bar's start. Otherwise
a 1-hour candle would leak up to an hour of hindsight into a 1-minute row.

USDTWD barely moves intraday (measured: 0.17% median intra-session range against
3.4% for each equity leg), so forward-filling through an outage is safer than
fragmenting sessions. Stale rows warn; sessions past the hard cap are dropped
(with one exception, below).

## Cost model

### Executable pricing (`--executable-displacement`)

The engine still fills at mid, and the displacement does two things — two halves
of one correction, not double counting:

- **Signal side** — each bar's threshold shifts by `displacement / spread_std`.
  Because the rolling standard deviation moves, this is a shift in z rather
  than a constant.
- **Fill side** — the same displacement is charged once per side as
  `crossing_cost_twd`, converting the mid fill into an executable one.

The mode requires the input to carry `spread_std*` columns. `--frozen-mean-exit`
and `--drift-bail-c` compare absolute spread levels and have no corresponding
displacement basis, so they are explicitly rejected rather than mixing two
pricing schemes.

Measured single-side displacement: CCF/UMC **0.2151** (43.0 bps round trip),
QFF/TSM **0.0755** (15.1 bps round trip).

### Commission and tax

```text
us_fee_bps               = 5.0 (TSM perp) / 2.5 (UMC ADR)
taifex_fee_per_contract  = 88.0 TWD
taifex_tax_rate          = 0.00002
contract_multiplier      = 100 (QFF) / 2000 (CCF)
```

Both are one-sided costs, charged once on entry and once on exit.

`--qff-fee-bps B` switches the futures leg to notional-proportional pricing,
**replacing** the flat amount. The two are not interchangeable: bps is a ratio
of price × multiplier, so a flat 88 TWD is 88 bps on a 10,000 TWD contract but
only 8.8 bps on a 100,000 TWD one. Always convert against the notional actually
traded. `parameters.qff_fee_mode` records which one took effect.

Not modelled: slippage, FX conversion cost, perpetual funding, margin interest,
broker surcharges.

## Position sizing

The TAIFEX leg is rounded to whole contracts at the entry fill bar's open, then
the US leg is matched to the resulting actual notional:

```text
raw_contracts = leg_notional_twd / (entry_open × contract_multiplier)
contracts     = floor(raw_contracts + 0.5)
actual_notional_twd = abs(contracts) × contract_multiplier × entry_open
us_units      = actual_notional_twd / entry_us_twd_fair_open
```

If `contracts == 0` the entry is cancelled. Position size is deliberately
allowed to float with the rounding (measured 858k–1,107k against a 1M target):
matching the legs exactly matters more than hitting a fixed size.

`--qff-lots N` switches to a fixed contract count, taking `leg_notional_twd` out
of sizing and leaving the rest unchanged. `raw_contracts` keeps the notional
fraction from the formula above in both modes, so the "rounding offset" column
does not collapse to the lot count in fixed mode. `parameters.sizing_mode`
records which one took effect.

Live orders are sent as a fixed number of contracts, so the two modes have
genuinely different cost structures: **per-contract fees** scale linearly and
cancel out in a comparison, while a **per-order minimum commission** does not —
it is a fixed cost spread over a smaller position. Sweep parameters at the lot
size you would actually trade; the grid search accepts `--qff-lots` too.

**Note:** `tsm_units` is in local-share equivalents, not ADRs. The IBKR order
quantity is `tsm_units / 5`.

## Entry and exit rules

**Entry** — evaluated only on bars where `entry_allowed` and `zscore_valid` are
both true. Flat and `z > entry_z` → short the US leg, long the Taiwan leg;
`z < −entry_z` → the reverse. The signal fills at the open of the next
`entry_allowed` bar, and is cancelled if that takes longer than
`max_entry_delay_minutes`. The z-score is not re-checked at the fill.

**Exit** — `should_exit` is `z < −exit_z`, so **a higher `exit_z` means waiting
for more overshoot past the mean, not exiting sooner.** The exit fills at the
open of the next `close_allowed` bar. Any position still open when the data ends
is closed on the final bar.

**Weekend policy** (`--weekend-policy`):

| Value | Behaviour |
|---|---|
| `flat` | No entries in the last session of an ISO week, and force-close on its final bar |
| `no-entry` | Keep the entry ban, drop the force-close |
| `none` | Neither rule |

The rule is inherited from QFF/TSM, where Binance trades 24/7 while QFF is
frozen over the weekend, leaving an uncovered leg. **CCF/UMC has no such
exposure** — TAIFEX and NYSE both shut — so it can be dropped, at the cost of
weekend gap risk and the Monday unhedged window. See
[ccf_umc_weekend_policy.md](ccf_umc_weekend_policy.md).

## Known differences between the two code paths

Two fields on the `ccf_umc` 1m configuration differ from 5m/15m, and they are
**deliberate rather than oversights**:

| Field | 1m | 5m / 15m |
|---|---|---|
| `fx_session_filter` | `False` | `True` |
| `range_includes_fx` | `False` | `True` |

The 1m script defined `MAX_FX_STALENESS_MINUTES = 720` and then **never used it
to filter**, and it also ignored FX coverage when choosing the start and end of
the range. The 5m/15m script does both.

Turning either on changes the published 1m spread file, and with it every
conclusion built on top. They were therefore preserved exactly during the
2026-08-17 reorganisation and documented in `scripts/lib/pairs.py`. **Fixing
them is a standalone data change with before/after figures attached, not
something to fold into a refactor.**
