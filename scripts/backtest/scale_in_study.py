"""The scale-in experiment, end to end.

    python scripts/backtest/scale_in_study.py

Three stages, each feeding the next:

1.  **MAE distribution.** Run the CCF/UMC headline configuration through the
    single-entry engine and measure every trade's maximum adverse excursion
    (MAE) from its entry fill, in entry-time spread-std units -- the same unit
    add_spacing_k is quoted in. This is what puts the spacing grid on
    empirical footing instead of a guess: the interesting spacings are the
    ones ordinary winners rarely reach and one-sided runs always cross.

2.  **Parity check.** run_scale_in with n_tranches=1 must reproduce the
    single-entry engine trade for trade on the real frame. The scale-in loop
    was written against the engine's conventions (signal at close, fill at
    next allowed bar, displaced z on both thresholds); this proves it.

3.  **Sweep.** n x spacing x exit-mode grid, every cell on the same TOTAL leg
    budget as the baseline -- n=3 means three tranches of a third each, so the
    fully-added basket carries the baseline's exposure and a partially-added
    one carries less. Results land in data/runs/scale_in_study/results.csv.

The headline configuration is duplicated from report/pair.py deliberately:
importing the report package for five constants would couple an experiment to
the report theme; if the headline moves, move HEADLINE below with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import engine  # noqa: E402
from backtest import grid  # noqa: E402
from backtest.scale_in import (  # noqa: E402
    ScaleInParams,
    default_be_cost_floor,
    run_scale_in,
    write_scale_in_outputs,
)
from features import zscore as zscore_calc  # noqa: E402
from lib import paths  # noqa: E402

HEADLINE = dict(
    window=2500,
    entry_z=1.0,
    exit_z=0.25,
    displacement=0.2317,
    # The report anchors the displacement at CCF's measurement price and scales
    # it 1/price per bar. Omitting this ran the whole study on a flat 0.2317 --
    # a cheaper market than the headline it claims to reproduce, and cheapest
    # exactly on the low-price months this experiment is about.
    ref_price=121.50,
)
CAPITAL = 2_000_000.0
LEG_NOTIONAL = 1_000_000.0
OUT_DIR = paths.run_dir("scale_in_study")

SPACINGS = [0.5, 0.75, 1.0, 1.5, 2.0]
TRANCHE_COUNTS = [2, 3]
# (exit_mode, be_offset_k); the cost floor covers crossings AND fees, set below.
EXIT_VARIANTS = [("basket", 0.0), ("breakeven", 0.0), ("breakeven", 0.5)]


def headline_params() -> engine.BacktestParams:
    return engine.BacktestParams(
        entry_z=HEADLINE["entry_z"],
        exit_z=HEADLINE["exit_z"],
        leg_notional_twd=LEG_NOTIONAL,
        initial_capital_twd=CAPITAL,
        max_entry_delay_minutes=15,
        tsm_fee_bps=2.5,
        tsm_fee_model="ibkr",
        qff_fee_per_contract_twd=88.0,
        qff_tax_rate=2e-5,
        qff_contract_multiplier=2000.0,
        executable_displacement=HEADLINE["displacement"],
        displacement_ref_price=HEADLINE["ref_price"],
    )


def build_frame() -> pd.DataFrame:
    spread = zscore_calc.read_spread_frame(paths.feature("ccf_umc", "spread_1m"))
    seed_path = paths.feature("ccf_umc", "seed")
    seed = zscore_calc.read_spread_frame(seed_path) if seed_path.exists() else None
    if seed is not None:
        first = pd.DatetimeIndex(spread["timestamp"]).min()
        seed = seed[pd.DatetimeIndex(seed["timestamp"]) < first]
        if seed.empty:
            seed = None
    return zscore_calc.calculate_zscore(
        spread, HEADLINE["window"], seed_frame=seed
    )


# --------------------------------------------------------------------------
# stage 1: MAE distribution
# --------------------------------------------------------------------------


def mae_table(frame: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    spread = frame["spread"].to_numpy(dtype=float)
    std_col = f"spread_std_{HEADLINE['window']}"
    spread_std = frame[std_col].to_numpy(dtype=float)
    rows = []
    for t in trades.itertuples():
        a, b = int(t.entry_idx), int(t.exit_idx)
        sgn = engine.direction_sign(t.direction)
        excursion = sgn * (spread[a : b + 1] - spread[a])
        std_ref = spread_std[int(t.entry_signal_idx)]
        mae = float(np.maximum(excursion, 0.0).max())
        rows.append(
            {
                "entry_time": t.entry_time,
                "direction": t.direction,
                "exit_reason": t.exit_reason,
                "net_pnl_twd": t.net_pnl_twd,
                "won": t.net_pnl_twd > 0,
                "holding_minutes": t.holding_minutes,
                "std_ref": std_ref,
                "mae_spread": mae,
                "mae_k": mae / std_ref,
                "bars_to_mae": int(np.argmax(excursion)),
            }
        )
    return pd.DataFrame(rows)


def print_mae_report(mae: pd.DataFrame) -> None:
    print("\n=== Stage 1: MAE of the single-entry baseline "
          "(entry-time std units) ===")
    print(f"trades: {len(mae)}, median std_ref {mae['std_ref'].median():.4f} "
          "spread units")
    for label, part in (
        ("all", mae),
        ("winners", mae[mae["won"]]),
        ("losers/forced", mae[~mae["won"]]),
    ):
        if part.empty:
            print(f"  {label:14s}: none")
            continue
        q = part["mae_k"].quantile
        print(
            f"  {label:14s}: n={len(part):3d}  "
            f"p50={q(0.5):.2f}k  p75={q(0.75):.2f}k  p90={q(0.9):.2f}k  "
            f"max={part['mae_k'].max():.2f}k"
        )
    print("  trades whose MAE reaches m x spacing "
          "(= tranche m+1 would have filled):")
    header = f"  {'spacing k':>10s} " + "".join(
        f"{label:>14s}" for label in ("adds@1k won", "adds@1k lost",
                                      "adds@2k won", "adds@2k lost")
    )
    print(header)
    for k in SPACINGS:
        cells = []
        for mult in (1, 2):
            hit = mae[mae["mae_k"] >= mult * k]
            cells.append(f"{int(hit['won'].sum()):>14d}")
            cells.append(f"{int((~hit['won']).sum()):>14d}")
        print(f"  {k:>10.2f} " + "".join(cells))


# --------------------------------------------------------------------------
# stage 2: n=1 parity against the single-entry engine
# --------------------------------------------------------------------------


def check_parity(frame: pd.DataFrame, base: engine.BacktestParams) -> None:
    print("\n=== Stage 2: n_tranches=1 parity against engine.run_backtest ===")
    single = engine.run_backtest(frame, base)
    scale = run_scale_in(frame, ScaleInParams(base=base, n_tranches=1))
    st, bt = single.trades, scale.tranches
    if len(st) != len(bt):
        raise RuntimeError(f"trade count differs: engine {len(st)}, "
                           f"scale_in {len(bt)}")
    for col in ("entry_idx", "exit_idx", "direction", "qff_contracts"):
        if not (st[col].to_numpy() == bt[col].to_numpy()).all():
            raise RuntimeError(f"column {col} differs between engines")
    net_diff = abs(float(st["net_pnl_twd"].sum()) - float(bt["net_pnl_twd"].sum()))
    if net_diff > 1e-6:
        raise RuntimeError(f"net PnL differs by {net_diff}")
    print(f"OK: {len(st)} trades, identical fills, net PnL matches "
          f"({float(st['net_pnl_twd'].sum()):,.0f} TWD)")


# --------------------------------------------------------------------------
# stage 3: the sweep
# --------------------------------------------------------------------------


def summarise_run(result, label: dict) -> dict:
    s = result.summary
    stats = grid.calculate_daily_return_stats(
        result.equity, initial_capital_twd=CAPITAL, annual_trading_days=252.0
    )
    hist = s["tranches_filled_histogram"]
    return {
        **label,
        "baskets": s["baskets"],
        "fills": "/".join(f"{k}:{v}" for k, v in sorted(hist.items())),
        "win_rate": round(s["win_rate"], 3),
        "net_twd": round(s["total_pnl_twd"]),
        "return_pct": round(100 * s["return_pct"], 2),
        "sharpe": (
            round(stats["sharpe_ratio"], 2)
            if stats["sharpe_ratio"] is not None
            else np.nan
        ),
        "maxdd_pct": round(100 * s["max_drawdown_pct"], 2),
        "worst_basket_twd": round(s["worst_basket_twd"]),
        "median_basket_twd": round(s["median_basket_twd"]),
        "total_fee_twd": round(s["total_fee_twd"]),
        "max_gross_notional_twd": round(s["max_gross_notional_twd"]),
        "exposure_pct": round(100 * s["exposure_ratio"], 1),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = headline_params()
    be_cost_floor = default_be_cost_floor(base)

    print("building z-score frame "
          f"(w{HEADLINE['window']}, CCF/UMC spread_1m)...")
    frame = build_frame()
    valid = int(frame["zscore_valid"].sum())
    print(f"{len(frame):,} rows, {valid:,} valid z rows")

    single = engine.run_backtest(frame, base)
    mae = mae_table(frame, single.trades)
    mae.to_csv(OUT_DIR / "mae_baseline.csv", index=False)
    print_mae_report(mae)

    check_parity(frame, base)

    print("\n=== Stage 3: sweep ===")
    rows = []
    baseline_row = summarise_run(
        run_scale_in(frame, ScaleInParams(base=base, n_tranches=1)),
        {"n": 1, "k": np.nan, "exit_mode": "basket", "be_offset_k": np.nan},
    )
    rows.append(baseline_row)
    summaries = {"baseline_n1": baseline_row}
    for n in TRANCHE_COUNTS:
        for k in SPACINGS:
            for exit_mode, be_offset in EXIT_VARIANTS:
                params = ScaleInParams(
                    base=base,
                    n_tranches=n,
                    add_spacing_k=k,
                    exit_mode=exit_mode,
                    be_offset_k=be_offset,
                    be_cost_floor=be_cost_floor,
                )
                result = run_scale_in(frame, params)
                tag = f"n{n}_k{k:g}_{exit_mode}"
                if exit_mode == "breakeven":
                    tag += f"_be{be_offset:g}"
                row = summarise_run(
                    result,
                    {"n": n, "k": k, "exit_mode": exit_mode,
                     "be_offset_k": be_offset},
                )
                rows.append(row)
                summaries[tag] = row
                print(
                    f"  {tag:28s} baskets {row['baskets']:3d} "
                    f"({row['fills']:12s}) net {row['net_twd']:>9,d} "
                    f"sharpe {row['sharpe']:>5} maxDD {row['maxdd_pct']:>6}% "
                    f"worst {row['worst_basket_twd']:>8,d}"
                )

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "results.csv", index=False)
    (OUT_DIR / "sweep_summaries.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'results.csv'}")

    # Keep full artifacts for the single most interesting cell of each n, by
    # Sharpe with the baseline's worst-basket as a tie context -- the study
    # report reads these for trade-level detail.
    swept = results[results["n"] > 1]
    for n in TRANCHE_COUNTS:
        best = swept[swept["n"] == n].nlargest(1, "sharpe")
        if best.empty:
            continue
        b = best.iloc[0]
        params = ScaleInParams(
            base=base,
            n_tranches=int(b["n"]),
            add_spacing_k=float(b["k"]),
            exit_mode=str(b["exit_mode"]),
            be_offset_k=float(b["be_offset_k"]),
            be_cost_floor=be_cost_floor,
        )
        result = run_scale_in(frame, params)
        tag = f"best_n{n}"
        write_scale_in_outputs(result, OUT_DIR / tag)
        print(f"kept full outputs for {tag}: n={b['n']:.0f} k={b['k']:g} "
              f"{b['exit_mode']} be={b['be_offset_k']:g}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
