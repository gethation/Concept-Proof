"""Sweep max_holding_minutes on one fixed configuration.

    python scripts/backtest/holding_sweep.py --pair ccf_umc
    python scripts/backtest/holding_sweep.py --pair ccf_umc --top 5

This is a DIAGNOSTIC, not a selection step, and the distinction is the whole
design. It deliberately does NOT add a fourth axis to report/pair.py's grid:
that grid already fits three parameters on 28-32 trades, and picking a holding
cap out of a fourth axis would be fitting one more on a sample that cannot
support the three it has. What this answers instead is narrower and safe --
holding the published configuration fixed, what does a cap do to trade count,
exposure and per-trade edge?

The motivating measurement, taken on the 28-trade CCF/UMC run that stood
before the 2026-08-30 data refresh (data/runs/fin_w2500_e10_x05):

  * the shortest holding quartile (median 297 min) earned 82,907 TWD and the
    longest (median 5,760 min) earned 80,708 -- the same money for roughly 20x
    the capital-time;
  * the book was occupied 86.2% of the elapsed window, the median gap between
    one exit and the next entry was 109 minutes, and 5 of 27 gaps were under
    5 minutes.

Those figures predate the refresh, the corrected open-price fills and the
directly measured displacement, so they are the hypothesis this script tests,
not evidence for it. The sweep re-derives everything from current data.

Together those say the binding constraint is not signal scarcity but the
single-position book plus long holds: every signal that fires while a position
is open is silently dropped. If that is real, a cap trades a little edge per
trade for a lot of capital-time back.

WHAT WOULD FALSIFY IT: a cap that cuts exposure without raising trade count
(the freed capital is not being re-used), or one that pushes the median edge
below the 2-tick screen (the cap is closing trades before they converge, so the
edge was in the tail after all). Both are printed.

A cap of 0 disables the stop, so the first row reproduces the published run and
is the baseline every other row is read against. If it does not match the
report, the inputs have moved and nothing below is comparable.

Note on the clock: holding_minutes is wall-clock, and the CCF/UMC index is
NYSE RTH only (a 390-minute session, then a 1,050-minute gap). So a cap below
~390 closes inside the same session, anything from there to ~1,440 closes at
the next session's open, and only caps above that span multiple days. The
regimes are not evenly spaced -- read the cap column, not the row spacing.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import engine as backtest  # noqa: E402
from backtest import grid  # noqa: E402
from features import zscore as zscore_calc  # noqa: E402
from lib import paths  # noqa: E402

# report/pair.py owns the per-pair strategy configuration -- fees, multiplier,
# tick size, measured displacement, spread and seed paths. Importing it keeps
# one source of truth; duplicating those constants here is exactly the drift
# this repo has spent two reorganisations removing. No cycle: report.pair
# imports backtest.engine, backtest.engine imports nothing from report.
from report import pair  # noqa: E402

# Caps in minutes, chosen to straddle the session structure described above
# rather than to be evenly spaced. 0 disables the stop.
DEFAULT_CAPS = [0, 60, 120, 240, 360, 480, 720, 1440, 2880, 4320]

# The configuration each report publishes as of the 2026-08-30 data. Both are
# currently the grid's highest linearly-annualised return AND unscreened -- the
# report ranks by return and labels the screen rather than filtering on it, so
# the baseline row below is a cell that fails at least one quality threshold.
# Defaults
# for convenience only -- the report re-picks from a fresh grid on every run,
# so if the data has moved these are stale. Pass --window/--entry-z/--exit-z to
# override, or --top to re-derive the pick from a grid run here.
PUBLISHED = {
    "ccf_umc": (1560, 1.0, 0.5),
    "qff_tsm": (2500, 1.5, 0.0),
}


def parse_caps(value: str) -> list[int]:
    caps = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not caps:
        raise argparse.ArgumentTypeError("Expected at least one cap")
    if any(cap < 0 for cap in caps):
        raise argparse.ArgumentTypeError("Caps must be non-negative (0 = no cap)")
    return caps


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", choices=sorted(pair.PAIRS), required=True)
    ap.add_argument("--caps", type=parse_caps, default=DEFAULT_CAPS)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--entry-z", type=float, default=None)
    ap.add_argument("--exit-z", type=float, default=None)
    ap.add_argument(
        "--top",
        type=int,
        default=0,
        help=(
            "Run the full grid, then sweep the N highest-Sharpe screened "
            "configurations instead of one. Slow (adds 100 backtests), but it "
            "is what tells you whether a cap helps the strategy or only helps "
            "one cell."
        ),
    )
    ap.add_argument(
        "--segment",
        type=int,
        default=0,
        help="Index into the pair's segments. 0 is the one the report leads with.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output. Default: data/runs/_grids/holding_sweep_<pair>.json",
    )
    return ap.parse_args(argv)


def sweep(
    spec: pair.PairSpec,
    spread: pd.DataFrame,
    seed: pd.DataFrame | None,
    seg: pair.Segment,
    window: int,
    entry_z: float,
    exit_z: float,
    caps: list[int],
) -> list[dict[str, Any]]:
    """One row per cap, all sharing a single z-score pass over the spread."""
    zframe = pair.slice_segment(
        zscore_calc.calculate_zscore(spread, window, seed_frame=seed), seg
    )
    # Segments spanning a tick-size change carry a per-bar displacement; without
    # this the whole span would be priced at the segment scalar.
    zframe = pair.with_displacement_column(spec, seg, zframe)
    base = pair.params_for(
        spec, entry_z, exit_z, seg.displacement, seg.ref_price
    )

    rows: list[dict[str, Any]] = []
    for cap in caps:
        result = backtest.run_backtest(zframe, replace(base, max_holding_minutes=cap))
        summary = result.summary
        trades = result.trades
        stats = grid.calculate_daily_return_stats(
            result.equity,
            initial_capital_twd=pair.CAPITAL,
            annual_trading_days=252.0,
        )

        span_days = (
            pd.Timestamp(summary["end"]) - pd.Timestamp(summary["start"])
        ).total_seconds() / 86400.0
        exposure = float(summary["exposure_ratio"])
        row: dict[str, Any] = {
            "cap_minutes": cap,
            "window": window,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "trades": int(summary["trade_count"]),
            "time_stop_exits": int(summary.get("time_stop_exits", 0)),
            "net_pnl_twd": float(summary["net_pnl_twd"]),
            "return_pct": float(summary["return_pct"]),
            "annualised_pct": float(summary["return_pct"]) * 365.0 / span_days,
            "sharpe": stats["sharpe_ratio"],
            "max_drawdown_pct": float(summary["max_drawdown_pct"]),
            "exposure_ratio": exposure,
            # Return per unit of capital-time. This is a capital-EFFICIENCY
            # ratio, not a realisable return: it is only collectable if the
            # freed capital can be redeployed, which needs either concurrent
            # positions or another pair. Read it as an upper bound on what a
            # cap is worth, not as a forecast.
            "return_per_exposure": (float(summary["return_pct"]) / exposure)
            if exposure > 0
            else None,
        }
        if len(trades):
            ticks = pair.trade_ticks(spec, seg, trades)
            row.update(
                med_ticks=float(ticks.median()),
                under1=float((ticks < 1).mean()),
                med_holding_minutes=float(trades["holding_minutes"].median()),
                win_rate=float((trades["net_pnl_twd"] > 0).mean()),
            )
        else:
            row.update(
                med_ticks=None, under1=None, med_holding_minutes=None, win_rate=None
            )
        rows.append(row)
        print(
            f"  cap={cap:>5}  trades={row['trades']:>3}  "
            f"ret={row['return_pct']:>7.2%}  expo={exposure:>6.1%}",
            flush=True,
        )
    return rows


def fmt(value: Any, spec: str, dash: str = "—") -> str:
    """Format, falling back to a dash padded to the spec's own column width.

    An unpadded dash shifts every column to its right, which on a table read
    for small differences between rows is worse than the missing value itself.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        digits = "".join(c for c in spec.lstrip("<>^") if c.isdigit() or c == ".")
        width = digits.split(".")[0]
        return dash.rjust(int(width)) if width else dash
    return format(value, spec)


def print_table(rows: list[dict[str, Any]], spec: pair.PairSpec) -> None:
    header = (
        f"{'上限(分)':>9} {'交易':>5} {'時停':>5} {'報酬':>9} {'Sharpe':>8} "
        f"{'maxDD':>8} {'曝險':>8} {'報酬/曝險':>10} {'中位ticks':>10} "
        f"{'<1tick':>8} {'中位持有':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    base = rows[0] if rows and rows[0]["cap_minutes"] == 0 else None
    for row in rows:
        label = "無上限" if row["cap_minutes"] == 0 else f"{row['cap_minutes']:,}"
        flag = ""
        if row["med_ticks"] is not None and row["med_ticks"] < 2.0:
            # The 2-tick screen is the report's own economic-significance floor;
            # a cap that breaks it has bought exposure with edge that was real.
            flag = "  ← 跌破 2 ticks 篩選門檻"
        elif base and row["return_per_exposure"] and base["return_per_exposure"]:
            if row["return_per_exposure"] > base["return_per_exposure"] * 1.25:
                flag = "  ← 資本效率 +25% 以上"
        print(
            f"{label:>9} {row['trades']:>5} {row['time_stop_exits']:>5} "
            f"{fmt(row['return_pct'], '>9.2%')} {fmt(row['sharpe'], '>8.2f')} "
            f"{fmt(row['max_drawdown_pct'], '>8.2%')} "
            f"{fmt(row['exposure_ratio'], '>8.1%')} "
            f"{fmt(row['return_per_exposure'], '>10.2%')} "
            f"{fmt(row['med_ticks'], '>10.2f')} "
            f"{fmt(row['under1'], '>8.0%')} "
            f"{fmt(row['med_holding_minutes'], '>9,.0f')}"
            f"{flag}"
        )
    print()
    print(
        "「報酬/曝險」是資本效率比，不是可實現報酬 —— 要收得到必須有地方"
        "重新部署釋放出來的資本（並存部位或另一個配對）。"
    )
    print(
        f"「中位 ticks」低於 2.0 代表該上限已經砍進真實邊際："
        f"{spec.futures_leg} 一個 tick = {spec.tick_desc}。"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    spec = pair.PAIRS[args.pair]
    spread, seed = pair.load_frames(spec)
    seg = spec.segments[args.segment]
    print(
        f"{spec.name} · segment「{seg.label}」· 位移 {seg.displacement:.4f} · "
        f"{len(spread):,} spread 列 · seed={'有' if seed is not None else '無'}"
    )

    configs: list[tuple[int, float, float]] = []
    if args.top > 0:
        print(f"\n跑完整網格以取出前 {args.top} 名（約 100 次回測）...")
        g = pair.run_grid(spec, spread, seed, seg)
        g["neighbour"] = pair.neighbour_sharpe(g)
        surv = pair.screen(g)
        if not len(surv):
            print(
                "WARNING: 沒有任何組態通過篩選；改用未篩選網格的前幾名，"
                "這些組態至少違反一道門檻。"
            )
        pool = surv if len(surv) else g
        for r in pool.nlargest(args.top, "ann").itertuples():
            configs.append((int(r.window), float(r.entry_z), float(r.exit_z)))
    else:
        window, entry_z, exit_z = PUBLISHED[args.pair]
        configs.append(
            (
                args.window if args.window is not None else window,
                args.entry_z if args.entry_z is not None else entry_z,
                args.exit_z if args.exit_z is not None else exit_z,
            )
        )

    all_rows: list[dict[str, Any]] = []
    for window, entry_z, exit_z in configs:
        print(f"\n=== w{window} / entry_z {entry_z:g} / exit_z {exit_z:g} ===")
        rows = sweep(spec, spread, seed, seg, window, entry_z, exit_z, args.caps)
        print_table(rows, spec)
        all_rows.extend(rows)

    out = args.out or (paths.RUNS / "_grids" / f"holding_sweep_{args.pair}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "pair": args.pair,
                "segment": seg.label,
                "displacement": seg.displacement,
                # Repo-relative: an absolute path here is the running
                # machine's, and this file is tracked.
                "spread_file": str(
                    spec.spread.relative_to(paths.REPO_ROOT)
                    if spec.spread.is_relative_to(paths.REPO_ROOT)
                    else spec.spread
                ),
                "caps": args.caps,
                "note": (
                    "Diagnostic sweep of max_holding_minutes on fixed "
                    "configurations. cap_minutes=0 is the unmodified published "
                    "run. return_per_exposure is a capital-efficiency ratio, "
                    "not a realisable return."
                ),
                "rows": all_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n寫入 {out}")

    baseline = next((r for r in all_rows if r["cap_minutes"] == 0), None)
    if baseline is not None:
        print(
            f"基準（無上限）：{baseline['trades']} 筆、報酬 "
            f"{baseline['return_pct']:.2%}、曝險 {baseline['exposure_ratio']:.1%}"
            " —— 請對照報告確認一致，不一致代表輸入已經變動。"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
