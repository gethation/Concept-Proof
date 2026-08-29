"""The perfect-foresight profit ceiling of the CCF/UMC spread.

How much money is there in this pair AT ALL? Run an oracle over the canonical
spread series: a trader holding at most one unit of spread (+1/0/-1, the
strategy's own 1M-per-leg shape) who sees the entire future and pays the
strategy's own costs -- the per-bar executable displacement (0.2317 at ref
price 121.50, scaled 1/price like engine.displacement_at) plus fees on both
legs. Dynamic programming over {flat, long, short} gives the exact maximum,
and the backtrace gives the oracle's trades.

What this is for:
  - the ceiling itself (in spread points, TWD at 1M/leg, and % of the 2M
    capital convention);
  - the capture ratio: what fraction the canonical w3900/e2.0/x-0.25 run
    actually took;
  - where the ceiling lives: profit by holding duration (the user's
    "if I could look ahead X, how much could I earn"), by month, and by
    per-trade size;
  - cost sensitivity: the same DP at 0x / 0.5x / 1x / 2x displacement.

Honesty notes: fills are at bar close (the engine fills next-bar-open; worth
1-3% on net historically, irrelevant at ceiling scale); the TWD conversion
uses 1 spread point ~= 1% of leg notional; the duration decomposition slices
the UNCONSTRAINED oracle's trades -- a re-optimised duration-capped oracle
could redistribute some profit across buckets.

Usage:
    python scripts/studies/spread_oracle.py

Outputs land in data/runs/spread_oracle/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import paths  # noqa: E402
from studies.spread_timing import CELL, canonical, trade_sign  # noqa: E402

OUT_DIR = paths.DATA / "runs" / "spread_oracle"

DISPLACEMENT = 0.2317
REF_PRICE = 121.50
US_FEE_PTS = 0.025  # ~2.5 bps per side; the ibkr per-ADR model sits 2.3-3.0
QFF_FEE_TWD = 88.0
MULTIPLIER = 2000.0
TAX_PTS = 0.002  # 2e-5 of notional
LEG_NOTIONAL = 1_000_000.0
CAPITAL = 2_000_000.0
PTS_TO_TWD = LEG_NOTIONAL / 100.0  # 1 spread point ~= 1% of leg notional

FLAT, LONG, SHORT = 0, 1, 2


def side_cost_pts(price: np.ndarray, disp_mult: float) -> np.ndarray:
    """One side-change's cost in spread points at each bar."""
    disp = DISPLACEMENT * REF_PRICE / price * disp_mult
    ccf_fee = QFF_FEE_TWD / (price * MULTIPLIER) * 100.0
    return disp + ccf_fee + US_FEE_PTS + TAX_PTS


def oracle_dp(s: np.ndarray, c: np.ndarray) -> tuple[float, list[tuple[int, int, int]]]:
    """Max profit of a +/-1-bounded perfect-foresight trader, and its trades.

    Buying the spread costs s+c, selling receives s-c, at every bar. Returns
    (profit, [(entry_index, exit_index, direction)]) with direction +1 for
    long-spread. The final bar force-liquidates any open position.
    """
    n = len(s)
    NEG = -1e18
    cash = np.array([0.0, NEG, NEG])
    # transitions[t, state] = state at t-1 that produced cash[state] at t
    trans = np.empty((n, 3), dtype=np.int8)
    for t in range(n):
        buy, sell = s[t] + c[t], s[t] - c[t]
        options_flat = (cash[FLAT], cash[LONG] + sell, cash[SHORT] - buy)
        options_long = (cash[LONG], cash[FLAT] - buy)
        options_short = (cash[SHORT], cash[FLAT] + sell)
        f = int(np.argmax(options_flat))
        l = int(np.argmax(options_long))
        h = int(np.argmax(options_short))
        trans[t, FLAT] = (FLAT, LONG, SHORT)[f]
        trans[t, LONG] = (LONG, FLAT)[l]
        trans[t, SHORT] = (SHORT, FLAT)[h]
        cash = np.array(
            [options_flat[f], options_long[l], options_short[h]]
        )
    profit = float(cash[FLAT])

    # backtrace
    state = FLAT
    trades = []
    exit_idx = None
    for t in range(n - 1, -1, -1):
        prev = int(trans[t, state])
        if state == FLAT and prev in (LONG, SHORT):
            exit_idx = t
            direction = +1 if prev == LONG else -1
        elif state in (LONG, SHORT) and prev == FLAT:
            trades.append((t, exit_idx, direction))
        state = prev
    trades.reverse()
    return profit, trades


def describe(profit_pts: float, days: float) -> dict:
    twd = profit_pts * PTS_TO_TWD
    return {
        "profit_pts": profit_pts,
        "profit_twd_at_1m_leg": twd,
        "return_pct_of_capital": twd / CAPITAL * 100.0,
        "annualised_linear_pct": twd / CAPITAL * 100.0 * 365.0 / days,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zframe, strat_trades = canonical()
    ts = pd.DatetimeIndex(zframe["timestamp"])
    days = (ts[-1] - ts[0]).total_seconds() / 86400.0
    s = zframe["spread"].to_numpy(float)
    price = zframe["qff_close_filled"].to_numpy(float)
    session_codes = pd.factorize(zframe["session"])[0]

    result = {
        "parameters": {
            "displacement": DISPLACEMENT,
            "ref_price": REF_PRICE,
            "us_fee_pts_per_side": US_FEE_PTS,
            "qff_fee_twd": QFF_FEE_TWD,
            "bars": int(len(s)),
            "days": days,
            "fill": "bar close (engine fills next-bar open; ~1-3% of net)",
        }
    }

    # cost sensitivity ladder; 1x is the headline
    trades_1x = None
    ladder = {}
    for mult, label in ((0.0, "fees_only"), (0.5, "half_disp"),
                        (1.0, "full_disp"), (2.0, "double_disp")):
        c = side_cost_pts(price, mult)
        profit, trades = oracle_dp(s, c)
        entry = describe(profit, days)
        entry["trades"] = len(trades)
        ladder[label] = entry
        if mult == 1.0:
            trades_1x = trades
            cost_1x = c
        print(f"oracle {label:12s}: {profit:8.1f} pts, {len(trades):5d} trades")
    result["cost_ladder"] = ladder

    # anatomy of the 1x oracle
    rows = []
    for i, j, direction in trades_1x:
        gross = direction * (s[j] - s[i])
        net = gross - cost_1x[i] - cost_1x[j]
        rows.append(
            {
                "entry_time": str(ts[i]),
                "exit_time": str(ts[j]),
                "direction": int(direction),
                "net_pts": float(net),
                "hold_minutes": float((ts[j] - ts[i]).total_seconds() / 60.0),
                "sessions_held": int(session_codes[j] - session_codes[i]) + 1,
                "month": str(ts[i].strftime("%Y-%m")),
            }
        )
    ot = pd.DataFrame(rows)
    ot.to_csv(OUT_DIR / "oracle_trades.csv", index=False)

    total = float(ot["net_pts"].sum())
    by_duration = {
        label: float(ot.loc[mask, "net_pts"].sum())
        for label, mask in (
            ("within_1_session", ot["sessions_held"] == 1),
            ("2_sessions", ot["sessions_held"] == 2),
            ("3_to_5_sessions", ot["sessions_held"].between(3, 5)),
            ("over_5_sessions", ot["sessions_held"] > 5),
        )
    }
    top = ot["net_pts"].sort_values(ascending=False)
    result["oracle_1x"] = {
        "total_pts": total,
        "trade_count": int(len(ot)),
        "net_pts_median": float(ot["net_pts"].median()),
        "hold_minutes_median": float(ot["hold_minutes"].median()),
        "sessions_held_median": float(ot["sessions_held"].median()),
        "profit_by_duration_pts": by_duration,
        "monthly_pts": {
            k: float(v) for k, v in ot.groupby("month")["net_pts"].sum().items()
        },
        "top_trades_cum_pts": {
            "top_21": float(top.head(21).sum()),
            "top_50": float(top.head(50).sum()),
            "top_100": float(top.head(100).sum()),
        },
    }

    # The slow-layer ceiling: an oracle that may only act once per session, at
    # the close. This is the layer the canonical strategy actually inhabits --
    # the minute-level 76% of the unconstrained ceiling is a different (and
    # per past studies, microstructure-treacherous) business.
    last_idx = np.flatnonzero(
        session_codes != np.roll(session_codes, -1)
    )
    s_daily = s[last_idx]
    c_daily = side_cost_pts(price[last_idx], 1.0)
    profit_daily, trades_daily = oracle_dp(s_daily, c_daily)
    daily_entry = describe(profit_daily, days)
    daily_entry["trades"] = len(trades_daily)
    daily_entry["decision_points"] = int(len(s_daily))
    result["oracle_daily_close_only"] = daily_entry
    print(f"oracle daily-close-only: {profit_daily:.1f} pts, "
          f"{len(trades_daily)} trades")

    # what the real strategy took of it
    strat_pts = float(strat_trades["net_pnl_twd"].sum()) / PTS_TO_TWD
    result["capture"] = {
        "canonical_trades": int(len(strat_trades)),
        "canonical_net_pts": strat_pts,
        "canonical_net_twd": float(strat_trades["net_pnl_twd"].sum()),
        "capture_ratio_vs_oracle": strat_pts / total,
        "oracle_top21_vs_canonical": float(top.head(21).sum()) / strat_pts,
    }

    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(result, fh, indent=2)

    # ---- digest ----------------------------------------------------------
    o = result["oracle_1x"]
    print(f"\n=== oracle ceiling (full costs) over {days:.0f} days ===")
    full = ladder["full_disp"]
    print(f"{full['profit_pts']:.1f} pts = {full['profit_twd_at_1m_leg']:,.0f} TWD "
          f"@1M/leg = {full['return_pct_of_capital']:.1f}% of 2M "
          f"({full['annualised_linear_pct']:.0f}%/yr linear)")
    print(f"{o['trade_count']} trades, median {o['net_pts_median']:.2f} pts, "
          f"median hold {o['hold_minutes_median']:.0f}min "
          f"({o['sessions_held_median']:.0f} session)")
    print("profit by holding duration: "
          + "  ".join(f"{k}={v:.0f} ({100 * v / total:.0f}%)"
                      for k, v in by_duration.items()))
    print("monthly pts: "
          + "  ".join(f"{k[-2:]}:{v:.0f}" for k, v in o["monthly_pts"].items()))
    print(f"top-21 oracle trades: {o['top_trades_cum_pts']['top_21']:.1f} pts; "
          f"top-50: {o['top_trades_cum_pts']['top_50']:.1f}; "
          f"top-100: {o['top_trades_cum_pts']['top_100']:.1f}")
    dc = result["oracle_daily_close_only"]
    print(f"daily-close-only oracle: {dc['profit_pts']:.1f} pts "
          f"({dc['trades']} trades over {dc['decision_points']} sessions) = "
          f"{dc['return_pct_of_capital']:.1f}% of 2M "
          f"({dc['annualised_linear_pct']:.0f}%/yr)")
    cap = result["capture"]
    print(f"\ncanonical strategy: {cap['canonical_net_pts']:.1f} pts "
          f"({cap['canonical_net_twd']:,.0f} TWD) with {cap['canonical_trades']} "
          f"trades = {100 * cap['capture_ratio_vs_oracle']:.1f}% of the full "
          f"ceiling, {100 * cap['canonical_net_pts'] / dc['profit_pts']:.0f}% "
          f"of the daily-close-only one")
    print(f"an oracle limited to 21 trades would take "
          f"{cap['oracle_top21_vs_canonical']:.1f}x the canonical net")
    print("\ncost ladder: "
          + "  ".join(f"{k}={v['profit_pts']:.0f}pts/{v['trades']}tr"
                      for k, v in ladder.items()))
    print(f"\nWrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
