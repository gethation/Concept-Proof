"""Two-account margin-transfer policy simulation (Binance perp + Fubon futures).

The pair strategy is market-neutral in TOTAL, but each account carries the
full single-leg directional risk: when TSM moves, one account gains and the
other loses roughly the same amount. Transfers between the accounts are
constrained: initiate only at 10:00 Taipei, funds arrive 17:00 the same day,
at most once per day. The no-help windows are therefore:

    7h  - transfer initiated today: survive 10:00 -> 17:00 on current margin
    31h - transfer skipped today: survive until tomorrow 17:00
    (Friday session-end force close keeps weekends flat -> Friday's window is
    ~19h and dominated by the 31h numbers; verified 0 weekend positions.)

This script (1) measures the empirical max adverse single-leg excursion over
the 7h/31h windows from every weekday 10:00 decision point, and (2) replays
the backtest's actual positions through a two-account balance simulation
under the policy "at 10:00, if an account's equity ratio is below its
trigger, transfer from the other account back up to the target ratio,
credited at 17:00", counting transfers and maintenance breaches.

Maintenance-rate defaults are PLACEHOLDERS - verify against the current
Binance margin-tier table and TAIFEX/Fubon stock-futures margin levels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAIPEI_TZ = "Asia/Taipei"
DEFAULT_ZSCORE_PATH = Path("data/processed/qff_tsm_spread_zscore_15m_w33.csv")
DEFAULT_EQUITY_PATH = Path("data/processed/qff_tsm_pair_backtest_equity_15m_best.csv")
DEFAULT_TRADES_PATH = Path("data/processed/qff_tsm_pair_backtest_trades_15m_best.csv")
DEFAULT_SUMMARY_OUT = Path("data/processed/margin_transfer_simulation.json")

DEFAULT_LEG_NOTIONAL_TWD = 1_000_000.0
DEFAULT_BINANCE_MAINT = 0.025   # PLACEHOLDER: Binance TSMUSDT.P maintenance tier
DEFAULT_FUBON_MAINT = 0.1035    # PLACEHOLDER: TAIFEX stock-futures maintenance level
DEFAULT_TRIGGER_BINANCE = 0.11  # maint + 31h max excursion, rounded up
DEFAULT_TRIGGER_FUBON = 0.20
DEFAULT_TARGETS = "0.20,0.25,0.30,0.35,0.40"
DECISION_HOUR = 10
ARRIVAL_HOUR = 17


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zscore-path", type=Path, default=DEFAULT_ZSCORE_PATH)
    p.add_argument("--equity-path", type=Path, default=DEFAULT_EQUITY_PATH)
    p.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES_PATH)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    p.add_argument("--leg-notional-twd", type=float, default=DEFAULT_LEG_NOTIONAL_TWD)
    p.add_argument("--binance-maint", type=float, default=DEFAULT_BINANCE_MAINT)
    p.add_argument("--fubon-maint", type=float, default=DEFAULT_FUBON_MAINT)
    p.add_argument("--trigger-binance", type=float, default=DEFAULT_TRIGGER_BINANCE)
    p.add_argument("--trigger-fubon", type=float, default=DEFAULT_TRIGGER_FUBON)
    p.add_argument(
        "--targets",
        default=DEFAULT_TARGETS,
        help="comma-separated funding targets to sweep, as fractions of leg notional",
    )
    p.add_argument("--skip-self-test", action="store_true")
    return p.parse_args(argv)


def load_frame(
    zscore_path: Path, equity_path: Path, trades_path: Path
) -> pd.DataFrame:
    z = pd.read_csv(
        zscore_path,
        parse_dates=["timestamp"],
        usecols=["timestamp", "tsm_twd_fair", "qff_close_filled"],
    )
    eq = pd.read_csv(
        equity_path,
        parse_dates=["timestamp"],
        usecols=["timestamp", "tsm_units", "qff_units"],
    )
    for frame in (z, eq):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
            TAIPEI_TZ
        )
    df = z.merge(eq, on="timestamp").sort_values("timestamp").reset_index(drop=True)
    if len(df) != len(z) or len(df) != len(eq):
        raise RuntimeError("zscore and equity files do not share the same bar grid")

    df["fee_binance"] = 0.0
    df["fee_fubon"] = 0.0
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    for column in ("entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True).dt.tz_convert(
            TAIPEI_TZ
        )
    # Crossing cost is the price paid for taking both books, so it is real cash
    # out of both accounts -- at the measured 0.2151 displacement it is ~43 bps
    # round trip, several times the commission schedule, and dropping it made
    # the simulated balances drift high and understate margin-call risk. The
    # trade row carries only the combined figure (the displacement bundles the
    # CCF tick and the UMC cent into one number), so it is split evenly; that is
    # an assumption about attribution, not about the total.
    crossing_split = 0.5
    has_crossing = "entry_crossing_cost_twd" in trades.columns
    if not has_crossing and len(trades):
        print(
            "WARNING: trades file predates crossing-cost accounting; "
            "book-crossing cash is not modelled"
        )
    idx_of = {t: i for i, t in enumerate(df["timestamp"])}
    for _, row in trades.iterrows():
        i = idx_of.get(row["entry_time"])
        j = idx_of.get(row["exit_time"])
        entry_cross = float(row["entry_crossing_cost_twd"]) if has_crossing else 0.0
        exit_cross = float(row["exit_crossing_cost_twd"]) if has_crossing else 0.0
        if i is not None:
            df.loc[i, "fee_binance"] += row["entry_tsm_fee_twd"] + entry_cross * crossing_split
            df.loc[i, "fee_fubon"] += (
                row["entry_qff_fee_twd"]
                + row["entry_qff_tax_twd"]
                + entry_cross * (1.0 - crossing_split)
            )
        if j is not None:
            df.loc[j, "fee_binance"] += row["exit_tsm_fee_twd"] + exit_cross * crossing_split
            df.loc[j, "fee_fubon"] += (
                row["exit_qff_fee_twd"]
                + row["exit_qff_tax_twd"]
                + exit_cross * (1.0 - crossing_split)
            )
    # The split is an attribution choice; the total is not. Prove nothing was
    # dropped on the way into the two buckets.
    if len(trades):
        modelled = float(df["fee_binance"].sum() + df["fee_fubon"].sum())
        booked = float(trades["total_fee_twd"].sum()) if has_crossing else None
        if booked is not None and modelled > booked + 1e-6:
            raise RuntimeError(
                f"Broker fee split {modelled:.2f} exceeds booked trade cost "
                f"{booked:.2f}"
            )
    return df


def max_excursion(values: np.ndarray) -> float:
    """Max |move| from the window-start value, both directions, as a fraction."""
    if len(values) < 2:
        return np.nan
    base = values[0]
    return max(values.max() - base, base - values.min()) / base


def window_excursions(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    decision = (ts.dt.hour == DECISION_HOUR) & (ts.dt.minute == 0) & (ts.dt.dayofweek < 5)
    rows = []
    for i in df.index[decision]:
        t0 = ts.iloc[i]
        end7 = t0.replace(hour=ARRIVAL_HOUR, minute=0)
        if t0.dayofweek == 4:  # Friday: forced flat by Saturday 05:00
            end31 = (t0 + pd.Timedelta(days=1)).replace(hour=5, minute=0)
        else:
            end31 = (t0 + pd.Timedelta(days=1)).replace(hour=ARRIVAL_HOUR, minute=0)
        j7 = ts.searchsorted(end7, side="right") - 1
        j31 = ts.searchsorted(end31, side="right") - 1
        rows.append(
            dict(
                friday=t0.dayofweek == 4,
                tsm7=max_excursion(df["tsm_twd_fair"].iloc[i : j7 + 1].to_numpy()),
                tsm31=max_excursion(df["tsm_twd_fair"].iloc[i : j31 + 1].to_numpy()),
                qff7=max_excursion(df["qff_close_filled"].iloc[i : j7 + 1].to_numpy()),
                qff31=max_excursion(df["qff_close_filled"].iloc[i : j31 + 1].to_numpy()),
            )
        )
    return pd.DataFrame(rows)


def quantile_stats(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    return {
        "p50": float(s.quantile(0.5)),
        "p90": float(s.quantile(0.9)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def simulate_policy(
    df: pd.DataFrame,
    *,
    leg_notional: float,
    target: float,
    trigger_binance: float,
    trigger_fubon: float,
    binance_maint: float,
    fubon_maint: float,
) -> dict[str, float]:
    ts = df["timestamp"]
    pnl_binance = (
        df["tsm_units"].shift(1).fillna(0.0) * df["tsm_twd_fair"].diff().fillna(0.0)
    ).to_numpy()
    pnl_fubon = (
        df["qff_units"].shift(1).fillna(0.0) * df["qff_close_filled"].diff().fillna(0.0)
    ).to_numpy()
    fee_binance = df["fee_binance"].to_numpy()
    fee_fubon = df["fee_fubon"].to_numpy()
    in_pos = (df["tsm_units"].abs() > 0).to_numpy()
    hour = ts.dt.hour.to_numpy()
    minute = ts.dt.minute.to_numpy()
    weekday = ts.dt.dayofweek.to_numpy()
    date = ts.dt.date.to_numpy()
    is_decision = (hour == DECISION_HOUR) & (minute == 0) & (weekday < 5)

    maint = {"binance": binance_maint, "fubon": fubon_maint}
    trigger = {"binance": trigger_binance, "fubon": trigger_fubon}
    balance = {"binance": target * leg_notional, "fubon": target * leg_notional}
    pending: tuple | None = None  # (credit_date, amount_twd, to_account)
    transfers = 0
    total_moved = 0.0
    breach_bars = {"binance": 0, "fubon": 0}
    min_ratio = {"binance": np.inf, "fubon": np.inf}

    for i in range(len(df)):
        balance["binance"] += pnl_binance[i] - fee_binance[i]
        balance["fubon"] += pnl_fubon[i] - fee_fubon[i]
        if pending is not None and date[i] == pending[0] and hour[i] >= ARRIVAL_HOUR:
            source = "binance" if pending[2] == "fubon" else "fubon"
            balance[source] -= pending[1]
            balance[pending[2]] += pending[1]
            pending = None
        if in_pos[i]:
            for account in ("binance", "fubon"):
                ratio = balance[account] / leg_notional
                min_ratio[account] = min(min_ratio[account], ratio)
                if ratio < maint[account]:
                    breach_bars[account] += 1
        if is_decision[i] and pending is None:
            for account in ("binance", "fubon"):
                ratio = balance[account] / leg_notional
                if ratio < trigger[account]:
                    amount = (target - ratio) * leg_notional
                    if amount > 0:
                        pending = (date[i], amount, account)
                        transfers += 1
                        total_moved += amount
                    break

    months = (ts.iloc[-1] - ts.iloc[0]).days / 30.4
    return {
        "target": target,
        "trigger_binance": trigger_binance,
        "trigger_fubon": trigger_fubon,
        "transfers": transfers,
        "transfers_per_month": transfers / months if months > 0 else np.nan,
        "breach_bars_binance": breach_bars["binance"],
        "breach_bars_fubon": breach_bars["fubon"],
        "min_ratio_binance": float(min_ratio["binance"]),
        "min_ratio_fubon": float(min_ratio["fubon"]),
        "avg_transfer_twd": total_moved / transfers if transfers else 0.0,
    }


def make_test_frame() -> pd.DataFrame:
    """3 weekdays of 15m bars, one always-open short-TSM position, with a
    controlled TSM rally so the binance account bleeds a known amount."""
    ts = pd.date_range(
        "2026-06-08 09:00", "2026-06-10 18:00", freq="15min", tz=TAIPEI_TZ
    )
    n = len(ts)
    price = np.full(n, 1000.0)
    # rally 12% spread evenly across day 1 10:15 -> day 2 09:45 (after the
    # day-1 decision, before the day-2 one)
    ramp = (ts > pd.Timestamp("2026-06-08 10:00", tz=TAIPEI_TZ)) & (
        ts <= pd.Timestamp("2026-06-09 09:45", tz=TAIPEI_TZ)
    )
    price[np.cumsum(ramp) > 0] = 1000.0 + 120.0 * np.minimum(
        np.cumsum(ramp)[np.cumsum(ramp) > 0] / ramp.sum(), 1.0
    )
    return pd.DataFrame(
        {
            "timestamp": ts,
            "tsm_twd_fair": price,
            # the hedge leg rallies identically: fubon (long) gains what
            # binance (short) loses, as in the real pair
            "qff_close_filled": price.copy(),
            "tsm_units": np.full(n, -1000.0),  # short 1M notional at 1000
            "qff_units": np.full(n, 1000.0),
            "fee_binance": 0.0,
            "fee_fubon": 0.0,
        }
    )


def run_self_test() -> None:
    frame = make_test_frame()
    # trigger_fubon deliberately below the 0.20 target: after the day-2
    # credit the source account sits exactly at target minus the moved
    # amount, and a knife-edge trigger would re-fire on float error
    common = dict(
        leg_notional=1_000_000.0,
        trigger_binance=0.11,
        trigger_fubon=0.18,
        binance_maint=0.025,
        fubon_maint=0.1035,
    )
    # 30% funding loses 12% -> 18% at day-2 10:00, above the 11% trigger
    calm = simulate_policy(frame, target=0.30, **common)
    if calm["transfers"] != 0:
        raise RuntimeError("Self-test failed: 30% funding should need no transfer")
    # 20% funding -> 8% at day-2 10:00, below trigger -> exactly one transfer,
    # and the balance must be restored to target after the 17:00 credit
    tight = simulate_policy(frame, target=0.20, **common)
    if tight["transfers"] != 1:
        raise RuntimeError(
            f"Self-test failed: expected exactly 1 transfer, got {tight['transfers']}"
        )
    if tight["min_ratio_binance"] < 0.025:
        raise RuntimeError("Self-test failed: 20% funding should not breach maintenance")
    # 13% funding -> 1% mid-ramp, below the 2.5% maintenance -> breaches counted
    breach = simulate_policy(frame, target=0.13, **common)
    if breach["breach_bars_binance"] == 0:
        raise RuntimeError("Self-test failed: 13% funding should breach maintenance")
    print("Self-test passed: no-transfer, one-transfer and breach scenarios behave")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_self_test:
        run_self_test()
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    if not targets:
        raise RuntimeError("--targets must list at least one funding target")

    df = load_frame(args.zscore_path, args.equity_path, args.trades_path)
    windows = window_excursions(df)
    if windows.empty:
        # An empty frame has no columns, so every downstream access raises a
        # bare KeyError instead of saying what went wrong. The usual cause is a
        # bar grid whose sessions never cover ARRIVAL_HOUR.
        raise RuntimeError(
            f"No decision points found in {args.zscore_path}: no bar sits at "
            f"{ARRIVAL_HOUR:02d}:00 Taipei on any session. This simulation is "
            "built for a session grid that spans the funding arrival hour."
        )
    print(f"decision points: {len(windows)} ({int(windows['friday'].sum())} Fridays)")
    stats = {}
    for column, label in [
        ("tsm7", "TSM leg 7h"),
        ("tsm31", "TSM leg 31h"),
        ("qff7", "QFF leg 7h"),
        ("qff31", "QFF leg 31h"),
    ]:
        stats[column] = quantile_stats(windows[column])
        s = stats[column]
        print(
            f"{label:12s} p50={s['p50']:.4f} p90={s['p90']:.4f} "
            f"p99={s['p99']:.4f} max={s['max']:.4f}"
        )

    sweep = []
    print(
        f"\n{'target':>7} {'xfers':>6} {'/mo':>5} {'breachB':>8} {'breachF':>8} "
        f"{'minB':>6} {'minF':>6} {'avg moved':>10}"
    )
    for target in targets:
        if target <= max(args.trigger_binance, args.trigger_fubon):
            continue
        row = simulate_policy(
            df,
            leg_notional=args.leg_notional_twd,
            target=target,
            trigger_binance=args.trigger_binance,
            trigger_fubon=args.trigger_fubon,
            binance_maint=args.binance_maint,
            fubon_maint=args.fubon_maint,
        )
        sweep.append(row)
        print(
            f"{row['target']:7.2f} {row['transfers']:6d} "
            f"{row['transfers_per_month']:5.1f} {row['breach_bars_binance']:8d} "
            f"{row['breach_bars_fubon']:8d} {row['min_ratio_binance']:6.3f} "
            f"{row['min_ratio_fubon']:6.3f} {row['avg_transfer_twd']:10,.0f}"
        )

    summary = {
        "parameters": {
            "leg_notional_twd": args.leg_notional_twd,
            "binance_maint_PLACEHOLDER": args.binance_maint,
            "fubon_maint_PLACEHOLDER": args.fubon_maint,
            "trigger_binance": args.trigger_binance,
            "trigger_fubon": args.trigger_fubon,
            "decision_hour_taipei": DECISION_HOUR,
            "arrival_hour_taipei": ARRIVAL_HOUR,
        },
        "sample": {
            "start": str(df["timestamp"].iloc[0]),
            "end": str(df["timestamp"].iloc[-1]),
            "decision_points": len(windows),
        },
        "window_excursions": stats,
        "policy_sweep": sweep,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote summary to {args.summary_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
