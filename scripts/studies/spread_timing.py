"""Timing studies on the CCF/UMC strategy, driven by the spread-drivers result.

The drivers study established that ALL of the spread's mean reversion is
realised in the overnight close-to-open gap (slope -0.35/session, t=-6.9) and
none during US RTH. Three questions follow, in the order the user asked:

  1. EXIT TIMING -- where do the canonical configuration's exits actually
     happen in the session, how much of each trade's capture arrived as
     overnight gaps vs intraday drift, and what would deferring mid-session
     exits to the next session open have done?
  2. SIGN ASYMMETRY -- does the reversion differ between spread-above-mean
     (fair > CCF) and spread-below-mean deviations? Institutional priors
     (one-way ADR convertibility, short-sale constraints) say it could.
  3. ENTRY HOUR -- bar-level forward capture of entry-threshold crossings by
     time of session, cross-checked against the canonical trades. Bar-level
     stats have flipped sign at trade level before (QFF/TSM volume probe), so
     the trade cross-check is part of the design, not decoration.

The canonical configuration is reproduced through report.pair's own code path
(w3900 / entry 2.0 / exit -0.25, displacement 0.2317 ref 121.50), so trades
here are the report's trades, not a reimplementation.

Usage:
    python scripts/studies/spread_timing.py

Outputs land in data/runs/spread_timing/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import engine as backtest  # noqa: E402
from features import zscore as zscore_calc  # noqa: E402
from lib import paths  # noqa: E402
from lib.sessions import us_session_day  # noqa: E402
from report.pair import (  # noqa: E402
    PAIRS,
    load_frames,
    slice_segment,
    trade_detail,
    with_displacement_column,
)
from studies.spread_drivers import (  # noqa: E402
    SESSION_ROLLING_MEAN,
    SPREAD_1M,
    add_minute_components,
    load_spread,
    nw_ols,
    session_frame,
)

CELL = SimpleNamespace(window=3900, entry_z=2.0, exit_z=-0.25)
OUT_DIR = paths.DATA / "runs" / "spread_timing"
OPEN_WINDOW_MIN = 30  # "at the open" = first this many minutes of a session
NW_LAGS_MIN = 30


def trade_sign(direction: str) -> float:
    """+1 if the trade profits when the spread RISES."""
    return -1.0 if direction == backtest.SHORT_TSM_LONG_QFF else 1.0


# --------------------------------------------------------------------------
# canonical run + session scaffolding
# --------------------------------------------------------------------------


def canonical() -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = PAIRS["ccf_umc"]
    spread, seed = load_frames(spec)
    seg = spec.segments[0]
    _, trades = trade_detail(spec, spread, seed, seg, CELL)

    zframe = zscore_calc.calculate_zscore(spread, CELL.window, seed_frame=seed)
    zframe = slice_segment(zframe, seg)
    zframe = with_displacement_column(spec, seg, zframe)
    zframe = zframe.reset_index(drop=True)
    ts = pd.DatetimeIndex(zframe["timestamp"])
    zframe["session"] = us_session_day(ts)
    zframe["min_in_session"] = zframe.groupby("session").cumcount()
    return zframe, trades


# --------------------------------------------------------------------------
# 1. exit timing
# --------------------------------------------------------------------------


def exit_timing(zframe: pd.DataFrame, trades: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    spread = pd.Series(
        zframe["spread"].to_numpy(), index=pd.DatetimeIndex(zframe["timestamp"])
    )
    minute_of = pd.Series(
        zframe["min_in_session"].to_numpy(), index=spread.index
    )
    session_of = pd.Series(zframe["session"].to_numpy(), index=spread.index)
    session_first_bar = zframe.groupby("session")["timestamp"].first()

    rows = []
    for t in trades.itertuples():
        sign = trade_sign(t.direction)
        window = spread.loc[t.entry_time:t.exit_time]
        d = window.diff().iloc[1:]
        boundary = session_of.loc[window.index].to_numpy()
        overnight = boundary[1:] != boundary[:-1]
        row = {
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time),
            "direction": t.direction,
            "net_pnl_twd": float(t.net_pnl_twd),
            "entry_min": int(minute_of.loc[t.entry_time]),
            "exit_min": int(minute_of.loc[t.exit_time]),
            "sessions_held": int(overnight.sum()) + 1,
            "capture_pts": float(sign * (window.iloc[-1] - window.iloc[0])),
            "overnight_pts": float(sign * d[overnight].sum()),
            "intraday_pts": float(sign * d[~overnight].sum()),
        }

        # counterfactual: a mid-session exit deferred to the next session open
        row["exit_at_open"] = row["exit_min"] < OPEN_WINDOW_MIN
        row["defer_delta_pts"] = np.nan
        row["defer_crosses_weekend"] = False
        if not row["exit_at_open"]:
            here = session_of.loc[t.exit_time]
            later = session_first_bar[session_first_bar.index > here]
            if not later.empty:
                alt_time = later.iloc[0]
                row["defer_delta_pts"] = float(
                    sign * (spread.loc[alt_time] - spread.loc[t.exit_time])
                )
                row["defer_crosses_weekend"] = bool(
                    (later.index[0] - here) > pd.Timedelta(days=2)
                )
        rows.append(row)

    per_trade = pd.DataFrame(rows)
    deferred = per_trade[per_trade["defer_delta_pts"].notna()]
    summary = {
        "trades": int(len(per_trade)),
        "open_window_minutes": OPEN_WINDOW_MIN,
        "entry_min_median": float(per_trade["entry_min"].median()),
        "exit_min_median": float(per_trade["exit_min"].median()),
        "exits_at_open": int(per_trade["exit_at_open"].sum()),
        "sessions_held_median": float(per_trade["sessions_held"].median()),
        "capture_pts_total": float(per_trade["capture_pts"].sum()),
        "overnight_pts_total": float(per_trade["overnight_pts"].sum()),
        "intraday_pts_total": float(per_trade["intraday_pts"].sum()),
        "overnight_share_of_capture": float(
            per_trade["overnight_pts"].sum() / per_trade["capture_pts"].sum()
        ),
        "trades_overnight_dominant": int(
            (per_trade["overnight_pts"] > per_trade["intraday_pts"]).sum()
        ),
        "defer_counterfactual": {
            "n": int(len(deferred)),
            "mean_delta_pts": float(deferred["defer_delta_pts"].mean()),
            "median_delta_pts": float(deferred["defer_delta_pts"].median()),
            "sum_delta_pts": float(deferred["defer_delta_pts"].sum()),
            "improved": int((deferred["defer_delta_pts"] > 0).sum()),
            "crossing_weekend": int(deferred["defer_crosses_weekend"].sum()),
        },
    }
    return summary, per_trade


# --------------------------------------------------------------------------
# 2. sign asymmetry
# --------------------------------------------------------------------------


def sign_asymmetry(zframe: pd.DataFrame) -> dict:
    out = {}

    # Session level, on the same frame the drivers study used.
    minutes = add_minute_components(load_spread(SPREAD_1M))
    sess = session_frame(minutes)
    dev = sess["s_close"] - sess["s_close"].rolling(
        SESSION_ROLLING_MEAN, min_periods=SESSION_ROLLING_MEAN
    ).mean()
    ds_on_next = sess["ds_overnight"].shift(-1)
    base = dev.notna() & ds_on_next.notna()
    for label, mask in (
        ("dev_positive", base & (dev > 0)),
        ("dev_negative", base & (dev < 0)),
    ):
        fit = nw_ols(ds_on_next[mask].to_numpy(), dev[mask].to_numpy(), lags=5)
        fit["mean_abs_dev"] = float(dev[mask].abs().mean())
        out[f"session_{label}"] = fit
    out["session_share_dev_positive"] = float((dev[base] > 0).mean())

    # Minute level, on the strategy's own gap-aware seeded w3900 mean.
    mean_col = f"spread_mean_{CELL.window}"
    valid = zframe["zscore_valid"].astype(bool)
    mdev = (zframe["spread"] - zframe[mean_col]).where(valid)
    ds = zframe.groupby("session")["spread"].diff()
    mdev_lag = mdev.shift(1)
    same = zframe["session"] == zframe["session"].shift(1)
    mbase = mdev_lag.notna() & ds.notna() & same
    for label, mask in (
        ("dev_positive", mbase & (mdev_lag > 0)),
        ("dev_negative", mbase & (mdev_lag < 0)),
    ):
        fit = nw_ols(ds[mask].to_numpy(), mdev_lag[mask].to_numpy(), lags=NW_LAGS_MIN)
        kappa = fit["slope"]
        fit["half_life_minutes"] = (
            float(np.log(2) / -kappa) if kappa < 0 else float("inf")
        )
        out[f"minute_{label}"] = fit
    out["minute_share_dev_positive"] = float((mdev_lag[mbase] > 0).mean())
    return out


# --------------------------------------------------------------------------
# 3. entry hour
# --------------------------------------------------------------------------


def entry_hour(zframe: pd.DataFrame, trades: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    z = zframe["spread_zscore"]
    valid = zframe["zscore_valid"].astype(bool) & zframe["entry_allowed"].astype(bool)
    prev = z.shift(1)
    same = zframe["session"] == zframe["session"].shift(1)
    crossing = valid & same & (
        ((z >= CELL.entry_z) & (prev < CELL.entry_z))
        | ((z <= -CELL.entry_z) & (prev > -CELL.entry_z))
    )

    spread = zframe["spread"]
    session = zframe["session"]
    first_bar_idx = zframe.groupby("session").head(1).index
    open_rows = zframe.loc[first_bar_idx, ["session", "spread"]]
    next_open_spread = dict(zip(open_rows["session"], open_rows["spread"]))
    close_rows = zframe.groupby("session").tail(1)[["session", "spread"]]
    next_close_spread = dict(zip(close_rows["session"], close_rows["spread"]))
    sessions_sorted = sorted(next_open_spread)
    next_session = {
        s: sessions_sorted[i + 1]
        for i, s in enumerate(sessions_sorted[:-1])
    }

    rows = []
    for i in np.flatnonzero(crossing.to_numpy()):
        sign = -1.0 if z.iloc[i] > 0 else 1.0  # high z -> profits when spread falls
        here = session.iloc[i]
        nxt = next_session.get(here)
        if nxt is None:
            continue
        rows.append(
            {
                "min_in_session": int(zframe["min_in_session"].iloc[i]),
                "abs_z": float(abs(z.iloc[i])),
                "to_next_open_pts": sign * (next_open_spread[nxt] - spread.iloc[i]),
                "to_next_close_pts": sign * (next_close_spread[nxt] - spread.iloc[i]),
            }
        )
    ev = pd.DataFrame(rows)
    ev["bucket_min"] = (ev["min_in_session"] // OPEN_WINDOW_MIN) * OPEN_WINDOW_MIN
    table = (
        ev.groupby("bucket_min")
        .agg(
            n=("abs_z", "size"),
            mean_abs_z=("abs_z", "mean"),
            mean_to_open=("to_next_open_pts", "mean"),
            median_to_open=("to_next_open_pts", "median"),
            mean_to_close=("to_next_close_pts", "mean"),
        )
        .reset_index()
    )

    # Trade-level cross-check (small n -- listed, not concluded from).
    tmin = []
    minute_of = pd.Series(
        zframe["min_in_session"].to_numpy(),
        index=pd.DatetimeIndex(zframe["timestamp"]),
    )
    for t in trades.itertuples():
        tmin.append(
            {
                "entry_min": int(minute_of.loc[t.entry_time]),
                "net_pnl_twd": float(t.net_pnl_twd),
            }
        )
    tframe = pd.DataFrame(tmin)
    tframe["bucket_min"] = (tframe["entry_min"] // 60) * 60
    trade_check = (
        tframe.groupby("bucket_min")
        .agg(n=("net_pnl_twd", "size"), mean_pnl=("net_pnl_twd", "mean"),
             total_pnl=("net_pnl_twd", "sum"))
        .reset_index()
    )

    summary = {
        "signal_bars": int(len(ev)),
        "note": (
            "bar-level forward stats; QFF/TSM history shows these can flip "
            "sign at trade level, see trade_check"
        ),
        "trade_check_by_entry_hour": trade_check.to_dict(orient="records"),
    }
    return summary, table


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zframe, trades = canonical()
    print(
        f"canonical run reproduced: {len(trades)} trades, "
        f"net {trades['net_pnl_twd'].sum():,.0f} TWD"
    )

    exit_summary, per_trade = exit_timing(zframe, trades)
    asym = sign_asymmetry(zframe)
    entry_summary, entry_table = entry_hour(zframe, trades)

    per_trade.to_csv(OUT_DIR / "trades_timing.csv", index=False)
    entry_table.to_csv(OUT_DIR / "entry_hour.csv", index=False)
    result = {
        "parameters": {
            "window": CELL.window,
            "entry_z": CELL.entry_z,
            "exit_z": CELL.exit_z,
            "open_window_minutes": OPEN_WINDOW_MIN,
            "trades": int(len(trades)),
        },
        "exit_timing": exit_summary,
        "sign_asymmetry": asym,
        "entry_hour": entry_summary,
    }
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(result, fh, indent=2)

    # ---- digest ----------------------------------------------------------
    et = exit_summary
    print(f"\n=== 1. exit timing ({et['trades']} trades) ===")
    print(f"entry minute median {et['entry_min_median']:.0f}, "
          f"exit minute median {et['exit_min_median']:.0f}; "
          f"{et['exits_at_open']}/{et['trades']} exits inside the first "
          f"{OPEN_WINDOW_MIN}min; median hold {et['sessions_held_median']:.0f} sessions")
    print(f"capture {et['capture_pts_total']:+.2f} pts = overnight "
          f"{et['overnight_pts_total']:+.2f} + intraday {et['intraday_pts_total']:+.2f} "
          f"(overnight share {100 * et['overnight_share_of_capture']:.0f}%; "
          f"overnight-dominant trades {et['trades_overnight_dominant']}/{et['trades']})")
    dc = et["defer_counterfactual"]
    print(f"defer mid-session exits to next open: n={dc['n']}, mean "
          f"{dc['mean_delta_pts']:+.3f} pts, median {dc['median_delta_pts']:+.3f}, "
          f"sum {dc['sum_delta_pts']:+.2f}, improved {dc['improved']}/{dc['n']} "
          f"({dc['crossing_weekend']} would cross a weekend)")

    print("\n=== 2. sign asymmetry ===")
    for lvl in ("session", "minute"):
        for side in ("dev_positive", "dev_negative"):
            f = asym[f"{lvl}_{side}"]
            extra = (f", HL {f['half_life_minutes']:.0f}m"
                     if "half_life_minutes" in f else "")
            print(f"{lvl:8s} {side:12s} slope {f['slope']:+.4f} "
                  f"(t={f['slope_t']:+.1f}, n={f['n']}{extra})")
        share = asym[f"{lvl}_share_dev_positive"]
        print(f"{lvl:8s} share of time dev>0: {100 * share:.0f}%")

    print(f"\n=== 3. entry hour ({entry_summary['signal_bars']} signal bars, "
          f"bar-level) ===")
    print(f"{'bucket':>7s}{'n':>6s}{'|z|':>6s}{'->open':>9s}{'med':>8s}{'->close':>9s}")
    for r in entry_table.itertuples():
        print(f"{r.bucket_min:>7.0f}{r.n:>6.0f}{r.mean_abs_z:>6.2f}"
              f"{r.mean_to_open:>9.3f}{r.median_to_open:>8.3f}{r.mean_to_close:>9.3f}")
    print("trade-level cross-check (entry hour, n tiny):")
    for r in entry_summary["trade_check_by_entry_hour"]:
        print(f"  min {r['bucket_min']:>4.0f}: n={r['n']}, "
              f"mean {r['mean_pnl']:>10,.0f}, total {r['total_pnl']:>11,.0f} TWD")

    print(f"\nWrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
