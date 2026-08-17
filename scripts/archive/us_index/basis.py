from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TAIPEI_TZ = "Asia/Taipei"

# Column names reuse the legacy qff_*/tsm_* schema so calculate_spread_zscore_1m.py,
# backtest_pair_strategy_1m.py and grid_search_pair_strategy_1m.py run unchanged:
#   qff_*     -> the TAIFEX US index future (UDF/SPF/UNF/SXF), quoted in index points
#   tsm_*     -> the US leg (CME future or cash index), also in index points
#   usdttwd_* -> held at the constant 5.0 that cancels the engine's hard-coded
#                ADR share ratio, so tsm_twd_fair == the raw US index level.
#
# There is genuinely no FX term in this pair.  TAIFEX US index futures are quanto
# contracts: they pay a fixed NT$ per index point regardless of USD/TWD, so the
# two legs are directly comparable in index points.  The FX exposure that remains
# is second order (it scales the US leg's PnL, not its notional) and is not
# modelled here.
FX_CANCELLING_CONSTANT = 5.0

# NT$ per index point, and the minimum tick, from the TAIFEX contract specs.
TAIFEX_SPECS = {
    "UDF": {"multiplier": 20.0, "tick": 1.0, "underlying": "Dow Jones Industrial"},
    "SPF": {"multiplier": 200.0, "tick": 0.25, "underlying": "S&P 500"},
    "UNF": {"multiplier": 50.0, "tick": 1.0, "underlying": "Nasdaq-100"},
    "SXF": {"multiplier": 80.0, "tick": 0.5, "underlying": "PHLX Semiconductor"},
}

DAY_SESSION = (8 * 60 + 45, 13 * 60 + 45)
NIGHT_SESSION = (15 * 60, 5 * 60)  # 15:00 -> 05:00 next day

MASK_COLUMNS = [
    "close_allowed",
    "entry_allowed",
    "friday_night_close_only",
    "weekend_session_close_only",
    "friday_session_end_force_close",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the TAIFEX-US-index-future vs US-leg basis dataset. Both legs "
            "are quoted in index points on the same index, so with matched "
            "contract months the cost of carry cancels and the spread is the "
            "pure venue basis. Writes the spread file plus the synthetic FX file "
            "the backtest engine expects, and precomputed session masks."
        )
    )
    parser.add_argument("--product", required=True, choices=sorted(TAIFEX_SPECS))
    parser.add_argument("--taifex-path", type=Path, required=True,
                        help="1m bars from build_qff1_1m.py --product <X>")
    parser.add_argument("--us-path", type=Path, required=True,
                        help="1m bars from download_ib_us_index_legs.py")
    parser.add_argument("--session", choices=["day", "night"], default="day")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tag", default=None,
                        help="Filename suffix. Defaults to <product>_<session>.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--max-us-staleness-minutes",
        type=float,
        default=5.0,
        help=(
            "Drop bars whose US price is older than this. The US leg trades "
            "nearly around the clock but goes quiet during Taiwan hours, so a "
            "small amount of forward-fill is needed and a lot is a red flag."
        ),
    )
    parser.add_argument(
        "--allow-stale-taifex-fills",
        action="store_true",
        help=(
            "Keep the full 1-minute clock grid and let forward-filled TAIFEX "
            "bars be tradeable. Off by default: these books print only every "
            "few minutes, so ~85%% of clock bars carry a stale TAIFEX price. "
            "Those bars both invent executions that could not have happened "
            "and inflate the rolling sigma the z-score is measured against, "
            "because a stale leg against a live one drifts apart mechanically. "
            "The default keeps one observation per TAIFEX print instead."
        ),
    )
    parser.add_argument(
        "--weekend-policy",
        choices=["flat", "noentry", "none"],
        default="flat",
        help=(
            "flat: last session of the ISO week is close-only and any position "
            "is force-closed on its final bar. noentry: close-only, no forced "
            "exit. none: no weekend restriction at all."
        ),
    )
    return parser.parse_args(argv)


def read_bars(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    frame = pd.read_csv(path)
    stamp = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={stamp: "timestamp"})
    missing = {"timestamp", "open", "close"}.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["close"]).drop_duplicates("timestamp")
    return frame.sort_values("timestamp").reset_index(drop=True)


def in_session(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    minute = index.hour * 60 + index.minute
    if session == "day":
        start, end = DAY_SESSION
        return (minute >= start) & (minute <= end)
    start, end = NIGHT_SESSION
    return (minute >= start) | (minute <= end)


def session_start_day(index: pd.DatetimeIndex, session: str) -> pd.Series:
    """The calendar day a session belongs to; a night session that runs past
    midnight belongs to the day it opened."""
    day = pd.Series(index.normalize(), index=range(len(index)))
    if session == "night":
        minute = index.hour * 60 + index.minute
        rolled = minute <= NIGHT_SESSION[1]
        day.loc[rolled] = day.loc[rolled] - pd.Timedelta(days=1)
    return day


def build_masks(
    frame: pd.DataFrame, session: str, policy: str, tradeable: np.ndarray
) -> pd.DataFrame:
    output = frame.copy()
    index = pd.DatetimeIndex(output["timestamp"])
    start_day = session_start_day(index, session)

    close_allowed = tradeable.copy()
    output["close_allowed"] = close_allowed
    output["friday_night_close_only"] = False

    if policy == "none":
        output["weekend_session_close_only"] = False
        output["friday_session_end_force_close"] = False
        output["entry_allowed"] = close_allowed
        return output

    # "last session of the ISO week" rather than "Friday", so a Friday holiday
    # moves the restriction onto Thursday instead of silently disabling it.
    iso = start_day.dt.isocalendar()
    week_key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    tradeable_rows = output.index[close_allowed]
    last_session_by_week = (
        start_day.loc[tradeable_rows].groupby(week_key.loc[tradeable_rows]).max()
    )
    is_last_session = (
        close_allowed
        & start_day.eq(week_key.map(last_session_by_week)).to_numpy()
    )
    output["weekend_session_close_only"] = is_last_session

    force_close = np.zeros(len(output), dtype=bool)
    if policy == "flat":
        last_rows = (
            output.index[is_last_session]
            .to_series()
            .groupby(start_day.loc[is_last_session].to_numpy())
            .max()
        )
        force_close[last_rows.to_numpy()] = True
    output["friday_session_end_force_close"] = force_close

    output["entry_allowed"] = close_allowed & ~is_last_session
    return output


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    spec = TAIFEX_SPECS[args.product]
    tag = args.tag or f"{args.product.lower()}_{args.session}"

    taifex = read_bars(args.taifex_path, "TAIFEX")
    us = read_bars(args.us_path, "US leg")
    print(
        f"{args.product}: {len(taifex):,} native 1m bars "
        f"{taifex['timestamp'].min()} -> {taifex['timestamp'].max()}"
    )
    print(
        f"US leg  : {len(us):,} native 1m bars "
        f"{us['timestamp'].min()} -> {us['timestamp'].max()}"
    )

    lo = max(taifex["timestamp"].min(), us["timestamp"].min())
    hi = min(taifex["timestamp"].max(), us["timestamp"].max())
    if args.start:
        lo = max(lo, pd.Timestamp(args.start, tz=TAIPEI_TZ))
    if args.end:
        hi = min(hi, pd.Timestamp(args.end, tz=TAIPEI_TZ))
    if lo >= hi:
        raise RuntimeError(f"No overlapping range: {lo} .. {hi}")

    grid = pd.date_range(lo.floor("min"), hi.ceil("min"), freq="1min", tz=TAIPEI_TZ)
    grid = grid[in_session(grid, args.session)]

    tw = taifex.set_index("timestamp").reindex(grid)
    us_re = us.set_index("timestamp").reindex(grid)

    # Only keep sessions where the TAIFEX leg traded at all: a session with no
    # prints is a Taiwan holiday, and carrying it forward would invent bars.
    start_day = session_start_day(grid, args.session)
    traded_days = set(start_day[tw["close"].notna().to_numpy()])
    # pandas hands back read-only views, and the mask is built up in place
    keep = np.array(start_day.isin(traded_days).to_numpy(), dtype=bool)

    us_close = us_re["close"].ffill()
    us_open = us_re["open"].fillna(us_close)
    # Age in grid steps since the last native US bar; the grid is 1-minute and
    # already restricted to the session, so a step is a minute of session time.
    last_native = pd.Series(
        np.where(us_re["close"].notna().to_numpy(), np.arange(len(grid)), np.nan)
    ).ffill()
    us_age_minutes = np.arange(len(grid)) - last_native.to_numpy()

    keep &= us_close.notna().to_numpy()
    keep &= us_age_minutes <= args.max_us_staleness_minutes

    grid = grid[keep]
    tw, us_re = tw[keep], us_re[keep]
    us_close, us_open = us_close[keep], us_open[keep]
    us_age_minutes = us_age_minutes[keep]

    qff_close = tw["close"]
    if args.allow_stale_taifex_fills:
        qff_close_filled = qff_close.groupby(
            session_start_day(grid, args.session).to_numpy()
        ).ffill()
        # A session cannot start with a filled value; drop the leading gap.
        valid = qff_close_filled.notna().to_numpy()
    else:
        # Event time: one observation per TAIFEX print. Every row is then a real
        # trade on both legs, so the rolling mean/sigma the z-score uses are
        # measured on prices that could actually have been dealt on.
        qff_close_filled = qff_close
        valid = qff_close.notna().to_numpy()
    grid = grid[valid]
    tw, qff_close = tw[valid], qff_close[valid]
    qff_close_filled = qff_close_filled[valid]
    us_close, us_open = us_close[valid], us_open[valid]
    us_age_minutes = us_age_minutes[valid]

    fair = us_close.to_numpy(dtype=float)
    twf = qff_close_filled.to_numpy(dtype=float)
    spread = (fair - twf) / (fair + twf) * 200.0

    frame = pd.DataFrame(
        {
            "timestamp": grid,
            "qff_close": qff_close.to_numpy(),
            "qff_close_filled": twf,
            "qff_volume": tw["volume"].to_numpy() if "volume" in tw else np.nan,
            "tsm_close": fair,
            "usdttwd_close": FX_CANCELLING_CONSTANT,
            "tsm_twd_fair": fair,
            "us_staleness_minutes": us_age_minutes,
            "spread": spread,
        }
    )

    tradeable = in_session(pd.DatetimeIndex(grid), args.session)
    if not args.allow_stale_taifex_fills:
        tradeable = tradeable & qff_close.notna().to_numpy()
    frame = build_masks(frame, args.session, args.weekend_policy, tradeable)

    # self-check: the engine's validator rejects these, so fail loudly here
    if (frame["entry_allowed"] & ~frame["close_allowed"]).any():
        raise RuntimeError("entry_allowed outside close_allowed")
    if (frame["friday_session_end_force_close"] & ~frame["close_allowed"]).any():
        raise RuntimeError("force-close bar outside close_allowed")
    recomputed = (
        (frame["tsm_twd_fair"] - frame["qff_close_filled"])
        / (frame["tsm_twd_fair"] + frame["qff_close_filled"])
        * 200.0
    )
    if (recomputed - frame["spread"]).abs().max() > 1e-9:
        raise RuntimeError("spread column does not match its own definition")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spread_path = args.out_dir / f"us_index_basis_{tag}.csv"
    fx_path = args.out_dir / f"us_index_basis_{tag}_fx.csv"
    us_path = args.out_dir / f"us_index_basis_{tag}_us_1m.csv"
    tw_path = args.out_dir / f"us_index_basis_{tag}_taifex_1m.csv"

    def stamp(values: pd.Series) -> pd.Series:
        text = pd.DatetimeIndex(values).strftime("%Y-%m-%d %H:%M:%S%z")
        return pd.Series(text).str.replace(
            r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True
        )

    out = frame.copy()
    out["timestamp"] = stamp(out["timestamp"]).to_numpy()
    out.to_csv(spread_path, index=False)

    # The engine reads entry OPEN prices from separate OHLCV files and computes
    # tsm_twd_fair_open = tsm_open * usdttwd_open / 5, so a constant 5.0 FX file
    # makes that identity collapse to the raw US open.
    pd.DataFrame(
        {"timestamp": out["timestamp"], "open": FX_CANCELLING_CONSTANT}
    ).to_csv(fx_path, index=False)
    pd.DataFrame(
        {"timestamp": out["timestamp"], "open": us_open.to_numpy()}
    ).to_csv(us_path, index=False)
    pd.DataFrame(
        {"timestamp": out["timestamp"], "open": tw["open"].to_numpy()}
    ).to_csv(tw_path, index=False)

    sessions = frame["timestamp"].dt.normalize().nunique()
    native = int(frame["qff_close"].notna().sum())
    print()
    print(f"session          : TAIFEX {args.session} session, "
          f"weekend policy {args.weekend_policy}")
    print(f"contract         : NT${spec['multiplier']:.0f}/point, tick "
          f"{spec['tick']} pt = NT${spec['multiplier'] * spec['tick']:.0f}")
    print(f"rows             : {len(frame):,} over {sessions} sessions")
    print(f"TAIFEX prints    : {native:,} ({native / max(len(frame), 1):.1%} of bars)")
    print(f"tradeable bars   : {int(frame['close_allowed'].sum()):,} close-allowed, "
          f"{int(frame['entry_allowed'].sum()):,} entry-allowed")
    print(f"US staleness     : median {np.median(us_age_minutes):.0f} min, "
          f"p95 {np.quantile(us_age_minutes, 0.95):.0f} min")
    print(f"spread (percent) : mean {frame['spread'].mean():.4f}, "
          f"std {frame['spread'].std():.4f}, "
          f"min {frame['spread'].min():.4f}, max {frame['spread'].max():.4f}")
    print(f"  in bps         : std {frame['spread'].std() * 100:.1f} bps")
    print()
    print(f"Wrote {spread_path}")
    print(f"Wrote {fx_path}  (constant {FX_CANCELLING_CONSTANT} FX, see module docstring)")
    print(f"Wrote {us_path}")
    print(f"Wrote {tw_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
