from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
ADR_SHARE_RATIO = 5.0

# Column names intentionally reuse the legacy qff_*/tsm_* schema so the
# existing z-score / backtest / grid-search scripts run unchanged:
#   qff_*     -> TAIFEX CCF1! (UMC stock futures, TWD)
#   tsm_*     -> NYSE:UMC ADR (USD)
#   usdttwd_* -> FX_IDC:USDTWD spliced series

INTERVAL_MINUTES = {"5m": 5, "15m": 15}

DEFAULT_CCF_PATHS = {
    "5m": Path("data/processed/ccf1_5m_taipei_tv.csv"),
    "15m": Path("data/processed/ccf1_15m_taipei_tv.csv"),
}
DEFAULT_UMC_PATHS = {
    "5m": Path("data/processed/nyse_umc_5m_taipei_tv.csv"),
    "15m": Path("data/processed/nyse_umc_15m_taipei_tv.csv"),
}
# priority order: finest first
DEFAULT_FX_PATHS = [
    (5, Path("data/processed/fxidc_usdtwd_5m_taipei_tv.csv")),
    (15, Path("data/processed/fxidc_usdtwd_15m_taipei_tv.csv")),
    (60, Path("data/processed/fxidc_usdtwd_1h_taipei_tv.csv")),
]

# The FX_IDC feed has occasional multi-hour outages across ALL intervals
# (verified 2026-06-30 16:00 -> 07-01 01:00 in the 5m/15m/1h files alike);
# USDTWD moves so little intraday that forward-filling through them is safer
# than fragmenting sessions, so the hard cap is generous and stale rows are
# reported instead.
MAX_FX_STALENESS_MINUTES = 720.0
WARN_FX_STALENESS_MINUTES = 60.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the UMC-session CCF/UMC spread dataset. The session index "
            "is NYSE:UMC's own RTH bars; CCF1! is aligned onto it (5m: exact "
            "grid with forward-fill; 15m: as-of alignment because TAIFEX "
            "night bars sit on a +10min offset grid); USDTWD is spliced from "
            "the finest available FX_IDC interval. Also writes the aligned FX "
            "open series and (15m only) a delayed-open CCF fill file, plus "
            "precomputed trading-mask columns consumed by the backtest."
        )
    )
    parser.add_argument("--interval", choices=["5m", "15m"], required=True)
    parser.add_argument("--ccf-path", type=Path, default=None)
    parser.add_argument("--umc-path", type=Path, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--min-ccf-bars-per-session",
        type=int,
        default=1,
        help="Sessions with fewer native CCF bars are excluded entirely.",
    )
    return parser.parse_args(argv)


def parse_taipei_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TAIPEI_TZ)
    return timestamp.tz_convert(TAIPEI_TZ)


def format_taipei_timestamps(timestamps: pd.Series) -> pd.Series:
    formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def read_ohlcv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["open", "close"]].isna().any().any():
        raise RuntimeError(f"{label} has invalid open/close values")
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if timestamps.has_duplicates:
        raise RuntimeError(f"{label} has duplicate timestamps")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["timestamp", "open", "high", "low", "close", "volume"]]


def us_session_day(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Group a 21:30->04:00 (or 22:30->05:00) Taipei session onto one key."""
    return (timestamps - pd.Timedelta(hours=12)).normalize()


def build_fx_series(interval_minutes: int) -> pd.DataFrame:
    """Splice FX_IDC intervals, finest first, into known-time close and
    as-of open series."""
    pieces = []
    for fx_minutes, path in DEFAULT_FX_PATHS:
        if not path.exists():
            continue
        frame = read_ohlcv(path, f"FX {fx_minutes}m")
        frame = frame[["timestamp", "open", "close"]].copy()
        frame["fx_interval_minutes"] = fx_minutes
        frame["known_time"] = frame["timestamp"] + pd.Timedelta(minutes=fx_minutes)
        pieces.append(frame)
    if not pieces:
        raise RuntimeError("No FX_IDC input files found")
    fx = pd.concat(pieces, ignore_index=True)
    # finest interval wins on identical bar-start timestamps
    fx = fx.sort_values(["timestamp", "fx_interval_minutes"])
    fx = fx.drop_duplicates(subset=["timestamp"], keep="first")
    return fx.sort_values("timestamp").reset_index(drop=True)


def asof_fx(
    fx: pd.DataFrame, session_index: pd.DatetimeIndex, interval_minutes: int
) -> pd.DataFrame:
    """FX close known by each session bar's close; FX open as of each session
    bar's start."""
    session_close_time = session_index + pd.Timedelta(minutes=interval_minutes)

    by_known = fx.sort_values("known_time")
    close_match = pd.merge_asof(
        pd.DataFrame({"session_close_time": session_close_time}),
        by_known[["known_time", "close", "timestamp"]],
        left_on="session_close_time",
        right_on="known_time",
        direction="backward",
    )
    by_start = fx.sort_values("timestamp")
    open_match = pd.merge_asof(
        pd.DataFrame({"session_start": session_index}),
        by_start[["timestamp", "open"]],
        left_on="session_start",
        right_on="timestamp",
        direction="backward",
    )
    result = pd.DataFrame(
        {
            "timestamp": session_index,
            "open": open_match["open"].to_numpy(),
            "close": close_match["close"].to_numpy(),
            "fx_close_staleness_minutes": (
                (session_close_time - close_match["known_time"]).dt.total_seconds()
                / 60.0
            ),
        }
    )
    if result[["open", "close"]].isna().any().any():
        raise RuntimeError("FX series does not cover the session index")
    max_staleness = float(result["fx_close_staleness_minutes"].max())
    stale_rows = int(
        (result["fx_close_staleness_minutes"] > WARN_FX_STALENESS_MINUTES).sum()
    )
    if stale_rows:
        print(
            f"WARNING: {stale_rows} session bars ({stale_rows / len(result):.1%}) "
            f"use FX closes more than {WARN_FX_STALENESS_MINUTES:.0f}min old "
            f"(max {max_staleness:.0f}min) - known FX_IDC feed outages are "
            "forward-filled"
        )
    return result


def align_ccf_close(
    ccf: pd.DataFrame, session_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """CCF close aligned to the session index.

    Both intervals reduce to the same rule: the latest CCF bar whose start is
    <= the session bar's start. On the 5m grid that is an exact match (with
    forward-fill over no-trade gaps); on the 15m grid the TAIFEX night bars
    sit 5 minutes earlier than the UMC grid point, so this is an as-of match
    whose close is known 5 minutes before the UMC bar closes (no lookahead).
    """
    closes = pd.Series(ccf["close"].to_numpy(), index=pd.DatetimeIndex(ccf["timestamp"]))
    exact = closes.reindex(session_index)
    union_index = closes.index.union(session_index)
    filled = closes.reindex(union_index).sort_index().ffill().reindex(session_index)

    match_time = pd.merge_asof(
        pd.DataFrame({"session_start": session_index}),
        ccf[["timestamp"]].assign(ccf_start=lambda d: d["timestamp"]),
        left_on="session_start",
        right_on="timestamp",
        direction="backward",
    )["ccf_start"]
    staleness = (
        (session_index - pd.DatetimeIndex(match_time)).total_seconds() / 60.0
    )
    return pd.DataFrame(
        {
            "timestamp": session_index,
            "ccf_close_exact": exact.to_numpy(),
            "ccf_close_aligned": filled.to_numpy(),
            "ccf_staleness_minutes": np.asarray(staleness),
        }
    )


def build_ccf_delayed_open(
    ccf: pd.DataFrame, session_index: pd.DatetimeIndex, interval_minutes: int
) -> pd.DataFrame:
    """15m variant: honest fill prices. For each session bar, the open of the
    first CCF bar starting at or after the session bar's start (night grid:
    ~10 minutes later). Session bars with no CCF bar inside them are omitted;
    the backtest then falls back to the as-of close."""
    match = pd.merge_asof(
        pd.DataFrame({"session_start": session_index}),
        ccf[["timestamp", "open"]].rename(columns={"timestamp": "ccf_start"}),
        left_on="session_start",
        right_on="ccf_start",
        direction="forward",
        tolerance=pd.Timedelta(minutes=interval_minutes - 1),
    )
    match = match.dropna(subset=["open"])
    return pd.DataFrame(
        {
            "timestamp": match["session_start"].to_numpy(),
            "open": match["open"].to_numpy(),
            "fill_delay_minutes": (
                (match["ccf_start"] - match["session_start"]).dt.total_seconds()
                / 60.0
            ).to_numpy(),
        }
    )


def add_trading_mask_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Precompute the mask columns the backtest engine otherwise derives from
    the QFF session clock. Every row is a tradable UMC-session bar; the last
    session of each ISO week is close-only and force-closed on its final bar
    (no weekend positions), mirroring the QFF baseline rules."""
    output = frame.copy()
    timestamps = pd.DatetimeIndex(output["timestamp"])
    n_rows = len(output)

    close_allowed = np.ones(n_rows, dtype=bool)
    force_close = np.zeros(n_rows, dtype=bool)
    iso = timestamps.isocalendar()
    week_key = list(zip(iso.year, iso.week))
    for index in range(n_rows - 1):
        if week_key[index] != week_key[index + 1]:
            force_close[index] = True

    session_key = us_session_day(timestamps)
    marked_sessions = set(session_key[force_close])
    weekend_close_only = np.isin(session_key, list(marked_sessions))
    entry_allowed = close_allowed & ~weekend_close_only

    output["session_day"] = session_key
    output["close_allowed"] = close_allowed
    output["entry_allowed"] = entry_allowed
    output["friday_night_close_only"] = False
    output["weekend_session_close_only"] = weekend_close_only
    output["friday_session_end_force_close"] = force_close
    return output


def validate_spread_output(frame: pd.DataFrame, interval_minutes: int) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise RuntimeError("Spread output timestamps must be unique and sorted")

    required = ["qff_close_filled", "tsm_close", "usdttwd_close", "tsm_twd_fair", "spread"]
    missing_counts = frame[required].isna().sum()
    bad = missing_counts[missing_counts > 0]
    if not bad.empty:
        raise RuntimeError(f"Spread output has missing values:\n{bad}")

    for position in sorted({0, len(frame) // 2, len(frame) - 1}):
        row = frame.iloc[position]
        fair = row["tsm_close"] * row["usdttwd_close"] / ADR_SHARE_RATIO
        expected = (fair - row["qff_close_filled"]) / (fair + row["qff_close_filled"]) * 200
        if abs(fair - row["tsm_twd_fair"]) > 1e-9:
            raise RuntimeError(f"Fair-value check failed at row {position}")
        if abs(expected - row["spread"]) > 1e-9:
            raise RuntimeError(f"Spread check failed at row {position}")

    if not bool(frame["close_allowed"].all()):
        raise RuntimeError("close_allowed must be True on every session row")
    expected_entry = frame["close_allowed"] & ~frame["weekend_session_close_only"]
    if not frame["entry_allowed"].equals(expected_entry):
        raise RuntimeError("entry_allowed must equal close_allowed & ~weekend close-only")
    if bool(frame["friday_night_close_only"].any()):
        raise RuntimeError("friday_night_close_only must be all False for UMC sessions")

    # exactly one force-close marker per ISO week transition, on that week's
    # final bar, and its whole session must be close-only
    marker = frame.loc[frame["friday_session_end_force_close"]]
    iso = pd.DatetimeIndex(frame["timestamp"]).isocalendar()
    week_key = pd.Series(list(zip(iso.year, iso.week)), index=frame.index)
    transitions = int((week_key != week_key.shift(-1)).iloc[:-1].sum())
    if len(marker) != transitions:
        raise RuntimeError(
            f"Expected {transitions} week-end force-close markers, got {len(marker)}"
        )
    if not frame.loc[frame["friday_session_end_force_close"], "weekend_session_close_only"].all():
        raise RuntimeError("Force-close bars must sit inside close-only sessions")

    # thin TAIFEX night trade means occasional multi-bar gaps; forward-filling
    # through them matches the QFF baseline convention, so gaps only warn.
    staleness_warn = 4 * interval_minutes
    max_staleness = float(frame["ccf_staleness_minutes"].max())
    stale_rows = int((frame["ccf_staleness_minutes"] > staleness_warn).sum())
    if stale_rows:
        print(
            f"WARNING: {stale_rows} session bars ({stale_rows / len(frame):.1%}) "
            f"use CCF closes more than {staleness_warn}min old "
            f"(max {max_staleness:.0f}min)"
        )
    if max_staleness > 24 * 60:
        raise RuntimeError(
            f"CCF staleness {max_staleness:.0f}min exceeds 24h - session filter broken"
        )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["timestamp"] = format_taipei_timestamps(output["timestamp"])
    output.to_csv(path, index=False)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    interval_minutes = INTERVAL_MINUTES[args.interval]
    ccf_path = args.ccf_path or DEFAULT_CCF_PATHS[args.interval]
    umc_path = args.umc_path or DEFAULT_UMC_PATHS[args.interval]

    ccf = read_ohlcv(ccf_path, "CCF")
    umc = read_ohlcv(umc_path, "UMC")
    fx = build_fx_series(interval_minutes)

    start = (
        parse_taipei_timestamp(args.start)
        if args.start
        else max(ccf["timestamp"].iloc[0], umc["timestamp"].iloc[0], fx["known_time"].iloc[0])
    )
    end = (
        parse_taipei_timestamp(args.end)
        if args.end
        else min(ccf["timestamp"].iloc[-1], umc["timestamp"].iloc[-1], fx["timestamp"].iloc[-1])
    )
    print(f"Interval {args.interval}; range {start} -> {end}")

    umc_in_range = umc[(umc["timestamp"] >= start) & (umc["timestamp"] <= end)].copy()
    if umc_in_range.empty:
        raise RuntimeError("No UMC bars inside the requested range")

    # drop UMC sessions (US trading days) with too few native CCF bars
    session_day = us_session_day(pd.DatetimeIndex(umc_in_range["timestamp"]))
    umc_in_range["session_day"] = session_day
    ccf_days = us_session_day(pd.DatetimeIndex(ccf["timestamp"]))
    ccf_bars_per_day = pd.Series(1, index=ccf_days).groupby(level=0).sum()
    kept_sessions = []
    dropped_sessions = []
    for day, group in umc_in_range.groupby("session_day"):
        first, last = group["timestamp"].iloc[0], group["timestamp"].iloc[-1]
        in_window = ccf[
            (ccf["timestamp"] >= first)
            & (ccf["timestamp"] < last + pd.Timedelta(minutes=interval_minutes))
        ]
        if len(in_window) >= args.min_ccf_bars_per_session:
            kept_sessions.append(day)
        else:
            dropped_sessions.append((day, len(group), len(in_window)))
    if dropped_sessions:
        for day, umc_bars, ccf_bars in dropped_sessions:
            print(
                f"Excluding session {day.date()}: {umc_bars} UMC bars but only "
                f"{ccf_bars} CCF bars (TAIFEX likely closed)"
            )
    umc_kept = umc_in_range[umc_in_range["session_day"].isin(kept_sessions)].copy()
    session_index = pd.DatetimeIndex(umc_kept["timestamp"])
    if session_index.empty:
        raise RuntimeError("No sessions left after the CCF-coverage filter")

    fx_probe = asof_fx(fx, session_index, interval_minutes)
    fx_probe_days = us_session_day(pd.DatetimeIndex(fx_probe["timestamp"]))
    fx_max_by_session = (
        pd.Series(fx_probe["fx_close_staleness_minutes"].to_numpy(), index=fx_probe_days)
        .groupby(level=0)
        .max()
    )
    fx_bad_sessions = set(fx_max_by_session[fx_max_by_session > MAX_FX_STALENESS_MINUTES].index)
    if fx_bad_sessions:
        for day in sorted(fx_bad_sessions):
            print(
                f"Excluding session {day.date()}: FX staleness "
                f"{fx_max_by_session[day]:.0f}min exceeds "
                f"{MAX_FX_STALENESS_MINUTES:.0f}min (FX_IDC feed outage)"
            )
        umc_kept = umc_kept[~umc_kept["session_day"].isin(fx_bad_sessions)].copy()
        kept_sessions = [day for day in kept_sessions if day not in fx_bad_sessions]
        session_index = pd.DatetimeIndex(umc_kept["timestamp"])
        if session_index.empty:
            raise RuntimeError("No sessions left after the FX-staleness filter")

    ccf_aligned = align_ccf_close(ccf, session_index)
    fx_aligned = asof_fx(fx, session_index, interval_minutes)

    umc_close = umc_kept["close"].to_numpy()
    fair = umc_close * fx_aligned["close"].to_numpy() / ADR_SHARE_RATIO
    ccf_close = ccf_aligned["ccf_close_aligned"].to_numpy()
    spread = (fair - ccf_close) / (fair + ccf_close) * 200

    frame = pd.DataFrame(
        {
            "timestamp": session_index,
            "qff_close": ccf_aligned["ccf_close_exact"].to_numpy(),
            "qff_close_filled": ccf_close,
            "qff_was_filled": ccf_aligned["ccf_close_exact"].isna().to_numpy(),
            "ccf_staleness_minutes": ccf_aligned["ccf_staleness_minutes"].to_numpy(),
            "tsm_close": umc_close,
            "usdttwd_close": fx_aligned["close"].to_numpy(),
            "fx_close_staleness_minutes": fx_aligned[
                "fx_close_staleness_minutes"
            ].to_numpy(),
            "tsm_twd_fair": fair,
            "spread": spread,
        }
    )
    frame = add_trading_mask_columns(frame)
    validate_spread_output(frame, interval_minutes)

    out_dir: Path = args.out_dir
    spread_path = out_dir / f"ccf_umc_spread_umc_session_{args.interval}.csv"
    fx_path = out_dir / f"ccf_umc_fx_umc_session_{args.interval}.csv"
    write_csv(frame.drop(columns=["session_day"]), spread_path)
    fx_out = fx_aligned.rename(
        columns={"fx_close_staleness_minutes": "close_staleness_minutes"}
    )
    fx_out.insert(1, "symbol", "FX_IDC:USDTWD(spliced)")
    write_csv(fx_out, fx_path)

    if args.interval == "15m":
        delayed = build_ccf_delayed_open(ccf, session_index, interval_minutes)
        delayed.insert(1, "symbol", "TAIFEX:CCF1!(delayed-open)")
        delayed_path = out_dir / "ccf_umc_ccf_delayed_open_15m.csv"
        write_csv(delayed, delayed_path)
        covered = len(delayed) / len(session_index)
        print(
            f"CCF delayed-open fills: {len(delayed):,} rows "
            f"({covered:.1%} of session bars; median delay "
            f"{delayed['fill_delay_minutes'].median():.0f}min) -> {delayed_path}"
        )

    sessions = frame["session_day"].nunique() if "session_day" in frame else None
    print(f"Sessions kept: {len(kept_sessions)}; dropped: {len(dropped_sessions)}")
    print(
        f"Session rows: {len(frame):,}; CCF native-bar coverage "
        f"{1 - frame['qff_was_filled'].mean():.1%}; CCF staleness median "
        f"{frame['ccf_staleness_minutes'].median():.0f}min / max "
        f"{frame['ccf_staleness_minutes'].max():.0f}min"
    )
    print(
        f"FX staleness median {frame['fx_close_staleness_minutes'].median():.0f}min "
        f"/ max {frame['fx_close_staleness_minutes'].max():.0f}min"
    )
    print(
        f"Spread: mean {frame['spread'].mean():.4f}, std {frame['spread'].std():.4f}, "
        f"min {frame['spread'].min():.4f}, max {frame['spread'].max():.4f}"
    )
    print(f"Wrote spread: {spread_path}")
    print(f"Wrote aligned FX: {fx_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
