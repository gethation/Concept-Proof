from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
ADR_SHARE_RATIO = 5.0

# Column names deliberately reuse the legacy qff_*/tsm_* schema so the existing
# z-score, backtest and grid-search scripts run unchanged:
#   qff_*     -> TAIFEX CCF (UMC stock futures, TWD, 1m from TAIFEX tick)
#   tsm_*     -> NYSE:UMC ADR (USD, 1m RTH from IBKR)
#   usdttwd_* -> FX_IDC:USDTWD spliced series
#
# The session index is UMC's own RTH minutes: that is when both legs can actually
# be traded together. CCF trades far longer (TAIFEX day + night), so it is aligned
# onto the UMC grid rather than the other way round.

DEFAULT_CCF = Path("data/processed/ccf1_1m_cumulative.csv")
DEFAULT_UMC = Path("data/processed/umc_1m_cumulative.csv")
DEFAULT_FX = [
    (5, Path("data/processed/fxidc_usdtwd_5m_taipei_tv.csv")),
    (15, Path("data/processed/fxidc_usdtwd_15m_taipei_tv.csv")),
    (60, Path("data/processed/fxidc_usdtwd_1h_taipei_tv.csv")),
]

# FX_IDC has multi-hour outages across every interval. USDTWD moves so little
# intraday (measured: 0.17% median intra-session range, versus 3.4% for each
# equity leg) that forward-filling through them is safer than fragmenting
# sessions. Stale rows are reported rather than silently accepted.
MAX_FX_STALENESS_MINUTES = 720.0
WARN_FX_STALENESS_MINUTES = 60.0

# CCF is thinly traded at times; a minute with no trade carries the previous
# close forward, matching the QFF baseline convention.
WARN_CCF_STALENESS_MINUTES = 15.0
MAX_CCF_STALENESS_MINUTES = 240.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 1-minute CCF/UMC spread on UMC's RTH session index. "
            "CCF is aligned as-of (last close at or before each UMC minute) and "
            "USDTWD is spliced from the finest available FX_IDC interval."
        )
    )
    parser.add_argument("--ccf-path", type=Path, default=DEFAULT_CCF)
    parser.add_argument("--umc-path", type=Path, default=DEFAULT_UMC)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/ccf_umc_spread_1m.csv"),
    )
    parser.add_argument(
        "--min-ccf-bars-per-session",
        type=int,
        default=30,
        help="Sessions with fewer native CCF minutes are excluded entirely.",
    )
    parser.add_argument(
        "--weekend-policy",
        choices=("flat", "no-entry", "none"),
        default="flat",
        help=(
            "How the last session of an ISO week is treated. "
            "'flat' (default, inherited from QFF/TSM): no entries in that session "
            "and force-close on its final bar. "
            "'no-entry': keep the entry ban, drop the force-close. "
            "'none': neither. "
            "The rule exists because Binance TSM trades 24/7 while QFF is frozen "
            "over the weekend, leaving an uncovered leg. CCF/UMC has no such "
            "exposure -- TAIFEX and NYSE both shut -- so the rule can be dropped, "
            "at the cost of carrying weekend gap risk on a fully hedged position."
        ),
    )
    return parser.parse_args(argv)


def parse_taipei(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TAIPEI_TZ)
    return timestamp.tz_convert(TAIPEI_TZ)


def format_taipei(values: pd.Series) -> pd.Series:
    formatted = values.dt.strftime("%Y-%m-%d %H:%M:%S%z")
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
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["open", "close"]].isna().any().any():
        raise RuntimeError(f"{label} has invalid open/close values")
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if timestamps.has_duplicates:
        raise RuntimeError(f"{label} has duplicate timestamps")
    return frame.sort_values("timestamp").reset_index(drop=True)


def us_session_day(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Group a 21:30->04:00 (or 22:30->05:00) Taipei session onto one key."""
    return (timestamps - pd.Timedelta(hours=12)).normalize()


def build_fx_series() -> pd.DataFrame:
    """Splice FX_IDC intervals, finest first, into a known-time close series."""
    pieces = []
    for minutes, path in DEFAULT_FX:
        if not path.exists():
            continue
        frame = read_ohlcv(path, f"FX {minutes}m")[["timestamp", "open", "close"]]
        frame = frame.copy()
        frame["fx_interval_minutes"] = minutes
        # a bar's close is only known once the bar has ended
        frame["known_time"] = frame["timestamp"] + pd.Timedelta(minutes=minutes)
        pieces.append(frame)
    if not pieces:
        raise RuntimeError("No FX_IDC input files found")
    fx = pd.concat(pieces, ignore_index=True)
    fx = fx.sort_values(["timestamp", "fx_interval_minutes"])
    fx = fx.drop_duplicates(subset=["timestamp"], keep="first")
    return fx.sort_values("known_time").reset_index(drop=True)


def asof_fx(fx: pd.DataFrame, session_index: pd.DatetimeIndex) -> pd.DataFrame:
    """FX close known by the end of each 1-minute session bar, plus the FX level
    as of each bar's start (used as the entry-fill rate)."""
    close_time = session_index + pd.Timedelta(minutes=1)
    matched = pd.merge_asof(
        pd.DataFrame({"close_time": close_time}),
        fx[["known_time", "close"]],
        left_on="close_time",
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
    staleness = (
        (matched["close_time"] - matched["known_time"]).dt.total_seconds() / 60.0
    )
    result = pd.DataFrame(
        {
            "timestamp": session_index,
            "open": open_match["open"].to_numpy(),
            "close": matched["close"].to_numpy(),
            "staleness_minutes": staleness.to_numpy(),
        }
    )
    if result["close"].isna().any():
        raise RuntimeError("FX series does not cover the session index")
    stale = int((result["staleness_minutes"] > WARN_FX_STALENESS_MINUTES).sum())
    if stale:
        print(
            f"WARNING: {stale} bars ({stale / len(result):.1%}) use FX closes older "
            f"than {WARN_FX_STALENESS_MINUTES:.0f}min "
            f"(max {result['staleness_minutes'].max():.0f}min)"
        )
    return result


def align_ccf(ccf: pd.DataFrame, session_index: pd.DatetimeIndex) -> pd.DataFrame:
    """CCF close as of each UMC minute: the last CCF bar starting at or before it."""
    closes = pd.Series(
        ccf["close"].to_numpy(), index=pd.DatetimeIndex(ccf["timestamp"])
    )
    exact = closes.reindex(session_index)
    union = closes.index.union(session_index)
    filled = closes.reindex(union).sort_index().ffill().reindex(session_index)

    matched = pd.merge_asof(
        pd.DataFrame({"session_start": session_index}),
        ccf[["timestamp"]].assign(ccf_start=lambda d: d["timestamp"]),
        left_on="session_start",
        right_on="timestamp",
        direction="backward",
    )["ccf_start"]
    staleness = (
        session_index - pd.DatetimeIndex(matched)
    ).total_seconds() / 60.0
    return pd.DataFrame(
        {
            "timestamp": session_index,
            "ccf_close_exact": exact.to_numpy(),
            "ccf_close_aligned": filled.to_numpy(),
            "ccf_staleness_minutes": np.asarray(staleness),
        }
    )


def add_trading_masks(
    frame: pd.DataFrame, *, weekend_policy: str = "flat"
) -> pd.DataFrame:
    """Every row is a tradable UMC RTH minute.

    ``weekend_policy`` controls the last session of each ISO week:
      flat      no entries in it, and force-close on its final bar (QFF/TSM default)
      no-entry  keep the entry ban, drop the force-close
      none      neither rule

    The inherited rule exists because the Binance TSM leg trades 24/7 while QFF is
    frozen over the weekend, leaving an uncovered leg. **CCF/UMC has no such
    exposure** -- TAIFEX and NYSE both shut -- so it can be dropped, at the cost of
    carrying weekend gap risk on a position that stays fully hedged.
    """
    if weekend_policy not in {"flat", "no-entry", "none"}:
        raise ValueError(f"unknown weekend_policy: {weekend_policy!r}")

    output = frame.copy()
    timestamps = pd.DatetimeIndex(output["timestamp"])
    n = len(output)
    session_key = us_session_day(timestamps)

    week_end_bar = np.zeros(n, dtype=bool)
    iso = timestamps.isocalendar()
    week_key = list(zip(iso.year, iso.week))
    for i in range(n - 1):
        if week_key[i] != week_key[i + 1]:
            week_end_bar[i] = True
    marked = set(session_key[week_end_bar])
    week_end_session = np.isin(session_key, list(marked))

    force_close = week_end_bar if weekend_policy == "flat" else np.zeros(n, dtype=bool)
    close_only = week_end_session if weekend_policy in {"flat", "no-entry"} else np.zeros(n, dtype=bool)

    output["session_day"] = session_key
    output["close_allowed"] = True
    output["entry_allowed"] = ~close_only
    output["friday_night_close_only"] = False
    output["weekend_session_close_only"] = close_only
    output["friday_session_end_force_close"] = force_close
    return output


def validate(frame: pd.DataFrame) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise RuntimeError("Output timestamps must be unique and sorted")

    required = ["qff_close_filled", "tsm_close", "usdttwd_close", "tsm_twd_fair", "spread"]
    bad = frame[required].isna().sum()
    if bad.any():
        raise RuntimeError(f"Output has missing values:\n{bad[bad > 0]}")

    for position in sorted({0, len(frame) // 2, len(frame) - 1}):
        row = frame.iloc[position]
        fair = row["tsm_close"] * row["usdttwd_close"] / ADR_SHARE_RATIO
        expected = (fair - row["qff_close_filled"]) / (fair + row["qff_close_filled"]) * 200
        if abs(fair - row["tsm_twd_fair"]) > 1e-9:
            raise RuntimeError(f"Fair-value check failed at row {position}")
        if abs(expected - row["spread"]) > 1e-9:
            raise RuntimeError(f"Spread check failed at row {position}")

    max_stale = float(frame["ccf_staleness_minutes"].max())
    if max_stale > MAX_CCF_STALENESS_MINUTES:
        raise RuntimeError(
            f"CCF staleness {max_stale:.0f}min exceeds "
            f"{MAX_CCF_STALENESS_MINUTES:.0f}min - session filter is broken"
        )
    stale = int((frame["ccf_staleness_minutes"] > WARN_CCF_STALENESS_MINUTES).sum())
    if stale:
        print(
            f"WARNING: {stale} bars ({stale / len(frame):.1%}) use CCF closes older "
            f"than {WARN_CCF_STALENESS_MINUTES:.0f}min (max {max_stale:.0f}min)"
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ccf = read_ohlcv(args.ccf_path, "CCF 1m")
    umc = read_ohlcv(args.umc_path, "UMC 1m")
    fx = build_fx_series()

    start = (
        parse_taipei(args.start)
        if args.start
        else max(ccf["timestamp"].iloc[0], umc["timestamp"].iloc[0])
    )
    end = (
        parse_taipei(args.end)
        if args.end
        else min(ccf["timestamp"].iloc[-1], umc["timestamp"].iloc[-1])
    )
    print(f"Range: {start} -> {end}")

    umc_in = umc[(umc["timestamp"] >= start) & (umc["timestamp"] <= end)].copy()
    if umc_in.empty:
        raise RuntimeError("No UMC bars inside the requested range")
    umc_in["session_day"] = us_session_day(pd.DatetimeIndex(umc_in["timestamp"]))

    # drop sessions where CCF barely traded during UMC hours
    kept, dropped = [], []
    for day, group in umc_in.groupby("session_day"):
        first, last = group["timestamp"].iloc[0], group["timestamp"].iloc[-1]
        native = ccf[
            (ccf["timestamp"] >= first)
            & (ccf["timestamp"] <= last)
        ]
        if len(native) >= args.min_ccf_bars_per_session:
            kept.append(day)
        else:
            dropped.append((day, len(group), len(native)))
    for day, umc_bars, ccf_bars in dropped:
        print(
            f"Excluding session {day.date()}: {umc_bars} UMC minutes but only "
            f"{ccf_bars} native CCF minutes"
        )
    umc_kept = umc_in[umc_in["session_day"].isin(kept)].copy()
    if umc_kept.empty:
        raise RuntimeError("No sessions left after the CCF-coverage filter")

    session_index = pd.DatetimeIndex(umc_kept["timestamp"])
    ccf_aligned = align_ccf(ccf, session_index)
    fx_aligned = asof_fx(fx, session_index)

    umc_close = umc_kept["close"].to_numpy()
    ccf_close = ccf_aligned["ccf_close_aligned"].to_numpy()
    fair = umc_close * fx_aligned["close"].to_numpy() / ADR_SHARE_RATIO
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
            "fx_close_staleness_minutes": fx_aligned["staleness_minutes"].to_numpy(),
            "tsm_twd_fair": fair,
            "spread": spread,
        }
    )
    frame = add_trading_masks(frame, weekend_policy=args.weekend_policy)
    validate(frame)
    if args.weekend_policy != "flat":
        print(
            f"NOTE: weekend policy '{args.weekend_policy}' -- positions may be "
            "carried over a weekend. Both legs are shut then, so there is no "
            "uncovered leg, but weekend gap risk is real and unmodelled here."
        )

    out = frame.drop(columns=["session_day"]).copy()
    out["timestamp"] = format_taipei(out["timestamp"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # The backtest fills at the next bar's open rather than its close, so it needs
    # an open series per leg on this exact session index. CCF and UMC come from
    # their own 1m files; FX is written here because it has been resampled onto
    # the 1m grid from a coarser interval and exists nowhere else.
    fx_out = pd.DataFrame(
        {
            "timestamp": format_taipei(pd.Series(session_index)),
            "open": fx_aligned["open"].to_numpy(),
            "high": fx_aligned["close"].to_numpy(),
            "low": fx_aligned["close"].to_numpy(),
            "close": fx_aligned["close"].to_numpy(),
            "volume": 0.0,
        }
    )
    fx_path = args.out.with_name(args.out.stem + "_fx.csv")
    fx_out.to_csv(fx_path, index=False)
    print(f"Wrote aligned FX open/close: {fx_path}")

    sessions = frame["session_day"].nunique()
    print(
        f"\nRows: {len(frame):,} over {sessions} sessions "
        f"({frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[-1]})"
    )
    print(
        f"CCF native-minute coverage {1 - frame['qff_was_filled'].mean():.1%}; "
        f"staleness median {frame['ccf_staleness_minutes'].median():.0f}min / "
        f"max {frame['ccf_staleness_minutes'].max():.0f}min"
    )
    print(
        f"Spread: mean {frame['spread'].mean():.4f}, std {frame['spread'].std():.4f}, "
        f"min {frame['spread'].min():.4f}, max {frame['spread'].max():.4f}"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
