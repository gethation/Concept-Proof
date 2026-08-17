from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tvDatafeed import Interval, TvDatafeed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.qff_tsm_15m import print_summary  # noqa: E402
from lib import paths  # noqa: E402
from lib.barsio import (  # noqa: E402
    normalize_ohlcv_frame,
    validate_frame,
    write_bars_csv as write_csv,
)
from lib.timeutil import TAIPEI_TZ  # noqa: E402
from lib.timeutil import parse_taipei as parse_taipei_timestamp  # noqa: E402

DEFAULT_START = "2026-02-01 00:00:00+08:00"
DEFAULT_OUT_DIR = paths.BARS

INTERVALS = {
    "5m": Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "1h": Interval.in_1_hour,
}
INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60}

# (exchange, symbol, bars subdirectory, file stem, intervals, n_bars per
#  interval, expected Taipei bar-start hours - warn only)
CCF_SESSION_HOURS = set(range(8, 14)) | set(range(17, 24)) | set(range(0, 5))
UMC_SESSION_HOURS = set(range(21, 24)) | set(range(0, 6))
FX_SESSION_HOURS = set(range(0, 24))

TV_DOWNLOADS = [
    ("TAIFEX", "CCF1!", "taifex", "ccf1", ["5m", "15m"],
     {"5m": 50_000, "15m": 30_000}, CCF_SESSION_HOURS),
    ("NYSE", "UMC", "nyse", "umc", ["5m", "15m"],
     {"5m": 30_000, "15m": 15_000}, UMC_SESSION_HOURS),
    ("FX_IDC", "USDTWD", "fxidc", "usdtwd", ["5m", "15m", "1h"],
     {"5m": 60_000, "15m": 30_000, "1h": 10_000}, FX_SESSION_HOURS),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download TAIFEX:CCF1!, NYSE:UMC, and FX_IDC:USDTWD OHLCV via "
            "tvdatafeed at 5m/15m (FX also 1h), Taipei timestamps. The 5m "
            "data is the primary interval for the CCF-UMC pair study; 15m "
            "reaches further back and feeds the auxiliary variant; the FX 1h "
            "series backfills whatever the finer FX intervals do not cover."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None, help="default: now (Taipei)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args(argv)


def download_tv_ohlcv(
    tv: TvDatafeed,
    *,
    exchange: str,
    symbol: str,
    interval_key: str,
    n_bars: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int,
) -> pd.DataFrame:
    last_error: Exception | None = None
    frame = None
    for attempt in range(max_retries):
        try:
            frame = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=INTERVALS[interval_key],
                n_bars=n_bars,
            )
            if frame is not None and not frame.empty:
                break
        except Exception as exc:  # tvdatafeed raises bare Exceptions on timeouts
            last_error = exc
        time.sleep(2**attempt)
    if frame is None or frame.empty:
        raise RuntimeError(
            f"tvdatafeed returned no rows for {exchange}:{symbol} {interval_key}"
        ) from last_error

    data = frame.sort_index().reset_index()
    data = data.rename(columns={data.columns[0]: "timestamp"})
    timestamps = pd.to_datetime(data["timestamp"])
    # tvdatafeed returns tz-naive Taipei-local timestamps (verified for
    # TAIFEX/NYSE/FX_IDC symbols against their known session hours)
    if timestamps.dt.tz is None:
        data["timestamp"] = timestamps.dt.tz_localize(TAIPEI_TZ)
    else:
        data["timestamp"] = timestamps.dt.tz_convert(TAIPEI_TZ)

    full_first = data["timestamp"].iloc[0]
    data = data[(data["timestamp"] >= start) & (data["timestamp"] <= end)].copy()
    if data.empty:
        raise RuntimeError(
            f"{exchange}:{symbol} {interval_key} has no rows inside "
            f"{start} -> {end} (feed starts {full_first})"
        )
    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
    print(
        f"{exchange}:{symbol} {interval_key}: feed reaches back to {full_first}"
    )
    return normalize_ohlcv_frame(data)


def check_session_hours(
    frame: pd.DataFrame, expected_hours: set[int], label: str
) -> None:
    observed = set(pd.DatetimeIndex(frame["timestamp"]).hour.unique())
    unexpected = observed - expected_hours
    if unexpected:
        print(
            f"WARNING: {label} has bars outside expected session hours: "
            f"{sorted(unexpected)} (expected within {sorted(expected_hours)}); "
            f"check the timezone assumption before using this file"
        )


def check_grid_alignment(frame: pd.DataFrame, interval_key: str, label: str) -> None:
    minutes = pd.DatetimeIndex(frame["timestamp"]).minute
    step = INTERVAL_MINUTES[interval_key]
    off_grid = int((minutes % min(step, 60) != 0).sum()) if step <= 60 else 0
    if off_grid:
        print(
            f"WARNING: {label} has {off_grid} bars off the {interval_key} "
            "minute grid"
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    start = parse_taipei_timestamp(args.start)
    end = (
        parse_taipei_timestamp(args.end)
        if args.end
        else pd.Timestamp.now(tz=TAIPEI_TZ)
    )
    if start > end:
        raise ValueError("--start must be before --end")
    out_dir: Path = args.out_dir
    print(f"Range: {start} to {end}")

    tv = TvDatafeed()
    for exchange, symbol, source, stem, interval_keys, n_bars_map, hours in TV_DOWNLOADS:
        for interval_key in interval_keys:
            label = f"{exchange}:{symbol} {interval_key}"
            frame = download_tv_ohlcv(
                tv,
                exchange=exchange,
                symbol=symbol,
                interval_key=interval_key,
                n_bars=n_bars_map[interval_key],
                start=start,
                end=end,
                max_retries=args.max_retries,
            )
            validate_frame(frame, label=label, start=start, end=end)
            check_session_hours(frame, hours, label)
            check_grid_alignment(frame, interval_key, label)
            out_path = out_dir / source / f"{stem}_{interval_key}.csv"
            # Summarise what landed on disk, not what was fetched -- the write
            # merges with existing history, so the two differ whenever the feed
            # serves a shorter window than the archive already holds.
            written = write_csv(frame, out_path, f"{exchange}:{symbol}")
            print_summary(label, written, out_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
