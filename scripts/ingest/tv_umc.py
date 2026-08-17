"""Extend the UMC ADR 1-minute archive from tvdatafeed.

The archive was originally an IBKR export, which has no downloader here. This
fills the gap with tvdatafeed, whose agreement with IBKR was measured on 5,208
overlapping bars: median close difference 0.000, correlation 0.999989, mean
-0.08 bps. The caveat that matters is volume -- tvdatafeed reports roughly a
tenth of IBKR's -- and bar count, 374-390 per session against IBKR's exact 390.
Prices are what the spread is built from, so the splice is sound; do not read
the volume column across the seam.

The anonymous endpoint only serves a rolling window (~19 days at 1m), so this
always append-merges: rows outside the fetched span are kept, and inside it the
fresh rows win. Running it on a schedule is what keeps history that the feed
has already dropped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tvDatafeed import Interval, TvDatafeed

TAIPEI_TZ = "Asia/Taipei"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths  # noqa: E402

DEFAULT_OUT = paths.UMC_1M
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append the latest NYSE:UMC 1m bars from tvdatafeed into the "
            "cumulative archive."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--symbol", default="UMC")
    parser.add_argument("--exchange", default="NYSE")
    parser.add_argument(
        "--n-bars",
        type=int,
        default=20000,
        help="Bars to request. The feed caps well below this at 1m anyway.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite instead of merging. Destroys history the feed no longer serves.",
    )
    return parser.parse_args(argv)


def fetch(symbol: str, exchange: str, n_bars: int) -> pd.DataFrame:
    tv = TvDatafeed()
    raw = tv.get_hist(
        symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=n_bars
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"tvdatafeed returned no {exchange}:{symbol} 1m bars")

    frame = raw.reset_index().rename(columns={"datetime": "timestamp"})
    stamps = pd.DatetimeIndex(frame["timestamp"])
    # The feed hands back naive stamps already on the exchange's Taipei-local
    # wall clock (NYSE RTH shows as 21:30-04:00), so they are localised, not
    # converted.
    if stamps.tz is None:
        stamps = stamps.tz_localize(TAIPEI_TZ)
    else:
        stamps = stamps.tz_convert(TAIPEI_TZ)
    frame["timestamp"] = stamps
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    return frame.sort_values("timestamp")[COLUMNS].reset_index(drop=True)


def format_timestamps(series: pd.Series) -> pd.Series:
    return (
        series.dt.strftime("%Y-%m-%d %H:%M:%S%z")
        .str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    fresh = fetch(args.symbol, args.exchange, args.n_bars)
    fresh["timestamp"] = format_timestamps(fresh["timestamp"])
    print(
        f"Fetched {len(fresh):,} bars: "
        f"{fresh['timestamp'].iloc[0]} to {fresh['timestamp'].iloc[-1]}"
    )

    output = fresh
    if not args.no_merge and args.out.exists():
        existing = pd.read_csv(args.out)
        if list(existing.columns) != COLUMNS:
            raise RuntimeError(
                f"{args.out} has columns {list(existing.columns)}, expected {COLUMNS}"
            )
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        order = pd.to_datetime(combined["timestamp"], utc=True, format="mixed")
        output = combined.iloc[order.argsort().to_numpy()].reset_index(drop=True)
        print(
            f"  merged with {len(existing):,} existing rows: {len(output):,} total, "
            f"{len(output) - len(fresh):,} kept from disk"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Wrote {len(output):,} rows to {args.out}")
    print(f"Range: {output['timestamp'].iloc[0]} to {output['timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
