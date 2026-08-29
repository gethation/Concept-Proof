"""Fetch TWSE:2303 (UMC's home listing) daily bars from tvdatafeed.

The CCF/UMC spread is two layers glued together -- the ADR premium (UMC ADR vs
the 2303 home shares) and the futures basis (CCF vs 2303) -- and 2303 is the
benchmark that separates them. Daily bars are enough: the layer decomposition
runs at session level, and the 13:30 TWSE closing auction IS the daily close.

tvdatafeed serves years of daily history, so unlike the intraday ingests this
does not race a rolling window; it still append-merges for consistency.

    python scripts/ingest/tv_2303.py

NOTE: whether TradingView's daily closes are dividend-adjusted is not
documented; the layer study checks this empirically against CCF around the
ex-dividend date before trusting level comparisons across it.
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

# Not in lib/paths.py yet: that file currently carries unrelated uncommitted
# changes on the parent branch, and adding one constant would drag them into
# this branch's commits. Move this to paths.TWSE_2303_1D when it lands.
DEFAULT_OUT = paths.BARS / "twse" / "2303_1d.csv"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append TWSE:2303 daily bars from tvdatafeed."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--symbol", default="2303")
    parser.add_argument("--exchange", default="TWSE")
    parser.add_argument("--n-bars", type=int, default=500)
    return parser.parse_args(argv)


def fetch(symbol: str, exchange: str, n_bars: int) -> pd.DataFrame:
    tv = TvDatafeed()
    raw = tv.get_hist(
        symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=n_bars
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"tvdatafeed returned no {exchange}:{symbol} daily bars")

    frame = raw.reset_index().rename(columns={"datetime": "timestamp"})
    stamps = pd.DatetimeIndex(frame["timestamp"])
    if stamps.tz is None:
        stamps = stamps.tz_localize(TAIPEI_TZ)
    else:
        stamps = stamps.tz_convert(TAIPEI_TZ)
    frame["timestamp"] = stamps
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    return frame.sort_values("timestamp")[COLUMNS].reset_index(drop=True)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    fresh = fetch(args.symbol, args.exchange, args.n_bars)
    fresh["timestamp"] = (
        fresh["timestamp"]
        .dt.strftime("%Y-%m-%d %H:%M:%S%z")
        .str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)
    )
    print(
        f"Fetched {len(fresh):,} bars: "
        f"{fresh['timestamp'].iloc[0]} to {fresh['timestamp'].iloc[-1]}"
    )

    output = fresh
    if args.out.exists():
        existing = pd.read_csv(args.out)
        if list(existing.columns) != COLUMNS:
            raise RuntimeError(
                f"{args.out} has columns {list(existing.columns)}, expected {COLUMNS}"
            )
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        order = pd.to_datetime(combined["timestamp"], utc=True, format="mixed")
        output = combined.iloc[order.argsort().to_numpy()].reset_index(drop=True)
        print(f"  merged with {len(existing):,} existing rows: {len(output):,} total")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Wrote {len(output):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
