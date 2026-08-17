"""Download 1-minute OHLCV from a ccxt exchange into the canonical bar file.

    python scripts/ingest/ccxt_ohlcv.py --feed binance_tsmusdtp
    python scripts/ingest/ccxt_ohlcv.py --feed bitopro_usdttwd

Replaces download_binance_tsmusdtp_1m.py and download_bitopro_usdttwd_1m.py,
which were the same 198-line script twice: the only differences were the
exchange constructor, the symbol, the page size, and the string used to suggest
alternatives when a symbol lookup failed. Those four things are the FEEDS table
below.

The fetch window comes from a reference bar file (the TAIFEX leg), because the
point of these legs is to cover it. That coupling is also how history was lost
once: writing the fetch to a *new* filename bypassed the append-merge, and the
reference file's start had moved forward, so the early weeks simply vanished
from the new file. Writing to the canonical path keeps the merge in the loop.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import ccxt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths  # noqa: E402
from lib.barsio import merge_with_existing, write_bars_csv  # noqa: E402
from lib.timeutil import TAIPEI_TZ  # noqa: E402

TIMEFRAME = "1m"
ONE_MINUTE_MS = 60_000


@dataclass(frozen=True)
class Feed:
    exchange: str  # a ccxt exchange class name
    symbol: str
    out: Path
    limit: int  # candles per request, per that exchange's cap
    similar: list[str] = field(default_factory=list)  # substrings for a failed lookup


FEEDS: dict[str, Feed] = {
    "binance_tsmusdtp": Feed(
        exchange="binanceusdm",
        symbol="TSM/USDT:USDT",
        out=paths.TSMUSDTP_1M,
        limit=1500,
        similar=["TSM"],
    ),
    "bitopro_usdttwd": Feed(
        exchange="bitopro",
        symbol="USDT/TWD",
        out=paths.USDTTWD_1M,
        limit=1000,
        similar=["USDT", "TWD"],
    ),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feed", required=True, choices=sorted(FEEDS))
    parser.add_argument(
        "--reference",
        type=Path,
        default=paths.QFF1_1M,
        help="Bar file whose timestamp range defines the fetch window.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite instead of append-merging. Loses history; see module docstring.",
    )
    return parser.parse_args(argv)


def reference_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not path.exists():
        raise FileNotFoundError(f"Reference CSV does not exist: {path}")
    frame = pd.read_csv(path, usecols=["timestamp"])
    if frame.empty:
        raise RuntimeError(f"Reference CSV has no rows: {path}")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(TAIPEI_TZ)
    return timestamps.iloc[0], timestamps.iloc[-1]


def build_exchange(name: str, timeout_ms: int) -> ccxt.Exchange:
    exchange = getattr(ccxt, name)({"enableRateLimit": True, "timeout": timeout_ms})
    exchange.load_markets()
    return exchange


def fetch_with_retries(
    exchange: ccxt.Exchange,
    symbol: str,
    since_ms: int,
    limit: int,
    max_retries: int,
) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_ms, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"Failed to fetch OHLCV after {max_retries} retries from since={since_ms}"
    ) from last_error


def fetch_range(
    exchange: ccxt.Exchange,
    feed: Feed,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit: int,
    max_retries: int,
) -> pd.DataFrame:
    if symbol not in exchange.markets:
        matches = [
            market_symbol
            for market_symbol, market in exchange.markets.items()
            if any(
                token in market_symbol or token in market.get("id", "")
                for token in feed.similar
            )
        ]
        raise RuntimeError(f"Symbol {symbol} not found. Similar markets: {matches}")

    start_ms = int(start.tz_convert("UTC").timestamp() * 1000)
    end_ms = int(end.tz_convert("UTC").timestamp() * 1000)

    rows: list[list[float]] = []
    since_ms = start_ms
    requests = 0

    while since_ms <= end_ms:
        batch = fetch_with_retries(exchange, symbol, since_ms, limit, max_retries)
        requests += 1
        if not batch:
            break

        rows.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts < since_ms:
            raise RuntimeError(
                f"Exchange returned a non-advancing batch: since={since_ms}, "
                f"last_ts={last_ts}"
            )
        since_ms = last_ts + ONE_MINUTE_MS

        if requests % 10 == 0:
            current = pd.to_datetime(last_ts, unit="ms", utc=True).tz_convert(TAIPEI_TZ)
            print(f"Fetched through {current}")

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(
        rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
    )
    frame = frame.drop_duplicates(subset=["timestamp_ms"], keep="last").sort_values(
        "timestamp_ms"
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(TAIPEI_TZ)
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy()
    return frame[columns]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    feed = FEEDS[args.feed]
    symbol = args.symbol or feed.symbol
    out = args.out or feed.out
    limit = args.limit or feed.limit

    start, end = reference_range(args.reference)
    print(f"Reference range ({args.reference.name}): {start} to {end}")

    exchange = build_exchange(feed.exchange, args.timeout_ms)
    market = exchange.market(symbol)
    print(f"Using {feed.exchange} market: {symbol} ({market['id']})")

    frame = fetch_range(
        exchange, feed, symbol, start, end, limit, args.max_retries
    )
    if frame.empty:
        raise RuntimeError(
            f"No OHLCV rows returned for {symbol} between {start} and {end}"
        )

    written = write_bars_csv(frame, out, symbol, merge=not args.no_merge)
    print(f"Fetched {len(frame):,} rows; {len(written):,} rows now in {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
