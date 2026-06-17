from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_QFF_PATH = Path("data/processed/qff1_1m.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/bitopro_usdttwd_1m_taipei.csv")
DEFAULT_SYMBOL = "USDT/TWD"
TIMEFRAME = "1m"
ONE_MINUTE_MS = 60_000


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download BitoPro USDT/TWD 1m OHLCV via ccxt, using the same "
            "timestamp range as the TAIFEX QFF1 CSV."
        )
    )
    parser.add_argument("--qff-path", type=Path, default=DEFAULT_QFF_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args(argv)


def read_qff_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not path.exists():
        raise FileNotFoundError(f"QFF CSV does not exist: {path}")
    frame = pd.read_csv(path, usecols=["timestamp"])
    if frame.empty:
        raise RuntimeError(f"QFF CSV has no rows: {path}")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(TAIPEI_TZ)
    return timestamps.iloc[0], timestamps.iloc[-1]


def build_exchange(timeout_ms: int) -> ccxt.Exchange:
    exchange = ccxt.bitopro(
        {
            "enableRateLimit": True,
            "timeout": timeout_ms,
        }
    )
    exchange.load_markets()
    return exchange


def fetch_ohlcv_with_retries(
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
            if "USDT" in market_symbol and "TWD" in market_symbol
        ]
        raise RuntimeError(f"Symbol {symbol} not found. USDT/TWD-like markets: {matches}")

    start_utc = start.tz_convert("UTC")
    end_utc = end.tz_convert("UTC")
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    rows: list[list[float]] = []
    since_ms = start_ms
    request_count = 0

    while since_ms <= end_ms:
        batch = fetch_ohlcv_with_retries(
            exchange=exchange,
            symbol=symbol,
            since_ms=since_ms,
            limit=limit,
            max_retries=max_retries,
        )
        request_count += 1
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

        if request_count % 10 == 0:
            current = pd.to_datetime(last_ts, unit="ms", utc=True).tz_convert(TAIPEI_TZ)
            print(f"Fetched through {current}")

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    frame = pd.DataFrame(
        rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
    )
    frame = frame.drop_duplicates(subset=["timestamp_ms"], keep="last").sort_values(
        "timestamp_ms"
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(TAIPEI_TZ)
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy()
    return frame[["timestamp", "open", "high", "low", "close", "volume"]]


def write_csv(frame: pd.DataFrame, output_path: Path, symbol: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.copy()
    data.insert(1, "symbol", symbol)
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    data["timestamp"] = data["timestamp"].str.replace(
        r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True
    )
    data.to_csv(output_path, index=False)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    start, end = read_qff_range(args.qff_path)
    print(f"QFF range: {start} to {end}")

    exchange = build_exchange(args.timeout_ms)
    market = exchange.market(args.symbol)
    print(f"Using BitoPro market: {args.symbol} ({market['id']})")

    frame = fetch_range(
        exchange=exchange,
        symbol=args.symbol,
        start=start,
        end=end,
        limit=args.limit,
        max_retries=args.max_retries,
    )
    if frame.empty:
        raise RuntimeError(
            f"No OHLCV rows returned for {args.symbol} between {start} and {end}"
        )

    write_csv(frame, args.out, args.symbol)
    print(f"Wrote {len(frame):,} rows to {args.out}")
    print(f"Output range: {frame['timestamp'].iloc[0]} to {frame['timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
