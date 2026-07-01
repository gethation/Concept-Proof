from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import ccxt
import pandas as pd
from tvDatafeed import Interval, TvDatafeed


TAIPEI_TZ = "Asia/Taipei"
FIFTEEN_MINUTE_TIMEFRAME = "15m"
ONE_MINUTE_TIMEFRAME = "1m"
TIMEFRAME_MS = {
    FIFTEEN_MINUTE_TIMEFRAME: 15 * 60 * 1000,
    ONE_MINUTE_TIMEFRAME: 60 * 1000,
}

DEFAULT_QFF_START = "2026-03-01 00:00:00+08:00"
DEFAULT_EXTERNAL_START = "2026-03-12 00:00:00+08:00"
DEFAULT_END = "2026-06-29 23:59:59+08:00"

DEFAULT_QFF_OUT = Path("data/processed/qff1_15m_taipei_tv.csv")
DEFAULT_TSM_OUT = Path("data/processed/okx_tsmusdtp_15m_taipei.csv")
DEFAULT_USDTTWD_OUT = Path("data/processed/bitopro_usdttwd_15m_taipei.csv")
DEFAULT_TSM_1M_OUT = Path("data/processed/okx_tsmusdtp_1m_taipei_for_15m.csv")
DEFAULT_USDTTWD_1M_OUT = Path("data/processed/bitopro_usdttwd_1m_taipei_for_15m.csv")

QFF_EXCHANGE = "TAIFEX"
QFF_SYMBOL = "QFF1!"
OKX_TSM_SYMBOL = "TSM/USDT:USDT"
BITOPRO_USDTTWD_SYMBOL = "USDT/TWD"


def parse_taipei_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TAIPEI_TZ)
    return timestamp.tz_convert(TAIPEI_TZ)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download QFF1!, OKX TSMUSDT perpetual, and USDT/TWD 15m OHLCV "
            "with Taipei timestamps."
        )
    )
    parser.add_argument("--qff-start", default=DEFAULT_QFF_START)
    parser.add_argument("--external-start", default=DEFAULT_EXTERNAL_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--qff-out", type=Path, default=DEFAULT_QFF_OUT)
    parser.add_argument("--tsm-out", type=Path, default=DEFAULT_TSM_OUT)
    parser.add_argument("--usdttwd-out", type=Path, default=DEFAULT_USDTTWD_OUT)
    parser.add_argument("--tsm-1m-out", type=Path, default=DEFAULT_TSM_1M_OUT)
    parser.add_argument("--usdttwd-1m-out", type=Path, default=DEFAULT_USDTTWD_1M_OUT)
    parser.add_argument("--qff-n-bars", type=int, default=10_000)
    parser.add_argument("--ccxt-limit", type=int, default=1000)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args(argv)


def format_taipei_timestamps(timestamps: pd.Series) -> pd.Series:
    formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    output = frame.loc[:, expected_columns].copy()
    output = output.drop_duplicates(subset=["timestamp"], keep="last")
    output = output.sort_values("timestamp").reset_index(drop=True)
    return output


def validate_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    if frame.empty:
        raise RuntimeError(f"{label} returned no rows")

    missing = {"timestamp", "open", "high", "low", "close", "volume"}.difference(
        frame.columns
    )
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {sorted(missing)}")

    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if timestamps.tz is None:
        raise RuntimeError(f"{label} timestamps are timezone-naive")
    if str(timestamps.tz) != TAIPEI_TZ:
        raise RuntimeError(f"{label} timestamps are not Asia/Taipei: {timestamps.tz}")
    if not timestamps.is_unique:
        duplicates = timestamps[timestamps.duplicated()].unique()
        raise RuntimeError(
            f"{label} has duplicate timestamps, first examples: {list(duplicates[:5])}"
        )
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError(f"{label} timestamps are not sorted")
    if timestamps[0] < start:
        raise RuntimeError(f"{label} starts before requested start: {timestamps[0]}")
    if timestamps[-1] > end:
        raise RuntimeError(f"{label} ends after requested end: {timestamps[-1]}")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    invalid = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        raise RuntimeError(f"{label} has invalid numeric values:\n{invalid}")


def write_csv(frame: pd.DataFrame, output_path: Path, symbol: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output.insert(1, "symbol", symbol)
    output["timestamp"] = format_taipei_timestamps(output["timestamp"])
    output.to_csv(output_path, index=False)


def download_qff_tvdatafeed(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_bars: int,
) -> pd.DataFrame:
    if n_bars <= 0:
        raise ValueError("--qff-n-bars must be positive")

    tv = TvDatafeed()
    frame = tv.get_hist(
        symbol=QFF_SYMBOL,
        exchange=QFF_EXCHANGE,
        interval=Interval.in_15_minute,
        n_bars=n_bars,
    )
    if frame is None or frame.empty:
        raise RuntimeError("tvdatafeed returned no QFF rows")

    data = frame.sort_index().reset_index()
    index_column = data.columns[0]
    data = data.rename(columns={index_column: "timestamp"})
    timestamps = pd.to_datetime(data["timestamp"])
    if timestamps.dt.tz is None:
        data["timestamp"] = timestamps.dt.tz_localize(TAIPEI_TZ)
    else:
        data["timestamp"] = timestamps.dt.tz_convert(TAIPEI_TZ)

    data = data[(data["timestamp"] >= start) & (data["timestamp"] <= end)].copy()
    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
    return normalize_ohlcv_frame(data)


def build_exchange(factory: Callable[..., ccxt.Exchange], timeout_ms: int) -> ccxt.Exchange:
    exchange = factory(
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
    timeframe: str,
    since_ms: int,
    limit: int,
    max_retries: int,
) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=limit
            )
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"Failed to fetch {symbol} OHLCV after {max_retries} retries "
        f"from since={since_ms}"
    ) from last_error


def download_ccxt_ohlcv(
    *,
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit: int,
    max_retries: int,
    label: str,
) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("--ccxt-limit must be positive")
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if symbol not in exchange.markets:
        matches = [
            market_symbol
            for market_symbol, market in exchange.markets.items()
            if symbol.split("/")[0] in market_symbol or symbol.split("/")[0] in market.get("id", "")
        ]
        raise RuntimeError(f"{label} symbol {symbol} not found. Similar markets: {matches}")

    start_ms = int(start.tz_convert("UTC").timestamp() * 1000)
    end_ms = int(end.tz_convert("UTC").timestamp() * 1000)
    since_ms = start_ms
    rows: list[list[float]] = []
    request_count = 0

    while since_ms <= end_ms:
        batch = fetch_ohlcv_with_retries(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            since_ms=since_ms,
            limit=limit,
            max_retries=max_retries,
        )
        request_count += 1
        if not batch:
            if not rows:
                raise RuntimeError(f"{label} returned no rows from {start}")
            break

        rows.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts < since_ms:
            raise RuntimeError(
                f"{label} returned a non-advancing batch: since={since_ms}, "
                f"last_ts={last_ts}"
            )
        since_ms = last_ts + TIMEFRAME_MS[timeframe]

        if request_count % 10 == 0:
            current = pd.to_datetime(last_ts, unit="ms", utc=True).tz_convert(TAIPEI_TZ)
            print(f"{label} {timeframe}: fetched through {current}")

    frame = pd.DataFrame(
        rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
    )
    frame = frame.drop_duplicates(subset=["timestamp_ms"], keep="last")
    frame = frame.sort_values("timestamp_ms")
    frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(TAIPEI_TZ)
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy()
    frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
    return normalize_ohlcv_frame(frame)


def print_summary(label: str, frame: pd.DataFrame, output_path: Path) -> None:
    print(
        f"{label}: wrote {len(frame):,} rows to {output_path}; "
        f"range {frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[-1]}"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    qff_start = parse_taipei_timestamp(args.qff_start)
    external_start = parse_taipei_timestamp(args.external_start)
    end = parse_taipei_timestamp(args.end)

    if qff_start > end:
        raise ValueError("--qff-start must be before --end")
    if external_start > end:
        raise ValueError("--external-start must be before --end")

    print(f"QFF range: {qff_start} to {end}")
    qff = download_qff_tvdatafeed(
        start=qff_start,
        end=end,
        n_bars=args.qff_n_bars,
    )
    validate_frame(qff, label="QFF", start=qff_start, end=end)
    write_csv(qff, args.qff_out, f"{QFF_EXCHANGE}:{QFF_SYMBOL}")
    print_summary("QFF", qff, args.qff_out)

    print(f"External range: {external_start} to {end}")
    okx = build_exchange(ccxt.okx, args.timeout_ms)
    print(f"Using OKX market: {OKX_TSM_SYMBOL} ({okx.market(OKX_TSM_SYMBOL)['id']})")
    tsm = download_ccxt_ohlcv(
        exchange=okx,
        symbol=OKX_TSM_SYMBOL,
        timeframe=FIFTEEN_MINUTE_TIMEFRAME,
        start=external_start,
        end=end,
        limit=args.ccxt_limit,
        max_retries=args.max_retries,
        label="OKX TSM",
    )
    validate_frame(tsm, label="OKX TSM", start=external_start, end=end)
    write_csv(tsm, args.tsm_out, OKX_TSM_SYMBOL)
    print_summary("OKX TSM", tsm, args.tsm_out)

    tsm_1m = download_ccxt_ohlcv(
        exchange=okx,
        symbol=OKX_TSM_SYMBOL,
        timeframe=ONE_MINUTE_TIMEFRAME,
        start=external_start,
        end=end,
        limit=args.ccxt_limit,
        max_retries=args.max_retries,
        label="OKX TSM",
    )
    validate_frame(tsm_1m, label="OKX TSM 1m", start=external_start, end=end)
    write_csv(tsm_1m, args.tsm_1m_out, OKX_TSM_SYMBOL)
    print_summary("OKX TSM 1m", tsm_1m, args.tsm_1m_out)

    bitopro = build_exchange(ccxt.bitopro, args.timeout_ms)
    print(
        "Using BitoPro market: "
        f"{BITOPRO_USDTTWD_SYMBOL} ({bitopro.market(BITOPRO_USDTTWD_SYMBOL)['id']})"
    )
    usdttwd = download_ccxt_ohlcv(
        exchange=bitopro,
        symbol=BITOPRO_USDTTWD_SYMBOL,
        timeframe=FIFTEEN_MINUTE_TIMEFRAME,
        start=external_start,
        end=end,
        limit=args.ccxt_limit,
        max_retries=args.max_retries,
        label="BitoPro USDT/TWD",
    )
    validate_frame(usdttwd, label="BitoPro USDT/TWD", start=external_start, end=end)
    write_csv(usdttwd, args.usdttwd_out, BITOPRO_USDTTWD_SYMBOL)
    print_summary("BitoPro USDT/TWD", usdttwd, args.usdttwd_out)

    usdttwd_1m = download_ccxt_ohlcv(
        exchange=bitopro,
        symbol=BITOPRO_USDTTWD_SYMBOL,
        timeframe=ONE_MINUTE_TIMEFRAME,
        start=external_start,
        end=end,
        limit=args.ccxt_limit,
        max_retries=args.max_retries,
        label="BitoPro USDT/TWD",
    )
    validate_frame(
        usdttwd_1m, label="BitoPro USDT/TWD 1m", start=external_start, end=end
    )
    write_csv(usdttwd_1m, args.usdttwd_1m_out, BITOPRO_USDTTWD_SYMBOL)
    print_summary("BitoPro USDT/TWD 1m", usdttwd_1m, args.usdttwd_1m_out)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
