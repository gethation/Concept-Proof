from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tvDatafeed import Interval, TvDatafeed


DEFAULT_SYMBOLS = [
    ("BINANCE", "TSNUSDT.P"),
    ("BINANCE", "TSMUSDT.P"),
    ("TAIFEX", "QFF1!"),
]

INTERVALS = {
    "1m": Interval.in_1_minute,
    "3m": Interval.in_3_minute,
    "5m": Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "30m": Interval.in_30_minute,
    "45m": Interval.in_45_minute,
    "1h": Interval.in_1_hour,
    "2h": Interval.in_2_hour,
    "3h": Interval.in_3_hour,
    "4h": Interval.in_4_hour,
    "1d": Interval.in_daily,
    "1w": Interval.in_weekly,
    "1mo": Interval.in_monthly,
}


@dataclass
class AttemptResult:
    exchange: str
    symbol: str
    attempt: int
    ok: bool
    seconds: float
    rows: int = 0
    first_index: str | None = None
    last_index: str | None = None
    first_close: float | None = None
    last_close: float | None = None
    error_type: str | None = None
    error: str | None = None


def parse_symbol(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("symbol must be in EXCHANGE:SYMBOL format")
    exchange, symbol = value.split(":", 1)
    return exchange.strip().upper(), symbol.strip().upper()


def frame_summary(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None:
        raise RuntimeError("tvdatafeed returned None")
    if frame.empty:
        raise RuntimeError("tvdatafeed returned an empty DataFrame")

    data = frame.sort_index()
    first = data.iloc[0]
    last = data.iloc[-1]
    return {
        "rows": int(len(data)),
        "first_index": str(data.index[0]),
        "last_index": str(data.index[-1]),
        "first_close": float(first["close"]),
        "last_close": float(last["close"]),
    }


def run_attempt(
    tv: TvDatafeed,
    exchange: str,
    symbol: str,
    attempt: int,
    interval: Interval,
    n_bars: int,
) -> AttemptResult:
    started = time.perf_counter()
    try:
        frame = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=n_bars,
        )
        summary = frame_summary(frame)
        return AttemptResult(
            exchange=exchange,
            symbol=symbol,
            attempt=attempt,
            ok=True,
            seconds=round(time.perf_counter() - started, 3),
            **summary,
        )
    except Exception as exc:
        return AttemptResult(
            exchange=exchange,
            symbol=symbol,
            attempt=attempt,
            ok=False,
            seconds=round(time.perf_counter() - started, 3),
            error_type=type(exc).__name__,
            error=str(exc),
        )


def summarize(results: list[AttemptResult]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(item) for item in results],
        "summary": {},
    }
    for key in sorted({(item.exchange, item.symbol) for item in results}):
        exchange, symbol = key
        items = [item for item in results if (item.exchange, item.symbol) == key]
        successes = [item for item in items if item.ok]
        failures = [item for item in items if not item.ok]
        output["summary"][f"{exchange}:{symbol}"] = {
            "attempts": len(items),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": len(successes) / len(items) if items else 0,
            "median_seconds": (
                round(statistics.median(item.seconds for item in successes), 3)
                if successes
                else None
            ),
            "rows_each_success": [item.rows for item in successes],
            "last_success_range": (
                {
                    "first_index": successes[-1].first_index,
                    "last_index": successes[-1].last_index,
                    "last_close": successes[-1].last_close,
                }
                if successes
                else None
            ),
            "errors": [
                {
                    "attempt": item.attempt,
                    "error_type": item.error_type,
                    "error": item.error,
                }
                for item in failures
            ],
        }
    return output


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke/stability test TradingView downloads through tvdatafeed."
    )
    parser.add_argument(
        "--symbol",
        action="append",
        type=parse_symbol,
        help="Symbol in EXCHANGE:SYMBOL format. Can be passed multiple times.",
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--interval", choices=sorted(INTERVALS), default="1m")
    parser.add_argument("--n-bars", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/tvdatafeed_symbol_test.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.n_bars <= 0:
        raise ValueError("--n-bars must be positive")

    symbols = args.symbol or DEFAULT_SYMBOLS
    interval = INTERVALS[args.interval]
    tv = TvDatafeed()
    results: list[AttemptResult] = []

    for attempt in range(1, args.attempts + 1):
        for exchange, symbol in symbols:
            result = run_attempt(tv, exchange, symbol, attempt, interval, args.n_bars)
            results.append(result)
            status = "OK" if result.ok else "FAIL"
            detail = (
                f"rows={result.rows} range={result.first_index} -> {result.last_index}"
                if result.ok
                else f"{result.error_type}: {result.error}"
            )
            print(
                f"[{status}] {exchange}:{symbol} attempt={attempt} "
                f"interval={args.interval} seconds={result.seconds} {detail}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    output = summarize(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    return 0 if all(item.ok for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
