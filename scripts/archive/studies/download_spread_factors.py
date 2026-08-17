from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd
from tvDatafeed import Interval, TvDatafeed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ingest.qff_tsm_15m import (  # noqa: E402
    FIFTEEN_MINUTE_TIMEFRAME,
    build_exchange,
    download_ccxt_ohlcv,
    print_summary,
)
from lib.barsio import (  # noqa: E402
    normalize_ohlcv_frame,
    validate_frame,
    write_bars_csv as write_csv,
)
from lib.timeutil import TAIPEI_TZ  # noqa: E402
from lib.timeutil import parse_taipei as parse_taipei_timestamp  # noqa: E402

DEFAULT_START = "2026-03-12 00:00:00+08:00"

PERP_SYMBOL = "TSM/USDT:USDT"
PERP_EXCHANGES = {
    "binance": ccxt.binance,
    "okx": ccxt.okx,
    "bybit": ccxt.bybit,
}

# expected Taipei bar-start hours per TradingView symbol (sanity check, warn-only)
TV_SYMBOLS = {
    # (exchange, symbol, output stem, expected bar-start hours in Taipei)
    "nyse_tsm": ("NYSE", "TSM", set(range(21, 24)) | set(range(0, 5))),
    "twse_2330": ("TWSE", "2330", set(range(9, 14))),
    "taifex_cdf1": ("TAIFEX", "CDF1!", set(range(8, 14)) | set(range(15, 24)) | set(range(0, 5))),
    "taifex_qff1": ("TAIFEX", "QFF1!", set(range(8, 14)) | set(range(15, 24)) | set(range(0, 5))),
}

DEFAULT_OUT_DIR = Path("data/processed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the spread-factor dataset: TSM perp 15m OHLCV from "
            "Binance/OKX/Bybit (ccxt, volume in base units, aggregated), plus "
            "NYSE:TSM, TWSE:2330, TAIFEX:CDF1!, TAIFEX:QFF1! via tvdatafeed."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None, help="default: now (Taipei)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tv-n-bars", type=int, default=10_000)
    parser.add_argument("--tv-max-retries", type=int, default=3)
    parser.add_argument("--ccxt-limit", type=int, default=1000)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--skip-perps", action="store_true")
    parser.add_argument("--skip-tv", action="store_true")
    return parser.parse_args(argv)


def download_tv_ohlcv(
    tv: TvDatafeed,
    *,
    exchange: str,
    symbol: str,
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
                interval=Interval.in_15_minute,
                n_bars=n_bars,
            )
            if frame is not None and not frame.empty:
                break
        except Exception as exc:  # tvdatafeed raises bare Exceptions on timeouts
            last_error = exc
        time.sleep(2**attempt)
    if frame is None or frame.empty:
        raise RuntimeError(
            f"tvdatafeed returned no rows for {exchange}:{symbol}"
        ) from last_error

    data = frame.sort_index().reset_index()
    data = data.rename(columns={data.columns[0]: "timestamp"})
    timestamps = pd.to_datetime(data["timestamp"])
    # tvdatafeed returns tz-naive Taipei-local timestamps (verified for
    # TAIFEX/TWSE/NYSE symbols against their known session hours)
    if timestamps.dt.tz is None:
        data["timestamp"] = timestamps.dt.tz_localize(TAIPEI_TZ)
    else:
        data["timestamp"] = timestamps.dt.tz_convert(TAIPEI_TZ)

    data = data[(data["timestamp"] >= start) & (data["timestamp"] <= end)].copy()
    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
    return normalize_ohlcv_frame(data)


def check_session_hours(frame: pd.DataFrame, expected_hours: set[int], label: str) -> None:
    observed = set(pd.DatetimeIndex(frame["timestamp"]).hour.unique())
    unexpected = observed - expected_hours
    if unexpected:
        print(
            f"WARNING: {label} has bars outside expected session hours: "
            f"{sorted(unexpected)} (expected within {sorted(expected_hours)}); "
            f"check the timezone assumption before using this file"
        )


def aggregate_perps(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name, frame in frames.items():
        part = frame[["timestamp", "close", "volume"]].rename(
            columns={"close": f"{name}_close", "volume": f"{name}_volume"}
        )
        merged = part if merged is None else merged.merge(part, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    close_cols = [f"{name}_close" for name in frames]
    vol_cols = [f"{name}_volume" for name in frames]
    volumes = merged[vol_cols].fillna(0.0)
    closes = merged[close_cols]

    merged["agg_volume"] = volumes.sum(axis=1)
    merged["agg_notional_usdt"] = (volumes.values * closes.fillna(0.0).values).sum(axis=1)
    weighted = merged["agg_notional_usdt"] / merged["agg_volume"]
    merged["agg_vwap_close"] = weighted.where(merged["agg_volume"] > 0, closes.mean(axis=1))
    return merged


def validate_aggregate(agg: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> None:
    if agg["timestamp"].duplicated().any():
        raise RuntimeError("aggregate has duplicate timestamps")
    total_by_exchange = {name: frame["volume"].sum() for name, frame in frames.items()}
    if abs(agg["agg_volume"].sum() - sum(total_by_exchange.values())) > 1e-6 * max(
        sum(total_by_exchange.values()), 1.0
    ):
        raise RuntimeError("aggregate volume does not match sum of per-exchange volumes")
    # per-exchange closes should agree closely where both traded
    pairs = [("binance", "okx"), ("binance", "bybit"), ("okx", "bybit")]
    for a, b in pairs:
        both = agg.dropna(subset=[f"{a}_close", f"{b}_close"])
        if len(both) < 100:
            continue
        corr = both[f"{a}_close"].corr(both[f"{b}_close"])
        if corr < 0.999:
            print(f"WARNING: {a} vs {b} close correlation only {corr:.5f}")
    share = {
        name: total / sum(total_by_exchange.values())
        for name, total in total_by_exchange.items()
    }
    print(
        "perp volume share: "
        + "  ".join(f"{name}={value:.1%}" for name, value in share.items())
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

    if not args.skip_perps:
        perp_frames: dict[str, pd.DataFrame] = {}
        for name, factory in PERP_EXCHANGES.items():
            exchange = build_exchange(factory, args.timeout_ms)
            label = f"{name} TSM perp"
            market = exchange.market(PERP_SYMBOL)
            if market.get("contractSize") not in (None, 1, 1.0):
                raise RuntimeError(
                    f"{label} contractSize={market.get('contractSize')} != 1; "
                    "volume units are not base-token comparable"
                )
            frame = download_ccxt_ohlcv(
                exchange=exchange,
                symbol=PERP_SYMBOL,
                timeframe=FIFTEEN_MINUTE_TIMEFRAME,
                start=start,
                end=end,
                limit=args.ccxt_limit,
                max_retries=args.max_retries,
                label=label,
            )
            validate_frame(frame, label=label, start=start, end=end)
            out_path = out_dir / f"factor_{name}_tsmusdtp_15m_taipei.csv"
            write_csv(frame, out_path, PERP_SYMBOL)
            print_summary(label, frame, out_path)
            perp_frames[name] = frame

        agg = aggregate_perps(perp_frames)
        validate_aggregate(agg, perp_frames)
        agg_path = out_dir / "factor_tsmusdtp_agg_15m_taipei.csv"
        agg_out = agg.copy()
        agg_out["timestamp"] = agg_out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        agg_out["timestamp"] = agg_out["timestamp"].str.replace(
            r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True
        )
        agg_out.to_csv(agg_path, index=False)
        print(f"perp aggregate: wrote {len(agg):,} rows to {agg_path}")

    if not args.skip_tv:
        tv = TvDatafeed()
        for stem, (tv_exchange, tv_symbol, expected_hours) in TV_SYMBOLS.items():
            label = f"{tv_exchange}:{tv_symbol}"
            frame = download_tv_ohlcv(
                tv,
                exchange=tv_exchange,
                symbol=tv_symbol,
                n_bars=args.tv_n_bars,
                start=start,
                end=end,
                max_retries=args.tv_max_retries,
            )
            validate_frame(frame, label=label, start=start, end=end)
            check_session_hours(frame, expected_hours, label)
            out_path = out_dir / f"factor_{stem}_15m_taipei.csv"
            write_csv(frame, out_path, label)
            print_summary(label, frame, out_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
