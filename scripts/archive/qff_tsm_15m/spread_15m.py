from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_QFF_PATH = Path("data/processed/qff1_15m_taipei_tv.csv")
DEFAULT_TSM_1M_PATH = Path("data/processed/okx_tsmusdtp_1m_taipei_for_15m.csv")
DEFAULT_USDTTWD_1M_PATH = Path("data/processed/bitopro_usdttwd_1m_taipei_for_15m.csv")
DEFAULT_TSM_SESSION_OUT = Path("data/processed/okx_tsmusdtp_15m_taipei_qff_session.csv")
DEFAULT_USDTTWD_SESSION_OUT = Path(
    "data/processed/bitopro_usdttwd_15m_taipei_qff_session.csv"
)
DEFAULT_SPREAD_OUT = Path("data/processed/qff_tsm_spread_15m_taipei_qff_session.csv")
DEFAULT_START = "2026-03-12 00:00:00+08:00"
DEFAULT_END = "2026-06-29 23:59:59+08:00"

BAR_MINUTES = 15
DAY_START_MINUTE = 8 * 60 + 45
DAY_END_MINUTE = 13 * 60 + 45
NIGHT_START_MINUTE = 17 * 60 + 25
NIGHT_END_MINUTE = 5 * 60


def parse_taipei_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TAIPEI_TZ)
    return timestamp.tz_convert(TAIPEI_TZ)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build QFF-session-aligned 15m OKX TSM/USDT, USDT/TWD, and "
            "QFF/TSM spread data. External markets are aggregated from 1m bars "
            "onto QFF 15m timestamps."
        )
    )
    parser.add_argument("--qff-path", type=Path, default=DEFAULT_QFF_PATH)
    parser.add_argument("--tsm-1m-path", type=Path, default=DEFAULT_TSM_1M_PATH)
    parser.add_argument("--usdttwd-1m-path", type=Path, default=DEFAULT_USDTTWD_1M_PATH)
    parser.add_argument("--tsm-session-out", type=Path, default=DEFAULT_TSM_SESSION_OUT)
    parser.add_argument(
        "--usdttwd-session-out", type=Path, default=DEFAULT_USDTTWD_SESSION_OUT
    )
    parser.add_argument("--spread-out", type=Path, default=DEFAULT_SPREAD_OUT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    return parser.parse_args(argv)


def format_taipei_timestamps(timestamps: pd.Series) -> pd.Series:
    formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def read_ohlcv_frame(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")

    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise RuntimeError(f"{label} CSV has no rows: {path}")

    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    for column in ["open", "high", "low", "close", "volume"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    invalid = output[["open", "high", "low", "close", "volume"]].isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        raise RuntimeError(f"{label} has invalid OHLCV values:\n{invalid}")

    timestamps = pd.DatetimeIndex(output["timestamp"])
    if timestamps.has_duplicates:
        duplicates = timestamps[timestamps.duplicated()].unique()
        raise RuntimeError(
            f"{label} has duplicate timestamps, first examples: {list(duplicates[:5])}"
        )
    if not timestamps.is_monotonic_increasing:
        output = output.sort_values("timestamp").reset_index(drop=True)

    return output[["timestamp", "open", "high", "low", "close", "volume"]]


def read_close_series(path: Path, label: str) -> pd.Series:
    frame = read_ohlcv_frame(path, label)
    series = pd.Series(
        frame["close"].to_numpy(),
        index=pd.DatetimeIndex(frame["timestamp"]),
        name=f"{label}_close",
    )
    return series.sort_index()


def build_qff_session_index(
    qff_close: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    timestamps = qff_close.index
    minute = timestamps.hour * 60 + timestamps.minute
    local_day = timestamps.normalize()

    day_clock = (minute >= DAY_START_MINUTE) & (minute < DAY_END_MINUTE)
    night_clock = (minute >= NIGHT_START_MINUTE) | (minute < NIGHT_END_MINUTE)

    night_session_start = pd.Series(local_day, index=timestamps)
    night_session_start.loc[minute < NIGHT_END_MINUTE] = (
        night_session_start.loc[minute < NIGHT_END_MINUTE] - pd.Timedelta(days=1)
    )

    ranges: list[pd.DatetimeIndex] = []
    for session_day in sorted(set(local_day[day_clock])):
        session_start = session_day + pd.Timedelta(minutes=DAY_START_MINUTE)
        session_end = session_day + pd.Timedelta(minutes=DAY_END_MINUTE)
        ranges.append(
            pd.date_range(
                session_start, session_end - pd.Timedelta(minutes=1), freq="15min"
            )
        )

    for session_day in sorted(set(night_session_start.loc[night_clock])):
        session_start = session_day + pd.Timedelta(minutes=NIGHT_START_MINUTE)
        session_end = session_day + pd.Timedelta(days=1, minutes=NIGHT_END_MINUTE)
        ranges.append(
            pd.date_range(
                session_start, session_end - pd.Timedelta(minutes=1), freq="15min"
            )
        )

    if not ranges:
        raise RuntimeError("QFF data does not contain any recognized 15m session bars")

    lower = max(start, qff_close.index[0])
    upper = min(end, qff_close.index[-1])
    session_index = ranges[0].append(ranges[1:]).sort_values().unique()
    session_index = pd.DatetimeIndex(session_index)
    session_index = session_index[(session_index >= lower) & (session_index <= upper)]
    if session_index.empty:
        raise RuntimeError(f"No QFF session rows inside requested range {start} -> {end}")
    return session_index


def aggregate_1m_to_qff_intervals(
    frame: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    label: str,
) -> pd.DataFrame:
    source = frame.set_index("timestamp").sort_index()
    rows: list[dict[str, object]] = []
    required = ["open", "high", "low", "close", "volume"]

    for timestamp in session_index:
        minute_index = pd.date_range(timestamp, periods=BAR_MINUTES, freq="min")
        chunk = source.reindex(minute_index)
        missing = chunk[required].isna().any(axis=1)
        if missing.any():
            missing_times = list(minute_index[missing.to_numpy()][:5])
            raise RuntimeError(
                f"{label} is missing 1m rows for QFF interval starting {timestamp}; "
                f"first missing timestamps: {missing_times}"
            )

        rows.append(
            {
                "timestamp": timestamp,
                "open": float(chunk["open"].iloc[0]),
                "high": float(chunk["high"].max()),
                "low": float(chunk["low"].min()),
                "close": float(chunk["close"].iloc[-1]),
                "volume": float(chunk["volume"].sum()),
            }
        )

    return pd.DataFrame(rows)


def build_spread_frame(
    *,
    qff_close: pd.Series,
    tsm_session: pd.DataFrame,
    usdttwd_session: pd.DataFrame,
    session_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    qff_actual = qff_close.reindex(session_index)
    fill_index = qff_close.index.union(session_index)
    qff_filled = qff_close.reindex(fill_index).sort_index().ffill().reindex(session_index)
    if qff_filled.isna().any():
        raise RuntimeError("QFF close still has missing values after forward-fill")

    tsm_close = pd.Series(
        tsm_session["close"].to_numpy(), index=session_index, name="tsm_close"
    )
    usdttwd_close = pd.Series(
        usdttwd_session["close"].to_numpy(), index=session_index, name="usdttwd_close"
    )
    tsm_twd_fair = tsm_close * usdttwd_close / 5
    spread = (tsm_twd_fair - qff_filled) / (tsm_twd_fair + qff_filled) * 200

    return pd.DataFrame(
        {
            "timestamp": session_index,
            "qff_close": qff_actual.to_numpy(),
            "qff_close_filled": qff_filled.to_numpy(),
            "qff_was_filled": qff_actual.isna().to_numpy(),
            "tsm_close": tsm_close.to_numpy(),
            "usdttwd_close": usdttwd_close.to_numpy(),
            "tsm_twd_fair": tsm_twd_fair.to_numpy(),
            "spread": spread.to_numpy(),
        }
    )


def validate_session_ohlcv(
    frame: pd.DataFrame, session_index: pd.DatetimeIndex, label: str
) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if not timestamps.equals(session_index):
        raise RuntimeError(f"{label} timestamps do not match the QFF session index")
    if frame[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise RuntimeError(f"{label} contains missing OHLCV values")


def validate_spread_output(frame: pd.DataFrame) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if not timestamps.is_unique:
        raise RuntimeError("Spread output timestamps are not unique")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Spread output timestamps are not sorted")

    required_non_null = [
        "qff_close_filled",
        "tsm_close",
        "usdttwd_close",
        "tsm_twd_fair",
        "spread",
    ]
    missing_counts = frame[required_non_null].isna().sum()
    bad = missing_counts[missing_counts > 0]
    if not bad.empty:
        raise RuntimeError(f"Spread output has unexpected missing values:\n{bad}")

    expected_filled = frame["qff_close"].isna()
    actual_filled = frame["qff_was_filled"].astype(bool)
    if not actual_filled.equals(expected_filled):
        raise RuntimeError("qff_was_filled must match missing qff_close rows")

    sample_positions = sorted({0, len(frame) // 2, len(frame) - 1})
    for position in sample_positions:
        row = frame.iloc[position]
        expected_fair = row["tsm_close"] * row["usdttwd_close"] / 5
        expected_spread = (
            (expected_fair - row["qff_close_filled"])
            / (expected_fair + row["qff_close_filled"])
            * 200
        )
        if abs(expected_fair - row["tsm_twd_fair"]) > 1e-9:
            raise RuntimeError(f"Manual fair-value check failed at row {position}")
        if abs(expected_spread - row["spread"]) > 1e-9:
            raise RuntimeError(f"Manual spread check failed at row {position}")


def write_ohlcv_csv(frame: pd.DataFrame, output_path: Path, symbol: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output.insert(1, "symbol", symbol)
    output["timestamp"] = format_taipei_timestamps(output["timestamp"])
    output.to_csv(output_path, index=False)


def write_spread_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["timestamp"] = format_taipei_timestamps(output["timestamp"])
    output.to_csv(output_path, index=False)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    start = parse_taipei_timestamp(args.start)
    end = parse_taipei_timestamp(args.end)
    if start > end:
        raise ValueError("--start must be before --end")

    qff_close = read_close_series(args.qff_path, "qff")
    tsm_1m = read_ohlcv_frame(args.tsm_1m_path, "OKX TSM 1m")
    usdttwd_1m = read_ohlcv_frame(args.usdttwd_1m_path, "BitoPro USDT/TWD 1m")
    external_first = max(tsm_1m["timestamp"].iloc[0], usdttwd_1m["timestamp"].iloc[0])
    external_last = min(tsm_1m["timestamp"].iloc[-1], usdttwd_1m["timestamp"].iloc[-1])
    effective_start = max(start, external_first)
    effective_end = min(
        end, external_last - pd.Timedelta(minutes=BAR_MINUTES - 1)
    )
    session_index = build_qff_session_index(qff_close, effective_start, effective_end)

    tsm_session = aggregate_1m_to_qff_intervals(
        tsm_1m, session_index, "OKX TSM"
    )
    usdttwd_session = aggregate_1m_to_qff_intervals(
        usdttwd_1m, session_index, "BitoPro USDT/TWD"
    )
    validate_session_ohlcv(tsm_session, session_index, "OKX TSM session")
    validate_session_ohlcv(usdttwd_session, session_index, "BitoPro USDT/TWD session")

    spread = build_spread_frame(
        qff_close=qff_close,
        tsm_session=tsm_session,
        usdttwd_session=usdttwd_session,
        session_index=session_index,
    )
    validate_spread_output(spread)

    write_ohlcv_csv(tsm_session, args.tsm_session_out, "TSM/USDT:USDT")
    write_ohlcv_csv(usdttwd_session, args.usdttwd_session_out, "USDT/TWD")
    write_spread_csv(spread, args.spread_out)

    print(f"Requested range: {start} to {end}")
    print(f"External 1m complete interval range: {effective_start} to {effective_end}")
    print(f"QFF session range: {session_index[0]} to {session_index[-1]}")
    print(f"Wrote {len(session_index):,} QFF-session 15m rows")
    print(f"QFF forward-filled rows: {int(spread['qff_was_filled'].sum()):,}")
    print(f"Wrote OKX TSM session OHLCV: {args.tsm_session_out}")
    print(f"Wrote BitoPro USDT/TWD session OHLCV: {args.usdttwd_session_out}")
    print(f"Wrote spread: {args.spread_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
