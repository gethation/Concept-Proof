from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_QFF_PATH = Path("data/processed/qff1_1m.csv")
DEFAULT_TSM_PATH = Path("data/processed/binance_tsmusdtp_1m_taipei.csv")
DEFAULT_USDTTWD_PATH = Path("data/processed/bitopro_usdttwd_1m_taipei.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/qff_tsm_spread_1m_taipei_qff_session.csv")

DAY_START_MINUTE = 8 * 60 + 45
DAY_END_MINUTE = 13 * 60 + 45
NIGHT_START_MINUTE = 17 * 60 + 25
NIGHT_END_MINUTE = 5 * 60


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate a TradingView-style QFF-TSM 1m spread from local QFF, "
            "TSMUSDT perpetual, and USDT/TWD CSV files."
        )
    )
    parser.add_argument("--qff-path", type=Path, default=DEFAULT_QFF_PATH)
    parser.add_argument("--tsm-path", type=Path, default=DEFAULT_TSM_PATH)
    parser.add_argument("--usdttwd-path", type=Path, default=DEFAULT_USDTTWD_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--alignment",
        choices=["qff-session", "continuous"],
        default="qff-session",
        help=(
            "qff-session keeps only QFF trading sessions and forward-fills "
            "missing QFF bars inside those sessions. continuous preserves the "
            "legacy continuous 1m range."
        ),
    )
    return parser.parse_args(argv)


def read_close_series(path: Path, name: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    frame = pd.read_csv(path)
    missing = {"timestamp", "close"}.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise RuntimeError(f"Input CSV has no rows: {path}")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(TAIPEI_TZ)
    close = pd.to_numeric(frame["close"], errors="coerce")
    if close.isna().any():
        bad_rows = frame.loc[close.isna(), ["timestamp", "close"]].head(10)
        raise RuntimeError(f"{path} has invalid close values:\n{bad_rows}")

    series = pd.Series(close.to_numpy(), index=pd.DatetimeIndex(timestamps), name=name)
    if series.index.has_duplicates:
        duplicates = series.index[series.index.duplicated()].unique()
        raise RuntimeError(
            f"{path} has duplicate timestamps, first examples: {list(duplicates[:5])}"
        )

    return series.sort_index()


def assert_external_complete(series: pd.Series, full_index: pd.DatetimeIndex) -> None:
    aligned = series.reindex(full_index)
    missing = aligned[aligned.isna()]
    if not missing.empty:
        raise RuntimeError(
            f"{series.name} is missing {len(missing)} minutes in the QFF index; "
            f"first missing timestamp is {missing.index[0]}"
        )


def format_taipei_timestamps(timestamps: pd.Series) -> pd.Series:
    formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def build_qff_session_index(qff_close: pd.Series) -> pd.DatetimeIndex:
    timestamps = qff_close.index
    minute = timestamps.hour * 60 + timestamps.minute
    local_day = timestamps.normalize()

    day_clock = (minute >= DAY_START_MINUTE) & (minute <= DAY_END_MINUTE)
    night_clock = (minute >= NIGHT_START_MINUTE) | (minute <= NIGHT_END_MINUTE)

    night_session_start = pd.Series(local_day, index=timestamps)
    night_session_start.loc[minute <= NIGHT_END_MINUTE] = (
        night_session_start.loc[minute <= NIGHT_END_MINUTE] - pd.Timedelta(days=1)
    )

    ranges: list[pd.DatetimeIndex] = []
    for session_day in sorted(set(local_day[day_clock])):
        start = session_day + pd.Timedelta(minutes=DAY_START_MINUTE)
        end = session_day + pd.Timedelta(minutes=DAY_END_MINUTE)
        ranges.append(pd.date_range(start, end, freq="min"))

    for session_day in sorted(set(night_session_start.loc[night_clock])):
        start = session_day + pd.Timedelta(minutes=NIGHT_START_MINUTE)
        end = session_day + pd.Timedelta(days=1, minutes=NIGHT_END_MINUTE)
        ranges.append(pd.date_range(start, end, freq="min"))

    if not ranges:
        raise RuntimeError("QFF data does not contain any recognized trading-session bars")

    session_index = ranges[0].append(ranges[1:]).sort_values().unique()
    session_index = pd.DatetimeIndex(session_index)
    return session_index[
        (session_index >= qff_close.index[0]) & (session_index <= qff_close.index[-1])
    ]


def calculate_continuous_spread(
    qff_close: pd.Series, tsm_close: pd.Series, usdttwd_close: pd.Series
) -> pd.DataFrame:
    full_index = pd.date_range(qff_close.index[0], qff_close.index[-1], freq="min")
    return calculate_spread_for_index(qff_close, tsm_close, usdttwd_close, full_index)


def calculate_qff_session_spread(
    qff_close: pd.Series, tsm_close: pd.Series, usdttwd_close: pd.Series
) -> pd.DataFrame:
    session_index = build_qff_session_index(qff_close)
    return calculate_spread_for_index(qff_close, tsm_close, usdttwd_close, session_index)


def calculate_spread_for_index(
    qff_close: pd.Series,
    tsm_close: pd.Series,
    usdttwd_close: pd.Series,
    full_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    assert_external_complete(tsm_close, full_index)
    assert_external_complete(usdttwd_close, full_index)

    qff_aligned = qff_close.reindex(full_index)
    qff_filled = qff_aligned.ffill()
    if qff_filled.isna().any():
        raise RuntimeError("QFF close still has missing values after forward-fill")

    tsm_aligned = tsm_close.reindex(full_index)
    usdttwd_aligned = usdttwd_close.reindex(full_index)
    tsm_twd_fair = tsm_aligned * usdttwd_aligned / 5
    spread = (tsm_twd_fair - qff_filled) / (tsm_twd_fair + qff_filled) * 200

    return pd.DataFrame(
        {
            "timestamp": full_index,
            "qff_close": qff_aligned.to_numpy(),
            "qff_close_filled": qff_filled.to_numpy(),
            "qff_was_filled": qff_aligned.isna().to_numpy(),
            "tsm_close": tsm_aligned.to_numpy(),
            "usdttwd_close": usdttwd_aligned.to_numpy(),
            "tsm_twd_fair": tsm_twd_fair.to_numpy(),
            "spread": spread.to_numpy(),
        }
    )


def validate_output(frame: pd.DataFrame) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if not timestamps.is_unique:
        raise RuntimeError("Output timestamps are not unique")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Output timestamps are not sorted")
    if ((timestamps[1:] - timestamps[:-1]) < pd.Timedelta(minutes=1)).any():
        raise RuntimeError("Output timestamps must be at least 1 minute apart")

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
        raise RuntimeError(f"Output has unexpected missing values:\n{bad}")

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


def write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["timestamp"] = format_taipei_timestamps(output["timestamp"])
    output.to_csv(output_path, index=False)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    qff_close = read_close_series(args.qff_path, "qff_close")
    tsm_close = read_close_series(args.tsm_path, "tsm_close")
    usdttwd_close = read_close_series(args.usdttwd_path, "usdttwd_close")

    if args.alignment == "continuous":
        frame = calculate_continuous_spread(qff_close, tsm_close, usdttwd_close)
    else:
        frame = calculate_qff_session_spread(qff_close, tsm_close, usdttwd_close)
    validate_output(frame)
    write_csv(frame, args.out)

    print(f"QFF range: {frame['timestamp'].iloc[0]} to {frame['timestamp'].iloc[-1]}")
    print(f"Alignment: {args.alignment}")
    print(f"Wrote {len(frame):,} rows to {args.out}")
    print(f"QFF forward-filled rows: {int(frame['qff_was_filled'].sum()):,}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
