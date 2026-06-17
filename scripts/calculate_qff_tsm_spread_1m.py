from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_QFF_PATH = Path("data/processed/qff1_1m.csv")
DEFAULT_TSM_PATH = Path("data/processed/binance_tsmusdtp_1m_taipei.csv")
DEFAULT_USDTTWD_PATH = Path("data/processed/bitopro_usdttwd_1m_taipei.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/qff_tsm_spread_1m_taipei.csv")


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
            f"{series.name} is missing {len(missing)} minutes in the QFF range; "
            f"first missing timestamp is {missing.index[0]}"
        )


def format_taipei_timestamps(timestamps: pd.Series) -> pd.Series:
    formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def calculate_spread(
    qff_close: pd.Series, tsm_close: pd.Series, usdttwd_close: pd.Series
) -> pd.DataFrame:
    full_index = pd.date_range(qff_close.index[0], qff_close.index[-1], freq="min")

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
    expected_rows = int((timestamps[-1] - timestamps[0]).total_seconds() / 60) + 1
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, got {len(frame)}")
    if not timestamps.is_unique:
        raise RuntimeError("Output timestamps are not unique")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Output timestamps are not sorted")

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

    frame = calculate_spread(qff_close, tsm_close, usdttwd_close)
    validate_output(frame)
    write_csv(frame, args.out)

    print(f"QFF range: {frame['timestamp'].iloc[0]} to {frame['timestamp'].iloc[-1]}")
    print(f"Wrote {len(frame):,} rows to {args.out}")
    print(f"QFF forward-filled rows: {int(frame['qff_was_filled'].sum()):,}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
