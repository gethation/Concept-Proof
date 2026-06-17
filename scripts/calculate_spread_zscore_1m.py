from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_SPREAD_PATH = Path("data/processed/qff_tsm_spread_1m_taipei.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/qff_tsm_spread_zscore_1m_taipei.csv")
DEFAULT_WINDOW = 1440
FLOAT_TOLERANCE = 1e-9


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate rolling z-score for the 1m QFF-TSM spread series. "
            "The default uses a 1440-minute rolling mean/std with ddof=0."
        )
    )
    parser.add_argument("--spread-path", type=Path, default=DEFAULT_SPREAD_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    return parser.parse_args(argv)


def read_spread_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Spread CSV does not exist: {path}")

    frame = pd.read_csv(path)
    missing = {"timestamp", "spread"}.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise RuntimeError(f"Spread CSV has no rows: {path}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce")
    if frame["spread"].isna().any():
        bad_rows = frame.loc[frame["spread"].isna(), ["timestamp", "spread"]].head(10)
        raise RuntimeError(f"{path} has missing or invalid spread values:\n{bad_rows}")

    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if timestamps.has_duplicates:
        duplicates = timestamps[timestamps.duplicated()].unique()
        raise RuntimeError(
            f"{path} has duplicate timestamps, first examples: {list(duplicates[:5])}"
        )
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError(f"{path} timestamps are not sorted")

    expected_index = pd.date_range(timestamps[0], timestamps[-1], freq="min")
    if len(timestamps) != len(expected_index) or not timestamps.equals(expected_index):
        missing_minutes = expected_index.difference(timestamps)
        extra_minutes = timestamps.difference(expected_index)
        raise RuntimeError(
            "Spread CSV is not a continuous 1m series. "
            f"Missing={len(missing_minutes)}, extra={len(extra_minutes)}"
        )

    return frame


def calculate_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("--window must be greater than 1")
    if len(frame) < window:
        raise ValueError(
            f"Not enough rows for window={window}: input has only {len(frame)} rows"
        )

    output = frame.copy()
    spread = output["spread"]
    rolling = spread.rolling(window=window, min_periods=window)
    mean_col = f"spread_mean_{window}"
    std_col = f"spread_std_{window}"

    output[mean_col] = rolling.mean()
    output[std_col] = rolling.std(ddof=0)
    valid = output[mean_col].notna() & output[std_col].notna() & (output[std_col] != 0)
    output["spread_zscore"] = pd.NA
    output.loc[valid, "spread_zscore"] = (
        (output.loc[valid, "spread"] - output.loc[valid, mean_col])
        / output.loc[valid, std_col]
    )
    output["spread_zscore"] = pd.to_numeric(output["spread_zscore"], errors="coerce")
    output["zscore_valid"] = valid.to_numpy()
    return output


def validate_output(frame: pd.DataFrame, window: int) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    expected_rows = int((timestamps[-1] - timestamps[0]).total_seconds() / 60) + 1
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, got {len(frame)}")
    if not timestamps.is_unique:
        raise RuntimeError("Output timestamps are not unique")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Output timestamps are not sorted")

    if frame["spread"].isna().any():
        raise RuntimeError("Output spread contains missing values")

    mean_col = f"spread_mean_{window}"
    std_col = f"spread_std_{window}"
    expected_warmup = window - 1
    warmup = frame.iloc[:expected_warmup]
    if warmup["zscore_valid"].any() or warmup["spread_zscore"].notna().any():
        raise RuntimeError("Warmup rows should not have valid z-score values")

    expected_valid = len(frame) - expected_warmup
    actual_valid = int(frame["zscore_valid"].sum())
    zero_std_rows = int(
        (frame[std_col].notna() & (frame[std_col] == 0)).sum()
    )
    if actual_valid != expected_valid - zero_std_rows:
        raise RuntimeError(
            f"Expected {expected_valid - zero_std_rows} valid z-score rows, "
            f"got {actual_valid}"
        )

    valid_rows = frame[frame["zscore_valid"]]
    if valid_rows[["spread_zscore", mean_col, std_col]].isna().any().any():
        raise RuntimeError("Valid z-score rows contain missing z-score inputs")

    sample_positions = sorted({window - 1, len(frame) // 2, len(frame) - 1})
    spread = frame["spread"]
    for position in sample_positions:
        window_values = spread.iloc[position - window + 1 : position + 1]
        expected_mean = window_values.mean()
        expected_std = window_values.std(ddof=0)
        expected_zscore = (
            (spread.iloc[position] - expected_mean) / expected_std
            if expected_std != 0
            else pd.NA
        )

        row = frame.iloc[position]
        if abs(row[mean_col] - expected_mean) > FLOAT_TOLERANCE:
            raise RuntimeError(f"Manual rolling mean check failed at row {position}")
        if abs(row[std_col] - expected_std) > FLOAT_TOLERANCE:
            raise RuntimeError(f"Manual rolling std check failed at row {position}")
        if expected_std != 0 and abs(row["spread_zscore"] - expected_zscore) > 1e-9:
            raise RuntimeError(f"Manual z-score check failed at row {position}")


def format_taipei_timestamps(timestamps: pd.Series) -> pd.Series:
    formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["timestamp"] = format_taipei_timestamps(output["timestamp"])
    output.to_csv(output_path, index=False)


def print_summary(frame: pd.DataFrame, window: int, output_path: Path) -> None:
    valid = frame.loc[frame["zscore_valid"], "spread_zscore"]
    print(f"Wrote {len(frame):,} rows to {output_path}")
    print(f"Window: {window} minutes")
    print(f"Valid z-score rows: {len(valid):,}")
    print(f"Warmup rows: {window - 1:,}")
    print(
        "Z-score summary: "
        f"mean={valid.mean():.6f}, "
        f"std={valid.std(ddof=0):.6f}, "
        f"p5={valid.quantile(0.05):.6f}, "
        f"p95={valid.quantile(0.95):.6f}, "
        f"max_abs={valid.abs().max():.6f}"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    frame = read_spread_frame(args.spread_path)
    output = calculate_zscore(frame, args.window)
    validate_output(output, args.window)
    write_csv(output, args.out)
    print_summary(output, args.window, args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
