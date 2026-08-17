from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths  # noqa: E402
from lib.timeutil import TAIPEI_TZ  # noqa: E402

DEFAULT_SPREAD_PATH = paths.feature("qff_tsm", "spread_1m")
DEFAULT_OUTPUT_PATH = paths.feature("qff_tsm", "zscore_1m")
DEFAULT_WINDOW = 997
FLOAT_TOLERANCE = 1e-9


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate rolling z-score for the QFF-TSM spread series. "
            "The default uses one QFF trading day, 997 observations, with ddof=0."
        )
    )
    parser.add_argument("--spread-path", type=Path, default=DEFAULT_SPREAD_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--seed-spread-path",
        type=Path,
        default=None,
        help=(
            "Optional spread CSV whose rows are prepended (rolling-stat warmup "
            "only) and then dropped from the output, so every output row has a "
            "full window. Seed timestamps must all precede the main series."
        ),
    )
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

    return frame


def calculate_zscore(
    frame: pd.DataFrame, window: int, seed_frame: pd.DataFrame | None = None
) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("--window must be greater than 1")
    if seed_frame is None and len(frame) < window:
        raise ValueError(
            f"Not enough rows for window={window}: input has only {len(frame)} rows"
        )

    output = frame.copy()
    mean_col = f"spread_mean_{window}"
    std_col = f"spread_std_{window}"

    if seed_frame is None:
        spread = output["spread"]
        rolling = spread.rolling(window=window, min_periods=window)
        output[mean_col] = rolling.mean()
        output[std_col] = rolling.std(ddof=0)
    else:
        if len(seed_frame) < window - 1:
            raise ValueError(
                f"Seed has only {len(seed_frame)} rows; window={window} needs "
                f"at least {window - 1} so every output row has a full window"
            )
        if seed_frame["timestamp"].iloc[-1] >= output["timestamp"].iloc[0]:
            raise ValueError("Seed timestamps must all precede the main series")
        combined = pd.concat(
            [seed_frame["spread"], output["spread"]], ignore_index=True
        )
        rolling = combined.rolling(window=window, min_periods=window)
        output[mean_col] = rolling.mean().tail(len(output)).to_numpy()
        output[std_col] = rolling.std(ddof=0).tail(len(output)).to_numpy()
    valid = output[mean_col].notna() & output[std_col].notna() & (output[std_col] != 0)
    output["spread_zscore"] = pd.NA
    output.loc[valid, "spread_zscore"] = (
        (output.loc[valid, "spread"] - output.loc[valid, mean_col])
        / output.loc[valid, std_col]
    )
    output["spread_zscore"] = pd.to_numeric(output["spread_zscore"], errors="coerce")
    output["zscore_valid"] = valid.to_numpy()
    return output


def validate_output(
    frame: pd.DataFrame, window: int, seed_frame: pd.DataFrame | None = None
) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if not timestamps.is_unique:
        raise RuntimeError("Output timestamps are not unique")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Output timestamps are not sorted")

    if frame["spread"].isna().any():
        raise RuntimeError("Output spread contains missing values")

    mean_col = f"spread_mean_{window}"
    std_col = f"spread_std_{window}"
    expected_warmup = window - 1 if seed_frame is None else 0
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

    sample_positions = sorted({expected_warmup, len(frame) // 2, len(frame) - 1})
    if seed_frame is None:
        spread = frame["spread"]
        offset = 0
    else:
        spread = pd.concat(
            [seed_frame["spread"], frame["spread"]], ignore_index=True
        )
        offset = len(seed_frame)
    for position in sample_positions:
        combined_position = position + offset
        window_values = spread.iloc[
            combined_position - window + 1 : combined_position + 1
        ]
        expected_mean = window_values.mean()
        expected_std = window_values.std(ddof=0)
        expected_zscore = (
            (spread.iloc[combined_position] - expected_mean) / expected_std
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


def print_summary(
    frame: pd.DataFrame, window: int, output_path: Path, seeded: bool = False
) -> None:
    valid = frame.loc[frame["zscore_valid"], "spread_zscore"]
    print(f"Wrote {len(frame):,} rows to {output_path}")
    print(f"Window: {window} minutes")
    print(f"Valid z-score rows: {len(valid):,}")
    print(f"Warmup rows: {0 if seeded else window - 1:,}")
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
    seed_frame = (
        read_spread_frame(args.seed_spread_path)
        if args.seed_spread_path is not None
        else None
    )
    output = calculate_zscore(frame, args.window, seed_frame=seed_frame)
    validate_output(output, args.window, seed_frame=seed_frame)
    write_csv(output, args.out)
    print_summary(output, args.window, args.out, seeded=seed_frame is not None)
    if seed_frame is not None:
        print(
            f"Seeded with {len(seed_frame):,} rows "
            f"({seed_frame['timestamp'].iloc[0]} -> "
            f"{seed_frame['timestamp'].iloc[-1]}); no warmup rows in output"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
