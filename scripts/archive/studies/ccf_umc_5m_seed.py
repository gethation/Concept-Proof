from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_15M_SPREAD = Path("data/processed/ccf_umc_spread_umc_session_15m.csv")
DEFAULT_5M_SPREAD = Path("data/processed/ccf_umc_spread_umc_session_5m.csv")
DEFAULT_OUT = Path("data/processed/ccf_umc_spread_5m_seed.csv")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pseudo-5m seed series for long-window rolling z-scores on "
            "the 5m CCF-UMC spread. Each 15m spread observation before the 5m "
            "sample start is repeated 3x at +0/+5/+10 minute offsets. The seed "
            "rows only feed the rolling mean/std warmup (via "
            "--seed-spread-path); they are never traded and never appear in "
            "backtest output."
        )
    )
    parser.add_argument("--spread-15m", type=Path, default=DEFAULT_15M_SPREAD)
    parser.add_argument("--spread-5m", type=Path, default=DEFAULT_5M_SPREAD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--offset-minutes", type=int, default=5)
    return parser.parse_args(argv)


def read_spread(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")
    frame = pd.read_csv(path)
    missing = {"timestamp", "spread"}.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce")
    if frame["spread"].isna().any():
        raise RuntimeError(f"{label} has invalid spread values")
    return frame.sort_values("timestamp").reset_index(drop=True)


def format_taipei_timestamps(timestamps: pd.Series) -> pd.Series:
    formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")

    spread_15m = read_spread(args.spread_15m, "15m spread")
    spread_5m = read_spread(args.spread_5m, "5m spread")
    real_start = spread_5m["timestamp"].iloc[0]

    source = spread_15m[spread_15m["timestamp"] < real_start].copy()
    if source.empty:
        raise RuntimeError("No 15m spread rows before the 5m sample start")

    rows = []
    for offset_index in range(args.repeat):
        part = source[["timestamp", "spread"]].copy()
        part["source_timestamp"] = part["timestamp"]
        part["timestamp"] = part["timestamp"] + pd.Timedelta(
            minutes=args.offset_minutes * offset_index
        )
        rows.append(part)
    seed = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    seed = seed[seed["timestamp"] < real_start].reset_index(drop=True)

    timestamps = pd.DatetimeIndex(seed["timestamp"])
    if timestamps.has_duplicates:
        raise RuntimeError("Seed timestamps are not unique")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Seed timestamps are not sorted")
    if timestamps[-1] >= real_start:
        raise RuntimeError("Seed rows must all precede the 5m sample start")

    output = seed.copy()
    output["is_seed"] = True
    output["timestamp"] = format_taipei_timestamps(output["timestamp"])
    output["source_timestamp"] = format_taipei_timestamps(output["source_timestamp"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)

    print(
        f"Wrote {len(seed):,} pseudo-5m seed rows to {args.out} "
        f"({seed['timestamp'].iloc[0]} -> {seed['timestamp'].iloc[-1]}; "
        f"5m sample starts {real_start})"
    )
    print(
        f"Seed spread: mean {seed['spread'].mean():.4f}, "
        f"std {seed['spread'].std():.4f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
