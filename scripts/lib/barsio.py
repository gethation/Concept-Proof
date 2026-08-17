"""Reading and writing bar files.

The read side enforces the invariants every downstream stage assumes: Taipei
timestamps, unique, sorted, numeric OHLCV. The write side is where
``merge_with_existing`` lives -- previously it sat inside the
download_qff_tsm_15m_data *script*, which four other downloaders imported, so
running any of them executed that module's top-level code as a side effect.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.timeutil import TAIPEI_TZ, format_taipei

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def read_ohlcv(path: Path, label: str) -> pd.DataFrame:
    """Load an OHLCV CSV as a Taipei-timestamped frame of the standard columns.

    Extra columns (``symbol`` and friends) are dropped: nothing downstream reads
    them, and keeping them made two otherwise-identical readers differ.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")
    frame = pd.read_csv(path)
    missing = set(OHLCV_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["open", "close"]].isna().any().any():
        raise RuntimeError(f"{label} has invalid open/close values")
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if timestamps.has_duplicates:
        raise RuntimeError(f"{label} has duplicate timestamps")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[OHLCV_COLUMNS]


def read_close_series(path: Path, name: str) -> pd.Series:
    """Load just the close column, indexed by Taipei timestamp.

    Used by the TAIFEX-grid spread path, which treats each leg as a bare price
    series rather than a bar frame.
    """
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


def normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce a freshly fetched frame to sorted, de-duplicated standard columns."""
    output = frame.loc[:, OHLCV_COLUMNS].copy()
    output = output.drop_duplicates(subset=["timestamp"], keep="last")
    return output.sort_values("timestamp").reset_index(drop=True)


def validate_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    """Assert a fetched frame is well-formed and inside the requested window."""
    if frame.empty:
        raise RuntimeError(f"{label} returned no rows")

    missing = set(OHLCV_COLUMNS).difference(frame.columns)
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

    numeric = ["open", "high", "low", "close", "volume"]
    invalid = frame[numeric].apply(pd.to_numeric, errors="coerce").isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        raise RuntimeError(f"{label} has invalid numeric values:\n{invalid}")


def merge_with_existing(output: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Fold a freshly fetched frame into whatever is already on disk.

    Every feed behind these downloaders serves a bounded window -- tvdatafeed's
    anonymous endpoint is a rolling ~6 weeks -- so writing the fetch straight
    out silently truncates history to whatever the source still happens to
    carry, and for an aged-out day that loss is permanent. Rows outside the
    fetched span are kept; inside it the fresh rows win, so a re-fetch still
    repairs a partial capture.

    Timestamps are compared as the formatted strings both sides were written
    with, and sorted by parsed time rather than lexically so a tz-offset change
    cannot reorder the file.
    """
    if not output_path.exists():
        return output
    existing = pd.read_csv(output_path)
    if list(existing.columns) != list(output.columns):
        raise RuntimeError(
            f"{output_path} has columns {list(existing.columns)}, fetch produced "
            f"{list(output.columns)}; refusing to merge mismatched schemas"
        )
    combined = pd.concat([existing, output], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    order = pd.to_datetime(combined["timestamp"], utc=True, format="mixed")
    combined = combined.iloc[order.argsort().to_numpy()].reset_index(drop=True)
    kept = len(combined) - len(output)
    print(
        f"  merged with {len(existing):,} existing rows: "
        f"{len(combined):,} total, {kept:,} kept from disk"
    )
    return combined


def write_bars_csv(
    frame: pd.DataFrame, output_path: Path, symbol: str, *, merge: bool = True
) -> pd.DataFrame:
    """Write a bar file with a symbol column, returning what landed on disk.

    The return value is the merged frame, not the fetched one -- callers that
    summarise the result have to report the file's contents, or the log claims
    a row count the file does not have.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output.insert(1, "symbol", symbol)
    output["timestamp"] = format_taipei(output["timestamp"])
    if merge:
        output = merge_with_existing(output, output_path)
    output.to_csv(output_path, index=False)
    return output


def write_frame_csv(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a derived frame (spread, z-score, aligned FX) with no merge step."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["timestamp"] = format_taipei(output["timestamp"])
    output.to_csv(output_path, index=False)
