"""FX_IDC USDTWD splicing and as-of alignment.

FX_IDC publishes the same rate at several intervals and each has multi-hour
outages, so the usable series is a splice: finest interval wins on a shared bar
start, coarser intervals fill the holes. A bar's close is only *known* once the
bar has ended, so alignment is done on a known-time key rather than the bar
start -- otherwise a 1h bar would leak up to an hour of hindsight into a
1-minute session row.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.barsio import read_ohlcv

# USDTWD moves so little intraday (measured: 0.17% median intra-session range,
# versus 3.4% for each equity leg) that forward-filling through a feed outage is
# safer than fragmenting sessions. Stale rows are reported, and callers may
# additionally drop whole sessions past the hard cap.
MAX_STALENESS_MINUTES = 720.0
WARN_STALENESS_MINUTES = 60.0


def build_fx_series(paths: list[tuple[int, Path]]) -> pd.DataFrame:
    """Splice FX_IDC intervals, finest first, into one known-time close series.

    ``paths`` is ``[(interval_minutes, path), ...]`` in priority order. Missing
    files are skipped so the splice degrades to whatever is on disk.
    """
    pieces = []
    for minutes, path in paths:
        if not path.exists():
            continue
        frame = read_ohlcv(path, f"FX {minutes}m")[["timestamp", "open", "close"]].copy()
        frame["fx_interval_minutes"] = minutes
        frame["known_time"] = frame["timestamp"] + pd.Timedelta(minutes=minutes)
        pieces.append(frame)
    if not pieces:
        raise RuntimeError("No FX_IDC input files found")
    fx = pd.concat(pieces, ignore_index=True)
    # finest interval wins on identical bar-start timestamps
    fx = fx.sort_values(["timestamp", "fx_interval_minutes"])
    fx = fx.drop_duplicates(subset=["timestamp"], keep="first")
    return fx.sort_values("timestamp").reset_index(drop=True)


def asof_fx(
    fx: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    interval_minutes: int,
    *,
    warn_minutes: float = WARN_STALENESS_MINUTES,
) -> pd.DataFrame:
    """FX close known by each session bar's close, and FX open as of its start.

    The close feeds the spread; the open is the entry-fill rate, because the
    backtest fills at the next bar's open rather than its close.
    """
    close_time = session_index + pd.Timedelta(minutes=interval_minutes)

    by_known = fx.sort_values("known_time")
    close_match = pd.merge_asof(
        pd.DataFrame({"session_close_time": close_time}),
        by_known[["known_time", "close"]],
        left_on="session_close_time",
        right_on="known_time",
        direction="backward",
    )
    by_start = fx.sort_values("timestamp")
    open_match = pd.merge_asof(
        pd.DataFrame({"session_start": session_index}),
        by_start[["timestamp", "open"]],
        left_on="session_start",
        right_on="timestamp",
        direction="backward",
    )
    staleness = (
        close_match["session_close_time"] - close_match["known_time"]
    ).dt.total_seconds() / 60.0

    result = pd.DataFrame(
        {
            "timestamp": session_index,
            "open": open_match["open"].to_numpy(),
            "close": close_match["close"].to_numpy(),
            "fx_close_staleness_minutes": staleness.to_numpy(),
        }
    )
    if result[["open", "close"]].isna().any().any():
        raise RuntimeError("FX series does not cover the session index")

    stale_rows = int((result["fx_close_staleness_minutes"] > warn_minutes).sum())
    if stale_rows:
        print(
            f"WARNING: {stale_rows} session bars "
            f"({stale_rows / len(result):.1%}) use FX closes more than "
            f"{warn_minutes:.0f}min old "
            f"(max {result['fx_close_staleness_minutes'].max():.0f}min) - known "
            "FX_IDC feed outages are forward-filled"
        )
    return result
