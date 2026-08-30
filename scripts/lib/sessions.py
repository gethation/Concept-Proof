"""Session grids and the weekend trading rules.

Two market structures, two ways of building the index a spread lives on:

``taifex_grid``  TAIFEX runs a day session and an overnight session, so the
                 index is synthesised from the clock and the legs are required
                 to cover it completely.
``us_rth``       NYSE RTH is the binding window for an ADR pair, so the index is
                 the US leg's own bars and the TAIFEX leg is aligned as-of onto
                 it. TAIFEX trades far longer, so it is the one that bends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# TAIFEX equity-futures clock, Taipei local minutes past midnight.
DAY_START_MINUTE = 8 * 60 + 45
DAY_END_MINUTE = 13 * 60 + 45
NIGHT_START_MINUTE = 17 * 60 + 25
NIGHT_END_MINUTE = 5 * 60

WEEKEND_POLICIES = ("flat", "no-entry", "none")


def us_session_day(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Group a 21:30->04:00 (or 22:30->05:00) Taipei session onto one key."""
    return (timestamps - pd.Timedelta(hours=12)).normalize()


def build_taifex_session_index(
    close: pd.Series, interval_minutes: int = 1
) -> pd.DatetimeIndex:
    """Every TAIFEX session bar-start spanned by a close series.

    Built from the clock rather than from the observed bars, so a bar with no
    trade still gets a row (and is later forward-filled and flagged) instead of
    silently vanishing from the index.

    TAIFEX sessions do not start on the hour, so a coarser grid is anchored to
    the session open rather than to midnight: the day session runs 08:45 and the
    night session 17:25, which puts a 15-minute night grid on :25/:40/:55/:10
    and the day grid on :45/:00/:15/:30. Anchoring to midnight instead would
    miss every night bar.

    The closing bar differs by interval, and deliberately. At 1m the session's
    final minute is kept, because every stored 1m series has it and dropping it
    would move the spread. Above 1m a bar is labelled by its START, so one
    starting AT the close would cover only time after the session -- the day
    session's 13:45 does not exist as a 15m bar, while the night session's 04:55
    does, because it still holds five minutes of session.
    """
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")
    freq = f"{interval_minutes}min"
    last_offset = pd.Timedelta(minutes=0 if interval_minutes == 1 else 1)
    timestamps = close.index
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
        ranges.append(pd.date_range(start, end - last_offset, freq=freq))

    for session_day in sorted(set(night_session_start.loc[night_clock])):
        start = session_day + pd.Timedelta(minutes=NIGHT_START_MINUTE)
        end = session_day + pd.Timedelta(days=1, minutes=NIGHT_END_MINUTE)
        ranges.append(pd.date_range(start, end - last_offset, freq=freq))

    if not ranges:
        raise RuntimeError(
            "QFF data does not contain any recognized trading-session bars"
        )

    session_index = ranges[0].append(ranges[1:]).sort_values().unique()
    session_index = pd.DatetimeIndex(session_index)
    return session_index[
        (session_index >= close.index[0]) & (session_index <= close.index[-1])
    ]


def weekend_masks(
    timestamps: pd.DatetimeIndex, *, policy: str = "flat"
) -> dict[str, object]:
    """Trading masks for a US-RTH session index.

    Every row is a tradable minute; ``policy`` decides what happens in the last
    session of each ISO week:

      flat      no entries in it, and force-close on its final bar
      no-entry  keep the entry ban, drop the force-close
      none      neither rule

    The rule is inherited from QFF/TSM, where Binance trades 24/7 while QFF is
    frozen over the weekend, leaving an uncovered leg. A TAIFEX/NYSE pair has no
    such exposure -- both shut -- so it can be dropped, at the cost of carrying
    weekend gap risk on a position that stays fully hedged.
    """
    if policy not in WEEKEND_POLICIES:
        raise ValueError(f"unknown weekend_policy: {policy!r}")

    n = len(timestamps)
    session_key = us_session_day(timestamps)

    week_end_bar = np.zeros(n, dtype=bool)
    iso = timestamps.isocalendar()
    week_key = list(zip(iso.year, iso.week))
    for i in range(n - 1):
        if week_key[i] != week_key[i + 1]:
            week_end_bar[i] = True
    marked = set(session_key[week_end_bar])
    week_end_session = np.isin(session_key, list(marked))

    force_close = week_end_bar if policy == "flat" else np.zeros(n, dtype=bool)
    close_only = (
        week_end_session if policy in {"flat", "no-entry"} else np.zeros(n, dtype=bool)
    )

    return {
        "session_day": session_key,
        "close_allowed": True,
        "entry_allowed": ~close_only,
        "friday_night_close_only": False,
        "weekend_session_close_only": close_only,
        "friday_session_end_force_close": force_close,
    }
