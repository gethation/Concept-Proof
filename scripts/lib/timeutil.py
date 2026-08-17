"""Taipei-time parsing and the on-disk timestamp format.

Every CSV this project writes carries a Taipei-local timestamp with an explicit
offset and a colon in it (``2026-08-15 04:59:00+08:00``). pandas' strftime emits
``+0800`` without the colon, so the format is a strftime plus one regex -- which
is exactly why this was copy-pasted into six files before it lived here.
"""
from __future__ import annotations

import pandas as pd

TAIPEI_TZ = "Asia/Taipei"

_OFFSET_COLON = (r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3")


def parse_taipei(value: str) -> pd.Timestamp:
    """Parse a timestamp string as Taipei time; naive input is localized."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TAIPEI_TZ)
    return timestamp.tz_convert(TAIPEI_TZ)


def format_taipei(timestamps: pd.Series) -> pd.Series:
    """Format a tz-aware Series into the project's on-disk timestamp string."""
    formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(_OFFSET_COLON[0], _OFFSET_COLON[1], regex=True)


def to_taipei_index(values) -> pd.DatetimeIndex:
    """Read arbitrary timestamp values into a Taipei-localized DatetimeIndex."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_convert(TAIPEI_TZ)
