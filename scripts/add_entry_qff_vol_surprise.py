"""Add a causal QFF volume-surprise gate column to a spread/z-score CSV.

The 2026-07-05 volume probe found that Taiwan day-session dislocations on
THIN QFF volume revert ~2x better than those on elevated volume (thin =
stale-price/microstructure noise, the best fades; elevated = informed
2330/CDF flow propagating into QFF). To gate on that *causally*:

    slot_median_t          = trailing median QFF volume of the same
                             time-of-day slot (past --slot-window sessions,
                             excluding today)
    entry_qff_vol_surprise = volume_t / slot_median_t

The column is only defined for TAIFEX day-session bars (08:45-13:45,
Mon-Fri) — the probe evidence is day-session-specific and night-session
volume dynamics differ. Elsewhere it is NaN and the backtest gate passes.
The backtest skips an entry when entry_qff_vol_surprise exceeds
--max-entry-qff-vol-surprise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAIPEI_TZ = "Asia/Taipei"
DEFAULT_INPUT = Path("data/processed/qff_tsm_spread_zscore_15m_w33.csv")
DEFAULT_QFF_OHLCV = Path("data/processed/qff1_15m_taipei_tv.csv")
DEFAULT_OUT = Path("data/processed/qff_tsm_spread_zscore_15m_w33_qv.csv")
DEFAULT_SLOT_WINDOW = 10

DAY_START_MIN = 8 * 60 + 45
DAY_END_MIN = 13 * 60 + 45


def add_qff_vol_surprise(
    frame: pd.DataFrame,
    qff_ohlcv: pd.DataFrame,
    slot_window: int,
) -> pd.DataFrame:
    out = frame.copy()
    vol = qff_ohlcv[["timestamp", "volume"]].copy()
    vol["volume"] = pd.to_numeric(vol["volume"], errors="coerce")
    vol = vol.sort_values("timestamp").reset_index(drop=True)

    ts = pd.DatetimeIndex(vol["timestamp"])
    minute_of_day = ts.hour * 60 + ts.minute
    is_day = (
        (minute_of_day >= DAY_START_MIN)
        & (minute_of_day <= DAY_END_MIN)
        & (ts.dayofweek < 5)
    )
    vol["slot"] = ts.strftime("%H:%M")
    # trailing same-slot median, excluding the current bar (shift THEN roll)
    vol["slot_median"] = vol.groupby("slot")["volume"].transform(
        lambda s: s.shift(1).rolling(slot_window, min_periods=slot_window // 2).median()
    )
    surprise = vol["volume"] / vol["slot_median"]
    vol["entry_qff_vol_surprise"] = surprise.where(is_day, np.nan)

    merged = out.merge(
        vol[["timestamp", "entry_qff_vol_surprise"]], on="timestamp", how="left"
    )
    return merged


def run_self_test() -> None:
    # 12 sessions x 4 day-session slots, constant volume 100 -> surprise ~1;
    # one spiked bar -> surprise ~3; night bars -> NaN; causality: only past
    # sessions feed the median.
    slots = ["09:00", "09:15", "09:30", "21:30"]
    rows = []
    for day in range(12):
        date = pd.Timestamp("2026-06-01", tz=TAIPEI_TZ) + pd.Timedelta(days=day)
        if date.dayofweek >= 5:
            continue
        for s in slots:
            h, m = map(int, s.split(":"))
            rows.append(
                {
                    "timestamp": date + pd.Timedelta(hours=h, minutes=m),
                    "volume": 100.0,
                }
            )
    qff = pd.DataFrame(rows)
    qff.loc[qff.index[-4], "volume"] = 300.0  # spike a 09:00 bar late in sample
    frame = pd.DataFrame({"timestamp": qff["timestamp"]})
    res = add_qff_vol_surprise(frame, qff, DEFAULT_SLOT_WINDOW)

    ts = pd.DatetimeIndex(res["timestamp"])
    night = res.loc[(ts.hour == 21), "entry_qff_vol_surprise"]
    if not night.isna().all():
        raise RuntimeError("Self-test failed: night bars should be NaN")

    spiked = res["entry_qff_vol_surprise"].iloc[-4]
    if not (2.5 < spiked < 3.5):
        raise RuntimeError(f"Self-test failed: spike surprise should be ~3, got {spiked}")

    calm = res["entry_qff_vol_surprise"].iloc[-3]
    if not (0.9 < calm < 1.1):
        raise RuntimeError(f"Self-test failed: calm surprise should be ~1, got {calm}")

    # causality: doubling a FUTURE bar's volume must not change surprise at -4
    qff_future = qff.copy()
    qff_future.loc[qff_future.index[-1], "volume"] = 10_000.0
    res_future = add_qff_vol_surprise(frame, qff_future, DEFAULT_SLOT_WINDOW)
    if res_future["entry_qff_vol_surprise"].iloc[-4] != spiked:
        raise RuntimeError("Self-test failed: future volume leaked into the surprise")

    print("Self-test passed: calm ~1, spike ~3, night NaN, future bars invisible")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--qff-ohlcv", type=Path, default=DEFAULT_QFF_OHLCV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--slot-window", type=int, default=DEFAULT_SLOT_WINDOW)
    p.add_argument("--skip-self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_self_test:
        run_self_test()
    if args.slot_window <= 1:
        raise RuntimeError("--slot-window must be > 1")

    frame = pd.read_csv(args.input)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    qff = pd.read_csv(args.qff_ohlcv)
    qff["timestamp"] = pd.to_datetime(qff["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    if "volume" not in qff.columns:
        raise RuntimeError(f"{args.qff_ohlcv} has no 'volume' column")

    out = add_qff_vol_surprise(frame, qff, args.slot_window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    valid = out["entry_qff_vol_surprise"].dropna()
    print(f"Wrote {len(out):,} rows to {args.out}")
    print(
        f"entry_qff_vol_surprise ({len(valid):,} day-session bars): "
        f"p50={valid.quantile(0.5):.2f}  p75={valid.quantile(0.75):.2f}  "
        f"p90={valid.quantile(0.9):.2f}  frac>1.5={100 * (valid > 1.5).mean():.1f}%"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
