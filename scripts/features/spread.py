"""Build a pair's spread series.

    python scripts/features/spread.py --pair qff_tsm --interval 1m
    python scripts/features/spread.py --pair ccf_umc --interval 1m --weekend-policy none
    python scripts/features/spread.py --pair ccf_umc --interval 15m

Replaces calculate_qff_tsm_spread_1m.py, calculate_ccf_umc_spread_1m.py and
calculate_ccf_umc_spread.py, which were the same pipeline three times over:
read the legs, build an index, align, convert to TWD, difference, validate,
write. What actually differed between them was market structure, and that now
lives in lib.pairs as configuration instead of as three files.

The spread itself is one formula everywhere:

    fair  = us_leg_close x usdtwd / share_ratio
    spread = (fair - tw_leg_close) / (fair + tw_leg_close) x 200

Two alignment strategies decide what index that formula is evaluated on --
see lib.sessions for why there are exactly two.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import paths  # noqa: E402
from lib.barsio import read_close_series, read_ohlcv, write_frame_csv  # noqa: E402
from lib.fx import asof_fx, build_fx_series  # noqa: E402
from lib.pairs import TaifexGridSpec, UsRthSpec, get_spec  # noqa: E402
from lib.sessions import (  # noqa: E402
    build_taifex_session_index,
    us_session_day,
    weekend_masks,
)
from lib.timeutil import format_taipei, parse_taipei  # noqa: E402

SPREAD_COLUMNS = [
    "qff_close_filled",
    "tsm_close",
    "usdttwd_close",
    "tsm_twd_fair",
    "spread",
]


# --------------------------------------------------------------------------
# shared maths
# --------------------------------------------------------------------------


def fair_and_spread(
    us_close: np.ndarray, fx_close: np.ndarray, tw_close: np.ndarray, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """The TWD fair value of the US leg, and the spread against the TW leg."""
    fair = us_close * fx_close / ratio
    spread = (fair - tw_close) / (fair + tw_close) * 200
    return fair, spread


def check_spread_rows(frame: pd.DataFrame, ratio: float) -> None:
    """Recompute the formula on three rows and insist the file agrees.

    Cheap end-to-end check that the vectorised path did what the definition
    says, at the first, middle and last row.
    """
    for position in sorted({0, len(frame) // 2, len(frame) - 1}):
        row = frame.iloc[position]
        fair = row["tsm_close"] * row["usdttwd_close"] / ratio
        expected = (fair - row["qff_close_filled"]) / (fair + row["qff_close_filled"]) * 200
        if abs(fair - row["tsm_twd_fair"]) > 1e-9:
            raise RuntimeError(f"Fair-value check failed at row {position}")
        if abs(expected - row["spread"]) > 1e-9:
            raise RuntimeError(f"Spread check failed at row {position}")


def check_index(frame: pd.DataFrame, *, min_gap_minutes: int | None = None) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if not timestamps.is_unique:
        raise RuntimeError("Output timestamps are not unique")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Output timestamps are not sorted")
    if min_gap_minutes is not None:
        gap = pd.Timedelta(minutes=min_gap_minutes)
        if ((timestamps[1:] - timestamps[:-1]) < gap).any():
            raise RuntimeError(
                f"Output timestamps must be at least {min_gap_minutes} minute(s) apart"
            )


def check_non_null(frame: pd.DataFrame) -> None:
    counts = frame[SPREAD_COLUMNS].isna().sum()
    bad = counts[counts > 0]
    if not bad.empty:
        raise RuntimeError(f"Output has unexpected missing values:\n{bad}")


# --------------------------------------------------------------------------
# strategy 1: TAIFEX session grid, complete external legs
# --------------------------------------------------------------------------


def assert_external_complete(series: pd.Series, index: pd.DatetimeIndex) -> None:
    aligned = series.reindex(index)
    missing = aligned[aligned.isna()]
    if not missing.empty:
        raise RuntimeError(
            f"{series.name} is missing {len(missing)} minutes in the QFF index; "
            f"first missing timestamp is {missing.index[0]}"
        )


def build_taifex_grid(spec: TaifexGridSpec, args: argparse.Namespace) -> dict:
    tw_close = read_close_series(args.tw_path, "qff_close")
    us_close = read_close_series(args.us_path, "tsm_close")
    fx_close = read_close_series(args.fx_path, "usdttwd_close")

    if args.alignment == "continuous":
        index = pd.date_range(tw_close.index[0], tw_close.index[-1], freq="min")
    else:
        index = build_taifex_session_index(tw_close)

    assert_external_complete(us_close, index)
    assert_external_complete(fx_close, index)

    tw_aligned = tw_close.reindex(index)
    tw_filled = tw_aligned.ffill()
    if tw_filled.isna().any():
        raise RuntimeError("QFF close still has missing values after forward-fill")

    us_aligned = us_close.reindex(index)
    fx_aligned = fx_close.reindex(index)
    fair, spread = fair_and_spread(
        us_aligned.to_numpy(),
        fx_aligned.to_numpy(),
        tw_filled.to_numpy(),
        spec.share_ratio,
    )

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "qff_close": tw_aligned.to_numpy(),
            "qff_close_filled": tw_filled.to_numpy(),
            "qff_was_filled": tw_aligned.isna().to_numpy(),
            "tsm_close": us_aligned.to_numpy(),
            "usdttwd_close": fx_aligned.to_numpy(),
            "tsm_twd_fair": fair,
            "spread": spread,
        }
    )

    check_index(frame, min_gap_minutes=1)
    check_non_null(frame)
    expected_filled = frame["qff_close"].isna()
    if not frame["qff_was_filled"].astype(bool).equals(expected_filled):
        raise RuntimeError("qff_was_filled must match missing qff_close rows")
    check_spread_rows(frame, spec.share_ratio)

    return {"frame": frame, "extras": {}}


# --------------------------------------------------------------------------
# strategy 2: US RTH index, as-of TAIFEX leg, spliced FX
# --------------------------------------------------------------------------


def align_asof_close(
    tw: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """TAIFEX close as of each session bar: the last bar starting at or before it.

    On a matched grid that is an exact match with forward-fill over no-trade
    gaps; on the 15m grid TAIFEX night bars sit 5 minutes earlier than the UMC
    grid point, so it is an as-of match whose close is known before the UMC bar
    closes -- no lookahead either way.
    """
    closes = pd.Series(tw["close"].to_numpy(), index=pd.DatetimeIndex(tw["timestamp"]))
    exact = closes.reindex(index)
    union = closes.index.union(index)
    filled = closes.reindex(union).sort_index().ffill().reindex(index)

    matched = pd.merge_asof(
        pd.DataFrame({"session_start": index}),
        tw[["timestamp"]].assign(tw_start=lambda d: d["timestamp"]),
        left_on="session_start",
        right_on="timestamp",
        direction="backward",
    )["tw_start"]
    staleness = (index - pd.DatetimeIndex(matched)).total_seconds() / 60.0

    return pd.DataFrame(
        {
            "timestamp": index,
            "tw_close_exact": exact.to_numpy(),
            "tw_close_aligned": filled.to_numpy(),
            "tw_staleness_minutes": np.asarray(staleness),
        }
    )


def build_delayed_open(
    tw: pd.DataFrame, index: pd.DatetimeIndex, interval_minutes: int
) -> pd.DataFrame:
    """Honest fill prices: the open of the first TAIFEX bar at or after each
    session bar's start. Session bars with no TAIFEX bar inside them are
    omitted; the backtest then falls back to the as-of close."""
    match = pd.merge_asof(
        pd.DataFrame({"session_start": index}),
        tw[["timestamp", "open"]].rename(columns={"timestamp": "tw_start"}),
        left_on="session_start",
        right_on="tw_start",
        direction="forward",
        tolerance=pd.Timedelta(minutes=interval_minutes - 1),
    ).dropna(subset=["open"])
    return pd.DataFrame(
        {
            "timestamp": match["session_start"].to_numpy(),
            "open": match["open"].to_numpy(),
            "fill_delay_minutes": (
                (match["tw_start"] - match["session_start"]).dt.total_seconds() / 60.0
            ).to_numpy(),
        }
    )


def select_sessions(
    tw: pd.DataFrame,
    us_in_range: pd.DataFrame,
    *,
    interval_minutes: int,
    min_tw_bars: int,
) -> list:
    """Drop US sessions during which the TAIFEX leg barely traded."""
    kept, dropped = [], []
    for day, group in us_in_range.groupby("session_day"):
        first = group["timestamp"].iloc[0]
        last = group["timestamp"].iloc[-1]
        native = tw[
            (tw["timestamp"] >= first)
            & (tw["timestamp"] < last + pd.Timedelta(minutes=interval_minutes))
        ]
        if len(native) >= min_tw_bars:
            kept.append(day)
        else:
            dropped.append((day, len(group), len(native)))
    for day, us_bars, tw_bars in dropped:
        print(
            f"Excluding session {day.date()}: {us_bars} US bars but only "
            f"{tw_bars} native TAIFEX bars (TAIFEX likely closed)"
        )
    return kept


def build_us_rth(spec: UsRthSpec, args: argparse.Namespace) -> dict:
    tw = read_ohlcv(args.tw_path, "TAIFEX leg")
    us = read_ohlcv(args.us_path, "US leg")
    fx = build_fx_series(spec.fx_splice)
    interval = spec.interval_minutes

    lower = [tw["timestamp"].iloc[0], us["timestamp"].iloc[0]]
    upper = [tw["timestamp"].iloc[-1], us["timestamp"].iloc[-1]]
    if spec.range_includes_fx:
        lower.append(fx["known_time"].iloc[0])
        upper.append(fx["timestamp"].iloc[-1])
    start = parse_taipei(args.start) if args.start else max(lower)
    end = parse_taipei(args.end) if args.end else min(upper)
    print(f"Range: {start} -> {end}")

    us_in_range = us[(us["timestamp"] >= start) & (us["timestamp"] <= end)].copy()
    if us_in_range.empty:
        raise RuntimeError("No US bars inside the requested range")
    us_in_range["session_day"] = us_session_day(
        pd.DatetimeIndex(us_in_range["timestamp"])
    )

    kept = select_sessions(
        tw,
        us_in_range,
        interval_minutes=interval,
        min_tw_bars=args.min_tw_bars_per_session,
    )
    us_kept = us_in_range[us_in_range["session_day"].isin(kept)].copy()
    if us_kept.empty:
        raise RuntimeError("No sessions left after the TAIFEX-coverage filter")
    index = pd.DatetimeIndex(us_kept["timestamp"])

    if spec.fx_session_filter:
        probe = asof_fx(fx, index, interval)
        probe_days = us_session_day(pd.DatetimeIndex(probe["timestamp"]))
        worst = (
            pd.Series(probe["fx_close_staleness_minutes"].to_numpy(), index=probe_days)
            .groupby(level=0)
            .max()
        )
        bad = set(worst[worst > args.max_fx_staleness_minutes].index)
        if bad:
            for day in sorted(bad):
                print(
                    f"Excluding session {day.date()}: FX staleness "
                    f"{worst[day]:.0f}min exceeds "
                    f"{args.max_fx_staleness_minutes:.0f}min (FX_IDC feed outage)"
                )
            us_kept = us_kept[~us_kept["session_day"].isin(bad)].copy()
            index = pd.DatetimeIndex(us_kept["timestamp"])
            if index.empty:
                raise RuntimeError("No sessions left after the FX-staleness filter")

    tw_aligned = align_asof_close(tw, index)
    fx_aligned = asof_fx(fx, index, interval)

    us_close = us_kept["close"].to_numpy()
    tw_close = tw_aligned["tw_close_aligned"].to_numpy()
    fair, spread = fair_and_spread(
        us_close, fx_aligned["close"].to_numpy(), tw_close, spec.share_ratio
    )

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "qff_close": tw_aligned["tw_close_exact"].to_numpy(),
            "qff_close_filled": tw_close,
            "qff_was_filled": tw_aligned["tw_close_exact"].isna().to_numpy(),
            "ccf_staleness_minutes": tw_aligned["tw_staleness_minutes"].to_numpy(),
            "tsm_close": us_close,
            "usdttwd_close": fx_aligned["close"].to_numpy(),
            "fx_close_staleness_minutes": fx_aligned[
                "fx_close_staleness_minutes"
            ].to_numpy(),
            "tsm_twd_fair": fair,
            "spread": spread,
        }
    )
    for column, values in weekend_masks(index, policy=args.weekend_policy).items():
        frame[column] = values

    validate_us_rth(frame, spec)
    if args.weekend_policy != "flat":
        print(
            f"NOTE: weekend policy '{args.weekend_policy}' -- positions may be "
            "carried over a weekend. Both legs are shut then, so there is no "
            "uncovered leg, but weekend gap risk is real and unmodelled here."
        )

    extras = {"fx_aligned": fx_aligned, "index": index}
    if spec.write_delayed_open:
        extras["delayed_open"] = build_delayed_open(tw, index, interval)
    return {"frame": frame, "extras": extras}


def validate_us_rth(frame: pd.DataFrame, spec: UsRthSpec) -> None:
    check_index(frame)
    check_non_null(frame)
    check_spread_rows(frame, spec.share_ratio)

    if spec.validate_masks:
        if not bool(frame["close_allowed"].all()):
            raise RuntimeError("close_allowed must be True on every session row")
        expected_entry = frame["close_allowed"] & ~frame["weekend_session_close_only"]
        if not frame["entry_allowed"].equals(expected_entry):
            raise RuntimeError(
                "entry_allowed must equal close_allowed & ~weekend close-only"
            )
        if bool(frame["friday_night_close_only"].any()):
            raise RuntimeError(
                "friday_night_close_only must be all False for US sessions"
            )

        # exactly one force-close marker per ISO week transition, on that week's
        # final bar, and its whole session must be close-only
        marker = frame.loc[frame["friday_session_end_force_close"]]
        iso = pd.DatetimeIndex(frame["timestamp"]).isocalendar()
        week_key = pd.Series(list(zip(iso.year, iso.week)), index=frame.index)
        transitions = int((week_key != week_key.shift(-1)).iloc[:-1].sum())
        if len(marker) != transitions:
            raise RuntimeError(
                f"Expected {transitions} week-end force-close markers, got {len(marker)}"
            )
        if not marker["weekend_session_close_only"].all():
            raise RuntimeError("Force-close bars must sit inside close-only sessions")

    max_stale = float(frame["ccf_staleness_minutes"].max())
    if max_stale > spec.tw_staleness_max_minutes:
        raise RuntimeError(
            f"TAIFEX staleness {max_stale:.0f}min exceeds "
            f"{spec.tw_staleness_max_minutes:.0f}min - session filter is broken"
        )
    stale = int((frame["ccf_staleness_minutes"] > spec.staleness_warn).sum())
    if stale:
        print(
            f"WARNING: {stale} bars ({stale / len(frame):.1%}) use TAIFEX closes "
            f"older than {spec.staleness_warn:.0f}min (max {max_stale:.0f}min)"
        )


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------


def write_outputs(
    frame: pd.DataFrame, extras: dict, spec, out_path: Path
) -> None:
    body = frame.drop(columns=["session_day"], errors="ignore")
    write_frame_csv(body, out_path)

    fx_aligned = extras.get("fx_aligned")
    if fx_aligned is None:
        return

    fx_path = out_path.with_name(out_path.stem + "_fx.csv")
    if spec.fx_output == "ohlcv":
        # The backtest fills at the next bar's open, so it needs an open series
        # per leg on this exact index. FX is written here because it has been
        # resampled onto the session grid and exists nowhere else.
        close = fx_aligned["close"].to_numpy()
        pd.DataFrame(
            {
                "timestamp": format_taipei(pd.Series(extras["index"])),
                "open": fx_aligned["open"].to_numpy(),
                "high": close,
                "low": close,
                "close": close,
                "volume": 0.0,
            }
        ).to_csv(fx_path, index=False)
    else:
        fx_out = fx_aligned.rename(
            columns={"fx_close_staleness_minutes": "close_staleness_minutes"}
        )
        fx_out.insert(1, "symbol", "FX_IDC:USDTWD(spliced)")
        write_frame_csv(fx_out, fx_path)
    print(f"Wrote aligned FX: {fx_path}")

    delayed = extras.get("delayed_open")
    if delayed is not None:
        delayed = delayed.copy()
        delayed.insert(1, "symbol", "TAIFEX:CCF1!(delayed-open)")
        delayed_path = out_path.with_name(out_path.stem + "_delayed_open.csv")
        write_frame_csv(delayed, delayed_path)
        covered = len(delayed) / len(extras["index"])
        print(
            f"TAIFEX delayed-open fills: {len(delayed):,} rows ({covered:.1%} of "
            f"session bars; median delay "
            f"{delayed['fill_delay_minutes'].median():.0f}min) -> {delayed_path}"
        )


def summarise(frame: pd.DataFrame, out_path: Path) -> None:
    if "session_day" in frame:
        print(
            f"\nRows: {len(frame):,} over {frame['session_day'].nunique()} sessions "
            f"({frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[-1]})"
        )
    else:
        print(
            f"\nRows: {len(frame):,} "
            f"({frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[-1]})"
        )
    print(f"TAIFEX-leg forward-filled rows: {int(frame['qff_was_filled'].sum()):,}")
    if "ccf_staleness_minutes" in frame:
        print(
            f"Native-bar coverage {1 - frame['qff_was_filled'].mean():.1%}; "
            f"staleness median {frame['ccf_staleness_minutes'].median():.0f}min / "
            f"max {frame['ccf_staleness_minutes'].max():.0f}min"
        )
    print(
        f"Spread: mean {frame['spread'].mean():.4f}, std {frame['spread'].std():.4f}, "
        f"min {frame['spread'].min():.4f}, max {frame['spread'].max():.4f}"
    )
    print(f"Wrote {out_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", required=True, choices=["qff_tsm", "ccf_umc"])
    parser.add_argument("--interval", default="1m", choices=["1m", "5m", "15m"])
    parser.add_argument("--tw-path", type=Path, default=None)
    parser.add_argument("--us-path", type=Path, default=None)
    parser.add_argument("--fx-path", type=Path, default=None,
                        help="TAIFEX-grid pairs only: the FX close series.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--alignment",
        choices=["qff-session", "continuous"],
        default="qff-session",
        help="TAIFEX-grid pairs only: keep session bars, or a continuous range.",
    )
    parser.add_argument("--weekend-policy", choices=["flat", "no-entry", "none"],
                        default=None, help="US-RTH pairs only. See lib.sessions.")
    parser.add_argument("--min-tw-bars-per-session", type=int, default=None)
    parser.add_argument("--max-fx-staleness-minutes", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    spec = get_spec(args.pair, args.interval)

    args.tw_path = args.tw_path or spec.tw_leg
    args.us_path = args.us_path or spec.us_leg
    args.out = args.out or paths.feature(spec.pair, spec.default_out)

    if isinstance(spec, TaifexGridSpec):
        args.fx_path = args.fx_path or spec.fx_leg
        result = build_taifex_grid(spec, args)
    elif isinstance(spec, UsRthSpec):
        if args.weekend_policy is None:
            args.weekend_policy = spec.weekend_policy
        if args.min_tw_bars_per_session is None:
            args.min_tw_bars_per_session = spec.min_tw_bars_per_session
        if args.max_fx_staleness_minutes is None:
            from lib.fx import MAX_STALENESS_MINUTES

            args.max_fx_staleness_minutes = MAX_STALENESS_MINUTES
        result = build_us_rth(spec, args)
    else:  # pragma: no cover - the registry only holds these two
        raise RuntimeError(f"Unknown spec type: {type(spec).__name__}")

    write_outputs(result["frame"], result["extras"], spec, args.out)
    summarise(result["frame"], args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
