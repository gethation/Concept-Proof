from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import taifex_ticks as build_qff1_1m  # noqa: E402
from lib import paths  # noqa: E402


TAIPEI_TZ = "Asia/Taipei"
BAR_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "contract_month",
    "trade_count",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Accumulate a TAIFEX 1m continuous front-month series across runs. "
            "TAIFEX only publishes the previous 30 trading days, so the window "
            "rolls forward and older days are lost permanently unless captured. "
            "This fetches the current window via build_qff1_1m and merges it "
            "into a cumulative CSV, so running it periodically grows history "
            "beyond what any single fetch can reach."
        )
    )
    parser.add_argument("--product", default="CCF")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--expiry-rule",
        choices=sorted(build_qff1_1m.EXPIRY_RULES),
        default=build_qff1_1m.DEFAULT_EXPIRY_RULE,
        help="Front-month convention; US index futures need third_friday.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Cumulative CSV. Default: data/bars/taifex/<product>1_1m.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/taifex_time_sales"),
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download zip files even when cached copies exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report what would change without writing the cumulative file.",
    )
    return parser.parse_args(argv)


def read_bars(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BAR_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(BAR_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"{label} is missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    return frame[BAR_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def format_timestamps(values: pd.Series) -> pd.Series:
    formatted = values.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return formatted.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def report_conflicts(existing: pd.DataFrame, fresh: pd.DataFrame) -> int:
    """Same minute present in both with different content.

    A settled trading day should re-aggregate identically, so a conflict means
    either an earlier partial capture or a front-month roll that reassigned the
    bar. Fresh data wins, but never silently.
    """
    if existing.empty or fresh.empty:
        return 0
    overlap = existing.merge(fresh, on="timestamp", suffixes=("_old", "_new"))
    if overlap.empty:
        return 0
    differs = pd.Series(False, index=overlap.index)
    for column in ("open", "high", "low", "close", "volume", "contract_month"):
        old, new = overlap[f"{column}_old"], overlap[f"{column}_new"]
        if column == "contract_month":
            differs |= old.astype(str) != new.astype(str)
        else:
            differs |= ~pd.Series(
                [
                    abs(float(a) - float(b)) < 1e-9
                    for a, b in zip(old.to_numpy(), new.to_numpy())
                ],
                index=overlap.index,
            )
    conflicts = overlap[differs]
    if conflicts.empty:
        return 0

    print(
        f"WARNING: {len(conflicts):,} overlapping minutes differ between the "
        "cumulative file and this fetch; the fresh values win. Sample:"
    )
    for _, row in conflicts.head(5).iterrows():
        print(
            f"  {row['timestamp']}  close {row['close_old']} -> {row['close_new']}"
            f"  contract {row['contract_month_old']} -> {row['contract_month_new']}"
        )
    return len(conflicts)


def session_days(frame: pd.DataFrame) -> pd.Series:
    """Group a night session onto the day it started, matching TAIFEX convention."""
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    shifted = timestamps - pd.Timedelta(hours=6)
    return pd.Series(shifted.date, index=frame.index)


def summarize(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        print(f"{label}: empty")
        return
    days = session_days(frame)
    print(
        f"{label}: {len(frame):,} bars, {days.nunique()} session days, "
        f"{frame['timestamp'].min()} -> {frame['timestamp'].max()}"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    product = args.product.strip().upper()
    out_path = args.out or paths.TAIFEX / f"{product.lower()}1_1m.csv"

    existing = read_bars(out_path, "cumulative file")
    summarize(existing, "Existing")

    with tempfile.TemporaryDirectory() as tmp_dir:
        fetch_path = Path(tmp_dir) / "fetch.csv"
        fetch_argv = [
            "--product", product,
            "--days", str(args.days),
            "--raw-dir", str(args.raw_dir),
            "--out", str(fetch_path),
            "--timeout", str(args.timeout),
            "--expiry-rule", args.expiry_rule,
        ]
        if args.refresh:
            fetch_argv.append("--refresh")

        print(f"\nFetching the current TAIFEX window for {product}...")
        exit_code = build_qff1_1m.main(fetch_argv)
        if exit_code != 0:
            print(f"ERROR: fetch failed with exit code {exit_code}", file=sys.stderr)
            return exit_code
        fresh = read_bars(fetch_path, "fetched window")

    summarize(fresh, "Fetched")
    if fresh.empty:
        print("\nNothing fetched; cumulative file left unchanged.")
        return 1

    conflicts = report_conflicts(existing, fresh)

    # Replace whole session days, not individual minutes. A per-minute union
    # only overwrites bars the fresh capture also produced, so any bar the old
    # capture had and the new one does not -- a different contract month trading
    # at different minutes, which is exactly what a wrong --expiry-rule
    # produces -- survives untouched and unreported. Since TAIFEX publishes only
    # 30 trading days, a day that ages out that way can never be repaired.
    #
    # Only days the fresh capture covers IN FULL are replaced. The oldest zip in
    # the window opens mid-session -- TAIFEX files a night session under the
    # following trade date, so the earliest session day arrives with its night
    # half only -- and replacing that day would delete the day session the fetch
    # was never going to supply.
    fresh_start = pd.DatetimeIndex(fresh["timestamp"]).min()
    refetched_days = set(session_days(fresh))
    partial_days = set(
        session_days(fresh[pd.DatetimeIndex(fresh["timestamp"]) == fresh_start])
    )
    superseded = 0
    orphaned = 0
    if not existing.empty:
        existing_days = session_days(existing)
        existing_ts = pd.DatetimeIndex(existing["timestamp"])
        replaced_mask = existing_days.isin(refetched_days) & (
            ~existing_days.isin(partial_days) | (existing_ts >= fresh_start)
        )
        superseded = int(replaced_mask.sum())
        # Bars the old capture had on a re-fetched day that the fresh capture
        # does not produce at all. Under the old per-minute union these were the
        # rows that quietly survived; now they are removed, and counted so a
        # wrong-rule capture is visible instead of silent.
        fresh_minutes = set(pd.DatetimeIndex(fresh["timestamp"]))
        replaced_rows = existing[replaced_mask]
        orphaned = int(
            (~pd.DatetimeIndex(replaced_rows["timestamp"]).isin(fresh_minutes)).sum()
        )
        existing_kept = existing[~replaced_mask]
    else:
        existing_kept = existing

    combined = pd.concat([existing_kept, fresh], ignore_index=True)
    # concat with an empty object-dtype frame (first run) degrades the column
    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"], utc=True
    ).dt.tz_convert(TAIPEI_TZ)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    timestamps = pd.DatetimeIndex(combined["timestamp"])
    if timestamps.has_duplicates:
        raise RuntimeError("Merged output has duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Merged output is not sorted")

    added = len(combined) - len(existing)
    new_days = set(session_days(combined)) - set(session_days(existing))
    if superseded:
        replaced_days = len(refetched_days & set(session_days(existing)))
        print(
            f"Replaced {superseded:,} existing bars across {replaced_days} "
            "re-fetched session day(s)"
        )
    if orphaned:
        print(
            f"WARNING: {orphaned:,} of the replaced bars had no counterpart in "
            "the fresh capture. A clean re-aggregation reproduces the same "
            "minutes, so this usually means the earlier run used a different "
            "--expiry-rule or --contract-month and stored another contract."
        )
    print()
    summarize(combined, "Merged")
    print(
        f"Added {added:,} bars and {len(new_days)} new session days"
        + (f"; {conflicts:,} minutes corrected" if conflicts else "")
    )

    if args.dry_run:
        print(f"\n--dry-run: {out_path} not written")
        return 0

    output = combined.copy()
    output["timestamp"] = format_timestamps(output["timestamp"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    if added == 0 and not conflicts:
        print(
            "NOTE: nothing new. If this repeats for several days, the fetch "
            "window may no longer overlap the cumulative file - check for a gap."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
