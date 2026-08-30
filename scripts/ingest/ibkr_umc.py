"""UMC 1-minute history from Interactive Brokers, paged backwards.

WHY THIS EXISTS. The UMC archive is currently filled by tvdatafeed, whose
anonymous endpoint serves a rolling ~19 days at 1m. That is enough to keep the
series alive with a scheduled run and not enough to ever deepen it: every day
the feed drops is gone unless something captured it. IBKR serves years of
1-minute bars for a US listing, so this is the one leg of this pair whose
history can actually be extended backwards rather than only forward.

It does not fix the pair. CCF is the binding leg -- TAIFEX publishes 30 trading
days of time-and-sales and no quotes at all -- so a deeper UMC series widens no
spread on its own. What it buys is independence from a rolling window, and the
volume column the tvdatafeed splice cannot be trusted on (measured at roughly a
tenth of IBKR's).

PACING IS AN ACCOUNT-WIDE RESOURCE, AND THAT IS THE WHOLE SAFETY STORY HERE.
IBKR meters historical requests per account, not per client id, so a bulk
backfill run against the same account as a live trading session can throttle
that session's market data while it holds a position. This module therefore
refuses to start while Project Lux reports a live run in progress, and paces
itself well inside the published limit even when it does run. --force exists,
and using it during a US session is a decision to slow the live system down.

WHAT-TO-SHOW IS A PRICE DEFINITION, NOT A FLAG. TRADES and MIDPOINT are
different series. Every stored UMC bar, the spread built on it, and the z-score
fitted to that spread are all traded prices; merging MIDPOINT bars into the same
file would redefine the history underneath a strategy fitted on trades, and the
run would look healthy while doing it. MIDPOINT is offered because the mid basis
is worth having -- see scripts/ingest/lux_quotes.py -- but it has no default
output path and will not write to the canonical file.

    python scripts/ingest/ibkr_umc.py --duration-weeks 52
    python scripts/ingest/ibkr_umc.py --what-to-show MIDPOINT --out data/bars/nyse/umc_1m_mid.csv
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths  # noqa: E402
from lib.timeutil import TAIPEI_TZ  # noqa: E402

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Project Lux holds 17002. Anything else avoids evicting its connection.
DEFAULT_CLIENT_ID = 17311
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4001

LUX_STORE = Path(
    r"C:\Users\huang\workplace\Project-Lux\data\live_ccf_umc_execute.sqlite3"
)

# IBKR allows 60 historical requests per 10 minutes. One every 11 seconds sits
# inside that with room for a retry, and a week of 1-minute bars per request
# means a year costs about ten minutes rather than an hour.
SECONDS_BETWEEN_REQUESTS = 11.0
CHUNK = "1 W"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--duration-weeks", type=int, default=52)
    parser.add_argument(
        "--what-to-show", choices=["TRADES", "MIDPOINT"], default="TRADES"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even while a Project Lux live session is active. This shares "
             "the account's historical-request budget with it.",
    )
    parser.add_argument("--lux-store", type=Path, default=LUX_STORE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and report the diff without writing.")
    return parser.parse_args(argv)


def live_run_active(store: Path, *, stale_minutes: float = 5.0) -> str | None:
    """Describe an in-progress Lux run, or None if nothing is trading.

    Read-only, and a missing or unreadable store is not an error: this is a
    courtesy check against a system that may not be installed, not a dependency.
    """
    if not store.exists():
        return None
    try:
        uri = f"file:{store.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute(
                "SELECT run_id, status FROM live_runs "
                "WHERE finished_at IS NULL ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            last_tick = conn.execute(
                "SELECT MAX(observed_at) FROM market_ticks"
            ).fetchone()[0]
    except sqlite3.Error:
        return None
    if not last_tick:
        return None
    age = (
        pd.Timestamp.now(tz=TAIPEI_TZ) - pd.Timestamp(last_tick)
    ).total_seconds() / 60.0
    if age > stale_minutes:
        return None
    return f"run {row[0]} ({row[1]}), last quote {age:.1f} minutes ago"


def fetch(args: argparse.Namespace) -> pd.DataFrame:
    from ib_async import IB, Stock

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=30)
    try:
        contract = Stock("UMC", "NYSE", "USD")
        ib.qualifyContracts(contract)
        pieces: list[pd.DataFrame] = []
        end = ""
        for i in range(args.duration_weeks):
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end,
                durationStr=CHUNK,
                barSizeSetting="1 min",
                whatToShow=args.what_to_show,
                useRTH=True,
                # formatDate=2 returns epoch seconds. Any other value hands back
                # whatever timezone the Gateway login screen is set to, which
                # would silently redefine every stored bar.
                formatDate=2,
                timeout=90,
            )
            if not bars:
                print(f"  chunk {i + 1}: empty, feed exhausted")
                break
            frame = pd.DataFrame(
                {
                    "timestamp": [b.date for b in bars],
                    "open": [float(b.open) for b in bars],
                    "high": [float(b.high) for b in bars],
                    "low": [float(b.low) for b in bars],
                    "close": [float(b.close) for b in bars],
                    "volume": [float(b.volume) for b in bars],
                }
            )
            frame["timestamp"] = pd.to_datetime(
                frame["timestamp"], utc=True
            ).dt.tz_convert(TAIPEI_TZ)
            pieces.append(frame)
            first = frame["timestamp"].min()
            print(
                f"  chunk {i + 1}/{args.duration_weeks}: {len(frame):,} bars back "
                f"to {first}"
            )
            end = first.tz_convert("UTC").strftime("%Y%m%d-%H:%M:%S")
            if i + 1 < args.duration_weeks:
                time.sleep(SECONDS_BETWEEN_REQUESTS)
    finally:
        ib.disconnect()

    if not pieces:
        raise RuntimeError("IBKR returned no bars")
    out = pd.concat(pieces, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out.sort_values("timestamp")[COLUMNS].reset_index(drop=True)


def format_timestamps(series: pd.Series) -> pd.Series:
    return (
        series.dt.strftime("%Y-%m-%d %H:%M:%S%z")
        .str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)
    )


def report_overlap(existing: pd.DataFrame, fresh: pd.DataFrame) -> None:
    """How far the incoming source disagrees with what is already stored.

    The overlap is a splice seam between two vendors, and the merge below lets
    the fresh rows win. That is the right default -- IBKR is the executing
    broker -- but it silently rewrites history the current backtest results were
    computed on, so the size of the rewrite gets printed rather than assumed.
    """
    merged = existing.merge(fresh, on="timestamp", suffixes=("_old", "_new"))
    if merged.empty:
        print("  no overlap with the stored series")
        return
    diff = (merged["close_old"] - merged["close_new"]).abs()
    changed = int((diff > 1e-9).sum())
    print(
        f"  overlap {len(merged):,} minutes: {changed:,} closes differ "
        f"({changed / len(merged) * 100:.1f}%), median {diff.median():.4f}, "
        f"max {diff.max():.4f}"
    )
    vol = merged["volume_new"].sum()
    if vol > 0:
        print(
            f"  volume ratio stored/fresh: "
            f"{merged['volume_old'].sum() / vol:.2f}x"
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.what_to_show == "MIDPOINT" and args.out is None:
        print(
            "ERROR: MIDPOINT needs an explicit --out. It is a different price "
            "definition from the traded-price series in "
            f"{paths.UMC_1M}, and merging the two would redefine the history "
            "the spread and z-score are fitted on.",
            file=sys.stderr,
        )
        return 2
    out_path = args.out or paths.UMC_1M

    active = live_run_active(args.lux_store)
    if active and not args.force:
        print(
            f"ERROR: Project Lux is trading right now -- {active}.\n"
            "IBKR meters historical requests per ACCOUNT, so a backfill now "
            "competes with the live session's market data. Wait for the US "
            "session to close, or pass --force to accept that cost.",
            file=sys.stderr,
        )
        return 1
    if active:
        print(f"WARNING: --force with a live run in progress -- {active}")

    print(f"Fetching UMC 1m {args.what_to_show} from IBKR {args.host}:{args.port}")
    fresh = fetch(args)
    print(
        f"Fetched {len(fresh):,} bars: {fresh['timestamp'].min()} -> "
        f"{fresh['timestamp'].max()}"
    )

    existing = pd.DataFrame(columns=COLUMNS)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if list(existing.columns) != COLUMNS:
            raise RuntimeError(
                f"{out_path} has columns {list(existing.columns)}, expected {COLUMNS}"
            )
        report_overlap(existing.assign(
            timestamp=pd.to_datetime(existing["timestamp"], utc=True, format="mixed")
            .dt.tz_convert(TAIPEI_TZ)
        ), fresh)

    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    fresh_out = fresh.copy()
    fresh_out["timestamp"] = format_timestamps(fresh_out["timestamp"])
    output = fresh_out
    if not existing.empty:
        combined = pd.concat([existing, fresh_out], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        order = pd.to_datetime(combined["timestamp"], utc=True, format="mixed")
        output = combined.iloc[order.argsort().to_numpy()].reset_index(drop=True)
        print(
            f"  merged with {len(existing):,} existing rows: {len(output):,} total, "
            f"{len(output) - len(fresh_out):,} kept from disk"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"Wrote {len(output):,} rows to {out_path}")
    print(f"Range: {output['timestamp'].iloc[0]} to {output['timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
