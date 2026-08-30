"""Mid-price minute bars, built from Project Lux's live quote stream.

WHY THIS EXISTS. Every price this repository has ever fed the spread builder is
a LAST TRADED price. TAIFEX publishes time-and-sales with no quote columns at
all -- Date, Product Code, Contract Month, Time of Trades, Trade Price, Volume,
and nothing else -- and tvdatafeed serves OHLC bars, not a book. The live system
scores on the book MID.

For a book that is one tick wide those two definitions differ by exactly half a
tick, in whichever direction the last print happened to land. Measured across
3,807 minutes shared by both systems, 94.7% of all CCF close differences were
exactly +/-0.25 TWD, and the resulting spread noise had a standard deviation of
0.238 -- the same size as the 0.2317 executable-displacement correction the
whole cost model is built around. The backtest was correcting carefully for the
width of a book while pricing off a series displaced from its mid by the same
amount, at random sign.

Project Lux keeps every quote it ever received in ``market_ticks``. This turns
that stream into the bar schema ``data/bars/`` already uses, so the existing
spread builder reads it through --tw-path / --us-path / --fx-path and needs no
change:

    python scripts/features/spread.py --pair ccf_umc --interval 1m \
        --weekend-policy none \
        --tw-path data/bars/lux/ccf1_1m_mid.csv \
        --us-path data/bars/lux/umc_1m_mid.csv \
        --fx-path data/bars/lux/usdtwd_1m_mid.csv \
        --out data/features/ccf_umc/spread_1m_mid.csv

COVERAGE IS THE CATCH, and it is not a small one. Quotes begin 2026-08-07, when
the CCF/UMC live run started. Nothing can reconstruct a book before that date,
because no source in this project ever recorded one. A mid-basis spread
therefore cannot cover the backtest's full sample, and will not until the live
store has accumulated enough sessions to fill a rolling window plus a testable
remainder. Until then this series is for CALIBRATION -- measuring how far the
traded-price basis sits from the mid one -- not for scoring a strategy.

The store is opened read-only. A live run may hold it, and this must never be
the process that disturbs one.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths  # noqa: E402
from lib.barsio import write_bars_csv  # noqa: E402
from lib.timeutil import TAIPEI_TZ  # noqa: E402

DEFAULT_STORE = Path(
    r"C:\Users\huang\workplace\Project-Lux\data\live_ccf_umc_execute.sqlite3"
)
LUX_BARS = paths.BARS / "lux"

# (tick source, output stem, continuous-series label, has a two-sided book)
SOURCES = [
    ("fubon_ccf", "ccf1_1m_mid", "TAIFEX:CCF1!", True),
    ("ibkr_umc", "umc_1m_mid", "NYSE:UMC", True),
    ("twelvedata", "usdtwd_1m_mid", "TWELVEDATA:USDTWD", False),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Project Lux's market_ticks into mid-price minute bars "
            "under data/bars/lux/. Opens the live store read-only."
        )
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out-dir", type=Path, default=LUX_BARS)
    parser.add_argument(
        "--source",
        action="append",
        choices=[s[0] for s in SOURCES],
        help="Build only these tick sources (repeatable). Default: all three.",
    )
    return parser.parse_args(argv)


def read_ticks(store: Path, source: str) -> pd.DataFrame:
    """Every quote this source ever produced, oldest first.

    ``mode=ro`` rather than a plain path: the execute store is routinely held by
    a running live loop, and a writer handle would contend with it.
    """
    if not store.exists():
        raise FileNotFoundError(f"live store does not exist: {store}")
    uri = f"file:{store.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        frame = pd.read_sql(
            "SELECT observed_at, symbol, price, bid, ask FROM market_ticks "
            "WHERE source = ? ORDER BY observed_at",
            conn,
            params=(source,),
        )
    if frame.empty:
        raise RuntimeError(f"no ticks for source {source!r} in {store}")
    frame["observed_at"] = pd.to_datetime(
        frame["observed_at"], format="ISO8601"
    ).dt.tz_convert(TAIPEI_TZ)
    return frame


def mid_series(frame: pd.DataFrame, two_sided: bool) -> pd.DataFrame:
    """Reduce a tick stream to one mid price per tick.

    A source with no book -- the FX reference rate -- has only ``price``, and
    that is already the number the spread wants. Asking it for a mid would
    silently produce NaN for every row, so the two cases are separated here
    rather than papered over with a fillna.
    """
    out = frame.copy()
    if two_sided:
        usable = out["bid"].notna() & out["ask"].notna() & (out["ask"] >= out["bid"])
        dropped = int((~usable).sum())
        if dropped:
            print(f"    dropped {dropped:,} ticks with no usable two-sided quote")
        out = out.loc[usable].copy()
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
    else:
        out = out.loc[out["price"].notna()].copy()
        out["mid"] = out["price"]
        out["bid"] = pd.NA
        out["ask"] = pd.NA
    if out.empty:
        raise RuntimeError("no usable quotes after filtering")
    return out


def to_minute_bars(ticks: pd.DataFrame, two_sided: bool) -> pd.DataFrame:
    """OHLC of the mid within each minute, labelled by the minute it starts.

    Bar-start labelling is what every other series in data/bars/ uses, so a mid
    bar and a traded bar for the same minute carry the same timestamp and the
    spread builder can align them without a special case.

    ``volume`` is 0 rather than absent: quotes carry no size, and the column has
    to exist for lib.barsio to read the file back. Nothing downstream reads the
    volume of these bars, and the file is not the place to argue about it -- but
    do not compare it across the seam with a traded-bar series.
    """
    work = ticks.copy()
    work["minute"] = work["observed_at"].dt.floor("min")
    grouped = work.groupby("minute", sort=True)
    bars = pd.DataFrame(
        {
            "open": grouped["mid"].first(),
            "high": grouped["mid"].max(),
            "low": grouped["mid"].min(),
            "close": grouped["mid"].last(),
        }
    )
    bars["volume"] = 0.0
    bars["tick_count"] = grouped.size()
    if two_sided:
        bars["bid_close"] = grouped["bid"].last()
        bars["ask_close"] = grouped["ask"].last()
    # The traded contract travels with the bar. CCF rolls mid-sample, and a
    # continuous series that does not say where the seam is cannot be checked
    # against one built under a different front-month rule.
    bars["contract"] = grouped["symbol"].last()
    bars = bars.reset_index().rename(columns={"minute": "timestamp"})
    ordered = ["timestamp", "open", "high", "low", "close", "volume", "tick_count"]
    if two_sided:
        ordered += ["bid_close", "ask_close"]
    return bars[ordered + ["contract"]]


def summarize(bars: pd.DataFrame, label: str) -> None:
    contracts = bars["contract"].nunique()
    note = f", {contracts} contracts" if contracts > 1 else ""
    print(
        f"    {len(bars):,} minute bars, {bars['timestamp'].min()} -> "
        f"{bars['timestamp'].max()}{note}"
    )
    thin = int((bars["tick_count"] < 10).sum())
    if thin:
        print(
            f"    {thin:,} bars ({thin / len(bars) * 100:.1f}%) built from fewer "
            "than 10 quotes"
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    wanted = set(args.source) if args.source else {s[0] for s in SOURCES}

    for source, stem, symbol, two_sided in SOURCES:
        if source not in wanted:
            continue
        print(f"\n{source} -> {stem}.csv")
        ticks = read_ticks(args.store, source)
        print(f"    {len(ticks):,} ticks in the store")
        bars = to_minute_bars(mid_series(ticks, two_sided), two_sided)
        summarize(bars, stem)
        written = write_bars_csv(bars, args.out_dir / f"{stem}.csv", symbol)
        print(f"    wrote {len(written):,} rows to {args.out_dir / (stem + '.csv')}")

    print(
        "\nMid bars cover only what the live store has seen. Check the range "
        "above before running a backtest against them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
