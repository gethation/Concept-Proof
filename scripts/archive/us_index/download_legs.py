from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    from ib_async import IB, Contract, Future, Index, Stock, util
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "ib_async is required: conda run -n Quant pip install ib_async"
    ) from exc


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4001  # IB Gateway live; paper is 4002, TWS is 7496/7497

# The US legs for the TAIFEX US index futures study.
#
# Matched-maturity CME futures are the preferred hedge: TAIFEX UDF/SPF/UNF and
# CBOT/CME YM/ES/NQ settle to the same index on the same third Friday, so the
# cost-of-carry term cancels in the basis and the quarterly roll happens on both
# legs at once.  The cash indices are reference only - they are not tradeable and
# their basis carries a carry term that decays to zero over the quarter.
#
# SOX has no listed future, so SXF has no matched hedge at all.  SOXX has tracked
# the ICE Semiconductor Index (not PHLX SOX) since 2021 and SMH tracks MVIS 25,
# so both carry index tracking error that does not mean-revert.
SPECS: dict[str, dict] = {
    "NQ": {"kind": "future", "symbol": "NQ", "exchange": "CME",
           "note": "matched hedge for TAIFEX UNF (Nasdaq-100)"},
    "ES": {"kind": "future", "symbol": "ES", "exchange": "CME",
           "note": "matched hedge for TAIFEX SPF (S&P 500)"},
    "YM": {"kind": "future", "symbol": "YM", "exchange": "CBOT",
           "note": "matched hedge for TAIFEX UDF (Dow)"},
    "MNQ": {"kind": "future", "symbol": "MNQ", "exchange": "CME",
            "note": "micro Nasdaq-100, the size that actually fits one UNF"},
    "MES": {"kind": "future", "symbol": "MES", "exchange": "CME"},
    "MYM": {"kind": "future", "symbol": "MYM", "exchange": "CBOT"},
    "NDX": {"kind": "index", "symbol": "NDX", "exchange": "NASDAQ"},
    "SPX": {"kind": "index", "symbol": "SPX", "exchange": "CBOE"},
    "SOX": {"kind": "index", "symbol": "SOX", "exchange": "PHLX"},
    "INDU": {"kind": "index", "symbol": "INDU", "exchange": "CME",
             "note": "needs a CME index subscription; YM works without it"},
    "SOXX": {"kind": "stock", "symbol": "SOXX"},
    "SMH": {"kind": "stock", "symbol": "SMH"},
    "QQQ": {"kind": "stock", "symbol": "QQQ"},
    "DIA": {"kind": "stock", "symbol": "DIA"},
    "SPY": {"kind": "stock", "symbol": "SPY"},
}

# IB caps the span of a single historical request by bar size.  These are
# conservative: exceeding them returns an error rather than a truncated frame.
CHUNK_DAYS = {"1 min": 3, "5 mins": 10, "15 mins": 20, "30 mins": 30, "1 hour": 30}
# IB allows 60 historical requests per 10 minutes; 6s between calls stays under it.
REQUEST_SPACING_SECONDS = 6.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the US legs of the TAIFEX US-index-futures pair study "
            "from a running IB Gateway / TWS. Timestamps are written in Taipei "
            "time to match the rest of the repo. Note IB carries no TAIFEX "
            "products at all, so the Taiwan leg must come from TAIFEX itself "
            "(see build_qff1_1m.py --product UNF --expiry-rule third_friday)."
        )
    )
    parser.add_argument(
        "--symbols",
        default="NQ,NDX",
        help=f"Comma-separated. Known: {','.join(sorted(SPECS))}",
    )
    parser.add_argument(
        "--expiry",
        default=None,
        help=(
            "Contract month/day for futures, e.g. 20260918 or 202609. Omit to "
            "take the front month IB returns first. IB serves little or no "
            "history for already-expired contracts, so back-fills are limited "
            "to the contracts still listed."
        ),
    )
    parser.add_argument("--bar-size", default="1 min", choices=sorted(CHUNK_DAYS))
    parser.add_argument("--start", default=None, help="YYYY-MM-DD, inclusive floor.")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, exclusive-ish ceiling.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=71)
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Merge into any existing output instead of overwriting it. Existing "
            "rows win on conflict. Use this for periodic top-ups."
        ),
    )
    parser.add_argument("--timeout", type=int, default=90)
    return parser.parse_args(argv)


def build_contract(name: str, expiry: str | None) -> Contract:
    if name not in SPECS:
        raise SystemExit(f"unknown symbol {name!r}; known: {sorted(SPECS)}")
    spec = SPECS[name]
    if spec["kind"] == "future":
        return Future(spec["symbol"], expiry or "", spec["exchange"])
    if spec["kind"] == "index":
        return Index(spec["symbol"], spec["exchange"])
    return Stock(spec["symbol"], "SMART", "USD")


def what_to_show(contract: Contract) -> str:
    return "MIDPOINT" if contract.secType == "CASH" else "TRADES"


def read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    return frame


def format_timestamps(values: pd.Series) -> pd.Series:
    text = values.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return text.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def download_one(
    ib: IB, name: str, args: argparse.Namespace
) -> tuple[pd.DataFrame, Contract] | None:
    contract = build_contract(name, args.expiry)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        print(f"{name}: contract did not resolve", file=sys.stderr)
        return None
    resolved = qualified[0]

    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )
    floor = (
        datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        if args.start
        else end - timedelta(days=90)
    )
    chunk = CHUNK_DAYS[args.bar_size]

    frames: list[pd.DataFrame] = []
    empty_streak = 0
    while end > floor and empty_streak < 3:
        try:
            bars = ib.reqHistoricalData(
                resolved,
                endDateTime=end,
                durationStr=f"{chunk} D",
                barSizeSetting=args.bar_size,
                whatToShow=what_to_show(resolved),
                useRTH=False,
                formatDate=2,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"{name} @ {end:%Y-%m-%d}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            empty_streak += 1
            time.sleep(REQUEST_SPACING_SECONDS)
            continue
        if not bars:
            # A gap (holiday run, or before the contract listed) is not fatal;
            # step back and keep going until several chunks in a row are empty.
            empty_streak += 1
            end -= timedelta(days=chunk)
            time.sleep(REQUEST_SPACING_SECONDS)
            continue
        empty_streak = 0
        frame = util.df(bars)
        frames.append(frame)
        end = pd.to_datetime(frame["date"]).min().to_pydatetime() - timedelta(minutes=1)
        time.sleep(REQUEST_SPACING_SECONDS)

    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert(TAIPEI_TZ)
    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
    return data.drop_duplicates("timestamp").sort_values("timestamp"), resolved


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    names = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id,
               timeout=30, readonly=True)
    print(f"connected to {args.host}:{args.port}, accounts={ib.managedAccounts()}")

    try:
        for name in names:
            result = download_one(ib, name, args)
            if result is None:
                print(f"{name}: no data")
                continue
            fresh, resolved = result
            suffix = args.bar_size.replace(" ", "").replace("mins", "m")
            suffix = suffix.replace("min", "m").replace("hour", "h")
            local = (resolved.localSymbol or name).replace(" ", "")
            out_path = args.out_dir / f"ib_{local.lower()}_{suffix}_taipei.csv"

            merged = fresh
            if args.append:
                existing = read_existing(out_path)
                if not existing.empty:
                    # existing rows win: a settled bar should not change, and a
                    # re-fetch must never silently rewrite captured history
                    merged = pd.concat([fresh, existing], ignore_index=True)
                    merged = merged.drop_duplicates("timestamp", keep="last")
                    merged = merged.sort_values("timestamp").reset_index(drop=True)
                    print(f"{name}: merged {len(merged) - len(existing):,} new rows "
                          f"into {len(existing):,} existing")

            output = merged.copy()
            output["timestamp"] = format_timestamps(output["timestamp"])
            output.to_csv(out_path, index=False)
            print(
                f"{name}: {len(merged):,} bars "
                f"{merged['timestamp'].min()} -> {merged['timestamp'].max()}  "
                f"conId={resolved.conId} {resolved.localSymbol}  -> {out_path}"
            )
    finally:
        ib.disconnect()
    return 0


if __name__ == "__main__":
    util.logToConsole(level=40)
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
