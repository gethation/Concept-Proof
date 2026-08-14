from __future__ import annotations

import argparse
import html
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


PREVIOUS_30_URL = (
    "https://www.taifex.com.tw/enl/eng3/"
    "futPrevious30DaysSalesData?menuid1=03"
)
DAILY_REPORT_URL = "https://www.taifex.com.tw/enl/eng3/futDataDown"
DAILY_ZIP_RE = re.compile(
    r"https://www\.taifex\.com\.tw/file/taifex/Dailydownload/"
    r"DailydownloadCSV_eng/Daily_(\d{4})_(\d{2})_(\d{2})\.zip"
)
TAIPEI_TZ = "Asia/Taipei"
REQUIRED_COLUMNS = {
    "Date",
    "Product Code",
    "Contract Month(Week)",
    "Time of Trades",
    "Trade Price",
    "Volume(Buy+Sell)",
    "Price for Nearer Delivery Month Contract",
    "Price for Further Delivery Month Contract",
}


@dataclass(frozen=True)
class DailyZipLink:
    trade_date: date
    url: str

    @property
    def filename(self) -> str:
        return f"Daily_{self.trade_date:%Y_%m_%d}.zip"

    @property
    def source_trade_date(self) -> str:
        return self.trade_date.isoformat()

    @classmethod
    def from_path(cls, path: Path) -> "DailyZipLink":
        """Inverse of `filename`, kept next to it so the two cannot drift.

        url is empty because a cached zip has no fetch URL; nothing downstream
        reads it (download_zip is the only consumer and the cache path skips
        it).
        """
        return cls(datetime.strptime(path.stem, "Daily_%Y_%m_%d").date(), "")


def build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        }
    )
    return session


def fetch_daily_zip_links(
    session: requests.Session, days: int, timeout: int
) -> list[DailyZipLink]:
    response = session.get(PREVIOUS_30_URL, timeout=timeout)
    response.raise_for_status()
    page = html.unescape(response.text)

    links: list[DailyZipLink] = []
    seen: set[str] = set()
    for match in DAILY_ZIP_RE.finditer(page):
        url = match.group(0)
        if url in seen:
            continue
        seen.add(url)
        year, month, day = (int(part) for part in match.groups())
        links.append(DailyZipLink(date(year, month, day), url))

    if len(links) < days:
        raise RuntimeError(
            f"TAIFEX page exposed only {len(links)} unique CSV zip links; "
            f"{days} were requested."
        )

    return sorted(links[:days], key=lambda item: item.trade_date)


def download_zip(
    session: requests.Session,
    link: DailyZipLink,
    raw_dir: Path,
    refresh: bool,
    timeout: int,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / link.filename
    if path.exists() and not refresh:
        return path

    response = session.get(link.url, timeout=timeout)
    response.raise_for_status()
    if "zip" not in response.headers.get("Content-Type", "").lower():
        raise RuntimeError(
            f"Unexpected content type for {link.url}: "
            f"{response.headers.get('Content-Type')}"
        )

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(response.content)
    with zipfile.ZipFile(tmp_path) as archive:
        archive.testzip()
    tmp_path.replace(path)
    return path


def read_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"{path} should contain exactly one CSV, got {csv_names}")
        raw = archive.read(csv_names[0])

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "ms950", "big5"):
        try:
            frame = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=encoding)
            frame.columns = [str(column).strip() for column in frame.columns]
            return frame
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Unable to decode {path}") from last_error


def require_columns(frame: pd.DataFrame, source: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{source} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )


def classify_session(time_text: pd.Series) -> pd.Series:
    time_text = time_text.str.zfill(6)
    regular = time_text.between("084500", "134500")
    return pd.Series(
        ["Regular" if is_regular else "After-Hours" for is_regular in regular],
        index=time_text.index,
    )


def parse_timestamps(date_text: pd.Series, time_text: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        date_text.str.strip() + time_text.str.strip().str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="raise",
    )
    return parsed.dt.tz_localize(TAIPEI_TZ)


def normalize_time_sales(
    frame: pd.DataFrame, source: Path, source_trade_date: str, product: str
) -> pd.DataFrame:
    require_columns(frame, source)
    data = frame.copy()
    data["_row_order"] = range(len(data))
    data = data[data["Product Code"].str.strip().eq(product)].copy()
    if data.empty:
        return empty_ticks_frame()

    data["contract_month_raw"] = data["Contract Month(Week)"].str.strip()
    data["time_text"] = data["Time of Trades"].str.strip().str.zfill(6)
    data["timestamp"] = parse_timestamps(data["Date"], data["time_text"])
    data["session"] = classify_session(data["time_text"])
    data["raw_volume"] = pd.to_numeric(
        data["Volume(Buy+Sell)"].str.strip(), errors="coerce"
    )
    if data["raw_volume"].isna().any():
        bad = data[data["raw_volume"].isna()].head(5)
        raise RuntimeError(f"Invalid volume values in {source}:\n{bad}")

    outright_mask = (
        ~data["contract_month_raw"].str.contains("/", regex=False)
        & data["contract_month_raw"].str.fullmatch(r"\d{6}")
    )
    outright = data.loc[outright_mask].copy()
    outright["contract_month"] = outright["contract_month_raw"]
    outright["price"] = pd.to_numeric(outright["Trade Price"].str.strip(), errors="coerce")
    outright["volume"] = outright["raw_volume"] / 2
    outright["leg_order"] = 0

    spread = data.loc[data["contract_month_raw"].str.contains("/", regex=False)].copy()
    legs: list[pd.DataFrame] = []
    if not spread.empty:
        split_months = spread["contract_month_raw"].str.split("/", expand=True)
        for leg_order, (month_column, price_column) in enumerate(
            (
                (0, "Price for Nearer Delivery Month Contract"),
                (1, "Price for Further Delivery Month Contract"),
            )
        ):
            leg = spread.copy()
            leg["contract_month"] = split_months[month_column].str.strip()
            leg["price"] = pd.to_numeric(
                leg[price_column].str.strip().replace("-", pd.NA), errors="coerce"
            )
            leg["volume"] = leg["raw_volume"] / 4
            leg["leg_order"] = leg_order + 1
            legs.append(leg)

    normalized = pd.concat([outright, *legs], ignore_index=True)
    normalized = normalized[
        normalized["contract_month"].str.fullmatch(r"\d{6}")
        & normalized["price"].notna()
        & normalized["volume"].notna()
    ].copy()
    normalized["source_trade_date"] = source_trade_date

    columns = [
        "source_trade_date",
        "timestamp",
        "session",
        "contract_month",
        "price",
        "volume",
        "_row_order",
        "leg_order",
    ]
    return normalized[columns]


def empty_ticks_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_trade_date",
            "timestamp",
            "session",
            "contract_month",
            "price",
            "volume",
            "_row_order",
            "leg_order",
        ]
    )


def find_trade_date_gaps(
    links: list[DailyZipLink], max_gap_days: int
) -> list[tuple[str, str, int]]:
    """Consecutive cached trade dates further apart than max_gap_days.

    Calendar days, not trading days: weekends and short holidays are normal, so
    the default threshold has to clear a long weekend. Anything wider is a
    missing capture, and it has to be reported rather than silently bridged.
    """
    gaps: list[tuple[str, str, int]] = []
    ordered = sorted(links, key=lambda item: item.trade_date)
    for previous, current in zip(ordered, ordered[1:]):
        delta = (current.trade_date - previous.trade_date).days
        if delta > max_gap_days:
            gaps.append(
                (previous.source_trade_date, current.source_trade_date, delta)
            )
    return gaps


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` of the month, weekday in Monday=0 .. Sunday=6."""
    first_day = date(year, month, 1)
    offset = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=offset + 7 * (n - 1))


def third_wednesday(year: int, month: int) -> date:
    return nth_weekday(year, month, 2, 3)


def third_friday(year: int, month: int) -> date:
    return nth_weekday(year, month, 4, 3)


# TAIFEX stock futures (QFF/CCF/CDF) expire on the third Wednesday; the US index
# futures (UDF/SPF/UNF/SXF) expire on the third Friday, matching the US SOQ.
EXPIRY_RULES = {
    "third_wednesday": third_wednesday,
    "third_friday": third_friday,
}
DEFAULT_EXPIRY_RULE = "third_wednesday"


def contract_expiry(contract_month: str, rule: str = DEFAULT_EXPIRY_RULE) -> date:
    year = int(contract_month[:4])
    month = int(contract_month[4:6])
    return EXPIRY_RULES[rule](year, month)


def select_front_month_ticks(
    ticks: pd.DataFrame, expiry_rule: str = DEFAULT_EXPIRY_RULE
) -> pd.DataFrame:
    if ticks.empty:
        return ticks.copy()

    months = sorted(ticks["contract_month"].unique())
    expiry_by_month = {
        month: contract_expiry(month, expiry_rule) for month in months
    }

    trade_dates = sorted(ticks["source_trade_date"].unique())
    front_by_trade_date: dict[str, str] = {}
    for trade_date_text in trade_dates:
        trade_date = datetime.strptime(trade_date_text, "%Y-%m-%d").date()
        eligible = [
            month for month in months if expiry_by_month[month] >= trade_date
        ]
        if not eligible:
            eligible = [month for month in months if month >= trade_date.strftime("%Y%m")]
        if not eligible:
            eligible = [months[-1]]
        front_by_trade_date[trade_date_text] = min(eligible)

    selected = ticks.copy()
    selected["front_contract_month"] = selected["source_trade_date"].map(
        front_by_trade_date
    )
    selected = selected[
        selected["contract_month"].eq(selected["front_contract_month"])
    ].copy()
    return selected.drop(columns=["front_contract_month"])


def aggregate_1m(ticks: pd.DataFrame) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "contract_month",
                "trade_count",
            ]
        )

    ordered = ticks.sort_values(
        ["timestamp", "_row_order", "leg_order"], kind="mergesort"
    ).copy()
    ordered["minute"] = ordered["timestamp"].dt.floor("min")

    bars = (
        ordered.groupby("minute", sort=True)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
            contract_month=("contract_month", "last"),
            trade_count=("price", "size"),
        )
        .reset_index()
        .rename(columns={"minute": "timestamp"})
    )

    if not bars.empty and ((bars["volume"] % 1).abs() < 1e-9).all():
        bars["volume"] = bars["volume"].round().astype("int64")
    return bars


def read_daily_report(
    session: requests.Session, trade_date: date, product: str, timeout: int
) -> pd.DataFrame:
    date_text = trade_date.strftime("%Y/%m/%d")
    response = session.post(
        DAILY_REPORT_URL,
        data={
            "queryStartDate": date_text,
            "queryEndDate": date_text,
            "commodity_id": "specialid",
            "commodity_id2": product,
            "down_type": "1",
        },
        timeout=timeout,
    )
    response.raise_for_status()

    last_error: Exception | None = None
    for encoding in ("ms950", "big5", "utf-8-sig", "utf-8"):
        try:
            frame = pd.read_csv(io.BytesIO(response.content), dtype=str, encoding=encoding)
            frame.columns = [str(column).strip() for column in frame.columns]
            return frame
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Unable to decode daily report for {date_text}") from last_error


def validate_against_daily_reports(
    session: requests.Session,
    ticks: pd.DataFrame,
    links: list[DailyZipLink],
    product: str,
    timeout: int,
) -> None:
    local = (
        ticks.groupby(["source_trade_date", "contract_month", "session"], as_index=False)[
            "volume"
        ]
        .sum()
        .rename(columns={"volume": "local_volume"})
    )

    reports: list[pd.DataFrame] = []
    for link in links:
        report = read_daily_report(session, link.trade_date, product, timeout)
        needed = {
            "contract",
            "contract month(Week)",
            "Volume",
            "Trading Session",
        }
        missing = needed.difference(report.columns)
        if missing:
            raise RuntimeError(
                f"Daily report for {link.source_trade_date} missing {sorted(missing)}"
            )

        report = report[report["contract"].str.strip().eq(product)].copy()
        report["contract_month"] = report["contract month(Week)"].str.strip()
        report = report[
            report["contract_month"].str.fullmatch(r"\d{6}")
            & report["Trading Session"].str.strip().isin(["Regular", "After-Hours"])
        ].copy()
        report["source_trade_date"] = link.source_trade_date
        report["session"] = report["Trading Session"].str.strip()
        report["report_volume"] = pd.to_numeric(
            report["Volume"].str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
        reports.append(
            report[
                ["source_trade_date", "contract_month", "session", "report_volume"]
            ]
        )

    expected = pd.concat(reports, ignore_index=True)
    merged = expected.merge(
        local,
        on=["source_trade_date", "contract_month", "session"],
        how="left",
    )
    merged["local_volume"] = merged["local_volume"].fillna(0)
    merged["diff"] = merged["local_volume"] - merged["report_volume"]
    failures = merged[merged["diff"].abs() > 1e-6]
    if not failures.empty:
        raise RuntimeError(
            "Normalized time-and-sales volume does not match TAIFEX daily report:\n"
            + failures.head(20).to_string(index=False)
        )


def write_bars(bars: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = bars.copy()
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    data["timestamp"] = data["timestamp"].str.replace(
        r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True
    )
    data.to_csv(output_path, index=False)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download TAIFEX QFF previous-30-trading-day tick CSV files and "
            "aggregate a TradingView-like QFF1! 1m continuous front-month series."
        )
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--product", default="QFF")
    parser.add_argument(
        "--expiry-rule",
        choices=sorted(EXPIRY_RULES),
        default=DEFAULT_EXPIRY_RULE,
        help=(
            "Last trading day convention used to pick the front month. "
            "Stock futures use third_wednesday (default); the US index "
            "futures UDF/SPF/UNF/SXF use third_friday."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/taifex_time_sales"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/qff1_1m.csv"),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download zip files even when cached files exist.",
    )
    parser.add_argument(
        "--contract-month",
        default=None,
        help=(
            "Pin one contract month (YYYYMM) instead of rolling the front "
            "month. Useful when the paired US leg only has history for one "
            "expiry, since matched maturities are what makes the basis clean."
        ),
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Build from every zip already in --raw-dir instead of fetching the "
            "current 30-trading-day window. TAIFEX only publishes 30 days, so "
            "cached zips are the only route to older history. --days is ignored "
            "in this mode; continuity is checked with --max-gap-days instead."
        ),
    )
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=5,
        help=(
            "With --from-cache, the widest calendar gap tolerated between "
            "consecutive cached trade dates. 5 clears a long weekend; raise it "
            "deliberately for a known market closure such as Lunar New Year."
        ),
    )
    parser.add_argument(
        "--validate-daily",
        action="store_true",
        help="Compare normalized contract/session volumes with TAIFEX daily reports.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.days <= 0:
        raise ValueError("--days must be positive")

    # --expiry-rule only feeds the front-month roll, which pinning bypasses.
    # Accepting both silently would look composable and do nothing.
    if args.contract_month and args.expiry_rule != DEFAULT_EXPIRY_RULE:
        raise RuntimeError(
            "--expiry-rule has no effect with --contract-month: pinning a month "
            "skips the front-month roll that the rule drives"
        )

    product = args.product.strip().upper()
    session = build_http_session()

    if args.from_cache:
        zip_paths = sorted(args.raw_dir.glob("Daily_*.zip"))
        if not zip_paths:
            raise RuntimeError(f"No cached zips under {args.raw_dir}")
        links = [DailyZipLink.from_path(path) for path in zip_paths]
        # fetch_daily_zip_links refuses to return fewer links than requested, so
        # the network path could never produce a hole. The cache path takes
        # whatever is on disk, and a hole is not benign: the rolling z-score
        # window is bar-count based, so a gap splices price levels weeks apart
        # and manufactures z-scores at the seam.
        gaps = find_trade_date_gaps(links, args.max_gap_days)
        if gaps:
            detail = "; ".join(
                f"{start} -> {end} ({days} days)" for start, end, days in gaps
            )
            raise RuntimeError(
                f"Cached zips under {args.raw_dir} have {len(gaps)} gap(s) wider "
                f"than --max-gap-days {args.max_gap_days}: {detail}. Re-fetch the "
                "missing days, or raise --max-gap-days if the gap is a known "
                "market closure."
            )
        print(
            f"Using {len(zip_paths)} cached TAIFEX CSV zips: "
            f"{links[0].source_trade_date} to {links[-1].source_trade_date}"
        )
    else:
        links = fetch_daily_zip_links(session, args.days, args.timeout)
        print(
            f"Found {len(links)} TAIFEX CSV zips: "
            f"{links[0].source_trade_date} to {links[-1].source_trade_date}"
        )

        zip_paths = [
            download_zip(session, link, args.raw_dir, args.refresh, args.timeout)
            for link in links
        ]
        print(f"Cached {len(zip_paths)} zip files under {args.raw_dir}")

    normalized_parts: list[pd.DataFrame] = []
    for link, path in zip(links, zip_paths):
        frame = read_csv_from_zip(path)
        normalized_parts.append(
            normalize_time_sales(frame, path, link.source_trade_date, product)
        )

    ticks = pd.concat(normalized_parts, ignore_index=True)
    print(f"Loaded {len(ticks):,} normalized {product} tick/leg rows")

    if args.validate_daily:
        validate_against_daily_reports(session, ticks, links, product, args.timeout)
        print("Daily report volume validation passed")

    if args.contract_month:
        front_ticks = ticks[ticks["contract_month"].eq(args.contract_month)].copy()
        if front_ticks.empty:
            available = sorted(ticks["contract_month"].unique())
            raise RuntimeError(
                f"No {product} rows for contract month {args.contract_month}; "
                f"available: {available}"
            )
        # Pinning skips the front-month roll, so it also skips its liquidity
        # guarantee: on days when the pinned month is still a distant back
        # month it barely trades, and the sparse bars forward-fill downstream
        # into quotes that were never executable. The empty check above only
        # catches a month missing outright, so report per-day coverage.
        covered = front_ticks["source_trade_date"].nunique()
        all_days = ticks["source_trade_date"].nunique()
        print(
            f"Contract month {args.contract_month} present on {covered}/{all_days} "
            "trade dates"
        )
        thin = (
            front_ticks.groupby("source_trade_date").size().pipe(lambda s: s[s < 100])
        )
        if covered < all_days or not thin.empty:
            print(
                "WARNING: pinned month has "
                f"{all_days - covered} day(s) with no prints and "
                f"{len(thin)} day(s) under 100 prints; the front-month roll "
                "exists to avoid exactly this"
            )
    else:
        front_ticks = select_front_month_ticks(ticks, args.expiry_rule)
    bars = aggregate_1m(front_ticks)
    write_bars(bars, args.out)
    label = args.contract_month or "front-month"
    print(f"Selected {len(front_ticks):,} {label} rows")
    print(f"Wrote {len(bars):,} 1m bars to {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
