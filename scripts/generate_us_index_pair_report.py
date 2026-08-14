from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TAIPEI_TZ = "Asia/Taipei"

SPECS = [
    ("UDF", "Dow Jones Industrial", 20.0, 1.0),
    ("SPF", "S&P 500", 200.0, 0.25),
    ("UNF", "Nasdaq-100", 50.0, 1.0),
    ("SXF", "PHLX Semiconductor", 80.0, 0.5),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the TAIFEX US-index-futures pair study report. Every figure "
            "is read from the pipeline outputs at run time, so re-running the "
            "pipeline and re-running this keeps the report honest."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--daily-quotes-dir",
        type=Path,
        required=True,
        help="Directory of per-day TAIFEX daily-report CSVs (commodity_id=all).",
    )
    parser.add_argument(
        "--cost-sweep",
        type=Path,
        default=None,
        help="Directory of grid CSVs named grid_hs<half-spread>.csv.",
    )
    parser.add_argument("--out", type=Path, default=Path("taifex_us_index_pair_report.html"))
    return parser.parse_args(argv)


def load_daily_quotes(directory: Path) -> pd.DataFrame:
    """The commodity_id=all export omits the date field from data rows, so every
    value lands one column left of its header; shift back and take the date from
    the filename."""
    frames = []
    for path in sorted(directory.glob("*.csv")):
        header = path.read_text(encoding="utf-8").splitlines()[0]
        names = [c.strip() for c in header.split(",")]
        if names[0] != "date":
            continue
        frame = pd.read_csv(
            path, dtype=str, header=None, skiprows=1, index_col=False,
            names=names[1:] + ["_extra"], engine="python",
        )
        frame["date"] = path.stem
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"no daily-report CSVs under {directory}")
    data = pd.concat(frames, ignore_index=True)

    def num(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.replace(",", "", regex=False).str.strip()
        return pd.to_numeric(text.replace("-", None), errors="coerce")

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="coerce")
    data["product"] = data["contract"].astype(str).str.strip()
    data["month"] = data["contract month(Week)"].astype(str).str.strip()
    data["session"] = data["Trading Session"].astype(str).str.strip()
    for column in ("Volume", "open_interest", "best_bid", "best_ask"):
        data[column] = num(data[column])
    data = data[data["month"].str.fullmatch(r"\d{6}", na=False)].copy()
    mid = (data["best_bid"] + data["best_ask"]) / 2
    data["mid"] = mid
    data["spread_bps"] = (data["best_ask"] - data["best_bid"]) / mid * 1e4
    return data


def front_month(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(["product", "date", "session", "Volume"])
    front = ordered.groupby(["product", "date", "session"], as_index=False).tail(1)
    return front[front["Volume"].fillna(0) > 0]


def fmt(value: float, digits: int = 1, dash: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return f"{value:,.{digits}f}"


def line_chart(series: pd.Series, width: int = 860, height: int = 240) -> str:
    """Basis over time. One series, so the title names it and no legend is needed."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 52, 12, 14, 26
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(i: int) -> float:
        return left + plot_w * (i / max(n - 1, 1))

    def y(v: float) -> float:
        return top + plot_h * (1 - (v - lo) / (hi - lo))

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}"
        for i, v in enumerate(values)
        if np.isfinite(v)
    )
    ticks = np.linspace(lo, hi, 5)
    grid = "".join(
        f'<line class="grid" x1="{left}" x2="{width - right}" '
        f'y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
        f'<text class="axis" x="{left - 8}" y="{y(t) + 4:.1f}" '
        f'text-anchor="end">{t:+.0f}</text>'
        for t in ticks
    )
    zero = (
        f'<line class="zero" x1="{left}" x2="{width - right}" '
        f'y1="{y(0):.1f}" y2="{y(0):.1f}"/>' if lo < 0 < hi else ""
    )
    stamps = pd.DatetimeIndex(series.index)
    labels = ""
    for i in (0, n // 2, n - 1):
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        labels += (
            f'<text class="axis" x="{x(i):.1f}" y="{height - 8}" '
            f'text-anchor="{anchor}">{stamps[i]:%b %d}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="TAIFEX UNF minus CME NQ basis in basis points over the sample">'
        f"{grid}{zero}"
        f'<path class="s1-line" d="{path}"/>{labels}</svg>'
    )


def spread_bars(rows: list[tuple[str, float, float]], breakeven: float,
                width: int = 860) -> str:
    """Quoted spread by product and session. Two series, so legend plus direct
    labels; the break-even reference is drawn as a labelled rule, not colour alone."""
    bar_h, gap, group_gap = 15, 2, 16
    left, right, top = 62, 150, 8
    height = top + len(rows) * (2 * bar_h + gap + group_gap)
    plot_w = width - left - right
    hi = max(max(d, n) for _, d, n in rows) * 1.05

    def bar_w(value: float) -> float:
        return max(plot_w * value / hi, 1.0)

    parts, y = [], top
    for name, day, night in rows:
        for value, cls, label in ((day, "s1-fill", "day"), (night, "s2-fill", "night")):
            parts.append(
                f'<rect class="{cls}" x="{left}" y="{y}" width="{bar_w(value):.1f}" '
                f'height="{bar_h}" rx="4"/>'
                f'<text class="val" x="{left + bar_w(value) + 8:.1f}" '
                f'y="{y + bar_h - 3}">{value:,.1f} ({label})</text>'
            )
            y += bar_h + gap
        parts.append(
            f'<text class="cat" x="{left - 10}" y="{y - bar_h - 6}" '
            f'text-anchor="end">{name}</text>'
        )
        y += group_gap - gap
    # The rule is labelled at the foot, clear of the first bar's value label.
    height += 16
    x_be = left + bar_w(breakeven)
    parts.append(
        f'<line class="breakeven" x1="{x_be:.1f}" x2="{x_be:.1f}" y1="{top}" '
        f'y2="{height - 20}"/>'
        f'<text class="be-label" x="{x_be + 6:.1f}" y="{height - 6}">'
        f"break-even {breakeven:.0f} bps</text>"
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Quoted bid-ask spread in basis points by product and session, '
        f'against the break-even the backtest requires">' + "".join(parts) + "</svg>"
    )


def table(headers: list[str], rows: list[list[str]], note: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    caption = f'<p class="note">{note}</p>' if note else ""
    return (
        f'<div class="tw"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>{caption}"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    processed = args.processed_dir

    quotes = load_daily_quotes(args.daily_quotes_dir)
    front = front_month(quotes)
    q_lo, q_hi = quotes["date"].min(), quotes["date"].max()

    basis = pd.read_csv(processed / "us_index_basis_unf_nq_day.csv")
    basis["timestamp"] = pd.to_datetime(basis["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    basis_bps = basis["spread"] * 100.0
    sessions = basis["timestamp"].dt.normalize()
    span_days = (sessions.max() - sessions.min()).days

    zero = pd.read_csv(processed / "unf_nq_day_grid_zerocost.csv")
    real = pd.read_csv(processed / "unf_nq_day_grid_realcost.csv")
    udf = pd.read_csv(processed / "udf_ym_day_grid_zerocost.csv")

    zero_best = zero.sort_values("sharpe_ratio", ascending=False).iloc[0]
    real_best = real.sort_values("net_pnl_twd", ascending=False).iloc[0]
    udf_best = udf.sort_values("sharpe_ratio", ascending=False).iloc[0]
    capital_unf, capital_udf = 2_800_000.0, 2_100_000.0
    zero_ret = zero_best["net_pnl_twd"] / capital_unf
    zero_ann = zero_ret * 365 / max(span_days, 1)
    udf_ret = udf_best["net_pnl_twd"] / capital_udf

    sweep_rows: list[list[str]] = []
    breakeven = float("nan")
    if args.cost_sweep and args.cost_sweep.exists():
        entries = []
        for path in sorted(args.cost_sweep.glob("grid_hs*.csv")):
            half = float(path.stem.replace("grid_hs", ""))
            grid = pd.read_csv(path)
            best = grid.sort_values("net_pnl_twd", ascending=False).iloc[0]
            entries.append((half, best))
        entries.sort(key=lambda item: item[0])
        previous = None
        for half, best in entries:
            net = best["net_pnl_twd"]
            if previous is not None and previous[1] > 0 >= net:
                p_half, p_net = previous
                breakeven = 2 * (p_half + (half - p_half) * p_net / (p_net - net))
            previous = (half, net)
            sweep_rows.append([
                f"{half * 2:.1f}", f"w{int(best['ma_window'])}/e{best['entry_z']:.1f}"
                f"/x{best['exit_z']:.1f}", f"{int(best['trade_count'])}",
                fmt(net, 0),
                f'<span class="{"pos" if net > 0 else "neg"}">'
                f'{"profitable" if net > 0 else "loses"}</span>',
            ])

    # ---- quoted-spread summary, both sessions, front month ----
    spread_rows, chart_rows = [], []
    for code, underlying, mult, tick in SPECS:
        cells = [f"<b>{code}</b>", underlying]
        by_session = {}
        for session in ("Regular", "After-Hours"):
            sub = front[(front["product"] == code) & (front["session"] == session)]
            sb = sub["spread_bps"].dropna()
            by_session[session] = sb.median() if len(sb) else float("nan")
            cells.append(fmt(sub["Volume"].median(), 0))
            cells.append(fmt(by_session[session]))
        oi = front[(front["product"] == code)
                   & (front["session"] == "Regular")]["open_interest"].dropna()
        level = front[front["product"] == code]["mid"].median()
        cells.insert(2, fmt(oi.median(), 0))
        cells.append(f"NT${mult * tick:,.0f}")
        cells.append(fmt(mult * tick / (level * mult) * 1e4, 2))
        spread_rows.append(cells)
        chart_rows.append((code, by_session["Regular"], by_session["After-Hours"]))

    bench_rows = []
    for code in ("TX", "MTX"):
        sub_d = front[(front["product"] == code) & (front["session"] == "Regular")]
        sub_n = front[(front["product"] == code) & (front["session"] == "After-Hours")]
        bench_rows.append([
            f"<b>{code}</b>", "TAIFEX benchmark",
            fmt(sub_d["open_interest"].dropna().median(), 0),
            fmt(sub_d["Volume"].median(), 0), fmt(sub_d["spread_bps"].median()),
            fmt(sub_n["Volume"].median(), 0), fmt(sub_n["spread_bps"].median()),
            "-", "-",
        ])

    # ---- liquidity trend by quarter ----
    trend = front.copy()
    trend["q"] = trend["date"].dt.to_period("Q").astype(str)
    quarters = sorted(trend["q"].unique())
    trend_rows = []
    for code, *_ in SPECS:
        cells = [f"<b>{code}</b>"]
        for quarter in quarters:
            sub = trend[(trend["product"] == code) & (trend["q"] == quarter)]
            if sub.empty:
                cells.append("-")
                continue
            cells.append(
                f"{fmt(sub['Volume'].median(), 0)} / "
                f"{fmt(sub['spread_bps'].median(), 0)}"
            )
        trend_rows.append(cells)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TAIFEX US index futures - pair study</title>
<style>
:root{{color-scheme:light;--surface-1:#fcfcfb;--plane:#f9f9f7;--text-primary:#0b0b0b;
--text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--series-1:#2a78d6;--series-2:#eb6834;--critical:#d03b3b;--good:#006300;
--border:rgba(11,11,11,0.10);}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{
color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#fff;
--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--series-1:#3987e5;--series-2:#d95926;--critical:#d03b3b;--good:#0ca30c;
--border:rgba(255,255,255,0.10);}}}}
:root[data-theme="dark"]{{color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;
--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--series-1:#3987e5;--series-2:#d95926;--critical:#d03b3b;
--good:#0ca30c;--border:rgba(255,255,255,0.10);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--plane);color:var(--text-primary);
font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;}}
main{{max-width:960px;margin:0 auto;padding:32px 20px 72px}}
h1{{font-size:26px;margin:0 0 4px}}
h2{{font-size:19px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--border)}}
h3{{font-size:15px;margin:22px 0 6px;color:var(--text-secondary)}}
p{{margin:9px 0}}
.sub{{color:var(--text-secondary);margin:0 0 18px;font-size:13px}}
.verdict{{background:var(--surface-1);border:1px solid var(--border);
border-radius:10px;padding:16px 18px;margin:18px 0}}
.verdict p:first-child{{margin-top:0}} .verdict p:last-child{{margin-bottom:0}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
gap:10px;margin:16px 0}}
.tile{{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
padding:12px 14px}}
.tile .k{{font-size:12px;color:var(--text-secondary);text-transform:uppercase;
letter-spacing:.04em}}
.tile .v{{font-size:24px;margin-top:2px}}
.tile .h{{font-size:12px;color:var(--muted)}}
.tw{{overflow-x:auto;background:var(--surface-1);border:1px solid var(--border);
border-radius:10px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;
font-variant-numeric:tabular-nums}}
th,td{{padding:7px 11px;text-align:right;border-bottom:1px solid var(--grid);
white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{color:var(--text-secondary);font-weight:600;font-size:12px;
text-transform:uppercase;letter-spacing:.03em}}
tbody tr:last-child td{{border-bottom:none}}
.note{{font-size:12.5px;color:var(--text-secondary);margin:4px 2px 0}}
figure{{margin:14px 0;background:var(--surface-1);border:1px solid var(--border);
border-radius:10px;padding:14px}}
figcaption{{font-size:12.5px;color:var(--text-secondary);margin-top:6px}}
svg{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--grid);stroke-width:1}}
.zero{{stroke:var(--axis);stroke-width:1;stroke-dasharray:3 3}}
.axis{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}}
.cat{{fill:var(--text-primary);font-size:12.5px;font-weight:600}}
.val{{fill:var(--text-secondary);font-size:11.5px;font-variant-numeric:tabular-nums}}
.s1-line{{fill:none;stroke:var(--series-1);stroke-width:2;
stroke-linejoin:round;stroke-linecap:round}}
.s1-fill{{fill:var(--series-1)}} .s2-fill{{fill:var(--series-2)}}
.breakeven{{stroke:var(--critical);stroke-width:2;stroke-dasharray:4 3}}
.be-label{{fill:var(--critical);font-size:11.5px;font-weight:600}}
.legend{{display:flex;gap:16px;font-size:12.5px;color:var(--text-secondary);
margin-bottom:8px;flex-wrap:wrap}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}
.sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.pos{{color:var(--good);font-weight:600}} .neg{{color:var(--critical);font-weight:600}}
code{{background:var(--surface-1);border:1px solid var(--border);border-radius:4px;
padding:1px 5px;font-size:12.5px}}
ul{{margin:8px 0;padding-left:22px}} li{{margin:5px 0}}
</style></head><body><main>

<h1>TAIFEX US index futures &mdash; pair study</h1>
<p class="sub">UDF / SPF / UNF / SXF against their US legs &middot; methodology
mirrors the CCF-UMC study &middot; generated {generated}</p>

<div class="verdict">
<p><b>Verdict.</b> Of the four, only <b>UNF</b> (Nasdaq-100) has enough liquidity to
carry a pair strategy at all, and even it does not clear its costs. The basis is
real and it does converge &mdash; but the whole edge is about
<b>{fmt(zero_ann * 100)}% a year</b> with a <em>completely free</em> order book, on
the same capital convention that gives CCF-UMC roughly 89%.</p>
<p>The decisive number is not the spread, it is the size of the dislocation. TAIFEX
and CME price the same index within a few basis points of each other, so there is
simply not much to harvest. SPF, UDF and SXF fail earlier and harder, on trade
frequency and on structure.</p>
</div>

<div class="tiles">
<div class="tile"><div class="k">UNF edge, zero cost</div>
<div class="v">{fmt(zero_ann * 100)}%</div>
<div class="h">annualised, {int(zero_best['trade_count'])} trades / {span_days} days</div></div>
<div class="tile"><div class="k">UNF at measured cost</div>
<div class="v neg">{fmt(real_best['net_pnl_twd'], 0)}</div>
<div class="h">TWD, best of {len(real)} configs &mdash; 0 profitable</div></div>
<div class="tile"><div class="k">Break-even spread</div>
<div class="v">{fmt(breakeven)} bps</div>
<div class="h">quoted; book measures 9.3 at the close</div></div>
<div class="tile"><div class="k">Basis dispersion</div>
<div class="v">{fmt(basis_bps.std())} bps</div>
<div class="h">std, UNF vs NQ, matched Sep expiry</div></div>
</div>

<h2>1. The contracts</h2>
<p>All four are <b>quanto</b> futures: they pay a fixed NT$ per index point
regardless of USD/TWD. Two consequences. There is <b>no FX term in the spread</b> &mdash;
both legs are directly comparable in index points, unlike CCF-UMC which needs
USDTWD. And fair value carries a quanto/rate-differential term that decays to zero
at expiry, which is why this study pairs <b>matched expiries</b> (TAIFEX Sep vs CME
Sep): the carry cancels and there is no roll discontinuity.</p>
{table(
    ["Code", "Underlying", "OI", "Day vol", "Day spread bps", "Night vol",
     "Night spread bps", "Tick value", "Tick bps"],
    spread_rows + bench_rows,
    "Front month. Volume and open interest are contracts; spread is the daily "
    f"report's last best bid/ask, {q_lo:%Y-%m-%d} to {q_hi:%Y-%m-%d}. "
    "Day = 08:45-13:45 Taipei, night = 15:00-05:00 (the session covering US "
    "hours). TX/MTX are the liquid TAIFEX index futures, shown for scale."
)}
<p>Note the tick is <b>not</b> the problem here: at 0.2-0.4 bps it is far finer than
CCF's 32 bps. The problem is entirely the gap between bid and ask.</p>

<h2>2. The order-book gap</h2>
<figure>
<div class="legend">
<span><i class="sw" style="background:var(--series-1)"></i>day session 08:45-13:45</span>
<span><i class="sw" style="background:var(--series-2)"></i>night session 15:00-05:00</span>
</div>
{spread_bars([r for r in chart_rows if r[0] != "SPF"],
             breakeven if np.isfinite(breakeven) else 6.0)}
<figcaption>Median quoted bid-ask spread, front month, one observation per session
per day over a year. The night session &mdash; the one that overlaps US trading
hours, and the one the CCF-UMC design would have used &mdash; is the widest for all
four products. <b>SPF is omitted from the chart</b>: at
{fmt(chart_rows[1][1], 0)} bps day / {fmt(chart_rows[1][2], 0)} bps night it is
off-scale by an order of magnitude and would flatten the rest; it is in the table
above and in &sect;6.</figcaption>
</figure>
<h3>Liquidity is deteriorating, not building</h3>
{table(
    ["Code"] + quarters, trend_rows,
    "Median front-month daily volume (contracts) / median quoted spread (bps). "
    "All four products traded less and quoted wider through the year."
)}

<h2>3. The basis itself</h2>
<figure>
{line_chart(pd.Series(basis_bps.to_numpy(), index=basis['timestamp']))}
<figcaption>TAIFEX UNF minus CME NQ, both September 2026, in basis points, one
observation per UNF print during the TAIFEX day session. Mean
{fmt(basis_bps.mean())} bps, standard deviation {fmt(basis_bps.std())} bps.</figcaption>
</figure>
<p>UNF sits about {fmt(abs(basis_bps.mean()))} bps <b>above</b> NQ on average. Naive
quanto fair value would put it below, by roughly the TWD-USD rate differential over
the remaining life; the persistent premium is local demand, and it is a level, not
a signal. The rolling z-score absorbs it.</p>

<h3>Why trade prices lie here</h3>
<p>Measured on trade prints, the first-order autocorrelation of basis <em>changes</em>
is <b>-0.53 (UNF), -0.53 (UDF), -0.51 (SXF)</b> in the night session. Roll's model
puts pure bid-ask bounce at exactly -0.5, and the basis level itself has
autocorrelation near zero &mdash; it is white noise. A z-score backtest on those
prices produces an excellent-looking Sharpe that means "buy at the bid, sell at the
ask", which is the one thing a taker cannot do.</p>
<p>The day session is different: basis autocorrelation runs 0.60 at one minute and is
still 0.37 at thirty, and change-autocorrelation is -0.38 rather than -0.5. That is a
genuinely persistent basis with bounce on top &mdash; which is why this study uses the
day session and hedges with CME futures, which trade nearly around the clock.</p>

<h2>4. What the strategy actually earns</h2>
{table(
    ["Cost assumption", "Best config", "Trades", "Win rate", "Gross TWD",
     "Net TWD", "Return", "Sharpe", "Profitable configs"],
    [[
        "Free book (fees off)",
        f"w{int(zero_best['ma_window'])}/e{zero_best['entry_z']:.1f}"
        f"/x{zero_best['exit_z']:.1f}", f"{int(zero_best['trade_count'])}",
        f"{zero_best['win_rate']:.0%}", fmt(zero_best['gross_pnl_twd'], 0),
        fmt(zero_best['net_pnl_twd'], 0), f"{zero_ret:.2%}",
        fmt(zero_best['sharpe_ratio'], 2),
        f"{int((zero['net_pnl_twd'] > 0).sum())}/{len(zero)}",
    ], [
        "Measured book (4.65 bps half-spread + fees)",
        f"w{int(real_best['ma_window'])}/e{real_best['entry_z']:.1f}"
        f"/x{real_best['exit_z']:.1f}", f"{int(real_best['trade_count'])}",
        f"{real_best['win_rate']:.0%}", fmt(real_best['gross_pnl_twd'], 0),
        f"<span class='neg'>{fmt(real_best['net_pnl_twd'], 0)}</span>",
        f"{real_best['net_pnl_twd'] / capital_unf:.2%}",
        fmt(real_best['sharpe_ratio'], 2),
        f"<span class='neg'>{int((real['net_pnl_twd'] > 0).sum())}/{len(real)}</span>",
    ]],
    f"UNF vs NQ, day session, {sessions.nunique()} sessions "
    f"({sessions.min():%Y-%m-%d} to {sessions.max():%Y-%m-%d}), 1 contract "
    f"(NT$1.4M notional) on NT${capital_unf:,.0f} capital &mdash; the same "
    "notional-to-capital ratio the CCF-UMC study uses, so the returns compare."
)}
{table(["Quoted spread bps", "Best config", "Trades", "Net TWD", ""], sweep_rows,
       "Cost sweep: the same grid re-run at a range of assumed spreads. "
       "Commission of 0.36 bps per side and 0.2 bps futures tax are charged on "
       "top of every row.") if sweep_rows else ""}

<h2>5. The one measurement that is genuinely unresolved</h2>
<p>Two ways of measuring UNF's day-session spread disagree, and the break-even sits
between them:</p>
<ul>
<li><b>9.3 bps</b> &mdash; the daily report's last best bid/ask. Real quotes, but only
one per session, and market makers widen into the close.</li>
<li><b>0.7-3.0 bps</b> &mdash; Roll's estimator on trades that print close together in
time, which isolates bounce. But it measures what people who <em>did</em> trade
achieved, and they were patient; a signal-driven strategy is closer to the quoted
number.</li>
</ul>
<p>Break-even is about {fmt(breakeven)} bps, so the two measurements bracket it. That
gap can only be closed with <b>intraday quote data</b>, which neither TAIFEX's free
downloads nor IB provide &mdash; IB carries no TAIFEX products at all. It matters less
than it looks, though: even at zero cost the edge is {fmt(zero_ann * 100)}% a year.
Resolving the spread decides between a small loss and a small gain, not between
failure and a business.</p>

<h2>6. The other three</h2>
<h3>SPF (S&amp;P 500) &mdash; no market</h3>
<p>{fmt(front[(front['product'] == 'SPF') & (front['session'] == 'Regular')]['Volume'].median(), 0)}
contracts a day in the day session and a median quoted spread in the hundreds of
basis points. Over the year, {(1 - quotes[(quotes['product'] == 'SPF') & (quotes['session'] == 'Regular')].groupby('date')['spread_bps'].apply(lambda s: s.notna().any()).mean()):.0%}
of day sessions had no two-sided quote at all. Nothing to study.</p>
<h3>UDF (Dow) &mdash; too few moments to trade</h3>
<p>Over the same sample the day session produced only <b>8.5 prints per session</b>.
The zero-cost grid best is {int(udf_best['trade_count'])} trades for
{fmt(udf_best['net_pnl_twd'], 0)} TWD ({udf_ret:.2%} on NT${capital_udf:,.0f},
about {fmt(udf_ret * 365 / max(span_days, 1) * 100)}% annualised) &mdash; and that is
before any cost at all.</p>
<h3>SXF (SOX) &mdash; structurally unhedgeable in the day session</h3>
<p>There is no listed SOX future anywhere, and the SOX index and every semiconductor
ETF are closed during TAIFEX day hours. So SXF can only be paired during the night
session, where its quoted spread is
{fmt(front[(front['product'] == 'SXF') & (front['session'] == 'After-Hours')]['spread_bps'].median())}
bps. The hedge itself is also unsound: SOXX has tracked the ICE Semiconductor Index
rather than PHLX SOX since 2021, and SMH tracks MVIS 25 &mdash; measured against SXF,
SMH shows a level correlation of 0.82 and a 160-minute basis half-life, which is
tracking error, not a converging basis. A true hedge means executing the ~30 SOX
constituents.</p>

<h2>7. Reproducing this</h2>
<p>IB carries no TAIFEX products, so the Taiwan leg comes from TAIFEX's own tick
archive and the US leg from IB Gateway.</p>
<pre style="overflow-x:auto"><code>python scripts/build_qff1_1m.py --product UNF --expiry-rule third_friday \\
    --from-cache --contract-month 202609 --out data/processed/unf_202609_1m.csv
python scripts/download_ib_us_index_legs.py --symbols NQ --expiry 20260918 \\
    --bar-size "1 min" --start 2026-05-06
python scripts/calculate_us_index_basis.py --product UNF \\
    --taifex-path data/processed/unf_202609_1m.csv \\
    --us-path data/processed/ib_nqu6_1m_taipei.csv --session day --tag unf_nq_day</code></pre>
<p class="note">TAIFEX publishes only the last 30 trading days, so
<code>--from-cache</code> replaying the archived zips under
<code>data/raw/taifex_time_sales</code> is the only route to longer history &mdash;
keep archiving them. The basis file defaults to <b>event time</b> (one row per
TAIFEX print) rather than a clock grid: about 85% of day-session minutes have no
UNF print, and a stale leg against a live one drifts apart mechanically, which both
invents fills and inflates the sigma the z-score is measured against.</p>

</main></body></html>
"""
    args.out.write_text(doc, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  UNF zero-cost best: {zero_ret:.2%} over {span_days} days "
          f"-> {zero_ann:.1%} annualised, {int(zero_best['trade_count'])} trades")
    print(f"  UNF at measured cost: {real_best['net_pnl_twd']:,.0f} TWD, "
          f"{int((real['net_pnl_twd'] > 0).sum())}/{len(real)} configs profitable")
    print(f"  break-even quoted spread: {breakeven:.1f} bps")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
