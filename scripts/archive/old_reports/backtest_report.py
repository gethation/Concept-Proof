from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EQUITY_PATH = Path(
    "data/processed/qff_tsm_pair_backtest_equity_1m_qff_session.csv"
)
DEFAULT_TRADES_PATH = Path("data/processed/qff_tsm_pair_backtest_trades_qff_session.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/qff_tsm_pair_backtest_summary_qff_session.json")
DEFAULT_OUTPUT_PATH = Path("回測報告_qff_session.html")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Plotly HTML report for the QFF/TSM backtest."
    )
    parser.add_argument("--equity", type=Path, default=DEFAULT_EQUITY_PATH)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary JSON does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    return frame


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def css_class_for_pnl(value: Any) -> str:
    if value is None or pd.isna(value):
        return "neutral"
    numeric = float(value)
    if numeric > 0:
        return "positive"
    if numeric < 0:
        return "negative"
    return "neutral"


def escape(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return html.escape(str(value))


# Every component fill_costs charges, as (summary key, label). Derived from the
# summary rather than hard-coded three-at-a-time, so the breakdown keeps summing
# to total_fee_twd when a new cost lands: crossing cost was added to the total
# and this line went on showing TSM/QFF/Tax, hiding ~79% of the trading cost on
# executable-displacement runs.
FEE_COMPONENTS = (
    ("total_tsm_fee_twd", "TSM"),
    ("total_qff_fee_twd", "QFF"),
    ("total_qff_tax_twd", "Tax"),
    ("total_crossing_cost_twd", "Crossing"),
)


def sizing_text(parameters: dict[str, Any]) -> str:
    """Report the sizing input that actually drove the run. Fixed-lot mode
    ignores leg_notional_twd entirely, so printing it unconditionally labelled
    a 1-lot run as a 1,000,000 TWD one."""
    lots = parameters.get("qff_lots") or 0
    if lots:
        return f"Fixed {fmt_int(lots)} contracts/entry (leg notional ignored)"
    return f"Target leg notional {fmt_number(parameters.get('leg_notional_twd'))} TWD"


def qff_fee_text(parameters: dict[str, Any]) -> str:
    """Same story for the commission: the bps knob replaces the flat schedule,
    so showing the now-inert flat rate misstated the cost model."""
    bps = parameters.get("qff_fee_bps") or 0.0
    if bps:
        return f"{fmt_number(bps)} bps of contract notional per side"
    return (
        f"{fmt_number(parameters.get('qff_fee_per_contract_twd'))} "
        "TWD/contract/side"
    )


def fee_breakdown_text(summary: dict[str, Any]) -> str:
    parts = [
        f"{label} {fmt_number(summary[key])}"
        for key, label in FEE_COMPONENTS
        if summary.get(key) is not None
    ]
    total = summary.get("total_fee_twd")
    accounted = sum(
        float(summary[key]) for key, _ in FEE_COMPONENTS if summary.get(key) is not None
    )
    if total is not None and abs(float(total) - accounted) > 0.5:
        # A summary written by an older engine, or a cost this report does not
        # know about. Say so rather than quietly showing parts that do not add
        # up to the headline number.
        parts.append(f"Unattributed {fmt_number(float(total) - accounted)}")
    return " / ".join(parts)


def make_metric_cards(summary: dict[str, Any]) -> str:
    cards = [
        (
            "Net PnL",
            f"{fmt_number(summary.get('net_pnl_twd'))} TWD",
            f"Gross {fmt_number(summary.get('gross_pnl_twd'))} / Fee {fmt_number(summary.get('total_fee_twd'))}",
            css_class_for_pnl(summary.get("net_pnl_twd")),
        ),
        (
            "Return",
            fmt_pct(summary.get("return_pct")),
            f"Final equity {fmt_number(summary.get('final_equity_twd'))} TWD",
            css_class_for_pnl(summary.get("return_pct")),
        ),
        (
            "Trades",
            fmt_int(summary.get("trade_count")),
            f"Win rate {fmt_pct(summary.get('win_rate'))}",
            "neutral",
        ),
        (
            "Max Drawdown",
            fmt_pct(summary.get("max_drawdown_pct")),
            f"{fmt_number(summary.get('max_drawdown_twd'))} TWD",
            "negative",
        ),
        (
            "Total Fee",
            f"{fmt_number(summary.get('total_fee_twd'))} TWD",
            fee_breakdown_text(summary),
            "neutral",
        ),
        (
            "Exposure",
            fmt_pct(summary.get("exposure_ratio")),
            f"{fmt_int(summary.get('exposure_minutes'))} minutes",
            "neutral",
        ),
    ]
    return "\n".join(
        f"""
        <section class="metric-card {card_class}">
          <div class="metric-label">{escape(label)}</div>
          <div class="metric-value">{escape(value)}</div>
          <div class="metric-subtitle">{escape(subtitle)}</div>
        </section>
        """
        for label, value, subtitle, card_class in cards
    )


def make_parameters_table(summary: dict[str, Any]) -> str:
    parameters = summary.get("parameters", {})
    rows = [
        ("Data range", f"{summary.get('start')} to {summary.get('end')}"),
        ("Rows", fmt_int(summary.get("rows"))),
        ("Entry Z", fmt_number(parameters.get("entry_z"))),
        ("Exit Z", fmt_number(parameters.get("exit_z"))),
        ("Position sizing", sizing_text(parameters)),
        ("Initial capital", f"{fmt_number(parameters.get('initial_capital_twd'))} TWD"),
        ("Max entry delay", f"{fmt_int(parameters.get('max_entry_delay_minutes'))} minutes"),
        ("TSM fee", f"{fmt_number(parameters.get('tsm_fee_bps'))} bps per side"),
        ("QFF fee", qff_fee_text(parameters)),
        (
            "Executable displacement",
            f"{fmt_number(parameters.get('executable_displacement'), digits=4)} "
            "spread units per side",
        ),
        ("QFF tax rate", fmt_number(parameters.get("qff_tax_rate"), digits=6)),
        ("QFF multiplier", fmt_number(parameters.get("qff_contract_multiplier"))),
        (
            "Weekend close-only minutes",
            fmt_int(summary.get("weekend_session_close_only_minutes")),
        ),
        (
            "Week-end force-close points",
            fmt_int(summary.get("friday_session_end_force_close_minutes")),
        ),
        (
            "Week-end forced exits",
            fmt_int(summary.get("friday_session_forced_exits")),
        ),
        ("Fee defaults as of", summary.get("fee_defaults_as_of", "-")),
    ]
    return "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in rows
    )


def make_trades_table(trades: pd.DataFrame) -> str:
    if trades.empty:
        return '<p class="empty-state">No trades.</p>'

    trades = trades.copy()
    denominator = trades["actual_leg_notional_twd"].replace(0, pd.NA)
    trades["gross_pnl_pct"] = trades["gross_pnl_twd"] / denominator
    trades["fee_pct"] = trades["total_fee_twd"] / denominator
    trades["net_pnl_pct"] = trades["net_pnl_twd"] / denominator

    columns = [
        ("entry_time", "Entry"),
        ("exit_time", "Exit"),
        ("direction", "Direction"),
        ("entry_fill_zscore", "Entry Z"),
        ("exit_fill_zscore", "Exit Z"),
        ("holding_minutes", "Hold Min"),
        ("qff_contracts", "QFF Ctr"),
        ("actual_leg_notional_twd", "Leg TWD"),
        ("gross_pnl_pct", "Gross %"),
        ("fee_pct", "Fee %"),
        ("net_pnl_pct", "Net %"),
        ("exit_reason", "Exit Reason"),
    ]
    header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body_rows: list[str] = []
    for _, row in trades.iterrows():
        cells: list[str] = []
        for column, _ in columns:
            value = row[column]
            cell_class = ""
            if column in {"gross_pnl_pct", "net_pnl_pct"}:
                cell_class = f' class="number {css_class_for_pnl(value)}"'
                value = fmt_pct(value)
            elif column == "fee_pct":
                cell_class = ' class="number"'
                value = fmt_pct(value)
            elif column in {
                "entry_fill_zscore",
                "exit_fill_zscore",
                "actual_leg_notional_twd",
            }:
                cell_class = ' class="number"'
                value = fmt_number(value)
            elif column in {"holding_minutes", "qff_contracts"}:
                cell_class = ' class="number"'
                value = fmt_int(value)
            cells.append(f"<td{cell_class}>{escape(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="table-wrap">
      <table class="trades-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def make_report(equity: pd.DataFrame, trades: pd.DataFrame, summary: dict[str, Any]) -> str:
    equity = equity.copy()
    equity["_timestamp_dt"] = pd.to_datetime(equity["timestamp"], utc=True).dt.tz_convert(
        "Asia/Taipei"
    )
    daily_equity = (
        equity.set_index("_timestamp_dt")
        .resample("D")
        .last()
        .dropna(subset=["equity"])
        .copy()
    )
    daily_equity["daily_running_max_equity"] = daily_equity["equity"].cummax()
    daily_equity["daily_drawdown_pct"] = (
        daily_equity["equity"] / daily_equity["daily_running_max_equity"] - 1.0
    )
    daily_equity["date"] = daily_equity.index.strftime("%Y-%m-%d")

    equity_data = {
        "timestamp": daily_equity["date"].tolist(),
        "equity": daily_equity["equity"].round(6).tolist(),
        "runningMax": daily_equity["daily_running_max_equity"].round(6).tolist(),
        "drawdown": daily_equity["daily_drawdown_pct"].round(8).tolist(),
    }

    winning_exits = trades.loc[trades["net_pnl_twd"] > 0].copy()
    losing_exits = trades.loc[trades["net_pnl_twd"] <= 0].copy()
    daily_equity_by_date = daily_equity.set_index("date")["equity"]
    for frame in [winning_exits, losing_exits]:
        if not frame.empty:
            frame["exit_date"] = (
                pd.to_datetime(frame["exit_time"], utc=True)
                .dt.tz_convert("Asia/Taipei")
                .dt.strftime("%Y-%m-%d")
            )
            frame["exit_equity"] = frame["exit_date"].map(daily_equity_by_date)
            frame["net_pnl_pct"] = (
                frame["net_pnl_twd"] / frame["actual_leg_notional_twd"]
            )

    trade_markers = {
        "winning": {
            "x": winning_exits.get("exit_date", pd.Series(dtype=str)).astype(str).tolist(),
            "y": winning_exits.get("exit_equity", pd.Series(dtype=float)).round(6).tolist(),
            "text": winning_exits.get("net_pnl_pct", pd.Series(dtype=float))
            .map(lambda value: f"+{value * 100:.2f}%")
            .tolist(),
        },
        "losing": {
            "x": losing_exits.get("exit_date", pd.Series(dtype=str)).astype(str).tolist(),
            "y": losing_exits.get("exit_equity", pd.Series(dtype=float)).round(6).tolist(),
            "text": losing_exits.get("net_pnl_pct", pd.Series(dtype=float))
            .map(lambda value: f"{value * 100:.2f}%")
            .tolist(),
        },
    }

    equity_json = json.dumps(equity_data, ensure_ascii=False)
    marker_json = json.dumps(trade_markers, ensure_ascii=False)
    metric_cards = make_metric_cards(summary)
    parameter_rows = make_parameters_table(summary)
    trades_table = make_trades_table(trades)

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QFF-TSM 回測報告</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #637083;
      --line: #d9dee7;
      --blue: #2563eb;
      --green: #15803d;
      --red: #b91c1c;
      --amber: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
      line-height: 1.5;
    }}
    header {{
      padding: 28px 32px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    main {{ padding: 24px 32px 36px; }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 18px; margin-bottom: 14px; }}
    .subtitle {{ margin-top: 6px; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .metric-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    .metric-card {{ padding: 16px; }}
    .metric-label {{ color: var(--muted); font-size: 13px; }}
    .metric-value {{ margin-top: 4px; font-size: 24px; font-weight: 700; }}
    .metric-subtitle {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .positive .metric-value, .positive {{ color: var(--green); }}
    .negative .metric-value, .negative {{ color: var(--red); }}
    .neutral {{ color: inherit; }}
    .panel {{ padding: 16px; margin-bottom: 18px; }}
    .chart {{ width: 100%; height: 430px; }}
    #drawdown-chart {{ height: 300px; }}
    .two-col {{
      display: grid;
      grid-template-columns: minmax(280px, 420px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .params-table, .trades-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .params-table th, .params-table td,
    .trades-table th, .trades-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    .params-table th {{ width: 160px; color: var(--muted); font-weight: 600; }}
    .trades-table th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #eef2f7;
      color: #334155;
      font-weight: 700;
    }}
    .table-wrap {{
      max-height: 560px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .number {{ text-align: right !important; font-variant-numeric: tabular-nums; }}
    .empty-state {{ color: var(--muted); }}
    .section-note {{
      margin: -6px 0 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .two-col {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 640px) {{
      .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 23px; }}
      .metric-value {{ font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>QFF-TSM 回測報告</h1>
    <div class="subtitle">資料區間：{escape(summary.get("start"))} 到 {escape(summary.get("end"))}</div>
  </header>
  <main>
    <section class="grid">
      {metric_cards}
    </section>

    <section class="panel">
      <h2>Equity Curve (Daily)</h2>
      <div id="equity-chart" class="chart"></div>
    </section>

    <section class="panel">
      <h2>Drawdown (%)</h2>
      <div id="drawdown-chart" class="chart"></div>
    </section>

    <section class="two-col">
      <section class="panel">
        <h2>回測參數</h2>
        <table class="params-table"><tbody>{parameter_rows}</tbody></table>
      </section>

      <section class="panel">
        <h2>交易清單</h2>
        <p class="section-note">Gross %、Fee %、Net % 皆以該筆 Leg TWD 為分母。</p>
        {trades_table}
      </section>
    </section>
  </main>

  <script>
    const equityData = {equity_json};
    const tradeMarkers = {marker_json};
    const commonLayout = {{
      margin: {{ l: 72, r: 24, t: 12, b: 46 }},
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{ family: "-apple-system, BlinkMacSystemFont, Segoe UI, Noto Sans TC, sans-serif" }},
      hovermode: "x unified",
      xaxis: {{ gridcolor: "#edf0f5", rangeslider: {{ visible: false }} }},
      yaxis: {{ gridcolor: "#edf0f5", tickformat: ",.0f" }}
    }};

    Plotly.newPlot("equity-chart", [
      {{
        x: equityData.timestamp,
        y: equityData.equity,
        type: "scatter",
        mode: "lines",
        name: "Equity",
        line: {{ color: "#2563eb", width: 2 }}
      }},
      {{
        x: equityData.timestamp,
        y: equityData.runningMax,
        type: "scatter",
        mode: "lines",
        name: "Running Max",
        line: {{ color: "#94a3b8", width: 1, dash: "dot" }}
      }},
      {{
        x: tradeMarkers.winning.x,
        y: tradeMarkers.winning.y,
        text: tradeMarkers.winning.text,
        type: "scatter",
        mode: "markers",
        name: "Winning Exit",
        marker: {{ color: "#15803d", size: 7, symbol: "triangle-up" }},
        hovertemplate: "%{{x}}<br>%{{text}}<extra></extra>"
      }},
      {{
        x: tradeMarkers.losing.x,
        y: tradeMarkers.losing.y,
        text: tradeMarkers.losing.text,
        type: "scatter",
        mode: "markers",
        name: "Losing Exit",
        marker: {{ color: "#b91c1c", size: 7, symbol: "triangle-down" }},
        hovertemplate: "%{{x}}<br>%{{text}}<extra></extra>"
      }}
    ], {{
      ...commonLayout,
      yaxis: {{ ...commonLayout.yaxis, title: "Equity (TWD)" }},
      legend: {{ orientation: "h", y: 1.08, x: 0 }}
    }}, {{ responsive: true, displaylogo: false }});

    Plotly.newPlot("drawdown-chart", [
      {{
        x: equityData.timestamp,
        y: equityData.drawdown,
        type: "scatter",
        mode: "lines",
        name: "Drawdown",
        fill: "tozeroy",
        line: {{ color: "#b91c1c", width: 1.5 }}
      }}
    ], {{
      ...commonLayout,
      yaxis: {{
        ...commonLayout.yaxis,
        title: "Drawdown (%)",
        tickformat: ".2%"
      }},
      showlegend: false
    }}, {{ responsive: true, displaylogo: false }});
  </script>
</body>
</html>
"""


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    equity = read_csv(
        args.equity,
        {
            "timestamp",
            "equity",
            "running_max_equity",
            "drawdown_twd",
        },
    )
    trades = read_csv(
        args.trades,
        {
            "entry_time",
            "exit_time",
            "direction",
            "entry_fill_zscore",
            "exit_fill_zscore",
            "holding_minutes",
            "qff_contracts",
            "actual_leg_notional_twd",
            "gross_pnl_twd",
            "total_fee_twd",
            "net_pnl_twd",
            "exit_reason",
        },
    )
    summary = read_json(args.summary)
    report = make_report(equity, trades, summary)
    args.out.write_text(report, encoding="utf-8")
    safe_output = str(args.out).encode("unicode_escape").decode("ascii")
    print(f"Wrote report to {safe_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
