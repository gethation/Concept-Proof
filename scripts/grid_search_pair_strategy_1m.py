from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

import backtest_pair_strategy_1m as backtest
import calculate_spread_zscore_1m as zscore_calc


DEFAULT_WINDOWS = [249, 498, 997, 1994]
DEFAULT_ENTRY_Z = [1.5, 2.0, 2.5, 3.0]
DEFAULT_EXIT_Z = [0.0, 0.5, 1.0]
DEFAULT_SUMMARY_PATH = Path("data/processed/qff_tsm_parameter_grid_qff_session.csv")
DEFAULT_REPORT_PATH = Path("qff_tsm_parameter_grid_report_qff_session.html")
DEFAULT_ANNUAL_TRADING_DAYS = 252.0
DEFAULT_MIN_SHARPE_TRADES = 20


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value")
    if any(item <= 1 for item in values):
        raise argparse.ArgumentTypeError("All windows must be greater than 1")
    return values


def format_float_list(values: list[float]) -> str:
    return ",".join(f"{value:g}" for value in values)


def format_int_list(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a QFF/TSM parameter grid over MA window, entry z, and exit z."
    )
    parser.add_argument(
        "--spread-path",
        type=Path,
        default=zscore_calc.DEFAULT_SPREAD_PATH,
        help="QFF/TSM spread CSV before z-score columns are added.",
    )
    parser.add_argument("--qff-ohlcv", type=Path, default=backtest.DEFAULT_QFF_OHLCV_PATH)
    parser.add_argument("--tsm-ohlcv", type=Path, default=backtest.DEFAULT_TSM_OHLCV_PATH)
    parser.add_argument(
        "--usdttwd-ohlcv", type=Path, default=backtest.DEFAULT_USDTTWD_OHLCV_PATH
    )
    parser.add_argument(
        "--windows",
        type=parse_int_list,
        default=DEFAULT_WINDOWS,
        help=f"Comma-separated MA/z-score windows. Default: {format_int_list(DEFAULT_WINDOWS)}",
    )
    parser.add_argument(
        "--entry-z",
        type=parse_float_list,
        default=DEFAULT_ENTRY_Z,
        help=f"Comma-separated entry z thresholds. Default: {format_float_list(DEFAULT_ENTRY_Z)}",
    )
    parser.add_argument(
        "--exit-z",
        type=parse_float_list,
        default=DEFAULT_EXIT_Z,
        help=f"Comma-separated exit z thresholds. Default: {format_float_list(DEFAULT_EXIT_Z)}",
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--leg-notional-twd", type=float, default=1_000_000.0)
    parser.add_argument("--initial-capital-twd", type=float, default=2_000_000.0)
    parser.add_argument("--max-entry-delay-minutes", type=int, default=15)
    parser.add_argument("--tsm-fee-bps", type=float, default=backtest.DEFAULT_TSM_FEE_BPS)
    parser.add_argument(
        "--qff-fee-per-contract-twd",
        type=float,
        default=backtest.DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
    )
    parser.add_argument("--qff-tax-rate", type=float, default=backtest.DEFAULT_QFF_TAX_RATE)
    parser.add_argument(
        "--qff-contract-multiplier",
        type=float,
        default=backtest.DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )
    parser.add_argument(
        "--annual-trading-days",
        type=float,
        default=DEFAULT_ANNUAL_TRADING_DAYS,
        help="Annualization factor for daily-equity Sharpe ratio.",
    )
    parser.add_argument(
        "--min-sharpe-trades",
        type=int,
        default=DEFAULT_MIN_SHARPE_TRADES,
        help="Minimum trade count for inclusion in Sharpe primary ranking.",
    )
    parser.add_argument(
        "--title",
        default="QFF-TSM Parameter Grid",
        help="Report page title (e.g. \"CCF-UMC Parameter Grid (5m)\").",
    )
    parser.add_argument(
        "--seed-spread-path",
        type=Path,
        default=None,
        help=(
            "Optional spread CSV prepended for rolling-stat warmup only (see "
            "calculate_spread_zscore_1m.py --seed-spread-path); removes the "
            "per-window warmup so all windows trade the same period."
        ),
    )
    return parser.parse_args(argv)


def make_params(args: argparse.Namespace, entry_z: float, exit_z: float) -> backtest.BacktestParams:
    return backtest.BacktestParams(
        entry_z=entry_z,
        exit_z=exit_z,
        leg_notional_twd=args.leg_notional_twd,
        initial_capital_twd=args.initial_capital_twd,
        max_entry_delay_minutes=args.max_entry_delay_minutes,
        tsm_fee_bps=args.tsm_fee_bps,
        qff_fee_per_contract_twd=args.qff_fee_per_contract_twd,
        qff_tax_rate=args.qff_tax_rate,
        qff_contract_multiplier=args.qff_contract_multiplier,
    )


def calculate_daily_return_stats(
    equity: pd.DataFrame, initial_capital_twd: float, annual_trading_days: float
) -> dict[str, float | int | None]:
    data = equity[["timestamp", "equity"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert(
        "Asia/Taipei"
    )
    data["date"] = data["timestamp"].dt.normalize()
    daily_equity = data.groupby("date", sort=True)["equity"].last()
    if daily_equity.empty:
        return {
            "daily_return_count": 0,
            "daily_return_mean": None,
            "daily_return_std": None,
            "sharpe_ratio": None,
        }

    daily_returns = daily_equity.pct_change()
    daily_returns.iloc[0] = daily_equity.iloc[0] / initial_capital_twd - 1.0
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return {
            "daily_return_count": 0,
            "daily_return_mean": None,
            "daily_return_std": None,
            "sharpe_ratio": None,
        }

    mean = float(daily_returns.mean())
    std = float(daily_returns.std(ddof=1)) if len(daily_returns) >= 2 else None
    sharpe = None
    if std is not None and std > 0:
        sharpe = mean / std * (annual_trading_days**0.5)

    return {
        "daily_return_count": int(len(daily_returns)),
        "daily_return_mean": mean,
        "daily_return_std": std,
        "sharpe_ratio": sharpe,
    }


def return_over_max_drawdown(return_pct: Any, max_drawdown_pct: Any) -> float | None:
    if return_pct is None or max_drawdown_pct is None:
        return None
    if pd.isna(return_pct) or pd.isna(max_drawdown_pct):
        return None
    drawdown = abs(float(max_drawdown_pct))
    if drawdown == 0:
        return None
    return float(return_pct) / drawdown


def result_row(
    *,
    ma_window: int,
    entry_z: float,
    exit_z: float,
    zscore_valid_rows: int,
    daily_stats: dict[str, float | int | None],
    min_sharpe_trades: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    fields = [
        "trade_count",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "gross_pnl_twd",
        "net_pnl_twd",
        "return_pct",
        "max_drawdown_twd",
        "max_drawdown_pct",
        "profit_factor",
        "avg_trade_pnl_twd",
        "exposure_minutes",
        "exposure_ratio",
        "total_fee_twd",
        "start",
        "end",
    ]
    ret_over_dd = return_over_max_drawdown(
        summary.get("return_pct"), summary.get("max_drawdown_pct")
    )
    sharpe_ratio = daily_stats.get("sharpe_ratio")
    trade_count = int(summary.get("trade_count") or 0)
    net_pnl = float(summary.get("net_pnl_twd") or 0.0)
    sharpe_rank_eligible = (
        sharpe_ratio is not None
        and not pd.isna(sharpe_ratio)
        and net_pnl > 0
        and trade_count >= min_sharpe_trades
    )
    row = {
        "ma_window": ma_window,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "zscore_valid_rows": zscore_valid_rows,
        "sharpe_ratio": sharpe_ratio,
        "daily_return_count": daily_stats.get("daily_return_count"),
        "daily_return_mean": daily_stats.get("daily_return_mean"),
        "daily_return_std": daily_stats.get("daily_return_std"),
        "return_over_max_dd": ret_over_dd,
        "sharpe_rank_eligible": sharpe_rank_eligible,
    }
    row.update({field: summary.get(field) for field in fields})
    return row


def run_grid(args: argparse.Namespace) -> pd.DataFrame:
    if args.annual_trading_days <= 0:
        raise RuntimeError("--annual-trading-days must be positive")
    if args.min_sharpe_trades < 0:
        raise RuntimeError("--min-sharpe-trades must be non-negative")

    spread = zscore_calc.read_spread_frame(args.spread_path)
    seed_frame = (
        zscore_calc.read_spread_frame(args.seed_spread_path)
        if args.seed_spread_path is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    total = len(args.windows) * len(args.entry_z) * len(args.exit_z)
    completed = 0

    for window in args.windows:
        zframe = zscore_calc.calculate_zscore(spread, window, seed_frame=seed_frame)
        zscore_valid_rows = int(zframe["zscore_valid"].sum())
        backtest_frame = backtest.add_entry_open_prices(
            zframe,
            qff_ohlcv_path=args.qff_ohlcv,
            tsm_ohlcv_path=args.tsm_ohlcv,
            usdttwd_ohlcv_path=args.usdttwd_ohlcv,
        )

        for entry_z in args.entry_z:
            for exit_z in args.exit_z:
                params = make_params(args, entry_z=entry_z, exit_z=exit_z)
                result = backtest.run_backtest(backtest_frame, params)
                daily_stats = calculate_daily_return_stats(
                    result.equity,
                    initial_capital_twd=args.initial_capital_twd,
                    annual_trading_days=args.annual_trading_days,
                )
                rows.append(
                    result_row(
                        ma_window=window,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        zscore_valid_rows=zscore_valid_rows,
                        daily_stats=daily_stats,
                        min_sharpe_trades=args.min_sharpe_trades,
                        summary=result.summary,
                    )
                )
                completed += 1
                sharpe_text = (
                    f"{daily_stats['sharpe_ratio']:.4f}"
                    if daily_stats["sharpe_ratio"] is not None
                    else "-"
                )
                print(
                    "Completed "
                    f"{completed}/{total}: window={window}, "
                    f"entry_z={entry_z:g}, exit_z={exit_z:g}, "
                    f"net_pnl={result.summary['net_pnl_twd']:.2f}, "
                    f"sharpe={sharpe_text}"
                )

    frame = pd.DataFrame(rows)
    frame["rank_net_pnl"] = (
        frame["net_pnl_twd"].rank(method="min", ascending=False).astype("Int64")
    )
    frame["rank_return_over_max_dd"] = (
        frame["return_over_max_dd"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    frame = frame.sort_values(
        [
            "sharpe_rank_eligible",
            "sharpe_ratio",
            "net_pnl_twd",
            "max_drawdown_pct",
            "profit_factor",
        ],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame.insert(1, "rank_sharpe", frame["rank"])
    expected_rows = total
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} grid rows, got {len(frame)}")
    return frame


def write_summary_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def css_class_for_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "neutral"
    return "positive" if float(value) >= 0 else "negative"


def table_rows(frame: pd.DataFrame, limit: int | None = None) -> str:
    rows = frame if limit is None else frame.head(limit)
    html_rows = []
    for row in rows.itertuples(index=False):
        html_rows.append(
            "<tr>"
            f"<td>{fmt_int(row.rank)}</td>"
            f"<td>{fmt_int(row.rank_net_pnl)}</td>"
            f"<td>{fmt_int(row.ma_window)}</td>"
            f"<td>{fmt_number(row.entry_z, 2)}</td>"
            f"<td>{fmt_number(row.exit_z, 2)}</td>"
            f"<td>{fmt_int(row.trade_count)}</td>"
            f"<td>{fmt_number(row.sharpe_ratio, 2)}</td>"
            f"<td>{fmt_pct(row.win_rate)}</td>"
            f"<td class='{css_class_for_value(row.net_pnl_twd)}'>{fmt_number(row.net_pnl_twd)}</td>"
            f"<td>{fmt_number(row.return_over_max_dd, 2)}</td>"
            f"<td>{fmt_pct(row.return_pct)}</td>"
            f"<td class='negative'>{fmt_pct(row.max_drawdown_pct)}</td>"
            f"<td>{fmt_number(row.profit_factor, 2)}</td>"
            f"<td>{fmt_pct(row.exposure_ratio)}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def make_results_table(frame: pd.DataFrame, limit: int | None = None) -> str:
    return f"""
    <table>
      <thead>
        <tr>
          <th>Sharpe Rank</th><th>Net Rank</th><th>MA</th><th>Entry Z</th><th>Exit Z</th>
          <th>Trades</th><th>Sharpe</th><th>Win Rate</th><th>Net PnL</th>
          <th>Ret/DD</th><th>Return</th><th>Max DD %</th><th>PF</th><th>Exposure</th>
        </tr>
      </thead>
      <tbody>{table_rows(frame, limit)}</tbody>
    </table>
    """


def make_window_summary(frame: pd.DataFrame) -> str:
    rows = []
    for window, group in frame.groupby("ma_window", sort=True):
        best_sharpe = group.sort_values(
            ["sharpe_rank_eligible", "sharpe_ratio", "net_pnl_twd"],
            ascending=[False, False, False],
            na_position="last",
        ).iloc[0]
        best_net = group.sort_values("net_pnl_twd", ascending=False).iloc[0]
        rows.append(
            "<tr>"
            f"<td>{fmt_int(window)}</td>"
            f"<td>{fmt_int(len(group))}</td>"
            f"<td>{fmt_number(best_sharpe['entry_z'], 2)} / {fmt_number(best_sharpe['exit_z'], 2)}</td>"
            f"<td>{fmt_number(best_sharpe['sharpe_ratio'], 2)}</td>"
            f"<td class='{css_class_for_value(best_sharpe['net_pnl_twd'])}'>{fmt_number(best_sharpe['net_pnl_twd'])}</td>"
            f"<td>{fmt_number(best_sharpe['return_over_max_dd'], 2)}</td>"
            f"<td class='{css_class_for_value(best_net['net_pnl_twd'])}'>{fmt_number(best_net['net_pnl_twd'])}</td>"
            f"<td>{fmt_number(group['net_pnl_twd'].mean())}</td>"
            f"<td class='negative'>{fmt_pct(best_sharpe['max_drawdown_pct'])}</td>"
            f"<td>{fmt_int(int(best_sharpe['trade_count']))}</td>"
            "</tr>"
        )
    return f"""
    <table>
      <thead>
        <tr>
          <th>MA</th><th>Combos</th><th>Best Sharpe Entry/Exit</th><th>Best Sharpe</th>
          <th>Best Sharpe Net PnL</th><th>Best Sharpe Ret/DD</th><th>Best Net PnL</th>
          <th>Avg Net PnL</th><th>Best Sharpe Max DD %</th><th>Best Sharpe Trades</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def heat_color(value: float, min_value: float, max_value: float, positive_good: bool) -> str:
    if pd.isna(value) or min_value == max_value:
        return "rgba(148, 163, 184, 0.12)"
    if positive_good:
        scale = (value - min_value) / (max_value - min_value)
        red = int(239 - 205 * scale)
        green = int(68 + 117 * scale)
        blue = int(68 + 61 * scale)
    else:
        scale = (value - min_value) / (max_value - min_value)
        red = int(34 + 205 * scale)
        green = int(197 - 129 * scale)
        blue = int(94 - 26 * scale)
    return f"rgba({red}, {green}, {blue}, 0.34)"


def make_heatmap(frame: pd.DataFrame, metric: str, title: str, positive_good: bool) -> str:
    sections = []
    metric_min = float(frame[metric].min())
    metric_max = float(frame[metric].max())
    formatter = fmt_pct if metric.endswith("_pct") else fmt_number
    for window, group in frame.groupby("ma_window", sort=True):
        pivot = group.pivot(index="entry_z", columns="exit_z", values=metric).sort_index()
        header = "".join(f"<th>{fmt_number(exit_z, 2)}</th>" for exit_z in pivot.columns)
        rows = []
        for entry_z, values in pivot.iterrows():
            cells = []
            for value in values:
                color = heat_color(float(value), metric_min, metric_max, positive_good)
                cells.append(
                    f"<td style='background:{color}'>{formatter(value)}</td>"
                )
            rows.append(
                f"<tr><th>{fmt_number(entry_z, 2)}</th>{''.join(cells)}</tr>"
            )
        sections.append(
            f"""
            <section>
              <h3>{html.escape(title)} - MA {fmt_int(window)}</h3>
              <table class="heatmap">
                <thead><tr><th>Entry \\ Exit</th>{header}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </section>
            """
        )
    return "\n".join(sections)


def make_report(
    frame: pd.DataFrame,
    elapsed_seconds: float,
    min_sharpe_trades: int,
    annual_trading_days: float,
    title: str = "QFF-TSM Parameter Grid",
) -> str:
    best = frame.iloc[0]
    eligible_count = int(frame["sharpe_rank_eligible"].sum())
    generated_at = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S+08:00")
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #657282;
      --line: #d8dee6;
      --positive: #087f5b;
      --negative: #c92a2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    h3 {{ margin: 18px 0 10px; font-size: 15px; color: var(--muted); }}
    .subtitle {{ color: var(--muted); margin-bottom: 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 24px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    .value {{ display: block; margin-top: 8px; font-size: 22px; font-weight: 700; }}
    .subvalue {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    th {{ background: #eef2f6; color: #344054; font-size: 12px; font-weight: 650; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    .positive {{ color: var(--positive); font-weight: 650; }}
    .negative {{ color: var(--negative); font-weight: 650; }}
    .neutral {{ color: var(--muted); }}
    .table-wrap {{ overflow-x: auto; }}
    .heatmaps {{ display: grid; gap: 16px; }}
    .heatmap td {{ min-width: 104px; }}
    @media (max-width: 860px) {{
      main {{ padding: 16px; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">Generated {generated_at}. Ranked by daily-equity Sharpe ratio. Annualized with {annual_trading_days:g} trading days; Sharpe rank requires net PnL &gt; 0 and at least {min_sharpe_trades} trades. Runtime {elapsed_seconds:.2f}s.</div>
  <section class="cards">
    <div class="card">
      <div class="label">Combos</div>
      <span class="value">{fmt_int(len(frame))}</span>
      <div class="subvalue">{fmt_int(eligible_count)} Sharpe-ranked</div>
    </div>
    <div class="card">
      <div class="label">Best Sharpe</div>
      <span class="value {css_class_for_value(best['sharpe_ratio'])}">{fmt_number(best['sharpe_ratio'], 2)}</span>
      <div class="subvalue">MA {fmt_int(best['ma_window'])}, entry {fmt_number(best['entry_z'], 2)}, exit {fmt_number(best['exit_z'], 2)}</div>
    </div>
    <div class="card">
      <div class="label">Best Sharpe Net PnL</div>
      <span class="value {css_class_for_value(best['net_pnl_twd'])}">{fmt_number(best['net_pnl_twd'])}</span>
      <div class="subvalue">{fmt_pct(best['return_pct'])} return</div>
    </div>
    <div class="card">
      <div class="label">Best Max DD %</div>
      <span class="value negative">{fmt_pct(best['max_drawdown_pct'])}</span>
      <div class="subvalue">Ret/DD {fmt_number(best['return_over_max_dd'], 2)}, PF {fmt_number(best['profit_factor'], 2)}</div>
    </div>
  </section>

  <h2>Top 10 by Sharpe</h2>
  <div class="table-wrap">{make_results_table(frame, limit=10)}</div>

  <h2>MA Window Summary</h2>
  <div class="table-wrap">{make_window_summary(frame)}</div>

  <h2>Sharpe Heatmap</h2>
  <div class="heatmaps">{make_heatmap(frame, "sharpe_ratio", "Sharpe", positive_good=True)}</div>

  <h2>Return / Max Drawdown Heatmap</h2>
  <div class="heatmaps">{make_heatmap(frame, "return_over_max_dd", "Return / Max DD", positive_good=True)}</div>

  <h2>Net PnL Heatmap</h2>
  <div class="heatmaps">{make_heatmap(frame, "net_pnl_twd", "Net PnL", positive_good=True)}</div>

  <h2>Max Drawdown % Heatmap</h2>
  <div class="heatmaps">{make_heatmap(frame, "max_drawdown_pct", "Max Drawdown %", positive_good=True)}</div>

  <h2>Full Sharpe Ranking</h2>
  <div class="table-wrap">{make_results_table(frame)}</div>
</main>
</body>
</html>
"""


def write_report(
    frame: pd.DataFrame,
    path: Path,
    elapsed_seconds: float,
    min_sharpe_trades: int,
    annual_trading_days: float,
    title: str = "QFF-TSM Parameter Grid",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        make_report(
            frame,
            elapsed_seconds,
            min_sharpe_trades=min_sharpe_trades,
            annual_trading_days=annual_trading_days,
            title=title,
        ),
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    frame = run_grid(args)
    elapsed_seconds = time.perf_counter() - started

    write_summary_csv(frame, args.summary_out)
    write_report(
        frame,
        args.report_out,
        elapsed_seconds,
        min_sharpe_trades=args.min_sharpe_trades,
        annual_trading_days=args.annual_trading_days,
        title=args.title,
    )

    best = frame.iloc[0]
    print(f"Wrote {len(frame):,} grid rows to {args.summary_out}")
    print(f"Wrote grid report to {args.report_out}")
    print(
        "Best: "
        f"rank=1, ma_window={int(best['ma_window'])}, "
        f"entry_z={best['entry_z']:.2f}, exit_z={best['exit_z']:.2f}, "
        f"sharpe={best['sharpe_ratio']:.4f}, "
        f"net_pnl={best['net_pnl_twd']:.2f}, "
        f"ret_over_dd={best['return_over_max_dd']:.4f}, "
        f"return={best['return_pct']:.4%}, "
        f"max_dd={best['max_drawdown_pct']:.4%}"
    )
    print(f"Elapsed: {elapsed_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
