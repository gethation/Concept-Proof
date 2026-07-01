import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRADES_PATH = Path("data/processed/qff_tsm_pair_backtest_trades_qff_session.csv")
DEFAULT_OUTPUT_PATH = Path("hold_profit_report_qff_session.html")


DIRECTION_STYLE = {
    "short_tsm_long_qff": {"color": "#2563eb", "symbol": "circle"},
    "long_tsm_short_qff": {"color": "#d97706", "symbol": "diamond"},
}

HOLD_BUCKET_BINS = [0, 30, 120, 480, math.inf]
HOLD_BUCKET_LABELS = ["0-30", "31-120", "121-480", "481+"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a standalone Hold Min vs Gross Profit % Plotly report."
    )
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


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


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def beta_continued_fraction(a_value: float, b_value: float, x_value: float) -> float:
        max_iter = 200
        epsilon = 3.0e-14
        fpmin = 1.0e-300
        qab = a_value + b_value
        qap = a_value + 1.0
        qam = a_value - 1.0
        c = 1.0
        d = 1.0 - qab * x_value / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d

        for iteration in range(1, max_iter + 1):
            m2 = 2 * iteration
            aa = (
                iteration
                * (b_value - iteration)
                * x_value
                / ((qam + m2) * (a_value + m2))
            )
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c

            aa = (
                -(a_value + iteration)
                * (qab + iteration)
                * x_value
                / ((a_value + m2) * (qap + m2))
            )
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < epsilon:
                break
        return h

    log_beta_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    beta_term = math.exp(log_beta_term)
    threshold = (a + 1.0) / (a + b + 2.0)
    if x < threshold:
        return beta_term * beta_continued_fraction(a, b, x) / a
    return 1.0 - beta_term * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0 or pd.isna(value):
        return math.nan
    if value == 0:
        return 0.5
    x_value = degrees_of_freedom / (degrees_of_freedom + value * value)
    beta_value = regularized_incomplete_beta(
        degrees_of_freedom / 2.0, 0.5, x_value
    )
    if value > 0:
        return 1.0 - 0.5 * beta_value
    return 0.5 * beta_value


def two_sided_t_p_value(t_stat: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0 or pd.isna(t_stat):
        return math.nan
    tail = 1.0 - student_t_cdf(abs(t_stat), degrees_of_freedom)
    return min(1.0, max(0.0, 2.0 * tail))


def two_sided_normal_p_value(z_stat: float) -> float:
    if pd.isna(z_stat):
        return math.nan
    return math.erfc(abs(z_stat) / math.sqrt(2.0))


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    count = len(x_values)
    if count < 2:
        return math.nan
    mean_x = sum(x_values) / count
    mean_y = sum(y_values) / count
    centered_x = [value - mean_x for value in x_values]
    centered_y = [value - mean_y for value in y_values]
    ss_x = sum(value * value for value in centered_x)
    ss_y = sum(value * value for value in centered_y)
    if ss_x == 0.0 or ss_y == 0.0:
        return math.nan
    ss_xy = sum(x_value * y_value for x_value, y_value in zip(centered_x, centered_y))
    return ss_xy / math.sqrt(ss_x * ss_y)


def correlation_p_value(correlation: float, count: int) -> float:
    if count < 3 or pd.isna(correlation):
        return math.nan
    if abs(correlation) >= 1.0:
        return 0.0
    t_stat = correlation * math.sqrt((count - 2) / (1.0 - correlation * correlation))
    return two_sided_t_p_value(t_stat, count - 2)


def kendall_tau_b(x_values: list[float], y_values: list[float]) -> float:
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    count = len(x_values)
    if count < 2:
        return math.nan

    for left in range(count - 1):
        for right in range(left + 1, count):
            delta_x = x_values[left] - x_values[right]
            delta_y = y_values[left] - y_values[right]
            if delta_x == 0 and delta_y == 0:
                continue
            if delta_x == 0:
                ties_x += 1
            elif delta_y == 0:
                ties_y += 1
            elif delta_x * delta_y > 0:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denominator == 0.0:
        return math.nan
    return (concordant - discordant) / denominator


def kendall_p_value(tau: float, count: int) -> float:
    if count < 3 or pd.isna(tau):
        return math.nan
    variance = 2.0 * (2.0 * count + 5.0) / (9.0 * count * (count - 1.0))
    if variance <= 0.0:
        return math.nan
    z_stat = tau / math.sqrt(variance)
    return two_sided_normal_p_value(z_stat)


def describe_strength(value: float) -> str:
    if pd.isna(value):
        return "not available"
    absolute = abs(value)
    if absolute < 0.1:
        return "negligible"
    if absolute < 0.3:
        return "weak"
    if absolute < 0.5:
        return "moderate"
    return "strong"


def describe_direction(value: float) -> str:
    if pd.isna(value) or abs(value) < 1.0e-12:
        return "flat"
    if value > 0:
        return "positive"
    return "negative"


def describe_p_value(value: float) -> str:
    if pd.isna(value):
        return "p-value not available"
    if value < 0.01:
        return "statistically clear at 1%"
    if value < 0.05:
        return "statistically clear at 5%"
    if value < 0.10:
        return "borderline at 10%"
    return "not statistically clear"


def compute_relationship_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if len(trades) < 3:
        return {
            "count": len(trades),
            "pearson": math.nan,
            "pearson_p": math.nan,
            "spearman": math.nan,
            "spearman_p": math.nan,
            "kendall": math.nan,
            "kendall_p": math.nan,
            "ols_slope": math.nan,
            "ols_slope_pp_per_100_min": math.nan,
            "ols_intercept": math.nan,
            "ols_r_squared": math.nan,
            "ols_p": math.nan,
        }

    x_values = trades["holding_minutes"].astype(float).tolist()
    y_values = trades["gross_profit_pct"].astype(float).tolist()
    count = len(x_values)

    pearson = pearson_correlation(x_values, y_values)
    pearson_p = correlation_p_value(pearson, count)

    ranked_x = pd.Series(x_values).rank(method="average").astype(float).tolist()
    ranked_y = pd.Series(y_values).rank(method="average").astype(float).tolist()
    spearman = pearson_correlation(ranked_x, ranked_y)
    spearman_p = correlation_p_value(spearman, count)

    kendall = kendall_tau_b(x_values, y_values)
    kendall_p = kendall_p_value(kendall, count)

    mean_x = sum(x_values) / count
    mean_y = sum(y_values) / count
    centered_x = [value - mean_x for value in x_values]
    centered_y = [value - mean_y for value in y_values]
    ss_xx = sum(value * value for value in centered_x)
    ss_yy = sum(value * value for value in centered_y)
    ss_xy = sum(x_value * y_value for x_value, y_value in zip(centered_x, centered_y))
    slope = ss_xy / ss_xx if ss_xx else math.nan
    intercept = mean_y - slope * mean_x if not pd.isna(slope) else math.nan
    if pd.isna(slope) or ss_yy == 0.0:
        r_squared = math.nan
    else:
        residual_sum_squares = sum(
            (y_value - (intercept + slope * x_value)) ** 2
            for x_value, y_value in zip(x_values, y_values)
        )
        r_squared = 1.0 - residual_sum_squares / ss_yy

    return {
        "count": count,
        "pearson": pearson,
        "pearson_p": pearson_p,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "kendall": kendall,
        "kendall_p": kendall_p,
        "ols_slope": slope,
        "ols_slope_pp_per_100_min": slope * 100.0 * 100.0
        if not pd.isna(slope)
        else math.nan,
        "ols_intercept": intercept,
        "ols_r_squared": r_squared,
        "ols_p": pearson_p,
    }


def prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    output = trades.copy()
    numeric_columns = [
        "holding_minutes",
        "gross_pnl_twd",
        "actual_leg_notional_twd",
        "net_pnl_twd",
        "total_fee_twd",
    ]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    bad_numeric = output[numeric_columns].isna().any(axis=1)
    if bad_numeric.any():
        raise RuntimeError(
            "Trades CSV contains invalid numeric values:\n"
            f"{output.loc[bad_numeric, numeric_columns].head(10)}"
        )

    bad_denominator = output["actual_leg_notional_twd"] <= 0
    if bad_denominator.any():
        raise RuntimeError(
            "actual_leg_notional_twd must be positive for percentage calculations:\n"
            f"{output.loc[bad_denominator, ['entry_time', 'actual_leg_notional_twd']].head(10)}"
        )

    denominator = output["actual_leg_notional_twd"]
    output["gross_profit_pct"] = output["gross_pnl_twd"] / denominator
    output["net_profit_pct"] = output["net_pnl_twd"] / denominator
    output["fee_pct"] = output["total_fee_twd"] / denominator
    return output


def make_metric_cards(trades: pd.DataFrame) -> str:
    if trades.empty:
        cards = [
            ("Trades", "0", "No trades available", "neutral"),
            ("Average Hold", "-", "No holding period", "neutral"),
            ("Average Gross", "-", "No gross profit", "neutral"),
            ("Median Gross", "-", "No gross profit", "neutral"),
            ("Best / Worst Gross", "-", "No gross profit range", "neutral"),
        ]
    else:
        best = trades.loc[trades["gross_profit_pct"].idxmax()]
        worst = trades.loc[trades["gross_profit_pct"].idxmin()]
        cards = [
            ("Trades", fmt_int(len(trades)), "Each marker is one closed trade", "neutral"),
            (
                "Average Hold",
                f"{fmt_number(trades['holding_minutes'].mean())} min",
                f"Median {fmt_number(trades['holding_minutes'].median())} min",
                "neutral",
            ),
            (
                "Average Gross",
                fmt_pct(trades["gross_profit_pct"].mean()),
                "Gross PnL / actual leg notional",
                css_class_for_pnl(trades["gross_profit_pct"].mean()),
            ),
            (
                "Median Gross",
                fmt_pct(trades["gross_profit_pct"].median()),
                "Middle trade by gross return",
                css_class_for_pnl(trades["gross_profit_pct"].median()),
            ),
            (
                "Best / Worst Gross",
                f"{fmt_pct(best['gross_profit_pct'])} / {fmt_pct(worst['gross_profit_pct'])}",
                f"Best hold {fmt_int(best['holding_minutes'])} min, worst hold {fmt_int(worst['holding_minutes'])} min",
                "neutral",
            ),
        ]

    return "\n".join(
        f"""
        <div class="metric-card {escape(card_class)}">
          <div class="metric-label">{escape(label)}</div>
          <div class="metric-value">{escape(value)}</div>
          <div class="metric-subtitle">{escape(subtitle)}</div>
        </div>
        """
        for label, value, subtitle, card_class in cards
    )


def make_stat_rows(stats: dict[str, Any]) -> str:
    rows = [
        (
            "Pearson r",
            fmt_number(stats["pearson"], 3),
            fmt_number(stats["pearson_p"], 4),
            (
                f"{describe_strength(stats['pearson'])} "
                f"{describe_direction(stats['pearson'])} linear relationship."
            ),
        ),
        (
            "Spearman rho",
            fmt_number(stats["spearman"], 3),
            fmt_number(stats["spearman_p"], 4),
            (
                f"{describe_strength(stats['spearman'])} "
                f"{describe_direction(stats['spearman'])} monotonic relationship."
            ),
        ),
        (
            "Kendall tau",
            fmt_number(stats["kendall"], 3),
            fmt_number(stats["kendall_p"], 4),
            (
                f"{describe_strength(stats['kendall'])} "
                f"{describe_direction(stats['kendall'])} rank agreement."
            ),
        ),
        (
            "OLS slope",
            f"{fmt_number(stats['ols_slope_pp_per_100_min'], 3)} pp / 100 min",
            fmt_number(stats["ols_p"], 4),
            "Average gross return change for each additional 100 hold minutes.",
        ),
        (
            "R-squared",
            fmt_pct(stats["ols_r_squared"]),
            "-",
            "Share of gross-return variance explained by Hold Min alone.",
        ),
    ]
    return "\n".join(
        f"""
        <tr>
          <th>{escape(label)}</th>
          <td class="number">{escape(value)}</td>
          <td class="number">{escape(p_value)}</td>
          <td>{escape(note)}</td>
        </tr>
        """
        for label, value, p_value, note in rows
    )


def make_stat_summary(stats: dict[str, Any]) -> str:
    if stats["count"] < 3:
        return "Need at least 3 trades to estimate correlation and regression statistics."

    primary = stats["spearman"]
    primary_p = stats["spearman_p"]
    direction = describe_direction(primary)
    strength = describe_strength(primary)
    significance = describe_p_value(primary_p)
    slope = stats["ols_slope_pp_per_100_min"]
    r_squared = stats["ols_r_squared"]
    return (
        f"Primary read: Spearman rho is {fmt_number(primary, 3)}, a {strength} "
        f"{direction} monotonic relationship, and it is {significance}. "
        f"OLS slope is {fmt_number(slope, 3)} percentage points per +100 hold "
        f"minutes, while R-squared is {fmt_pct(r_squared)}, so Hold Min alone "
        "has limited explanatory power when this value is low."
    )


def make_stats_panel(stats: dict[str, Any]) -> str:
    return f"""
    <section class="stats-panel">
      <div class="section-title">
        <h2>Correlation and regression</h2>
        <span>{fmt_int(stats["count"])} trades</span>
      </div>
      <p class="explanation">{escape(make_stat_summary(stats))}</p>
      <div class="table-wrap">
        <table class="stats-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
              <th>p-value</th>
              <th>How to read it</th>
            </tr>
          </thead>
          <tbody>
            {make_stat_rows(stats)}
          </tbody>
        </table>
      </div>
      <p class="note">
        Pearson focuses on linear fit. Spearman and Kendall use ranks and are
        less sensitive to large outliers. p-values are two-sided approximations.
      </p>
    </section>
    """


def make_bucket_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "trade_count",
                "avg_hold",
                "mean_gross",
                "median_gross",
                "gross_win_rate",
            ]
        )

    output = trades.copy()
    output["hold_bucket"] = pd.cut(
        output["holding_minutes"],
        bins=HOLD_BUCKET_BINS,
        labels=HOLD_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    )
    grouped = (
        output.groupby("hold_bucket", observed=False)
        .agg(
            trade_count=("holding_minutes", "size"),
            avg_hold=("holding_minutes", "mean"),
            mean_gross=("gross_profit_pct", "mean"),
            median_gross=("gross_profit_pct", "median"),
            gross_win_rate=("gross_profit_pct", lambda values: (values > 0).mean()),
        )
        .reset_index()
        .rename(columns={"hold_bucket": "bucket"})
    )
    return grouped


def make_bucket_table(trades: pd.DataFrame) -> str:
    buckets = make_bucket_summary(trades)
    if buckets.empty:
        rows = """
        <tr>
          <td colspan="6" class="empty-cell">No trades available.</td>
        </tr>
        """
    else:
        row_html: list[str] = []
        for _, row in buckets.iterrows():
            count = int(row["trade_count"])
            row_html.append(
                f"""
                <tr>
                  <th>{escape(row["bucket"])}</th>
                  <td class="number">{fmt_int(count)}</td>
                  <td class="number">{fmt_number(row["avg_hold"]) if count else "-"}</td>
                  <td class="number {css_class_for_pnl(row["mean_gross"])}">{fmt_pct(row["mean_gross"]) if count else "-"}</td>
                  <td class="number {css_class_for_pnl(row["median_gross"])}">{fmt_pct(row["median_gross"]) if count else "-"}</td>
                  <td class="number">{fmt_pct(row["gross_win_rate"]) if count else "-"}</td>
                </tr>
                """
            )
        rows = "\n".join(row_html)

    return f"""
    <section class="stats-panel">
      <div class="section-title">
        <h2>Hold Min bucket summary</h2>
        <span>Gross Profit % by bucket</span>
      </div>
      <div class="table-wrap">
        <table class="stats-table">
          <thead>
            <tr>
              <th>Hold Min</th>
              <th>Trades</th>
              <th>Avg Hold</th>
              <th>Mean Gross %</th>
              <th>Median Gross %</th>
              <th>Gross Win Rate</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
      <p class="note">
        Buckets make the relationship easier to inspect when a few trades have
        unusually long holding periods or unusually large returns.
      </p>
    </section>
    """


def make_plot_data(trades: pd.DataFrame) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    if trades.empty:
        return traces

    for direction, group in trades.sort_values("holding_minutes").groupby("direction"):
        style = DIRECTION_STYLE.get(direction, {"color": "#475569", "symbol": "circle-open"})
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": direction,
                "x": group["holding_minutes"].round(6).tolist(),
                "y": group["gross_profit_pct"].round(10).tolist(),
                "customdata": group[
                    [
                        "entry_time",
                        "exit_time",
                        "direction",
                        "holding_minutes",
                        "gross_profit_pct",
                        "net_profit_pct",
                        "fee_pct",
                        "exit_reason",
                    ]
                ].values.tolist(),
                "marker": {
                    "size": 10,
                    "color": style["color"],
                    "symbol": style["symbol"],
                    "opacity": 0.82,
                    "line": {"color": "#0f172a", "width": 0.7},
                },
                "hovertemplate": (
                    "<b>%{customdata[2]}</b><br>"
                    "Entry: %{customdata[0]}<br>"
                    "Exit: %{customdata[1]}<br>"
                    "Hold Min: %{x:,}<br>"
                    "Gross Profit: %{y:.2%}<br>"
                    "Net Profit: %{customdata[5]:.2%}<br>"
                    "Fee: %{customdata[6]:.2%}<br>"
                    "Exit Reason: %{customdata[7]}"
                    "<extra></extra>"
                ),
            }
        )
    return traces


def make_report(trades: pd.DataFrame) -> str:
    prepared = prepare_trades(trades)
    metric_cards = make_metric_cards(prepared)
    relationship_stats = compute_relationship_stats(prepared)
    stats_panel = make_stats_panel(relationship_stats)
    bucket_table = make_bucket_table(prepared)
    plot_data_json = json.dumps(make_plot_data(prepared), ensure_ascii=False)
    trade_count = int(len(prepared))
    subtitle = (
        "Each point is one closed trade. Gross Profit (%) uses gross_pnl_twd / "
        "actual_leg_notional_twd."
    )

    empty_state = (
        '<div class="empty-state">No trades available.</div>' if prepared.empty else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hold Min vs Gross Profit %</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #d7dee8;
      --blue: #2563eb;
      --gold: #d97706;
      --positive: #047857;
      --negative: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 40px;
    }}
    header {{
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 3vw, 40px);
      line-height: 1.1;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
      max-width: 820px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 14px 13px;
      min-height: 104px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 22px;
      font-weight: 760;
      line-height: 1.15;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }}
    .metric-subtitle {{
      color: var(--muted);
      margin-top: 7px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .metric-card.positive .metric-value {{ color: var(--positive); }}
    .metric-card.negative .metric-value {{ color: var(--negative); }}
    .chart-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px 16px 10px;
      margin-bottom: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .stats-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .chart-title, .section-title {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin: 0 0 4px;
    }}
    .chart-title h2, .section-title h2 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.3;
      letter-spacing: 0;
    }}
    .chart-title span, .section-title span {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .explanation {{
      color: #334155;
      font-size: 14px;
      line-height: 1.55;
      max-width: 960px;
      margin: 8px 0 14px;
    }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      margin: 12px 0 0;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .stats-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
      background: #ffffff;
    }}
    .stats-table th, .stats-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
      line-height: 1.35;
    }}
    .stats-table thead th {{
      color: #334155;
      background: #f8fafc;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stats-table tbody tr:last-child th,
    .stats-table tbody tr:last-child td {{
      border-bottom: 0;
    }}
    .stats-table .number {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .stats-table .positive {{ color: var(--positive); }}
    .stats-table .negative {{ color: var(--negative); }}
    .empty-cell {{
      color: var(--muted);
      text-align: center;
    }}
    #hold-profit-chart {{
      width: 100%;
      height: 620px;
    }}
    .empty-state {{
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 42px 16px;
      text-align: center;
      margin-top: 12px;
    }}
    @media (max-width: 920px) {{
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      #hold-profit-chart {{ height: 520px; }}
    }}
    @media (max-width: 560px) {{
      main {{ padding: 22px 12px 30px; }}
      .metric-grid {{ grid-template-columns: 1fr; }}
      .chart-title, .section-title {{ display: block; }}
      .chart-title span, .section-title span {{ display: block; margin-top: 4px; white-space: normal; }}
      #hold-profit-chart {{ height: 460px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Hold Min vs Gross Profit %</h1>
      <div class="subtitle">{escape(subtitle)}</div>
    </header>

    <section class="metric-grid">
      {metric_cards}
    </section>

    {stats_panel}

    <section class="chart-panel">
      <div class="chart-title">
        <h2>Gross profit by holding time</h2>
        <span>{fmt_int(trade_count)} trades</span>
      </div>
      {empty_state}
      <div id="hold-profit-chart"></div>
    </section>

    {bucket_table}
  </main>

  <script>
    const plotData = {plot_data_json};
    const layout = {{
      margin: {{ l: 72, r: 28, t: 28, b: 68 }},
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      hovermode: "closest",
      legend: {{
        orientation: "h",
        y: 1.08,
        x: 0,
        font: {{ size: 12, color: "#334155" }}
      }},
      xaxis: {{
        title: {{ text: "Hold Min", standoff: 14 }},
        gridcolor: "#e2e8f0",
        zeroline: false,
        rangemode: "tozero",
        tickformat: ",d"
      }},
      yaxis: {{
        title: {{ text: "Gross Profit (%)", standoff: 16 }},
        tickformat: ".2%",
        gridcolor: "#e2e8f0",
        zeroline: false
      }},
      shapes: [
        {{
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          yref: "y",
          y0: 0,
          y1: 0,
          line: {{ color: "#475569", width: 1, dash: "dash" }}
        }}
      ],
      annotations: [
        {{
          xref: "paper",
          x: 1,
          yref: "y",
          y: 0,
          text: "0%",
          showarrow: false,
          xanchor: "left",
          yanchor: "bottom",
          font: {{ color: "#475569", size: 12 }},
          xshift: 4
        }}
      ]
    }};
    const config = {{ responsive: true, displayModeBar: true }};
    if (plotData.length) {{
      Plotly.newPlot("hold-profit-chart", plotData, layout, config);
    }} else {{
      document.getElementById("hold-profit-chart").style.display = "none";
    }}
  </script>
</body>
</html>"""


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    trades = read_csv(
        args.trades,
        {
            "entry_time",
            "exit_time",
            "direction",
            "holding_minutes",
            "actual_leg_notional_twd",
            "gross_pnl_twd",
            "total_fee_twd",
            "net_pnl_twd",
            "exit_reason",
        },
    )
    report = make_report(trades)
    args.out.write_text(report, encoding="utf-8")
    safe_output = str(args.out).encode("unicode_escape").decode("ascii")
    print(f"Wrote hold profit report to {safe_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
