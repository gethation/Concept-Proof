from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "data/processed/qff_tsm_spread_zscore_1m_taipei_qff_session.csv"
)
DEFAULT_SUMMARY_PATH = Path(
    "data/processed/qff_tsm_pair_backtest_summary_qff_session.json"
)
DEFAULT_OUTPUT_PATH = Path("spread_zscore_report_qff_session.html")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Lightweight Charts report for spread and z-score."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary JSON does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_indicator_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Indicator CSV does not exist: {path}")

    frame = pd.read_csv(path)
    required = {
        "timestamp",
        "spread",
        "spread_zscore",
        "zscore_valid",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise RuntimeError(f"Indicator CSV has no rows: {path}")

    mean_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("spread_mean_")
    )
    if not mean_columns:
        raise RuntimeError(
            f"{path} does not contain a spread_mean_* column. "
            f"Available columns: {list(frame.columns)}"
        )
    mean_column = mean_columns[-1]

    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    for column in ["spread", mean_column, "spread_zscore"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["spread_mean"] = output[mean_column]
    output.attrs["spread_mean_column"] = mean_column
    output["time"] = output["timestamp"].map(lambda value: int(value.timestamp()))

    if output["spread"].isna().any():
        raise RuntimeError("spread contains missing values")
    if not output["timestamp"].is_unique:
        raise RuntimeError("timestamps are not unique")
    if not output["timestamp"].is_monotonic_increasing:
        raise RuntimeError("timestamps are not sorted")
    return output


def to_series(
    frame: pd.DataFrame,
    column: str,
    digits: int = 8,
    include_whitespace: bool = False,
) -> list[dict[str, Any]]:
    if not include_whitespace:
        clean = frame.loc[frame[column].notna(), ["time", column]]
        return [
            {"time": int(row.time), "value": round(float(getattr(row, column)), digits)}
            for row in clean.itertuples(index=False)
        ]

    output: list[dict[str, Any]] = []
    for row in frame[["time", column]].itertuples(index=False):
        point: dict[str, Any] = {"time": int(row.time)}
        value = getattr(row, column)
        if not pd.isna(value):
            point["value"] = round(float(value), digits)
        output.append(point)
    return output


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def escape(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return html.escape(str(value))


def make_report(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    params = summary.get("parameters", {})
    entry_z = float(params.get("entry_z", 2.0))
    exit_z = float(params.get("exit_z", 0.0))
    mean_column = str(frame.attrs.get("spread_mean_column", "spread_mean"))
    mean_label = (
        f"MA({mean_column[len('spread_mean_'):]})"
        if mean_column.startswith("spread_mean_")
        else "MA"
    )
    start = frame["timestamp"].iloc[0].tz_convert("Asia/Taipei").strftime(
        "%Y-%m-%d %H:%M:%S+08:00"
    )
    end = frame["timestamp"].iloc[-1].tz_convert("Asia/Taipei").strftime(
        "%Y-%m-%d %H:%M:%S+08:00"
    )
    valid_z = frame["spread_zscore"].dropna()

    chart_payload = {
        "spread": to_series(frame, "spread"),
        "spreadMean": to_series(frame, "spread_mean"),
        "zscore": to_series(frame, "spread_zscore", include_whitespace=True),
        "levels": {
            "entryZ": entry_z,
            "exitZ": exit_z,
        },
        "summary": {
            "rows": int(len(frame)),
            "start": start,
            "end": end,
            "spreadLast": float(frame["spread"].iloc[-1]),
            "spreadMeanLast": (
                None
                if pd.isna(frame["spread_mean"].iloc[-1])
                else float(frame["spread_mean"].iloc[-1])
            ),
            "zscoreLast": (
                None
                if pd.isna(frame["spread_zscore"].iloc[-1])
                else float(frame["spread_zscore"].iloc[-1])
            ),
            "zscoreMin": None if valid_z.empty else float(valid_z.min()),
            "zscoreMax": None if valid_z.empty else float(valid_z.max()),
        },
    }
    payload_json = json.dumps(chart_payload, ensure_ascii=False, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QFF-TSM Spread / Z-Score</title>
  <script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111418;
      --panel: #171b21;
      --panel-2: #1d222a;
      --text: #edf2f7;
      --muted: #9aa6b2;
      --line: rgba(176, 187, 199, 0.18);
      --spread: #4fc3b1;
      --mean: #f1b84b;
      --zscore: #8ab4ff;
      --entry: rgba(239, 68, 68, 0.62);
      --exit: rgba(245, 158, 11, 0.58);
      --zero: rgba(226, 232, 240, 0.62);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", sans-serif;
    }}
    .shell {{
      min-height: 100vh;
      padding: 22px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      padding: 18px 18px 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .range {{
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      min-width: min(740px, 100%);
    }}
    .stat {{
      padding: 10px 12px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .stat strong {{
      display: block;
      margin-top: 3px;
      font-size: 17px;
      font-variant-numeric: tabular-nums;
    }}
    .charts {{
      display: grid;
      grid-template-rows: minmax(330px, 1fr) minmax(330px, 1fr);
      gap: 14px;
      min-height: calc(100vh - 150px);
    }}
    .panel {{
      min-height: 0;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .panel-head {{
      height: 30px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }}
    h2 {{
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .legend {{
      color: var(--muted);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    .chart {{
      width: 100%;
      height: calc(100% - 38px);
      min-height: 280px;
    }}
    @media (max-width: 980px) {{
      .shell {{ padding: 12px; }}
      .topbar {{ display: block; }}
      .stats {{
        margin-top: 14px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .charts {{
        min-height: 900px;
        grid-template-rows: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div>
        <h1>QFF-TSM Spread / Z-Score</h1>
        <div class="range">{escape(start)} - {escape(end)} Taipei</div>
      </div>
      <section class="stats" aria-label="summary">
        <div class="stat"><span>Rows</span><strong>{len(frame):,}</strong></div>
        <div class="stat"><span>Last Spread</span><strong>{fmt_number(frame["spread"].iloc[-1], 4)}%</strong></div>
        <div class="stat"><span>Last Z</span><strong>{fmt_number(chart_payload["summary"]["zscoreLast"], 4)}</strong></div>
        <div class="stat"><span>Entry / Exit Z</span><strong>{fmt_number(entry_z, 2)} / {fmt_number(exit_z, 2)}</strong></div>
      </section>
    </header>

    <section class="charts">
      <section class="panel">
        <div class="panel-head">
          <h2>Spread + {escape(mean_label)}</h2>
          <div id="spreadLegend" class="legend">--</div>
        </div>
        <div id="spreadChart" class="chart"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>Z-Score</h2>
          <div id="zLegend" class="legend">--</div>
        </div>
        <div id="zChart" class="chart"></div>
      </section>
    </section>
  </main>

  <script>
    const payload = {payload_json};
    const colors = {{
      bg: "#171b21",
      text: "#edf2f7",
      muted: "#9aa6b2",
      grid: "rgba(176, 187, 199, 0.14)",
      spread: "#4fc3b1",
      mean: "#f1b84b",
      zscore: "#8ab4ff",
      entry: "rgba(239, 68, 68, 0.62)",
      exit: "rgba(245, 158, 11, 0.58)",
      zero: "rgba(226, 232, 240, 0.62)"
    }};

    function formatTaipeiTime(unixSeconds) {{
      return new Intl.DateTimeFormat("en-CA", {{
        timeZone: "Asia/Taipei",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false
      }}).format(new Date(unixSeconds * 1000));
    }}

    function formatFullTaipeiTime(unixSeconds) {{
      return new Intl.DateTimeFormat("en-CA", {{
        timeZone: "Asia/Taipei",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false
      }}).format(new Date(unixSeconds * 1000));
    }}

    function formatNumber(value, digits = 4) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "--";
      return new Intl.NumberFormat("en-US", {{
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      }}).format(value);
    }}

    function chartOptions() {{
      return {{
        layout: {{
          background: {{ type: "solid", color: colors.bg }},
          textColor: colors.muted,
          fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
        }},
        grid: {{
          vertLines: {{ color: colors.grid }},
          horzLines: {{ color: colors.grid }}
        }},
        rightPriceScale: {{
          borderColor: "rgba(176, 187, 199, 0.24)",
          scaleMargins: {{ top: 0.12, bottom: 0.12 }}
        }},
        timeScale: {{
          borderColor: "rgba(176, 187, 199, 0.24)",
          timeVisible: true,
          secondsVisible: false,
          tickMarkFormatter: (time) => formatTaipeiTime(time)
        }},
        crosshair: {{
          mode: LightweightCharts.CrosshairMode.Normal,
          vertLine: {{ color: "rgba(237, 242, 247, 0.25)", labelBackgroundColor: "#2d3748" }},
          horzLine: {{ color: "rgba(237, 242, 247, 0.25)", labelBackgroundColor: "#2d3748" }}
        }},
        localization: {{
          timeFormatter: (time) => formatFullTaipeiTime(time)
        }}
      }};
    }}

    function resizeChart(chart, container) {{
      const rect = container.getBoundingClientRect();
      chart.applyOptions({{
        width: Math.max(320, Math.floor(rect.width)),
        height: Math.max(260, Math.floor(rect.height))
      }});
    }}

    function addDashedLine(series, price, color, title) {{
      series.createPriceLine({{
        price,
        color,
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title
      }});
    }}

    const spreadContainer = document.getElementById("spreadChart");
    const zContainer = document.getElementById("zChart");
    const spreadChart = LightweightCharts.createChart(spreadContainer, chartOptions());
    const zChart = LightweightCharts.createChart(zContainer, chartOptions());

    const spreadSeries = spreadChart.addLineSeries({{
      color: colors.spread,
      lineWidth: 2,
      priceFormat: {{ type: "price", precision: 4, minMove: 0.0001 }},
      lastValueVisible: true,
      priceLineVisible: true
    }});
    const meanSeries = spreadChart.addLineSeries({{
      color: colors.mean,
      lineWidth: 2,
      priceFormat: {{ type: "price", precision: 4, minMove: 0.0001 }},
      lastValueVisible: true,
      priceLineVisible: false
    }});
    spreadSeries.setData(payload.spread);
    meanSeries.setData(payload.spreadMean);

    const zSeries = zChart.addLineSeries({{
      color: colors.zscore,
      lineWidth: 2,
      priceFormat: {{ type: "price", precision: 2, minMove: 0.01 }},
      lastValueVisible: true,
      priceLineVisible: true
    }});
    zSeries.setData(payload.zscore);

    const entryZ = Math.abs(payload.levels.entryZ);
    const exitZ = Math.abs(payload.levels.exitZ);
    addDashedLine(zSeries, 0, colors.zero, exitZ === 0 ? "0 / exit_z" : "0");
    if (entryZ > 0) {{
      addDashedLine(zSeries, entryZ, colors.entry, "+entry_z");
      addDashedLine(zSeries, -entryZ, colors.entry, "-entry_z");
    }}
    if (exitZ > 0) {{
      addDashedLine(zSeries, exitZ, colors.exit, "+exit_z");
      addDashedLine(zSeries, -exitZ, colors.exit, "-exit_z");
    }}

    const spreadByTime = new Map(payload.spread.map((point) => [point.time, point.value]));
    const meanByTime = new Map(payload.spreadMean.map((point) => [point.time, point.value]));
    const zByTime = new Map(
      payload.zscore
        .filter((point) => point.value !== undefined)
        .map((point) => [point.time, point.value])
    );
    const lastSpread = payload.spread[payload.spread.length - 1];
    const lastZ = payload.zscore[payload.zscore.length - 1];

    function setSpreadLegend(time) {{
      const spread = spreadByTime.get(time);
      const mean = meanByTime.get(time);
      document.getElementById("spreadLegend").textContent =
        `${{formatFullTaipeiTime(time)}}  Spread ${{formatNumber(spread, 4)}}%  MA ${{formatNumber(mean, 4)}}%`;
    }}

    function setZLegend(time) {{
      const z = zByTime.get(time);
      document.getElementById("zLegend").textContent =
        `${{formatFullTaipeiTime(time)}}  Z ${{formatNumber(z, 4)}}`;
    }}

    setSpreadLegend(lastSpread.time);
    setZLegend(lastZ.time);

    spreadChart.subscribeCrosshairMove((param) => {{
      if (param.time) setSpreadLegend(param.time);
    }});
    zChart.subscribeCrosshairMove((param) => {{
      if (param.time) setZLegend(param.time);
    }});

    function syncRange(sourceChart, targetChart) {{
      sourceChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
        if (range) targetChart.timeScale().setVisibleLogicalRange(range);
      }});
    }}
    syncRange(spreadChart, zChart);
    syncRange(zChart, spreadChart);

    function resize() {{
      resizeChart(spreadChart, spreadContainer);
      resizeChart(zChart, zContainer);
    }}
    window.addEventListener("resize", resize);
    resize();
    spreadChart.timeScale().fitContent();
    zChart.timeScale().fitContent();
  </script>
</body>
</html>
"""


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    frame = read_indicator_frame(args.input)
    summary = read_json(args.summary)
    report = make_report(frame, summary)
    args.out.write_text(report, encoding="utf-8")
    safe_output = str(args.out).encode("unicode_escape").decode("ascii")
    print(f"Wrote indicator report to {safe_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
