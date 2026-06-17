const CSV_URL = "/data/processed/qff_tsm_spread_1m_taipei.csv";
const WINDOW_HOURS = 48;
const TAIPEI_TZ = "Asia/Taipei";

const colors = {
  bg: "#171a1e",
  text: "#eef2f6",
  muted: "#9da8b3",
  grid: "rgba(174, 184, 194, 0.12)",
  spread: "#44b7a8",
  qff: "#d6b15e",
  fair: "#7db1ff",
  zero: "#8b949e",
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines
    .filter(Boolean)
    .map((line) => {
      const cells = line.split(",");
      const row = {};
      headers.forEach((header, index) => {
        row[header] = cells[index];
      });
      return {
        timestamp: row.timestamp,
        time: Math.floor(Date.parse(row.timestamp) / 1000),
        qffClose: row.qff_close === "" ? null : Number(row.qff_close),
        qffCloseFilled: Number(row.qff_close_filled),
        qffWasFilled: row.qff_was_filled === "True",
        tsmClose: Number(row.tsm_close),
        usdttwdClose: Number(row.usdttwd_close),
        tsmTwdFair: Number(row.tsm_twd_fair),
        spread: Number(row.spread),
      };
    });
}

function latestWindow(rows, hours) {
  const lastTime = rows[rows.length - 1].time;
  const startTime = lastTime - hours * 60 * 60;
  return rows.filter((row) => row.time >= startTime && row.time <= lastTime);
}

function formatTaipeiTime(unixSeconds) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: TAIPEI_TZ,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(unixSeconds * 1000));
}

function formatFullTaipeiTime(unixSeconds) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: TAIPEI_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(unixSeconds * 1000));
}

function formatNumber(value, digits = 2) {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function baseChartOptions() {
  return {
    layout: {
      background: { type: "solid", color: colors.bg },
      textColor: colors.muted,
      fontFamily:
        "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    },
    grid: {
      vertLines: { color: colors.grid },
      horzLines: { color: colors.grid },
    },
    rightPriceScale: {
      borderColor: "rgba(174, 184, 194, 0.22)",
    },
    timeScale: {
      borderColor: "rgba(174, 184, 194, 0.22)",
      timeVisible: true,
      secondsVisible: false,
      tickMarkFormatter: (time) => formatTaipeiTime(time),
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: "rgba(238, 242, 246, 0.25)", labelBackgroundColor: "#303741" },
      horzLine: { color: "rgba(238, 242, 246, 0.25)", labelBackgroundColor: "#303741" },
    },
    localization: {
      timeFormatter: (time) => formatFullTaipeiTime(time),
    },
  };
}

function resizeChart(chart, container) {
  const rect = container.getBoundingClientRect();
  chart.applyOptions({
    width: Math.max(320, Math.floor(rect.width)),
    height: Math.max(220, Math.floor(rect.height)),
  });
}

function showError(error) {
  const box = document.getElementById("errorBox");
  box.hidden = false;
  box.textContent = error instanceof Error ? error.message : String(error);
}

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

function render(rows) {
  const data = latestWindow(rows, WINDOW_HOURS);
  if (data.length === 0) {
    throw new Error("No rows found for the latest 48h window.");
  }

  const start = data[0].time;
  const end = data[data.length - 1].time;
  const spreads = data.map((row) => row.spread);
  const minSpread = Math.min(...spreads);
  const maxSpread = Math.max(...spreads);
  const last = data[data.length - 1];
  const fillCount = data.filter((row) => row.qffWasFilled).length;

  setText("rangeLabel", `${formatFullTaipeiTime(start)} - ${formatFullTaipeiTime(end)} Taipei`);
  setText("lastSpread", `${formatNumber(last.spread, 4)}%`);
  setText("spreadRange", `${formatNumber(minSpread, 2)} / ${formatNumber(maxSpread, 2)}%`);
  setText("fillCount", `${fillCount.toLocaleString()} / ${data.length.toLocaleString()}`);

  const spreadContainer = document.getElementById("spreadChart");
  const priceContainer = document.getElementById("priceChart");
  const spreadChart = LightweightCharts.createChart(spreadContainer, baseChartOptions());
  const priceChart = LightweightCharts.createChart(priceContainer, baseChartOptions());

  const spreadSeries = spreadChart.addLineSeries({
    color: colors.spread,
    lineWidth: 2,
    priceFormat: { type: "price", precision: 4, minMove: 0.0001 },
    lastValueVisible: true,
    priceLineVisible: true,
  });
  spreadSeries.setData(data.map((row) => ({ time: row.time, value: row.spread })));
  spreadSeries.createPriceLine({
    price: 0,
    color: colors.zero,
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dotted,
    axisLabelVisible: true,
    title: "0",
  });

  const qffSeries = priceChart.addLineSeries({
    color: colors.qff,
    lineWidth: 2,
    priceFormat: { type: "price", precision: 2, minMove: 0.01 },
  });
  const fairSeries = priceChart.addLineSeries({
    color: colors.fair,
    lineWidth: 2,
    priceFormat: { type: "price", precision: 2, minMove: 0.01 },
  });
  qffSeries.setData(data.map((row) => ({ time: row.time, value: row.qffCloseFilled })));
  fairSeries.setData(data.map((row) => ({ time: row.time, value: row.tsmTwdFair })));

  const rowsByTime = new Map(data.map((row) => [row.time, row]));
  const updateLegends = (row) => {
    setText(
      "spreadLegend",
      `${formatFullTaipeiTime(row.time)}  Spread ${formatNumber(row.spread, 4)}%`
    );
    setText(
      "priceLegend",
      `${formatFullTaipeiTime(row.time)}  QFF ${formatNumber(row.qffCloseFilled, 2)}  Fair ${formatNumber(row.tsmTwdFair, 2)}`
    );
  };
  updateLegends(last);

  spreadChart.subscribeCrosshairMove((param) => {
    const row = param.time ? rowsByTime.get(param.time) : last;
    if (row) updateLegends(row);
  });
  priceChart.subscribeCrosshairMove((param) => {
    const row = param.time ? rowsByTime.get(param.time) : last;
    if (row) updateLegends(row);
  });

  const syncRange = (sourceChart, targetChart) => {
    sourceChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) targetChart.timeScale().setVisibleLogicalRange(range);
    });
  };
  syncRange(spreadChart, priceChart);
  syncRange(priceChart, spreadChart);

  const resize = () => {
    resizeChart(spreadChart, spreadContainer);
    resizeChart(priceChart, priceContainer);
  };
  window.addEventListener("resize", resize);
  resize();
  spreadChart.timeScale().fitContent();
  priceChart.timeScale().fitContent();
}

async function main() {
  const response = await fetch(CSV_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${CSV_URL}: ${response.status} ${response.statusText}`);
  }
  const rows = parseCsv(await response.text());
  render(rows);
}

main().catch(showError);
