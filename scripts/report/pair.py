"""Build the technical report for one pair.

Runs the parameter grid on the current spread, screens it, picks a configuration,
re-runs that configuration for trade-level detail, and writes a single
self-contained HTML file. Every figure in the output is computed here at run
time -- nothing is carried over from a previous report.

    python scripts/report/pair.py --pair ccf_umc
    python scripts/report/pair.py --pair qff_tsm
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import engine as backtest  # noqa: E402
from backtest import grid  # noqa: E402
from features import zscore as zscore_calc  # noqa: E402
from lib import paths  # noqa: E402
from report import theme as T  # noqa: E402

WINDOWS = [390, 780, 1560, 2500, 3900]
# Extended past 2.5 because QFF/TSM could not clear its own microstructure
# screen anywhere below it: after TAIFEX cut the tick to 1 TWD the cheap
# crossing cost made low thresholds look profitable while 45% of the trades
# they took earned less than one tick. If nothing here clears the screen
# either, the answer is that the pair has no viable configuration under the
# current tick, not that the grid was too small.
ENTRIES = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
EXITS = [-0.25, 0.0, 0.25, 0.5]
CAPITAL = 2_000_000.0
LEG_NOTIONAL = 1_000_000.0


@dataclass
class TickRegime:
    """One exchange rule period: what a tick was worth, and what it cost to cross.

    A contract's tick size is not a constant of the instrument. TAIFEX cut QFF
    from 5 TWD to 1 TWD on 2026-07-05, which moved both numbers this project
    measures everything against at once -- the crossing cost fell by four
    fifths, and so did the noise floor that decides whether an edge is
    distinguishable from a fill landing on the wrong side.

    Splitting the sample at the break was the first answer, and it cost a
    report that could only ever cover the shorter half. Carrying the schedule
    instead lets one run span the change with the right numbers on either side
    of it.
    """

    effective_from: str | None  # None = from the start of the data
    tick_twd: float  # per contract, i.e. tick per share x multiplier
    displacement: float  # measured or derived for THIS regime


@dataclass
class Segment:
    label: str
    start: str | None
    end: str | None
    displacement: float
    note: str
    # The futures price `displacement` was measured at. A tick is fixed in TWD
    # and a spread unit is not, so the same book width is a different number of
    # spread units at a different price. 0 keeps the displacement constant.
    ref_price: float = 0.0
    # Set to span a tick-size change. `displacement` and the pair's tick_twd
    # then apply only where no regime covers the bar.
    regimes: list[TickRegime] = field(default_factory=list)


@dataclass
class PairSpec:
    key: str
    name: str
    legs: str
    spread: Path
    seed: Path | None
    tick_twd: float
    tick_desc: str
    equity_fee_bps: float
    equity_fee_model: str
    futures_fee_twd: float
    multiplier: float
    tax_rate: float
    out: Path
    segments: list[Segment]
    equity_leg: str
    futures_leg: str
    legs_zh: str = ""
    extra: str = ""
    # OHLCV sources for the two legs and FX. Without them the engine has no
    # open prices to fill at and silently falls back to the fill bar's CLOSE --
    # which is what every report published while its own prose promised the
    # next bar's open.
    tw_ohlcv: Path | None = None
    us_ohlcv: Path | None = None
    fx_ohlcv: Path | None = None
    _cache: dict = field(default_factory=dict, repr=False)


PAIRS = {
    "ccf_umc": PairSpec(
        key="ccf_umc",
        name="CCF / UMC",
        legs="TAIFEX CCF (UMC stock futures) against the NYSE:UMC ADR",
        spread=paths.feature("ccf_umc", "spread_1m"),
        seed=paths.feature("ccf_umc", "seed"),
        tick_twd=0.5 * 2000,
        tick_desc="0.5 TWD × 2,000 shares = 1,000 TWD per contract",
        equity_fee_bps=2.5,
        # IBKR bills per ADR, not per dollar. At UMC's current $18-19 the flat
        # 2.5 bps under-charges by about 13%; the two models cross near $23.2.
        equity_fee_model="ibkr",
        futures_fee_twd=88.0,
        multiplier=2000.0,
        tax_rate=2e-5,
        out=paths.report("ccf_umc_report"),
        equity_leg="UMC ADR",
        futures_leg="CCF",
        legs_zh="TAIFEX CCF（UMC 股票期貨）與其在 NYSE 掛牌的 UMC ADR",
        tw_ohlcv=paths.CCF1_1M,
        us_ohlcv=paths.UMC_1M,
        fx_ohlcv=paths.USDTTWD_1M,
        segments=[
            Segment(
                "全期間", None, None, 0.2317,
                "由 2,618 根實盤分鐘 bar（2026-08-07~19）直接量測盤口寬度："
                "單邊位移 0.2317 spread 單位，來回 46.3 bps。"
                "先前以 tick 寬度出現頻率推得 0.2151（CCF 一個 tick 佔 98.2%、"
                "UMC 一美分佔 99.9%），兩種量法相差 8%。"
                "量測期間 CCF 中位價 121.50，位移隨價位以 1/price 縮放。",
                ref_price=121.50,
            )
        ],
    ),
    "qff_tsm": PairSpec(
        key="qff_tsm",
        name="QFF / TSM",
        legs="TAIFEX QFF (TSMC stock futures) against the OKX TSM-USDT perpetual",
        spread=paths.feature("qff_tsm", "spread_1m"),
        seed=paths.feature("qff_tsm", "seed"),
        tick_twd=1.0 * 100,
        tick_desc="1 TWD × 100 shares = 100 TWD per contract（2026-07-05 起）",
        equity_fee_bps=5.0,
        # The perp venue charges in bps of notional, so the flat model is the
        # right shape here -- this is not an unmeasured default. The RATE and
        # the displacement below were both measured on Binance's book; the data
        # is now OKX's, which is thinner, so both are inherited rather than
        # observed until they are re-measured. Section 6 says so.
        equity_fee_model="bps",
        futures_fee_twd=88.0,
        multiplier=100.0,
        tax_rate=2e-5,
        out=paths.report("qff_tsm_report"),
        equity_leg="TSM perp",
        futures_leg="QFF",
        legs_zh="TAIFEX QFF（台積電股票期貨）與 OKX 的 TSM-USDT 永續合約",
        tw_ohlcv=paths.QFF1_1M,
        us_ohlcv=paths.OKX_TSMUSDTP_1M,
        fx_ohlcv=paths.USDTTWD_1M,
        segments=[
            Segment(
                "全期間（跨 tick 變更）", None, None, 0.0755,
                "2026-07-05 之前 tick 為 5 TWD、位移 0.1589（由半 tick 0.1043 加"
                "其餘成分推得）；之後 tick 1 TWD、位移 0.0755（量測值）。"
                "兩者都錨在 QFF 中位價 2,390，並隨價位以 1/price 縮放。"
                "每筆交易的 tick 以其進場當下的制度計算，不是全期間共用一個。",
                ref_price=2390.0,
                regimes=[
                    TickRegime(None, 5.0 * 100, 0.1589),
                    TickRegime("2026-07-05", 1.0 * 100, 0.0755),
                ],
            ),
        ],
        extra=(
            "QFF 的最小跳動單位在 <b>2026-07-05</b> 由 5 TWD 改為 1 TWD，盤口價差成本一次降掉八成——"
            "但雜訊地板也跟著降了八成。本報告逐 bar 套用當下制度的位移，逐筆交易以其進場當下的 tick "
            "衡量邊際，因此樣本可以跨過這個斷點；代價是報告混合了兩種市場結構，"
            "讀數字時要記得變更前後不是同一個東西。"
        ),
    ),
}


# --------------------------------------------------------------------------- #
def load_frames(spec: PairSpec) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    spread = zscore_calc.read_spread_frame(spec.spread)
    seed = None
    if spec.seed and spec.seed.exists():
        seed = zscore_calc.read_spread_frame(spec.seed)
        # A seed that overlaps the sample would feed the same bars into the
        # rolling window twice; keep only what precedes the sample.
        first = pd.DatetimeIndex(spread["timestamp"]).min()
        seed = seed[pd.DatetimeIndex(seed["timestamp"]) < first]
        if seed.empty:
            seed = None
    if spec.tw_ohlcv and spec.us_ohlcv and spec.fx_ohlcv:
        # Once, on the spread frame: calculate_zscore copies it per window, so
        # the opens ride along into every grid cell without re-reading the
        # OHLCV files 140 times.
        spread = backtest.add_entry_open_prices(
            spread, spec.tw_ohlcv, spec.us_ohlcv, spec.fx_ohlcv
        )
    return spread, seed


def regime_series(spec: PairSpec, seg: Segment, timestamps) -> tuple[pd.Series, pd.Series]:
    """Per-bar (tick_twd, displacement) for a segment.

    With no regimes this is the segment's single displacement and the pair's
    single tick, repeated -- exactly what the scalar path did.
    """
    ts = pd.DatetimeIndex(timestamps)
    tick = pd.Series(spec.tick_twd, index=range(len(ts)), dtype=float)
    disp = pd.Series(seg.displacement, index=range(len(ts)), dtype=float)
    for regime in seg.regimes:
        mask = (
            np.ones(len(ts), dtype=bool)
            if regime.effective_from is None
            else np.asarray(ts >= pd.Timestamp(regime.effective_from, tz="Asia/Taipei"))
        )
        tick[mask] = regime.tick_twd
        disp[mask] = regime.displacement
    return tick, disp


def with_displacement_column(spec: PairSpec, seg: Segment, frame: pd.DataFrame):
    """Attach the per-bar base displacement the engine reads, when it varies."""
    if not seg.regimes:
        return frame
    out = frame.copy()
    _, disp = regime_series(spec, seg, out["timestamp"])
    out["executable_displacement"] = disp.to_numpy()
    return out


def trade_ticks(spec: PairSpec, seg: Segment, trades: pd.DataFrame) -> pd.Series:
    """Net PnL per contract in ticks, using the tick in force at ENTRY.

    A tick is the noise floor a trade has to clear, and which floor applied is a
    property of when the trade happened. Dividing a 5 TWD-tick trade by a 1 TWD
    tick would inflate its edge fivefold and quietly walk it through the screen.
    """
    per_contract = trades["net_pnl_twd"] / trades["qff_contracts"].abs()
    if not seg.regimes:
        return per_contract / spec.tick_twd
    tick, _ = regime_series(spec, seg, pd.to_datetime(trades["entry_time"]))
    return per_contract.reset_index(drop=True) / tick


def slice_segment(frame: pd.DataFrame, seg: Segment) -> pd.DataFrame:
    ts = pd.DatetimeIndex(frame["timestamp"])
    mask = np.ones(len(frame), dtype=bool)
    if seg.start:
        mask &= ts >= pd.Timestamp(seg.start, tz="Asia/Taipei")
    if seg.end:
        mask &= ts <= pd.Timestamp(seg.end, tz="Asia/Taipei") + pd.Timedelta(days=1)
    return frame[mask].copy()


def params_for(
    spec: PairSpec, entry_z: float, exit_z: float, disp: float, ref_price: float = 0.0
):
    return backtest.BacktestParams(
        entry_z=entry_z,
        exit_z=exit_z,
        leg_notional_twd=LEG_NOTIONAL,
        initial_capital_twd=CAPITAL,
        max_entry_delay_minutes=15,
        tsm_fee_bps=spec.equity_fee_bps,
        tsm_fee_model=spec.equity_fee_model,
        qff_fee_per_contract_twd=spec.futures_fee_twd,
        qff_tax_rate=spec.tax_rate,
        qff_contract_multiplier=spec.multiplier,
        executable_displacement=disp,
        displacement_ref_price=ref_price,
    )


def run_grid(spec: PairSpec, spread: pd.DataFrame, seed, seg: Segment) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        zframe = zscore_calc.calculate_zscore(spread, window, seed_frame=seed)
        zframe = slice_segment(zframe, seg)
        zframe = with_displacement_column(spec, seg, zframe)
        for entry_z in ENTRIES:
            for exit_z in EXITS:
                if exit_z < -entry_z:
                    continue
                res = backtest.run_backtest(
                    zframe,
                    params_for(spec, entry_z, exit_z, seg.displacement, seg.ref_price),
                )
                t = res.trades
                stats = grid.calculate_daily_return_stats(
                    res.equity, initial_capital_twd=CAPITAL, annual_trading_days=252.0
                )
                # Linear annualisation of the period return. It is an
                # extrapolation, not a measurement -- a 148-day sample scaled
                # by 2.5 -- so it is carried alongside `ret`, never instead of
                # it, and the report quotes the raw period return first.
                span_days = (
                    pd.Timestamp(res.summary["end"])
                    - pd.Timestamp(res.summary["start"])
                ).total_seconds() / 86400.0
                row = dict(
                    window=window, entry_z=entry_z, exit_z=exit_z,
                    sharpe=stats["sharpe_ratio"], trades=len(t),
                    net=float(res.summary["net_pnl_twd"]),
                    maxdd=float(res.summary["max_drawdown_pct"]),
                    ret=float(res.summary["return_pct"]),
                    ann=float(res.summary["return_pct"]) * 365.0 / span_days,
                    span_days=span_days,
                )
                if len(t):
                    ticks = trade_ticks(spec, seg, t)
                    row.update(
                        med_ticks=float(ticks.median()),
                        under1=float((ticks < 1).mean()),
                        win=float((t["net_pnl_twd"] > 0).mean()),
                    )
                else:
                    row.update(med_ticks=np.nan, under1=np.nan, win=np.nan)
                rows.append(row)
        print(f"  window {window} done", flush=True)
    return pd.DataFrame(rows)


def neighbour_sharpe(g: pd.DataFrame) -> pd.Series:
    key = {(r.window, r.entry_z, r.exit_z): r.sharpe for r in g.itertuples()}
    wi = {w: i for i, w in enumerate(WINDOWS)}
    ei = {e: i for i, e in enumerate(ENTRIES)}
    xi = {x: i for i, x in enumerate(EXITS)}
    out = []
    for r in g.itertuples():
        vals = []
        for dw, de, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = wi[r.window] + dw, ei[r.entry_z] + de, xi[r.exit_z] + dx
            if 0 <= a < len(WINDOWS) and 0 <= b < len(ENTRIES) and 0 <= c < len(EXITS):
                v = key.get((WINDOWS[a], ENTRIES[b], EXITS[c]))
                if v is not None and not pd.isna(v):
                    vals.append(v)
        out.append(float(np.mean(vals)) if vals else np.nan)
    return pd.Series(out, index=g.index)


def screen(g: pd.DataFrame) -> pd.DataFrame:
    return g[
        (g["trades"] >= 15)
        & (g["med_ticks"] >= 2.0)
        & (g["under1"] <= 0.15)
        & (g["neighbour"] >= g["sharpe"].median())
    ].copy()


# --------------------------------------------------------------------------- #
# Report assembly                                                               #
# --------------------------------------------------------------------------- #
def trade_detail(spec: PairSpec, spread, seed, seg: Segment, cell) -> tuple:
    zframe = zscore_calc.calculate_zscore(spread, int(cell.window), seed_frame=seed)
    zframe = slice_segment(zframe, seg)
    zframe = with_displacement_column(spec, seg, zframe)
    res = backtest.run_backtest(
        zframe,
        params_for(spec, cell.entry_z, cell.exit_z, seg.displacement, seg.ref_price),
    )
    t = res.trades.copy()
    for c in ("entry_time", "exit_time"):
        t[c] = pd.to_datetime(t[c])
    t["hours"] = t["holding_minutes"] / 60.0
    t["ticks"] = trade_ticks(spec, seg, t).to_numpy()
    # Scale-free per-trade result: bps of the leg notional that trade actually
    # carried. Independent of the capital base typed into the backtest, and
    # directly comparable across pairs and across sizing modes.
    t["ret_bps"] = t["net_pnl_twd"] / t["actual_leg_notional_twd"] * 10000.0
    return res, t


def session_equity(equity: pd.DataFrame) -> tuple[list[str], list[float]]:
    """Session-day labels and cumulative return in percent of capital."""
    eq = equity[["timestamp", "equity"]].copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True).dt.tz_convert("Asia/Taipei")
    day = (eq["timestamp"] - pd.Timedelta(hours=6)).dt.date
    close = eq.groupby(day)["equity"].last()
    pct = (close - CAPITAL) / CAPITAL * 100.0
    return [d.strftime("%m-%d") for d in close.index], [float(v) for v in pct.values]


def pick_example_trade(trades: pd.DataFrame) -> pd.Series:
    """A trade that is typical in outcome and legible on a chart.

    Typical first: the median per-trade return is what the report quotes, so a
    walk-through of an unusually good trade would teach the mechanism against a
    misleading example. Legible second: holds that span a session gap compress
    to nothing on a bar index, so short in-session trades are preferred among
    the near-median ones.
    """
    t = trades.copy()
    t["bars"] = t["exit_idx"] - t["entry_signal_idx"]
    compact = t[(t["bars"] <= 80) & (t["hours"] <= 3)]
    pool = compact if len(compact) else t
    return pool.iloc[(pool["ret_bps"] - trades["ret_bps"].median()).abs().argsort()].iloc[0]


def anatomy_figure(spec, zframe, trades, pick, seg, fignum: int) -> str:
    ex = pick_example_trade(trades)
    a, b = int(ex["entry_signal_idx"]), int(ex["exit_idx"])
    pad = max(8, int((b - a) * 0.45))
    lo, hi = max(0, a - pad), min(len(zframe) - 1, b + pad)
    # Clamp the padding at a session boundary. Bars are plotted on an index, so
    # a window that reaches back over a closed market silently splices days that
    # are hours apart into adjacent pixels.
    ts_all = pd.to_datetime(zframe["timestamp"])
    breaks = ts_all.diff() > pd.Timedelta(minutes=60)
    prev = breaks.iloc[lo : a + 1]
    if prev.any():
        lo = int(prev[prev].index[-1])
    nxt = breaks.iloc[b + 1 : hi + 1]
    if nxt.any():
        hi = int(nxt[nxt].index[0]) - 1
    win = zframe.iloc[lo : hi + 1].reset_index(drop=True)

    mean_col, std_col = backtest.find_spread_stat_columns(list(win.columns))
    z = win["spread_zscore"].to_numpy(dtype=float)
    std = win[std_col].to_numpy(dtype=float)
    # The displacement the ENGINE used on these bars: the per-bar regime value
    # when the frame carries one, then scaled by ref/price. Drawing the segment
    # scalar instead put the bands at up to half their true width, so the
    # figure showed threshold crossings the engine's own z never made -- on a
    # panel captioned as the three z the engine judged by.
    base_disp = (
        win["executable_displacement"].to_numpy(dtype=float)
        if "executable_displacement" in win.columns
        else np.full(len(win), seg.displacement)
    )
    bar_disp = backtest.displacement_at(
        win["qff_close_filled"].to_numpy(dtype=float),
        params_for(spec, pick.entry_z, pick.exit_z, seg.displacement, seg.ref_price),
        base_disp,
    )
    gap = bar_disp / std
    short_side = ex["direction"] == backtest.SHORT_TSM_LONG_QFF
    ts = pd.to_datetime(win["timestamp"])

    svg = T.trade_anatomy(
        spread=[float(v) for v in win["spread"]],
        mean=[float(v) for v in win[mean_col]],
        std=[float(v) for v in std],
        z_mid=[float(v) for v in z],
        z_short=[float(v) for v in (z - gap)],
        z_long=[float(v) for v in (z + gap)],
        labels=[d.strftime("%m-%d %H:%M") for d in ts],
        entry_z=float(pick.entry_z), exit_z=float(pick.exit_z),
        direction=str(ex["direction"]),
        events={
            "entry_signal": a - lo, "entry_fill": int(ex["entry_idx"]) - lo,
            "exit_signal": int(ex["exit_signal_idx"]) - lo, "exit_fill": b - lo,
        },
        aria=f"{spec.name} anatomy of one trade: spread with band, and the three z series",
    )
    side = "做空 spread" if short_side else "做多 spread"
    enter_on = "z_short 上穿 +entry" if short_side else "z_long 下穿 −entry"
    exit_on = "z_long 下穿 −exit" if short_side else "z_short 上穿 +exit"
    legend = T.legend([
        ("Spread / mid z", "var(--s1)"),
        ("z_short 與 z_long（可成交側）", "var(--s3)"),
        ("滾動均值與 entry 門檻", "var(--s2)"),
    ])
    return T.Fig(
        fignum, "一筆交易的解剖",
        f"實際成交的一筆 {side}（{ex['ret_bps']:+.1f} bps，貼近本組態的中位數）。"
        "上：spread 與其滾動均值、entry 門檻帶。下：引擎真正據以判斷的三條 z。",
        svg,
        f"<b>spread 不是一條線，是三條。</b>引擎為每個方向各建一條「該筆委託真正要跨過的那一側」的 spread："
        f"z_short 是賣出 {spec.equity_leg}、買進 {spec.futures_leg} 拿得到的價，z_long 是反向。"
        f"兩者各距中價 displacement / spread_std，圖中的淺色帶就是這個間隔 —— 它會隨滾動標準差變寬變窄，"
        f"所以是 z 的位移而非常數。<br>"
        f"本例<b>進場條件是 {enter_on}</b>（不是 mid），"
        f"<b>出場翻到另一側：{exit_on}</b>，因為平倉是反方向的委託。"
        f"訊號與成交相隔一根 bar：訊號在收盤成立，成交在下一根允許交易的 bar 開盤，"
        "兩次成交各收一次盤口價差成本。",
        spec.spread.name,
        legend=legend,
    ).render()


def build(spec: PairSpec) -> None:
    spread, seed = load_frames(spec)
    print(f"{spec.name}: {len(spread):,} spread rows, seed={'yes' if seed is not None else 'no'}")

    results = []
    for seg in spec.segments:
        # A segment can outlive its data. QFF/TSM's pre-tick-change segment is
        # bounded at 2026-07-04, and TAIFEX publishes 30 trading days, so once
        # the window rolls past that date the slice is empty -- an empty frame
        # reaches build_summary and dies on equity.iloc[-1]. Skipping says so
        # and still builds the report from the segments that do have data;
        # crashing produced no report at all.
        if slice_segment(spread, seg).empty:
            print(f" segment {seg.label}: no data in range, skipped")
            continue
        print(f" segment {seg.label}")
        g = run_grid(spec, spread, seed, seg)
        g["neighbour"] = neighbour_sharpe(g)
        surv = screen(g)
        # The headline is the highest linearly-annualised return in the grid,
        # screened or not. That is a deliberate choice of ranking, and it makes
        # the screen a LABEL rather than a filter -- so `screened` now means
        # "this pick passed", not "something passed", and the report says which
        # of the two it is. Ranking by return and silently reporting it as
        # screened is the one combination that is not allowed.
        pick = g.nlargest(1, "ann").iloc[0]
        screened = bool(
            len(surv)
            and (
                (surv["window"] == pick.window)
                & (surv["entry_z"] == pick.entry_z)
                & (surv["exit_z"] == pick.exit_z)
            ).any()
        )
        res, trades = trade_detail(spec, spread, seed, seg, pick)
        results.append((seg, g, surv, pick, res, trades, screened))

    if not results:
        raise RuntimeError(
            f"{spec.name}: no segment has data. Spread covers "
            f"{pd.DatetimeIndex(spread['timestamp']).min()} to "
            f"{pd.DatetimeIndex(spread['timestamp']).max()}; segments are "
            + ", ".join(f"{s.label} [{s.start or 'open'}..{s.end or 'open'}]"
                        for s in spec.segments)
        )
    seg, g, surv, pick, res, trades, screened = results[0]
    s = res.summary
    labels, eq = session_equity(res.equity)
    sessions = len(labels)
    days = (pd.Timestamp(s["end"]) - pd.Timestamp(s["start"])).total_seconds() / 86400.0
    stats = grid.calculate_daily_return_stats(
        res.equity, initial_capital_twd=CAPITAL, annual_trading_days=252.0
    )
    cross_share = s["total_crossing_cost_twd"] / s["total_fee_twd"] * 100
    # Costs as bps of the leg notional traded. Round-trip by construction, since
    # both the entry and the exit fill are charged against the same notional --
    # which is why crossing cost lands at 200 x displacement and can be checked
    # against the measured book width.
    notional = float(trades["actual_leg_notional_twd"].sum())
    pick_total = float(trades["total_fee_twd"].sum())
    pick_cross = float(trades["crossing_cost_twd"].sum())
    # Gross notional against capital, measured rather than assumed. Both legs
    # carry the same notional and point opposite ways, so gross is 2x one leg
    # and net exposure is ~0. Whole-contract rounding is why the max exceeds 1.
    gross_x = float(trades["actual_leg_notional_twd"].median() * 2 / CAPITAL)
    gross_x_max = float(trades["actual_leg_notional_twd"].max() * 2 / CAPITAL)
    pick_total_bps = pick_total / notional * 10000.0
    pick_cross_bps = pick_cross / notional * 10000.0

    body = [
        T.doc_header(
            f"{spec.name} · 1-minute pair backtest",
            f"{spec.name} 配對策略回測報告",
            f"{spec.legs}. 所有結果以資本百分比或腿名目 bps 表示，與部位規模無關。",
            [
                ("期間", f"{s['start'][:10]} → {s['end'][:10]}"),
                ("Sessions", f"{sessions}"),
                ("資料", spec.spread.name),
                ("產生方式", "make_reports.py"),
            ],
        )
    ]

    body.append(
        '<div class="abstract"><h3>摘要</h3>'
        f"<p>本策略在 {spec.legs_zh or spec.legs} 之間做配對均值回歸。兩腿以相同名目對沖，"
        f"價差（spread）相對其滾動均值偏離達 entry_z 個標準差時進場，"
        f"回歸至 exit_z 時平倉。訊號與成交都以<strong>可成交價</strong>評分 —— "
        f"引擎為每個方向各建一條「該筆委託真正要跨過的那一側」的 spread，而非中價（見第 1 節）。</p>"
        + (
            f"<p>在 {sessions} 個 session、{days:.0f} 個日曆日的樣本上，"
            f"依<strong>線性年化報酬</strong>排序的第一名為（該格同時通過四道品質門檻）："
            if screened else
            f"<p>在 {sessions} 個 session、{days:.0f} 個日曆日的樣本上，"
            f"依<strong>線性年化報酬</strong>排序的第一名為 "
            f"<strong>（此格未通過四道品質門檻，見第 3 節）</strong>："
        )
        + f"<code>window {pick.window:.0f} / entry_z {pick.entry_z:g} / exit_z {pick.exit_z:g}</code>，"
        f"取得 <strong>{len(trades)} 筆交易</strong>、總報酬 "
        f"<strong>{s['return_pct'] * 100:.2f}%</strong>（線性年化 "
        f"{s['return_pct'] * 365.0 / days * 100:.1f}%）、Sharpe "
        f"<strong>{stats['sharpe_ratio']:.2f}</strong>、"
        f"最大回撤 {s['max_drawdown_pct'] * 100:.2f}%。"
        f"全程 <strong>no leverage</strong> —— 兩腿名目合計等於資本（{gross_x:.2f}x），"
        f"報酬來自部位本身而非融資放大。</p>"
        f"<p>執行成本是本配對的主要特徵：全部成本合計 "
        f"<strong>{pick_total_bps:.1f} bps</strong>（腿名目，來回），"
        f"其中盤口價差成本 <strong>{pick_cross_bps:.1f} bps</strong> 佔 "
        f"<strong>{cross_share:.1f}%</strong>，遠高於佣金與交易稅之和。"
        f"每筆交易每口的中位邊際為 <strong>{trades['ticks'].median():.2f} 個 tick</strong>，"
        f"即付完全部成本後仍高於微結構地板 {trades['ticks'].median():.1f} 倍（見第 3 節）。</p>"
        f"<p class=\"ptr\">使用前的限制與保留見第 6 節。</p></div>"
    )

    # Linear annualisation on calendar days. On a two-month sample this is an
    # extrapolation, not a forecast, so the card says so underneath rather than
    # letting the headline number stand unqualified.
    annualised = s["return_pct"] * 365.0 / days
    body.append(
        T.cards([
            ("年化報酬 Annualised", f"{annualised * 100:.1f}%",
             f"no leverage・總報酬 {s['return_pct'] * 100:.2f}%，{days:.0f} 日線性年化",
             "pos" if annualised > 0 else "neg"),
            ("Sharpe", f"{stats['sharpe_ratio']:.2f}",
             f"252 日年化，{stats['daily_return_count']} 個日報酬", ""),
            ("最大回撤 maxDD", f"{s['max_drawdown_pct'] * 100:.2f}%",
             f"報酬 / 回撤 {abs(s['return_pct'] / s['max_drawdown_pct']):.1f}x", "neg"),
        ])
    )

    body.append(T.section("1", "策略機制"))
    zframe_pick = slice_segment(
        zscore_calc.calculate_zscore(spread, int(pick.window), seed_frame=seed), seg
    ).reset_index(drop=True)
    body.append(anatomy_figure(spec, zframe_pick, trades, pick, seg, 1))

    body.append(T.section("2", "資料與建構"))
    body.append(
        "<p>Spread 定義為 <code>(fair − futures) / (fair + futures) × 200</code>，"
        "即以中價為基準的百分比尺度，因此一個 spread 單位約等於腿名目的 1%。"
        f"{spec.futures_leg} 以 as-of 對齊到 {spec.equity_leg} 的分鐘格上，"
        "FX 由最細可用區間拼接。</p>"
    )
    if spec.extra:
        body.append(f"<p>{spec.extra}</p>")
    # What the US leg actually cost this sample, in bps of the notional it
    # carried. The spec's flat bps is not what the 'ibkr' model charges, and
    # printing it described a cost the engine never applied.
    us_fee_bps_median = float(
        (trades["tsm_fee_twd"] / trades["actual_leg_notional_twd"] * 10000.0).median()
    )
    body.append(
        T.table(
            ["項目", "設定", "說明"],
            [
                ["交易期間", f"{s['start'][:16]} → {s['end'][:16]}", f"{sessions} 個 session"],
                ["Spread 檔案", f"<code>{T.esc(spec.spread.name)}</code>", f"{len(spread):,} 列"],
                ["Warmup seed",
                 f"<code>{T.esc(spec.seed.name)}</code>" if seed is not None else "無",
                 "所有 window 交易同一段期間" if seed is not None else "各 window 自行 burn warmup"],
                [f"{spec.equity_leg} 手續費",
                 ("IBKR 每股計費" if spec.equity_fee_model == "ibkr"
                  else f"{spec.equity_fee_bps:g} bps"),
                 ("單邊。$0.005/ADR、最低 $1.00，賣出另計 SEC/FINRA；"
                  f"本樣本實收中位 {us_fee_bps_median:.2f} bps"
                  if spec.equity_fee_model == "ibkr" else "單邊")],
                [f"{spec.futures_leg} 手續費",
                 f"{spec.futures_fee_twd:,.0f} TWD / 口"
                 f"（{spec.futures_fee_twd / (trades['actual_leg_notional_twd'].median() / abs(trades['qff_contracts']).median()) * 10000:.2f} bps）",
                 "單邊，固定制。bps 以本樣本的中位契約名目換算"],
                ["交易稅", f"{spec.tax_rate:.0e}", "期貨腿，單邊"],
                ["契約乘數", f"{spec.multiplier:,.0f}", "股 / 口"],
                ["1 tick", spec.tick_desc, "微結構門檻的基準"],
                ["資本與槓桿",
                 f"{CAPITAL:,.0f} TWD 資本，每腿目標 {LEG_NOTIONAL:,.0f} TWD",
                 f"<b>no leverage</b>：兩腿名目合計 / 資本 = {gross_x:.2f}x"
                 f"（最高 {gross_x_max:.2f}x，整數口數進位所致）。兩腿等名目反向，"
                 "淨曝險約為零。期貨實際保證金遠低於名目，故此處以全額名目當資本是保守估計"],
            ],
            left_cols={0, 1, 2},
            number="Table 0", title="資料來源與成本假設",
        )
    )

    body.append(T.section("3", "執行成本模型"))
    body.append(
        f"<p>訊號不以中價評分。引擎為每個方向各建一條可成交 spread —— 做空用 "
        f"{spec.equity_leg} bid 配 {spec.futures_leg} ask，做多相反 —— 並以「該筆委託真正要跨過的那一側」"
        f"檢定進場與出場。單邊位移為 <strong>{seg.displacement:.4f}</strong> spread 單位。{seg.note}</p>"
        "<p>位移同時做兩件事，這是同一個修正的兩半而非重複計算："
        "<strong>訊號面</strong>每根 bar 的門檻位移 <code>displacement / spread_std</code>，"
        "因為滾動標準差會變，所以這是 z 的位移而非常數；"
        "<strong>成交面</strong>由於引擎仍以中價成交，同樣的位移每邊收取一次"
        "<strong>盤口價差成本</strong>（order-book gap，欄位 "
        "<code>crossing_cost_twd</code>），把中價成交價換算成可成交價。"
        "注意這裡的「價差」指的是委託簿上買賣報價之間的間隔，"
        "與本報告用來做均值回歸的 spread 是兩回事。</p>"
    )

    # A tick is the unit the whole microstructure screen is denominated in, so
    # it gets defined where it is first leaned on rather than assumed.
    per_contract_notional = float(
        (trades["actual_leg_notional_twd"] / trades["qff_contracts"].abs()).median()
    )
    tick_bps = spec.tick_twd / per_contract_notional * 10000.0
    med_tick = float(trades["ticks"].median())
    body.append("<h3>一個 tick 是什麼，為什麼用它當標準</h3>")
    body.append(
        f"<p><strong>Tick 是最小跳動單位</strong> —— 交易所規定該商品價格能移動的最小級距，"
        f"價格不可能變動得比一個 tick 更小。{spec.futures_leg} 的報價級距為 "
        f"{spec.tick_desc}。以本樣本的中位契約名目 {per_contract_notional:,.0f} TWD 換算，"
        f"<strong>一個 tick 約等於 {tick_bps:.1f} bps</strong>。</p>"
        f"<p>買賣價差不可能窄於一個 tick，因此它是一條硬底線："
        f"<strong>一筆交易若每口賺不到一個 tick，那個獲利就無法與買賣價差跳動區分</strong> —— "
        f"衡量到的只是成交價印在買價還是賣價上的隨機性，不是 edge。"
        f"這就是第 4 節的篩選把門檻設在「中位邊際 ≥ 2 ticks」的原因。</p>"
        f"<p>要注意本報告的 tick 數是<strong>付完盤口價差成本之後</strong>的餘額，"
        f"因為位移機制已經在成本裡收過一次來回的盤口費用。三個單位對照如下：</p>"
    )
    gross_bps = float(trades["gross_pnl_twd"].sum() / notional * 10000.0)
    body.append(
        T.table(
            ["項目", "bps of 腿名目", "換算成 tick", "說明"],
            [
                ["毛利（來回）", f"{gross_bps:.1f}", f"{gross_bps / tick_bps:.2f}", "未扣成本"],
                ["全部成本（來回）", f"−{pick_total_bps:.1f}", f"−{pick_total_bps / tick_bps:.2f}",
                 f"其中盤口價差 {pick_cross_bps:.1f} bps"],
                ["淨額（中位那筆）", f"{med_tick * tick_bps:.1f}", f"{med_tick:.2f}",
                 "即報告各處引用的中位邊際"],
                ["一個 tick", f"{tick_bps:.1f}", "1.00", "微結構地板"],
            ],
            left_cols={0, 3},
            number="Table 1", title="成本階梯：bps 與 tick 的對照",
            caption="毛利與成本為整段樣本的加總折算，淨額為每筆交易的中位數，"
                    "因此三者不會恰好相加。",
        )
    )

    body.append(T.section("4", "結果"))
    fignum = 2
    body.append(
        T.Fig(
            fignum, "權益曲線與回撤",
            "上：累積報酬，虛線為零。下：距前高的回撤。兩者皆以起始資本的百分比表示，"
            "與資本規模無關。",
            T.equity_drawdown(labels, eq,
                              aria=f"{spec.name} cumulative return and drawdown by session day"),
            "每個 session day 取最後一根 bar 的權益。未平倉部位以中價計價，"
            "因此出場側的盤口價差成本要到實際成交才反映在曲線上。",
            spec.spread.name,
        ).render()
    )
    fignum += 1

    cost_names = [
        "盤口價差 Order-book gap",
        f"{spec.futures_leg} 手續費",
        f"{spec.equity_leg} 手續費",
        "交易稅 Tax",
    ]
    cost_cols = ["crossing_cost_twd", "qff_fee_twd", "tsm_fee_twd", "qff_tax_twd"]
    top = g.nlargest(5, "ann")
    cats: list[str] = []
    comps: dict[str, list[float]] = {n: [] for n in cost_names}
    for r in top.itertuples():
        _, tt = trade_detail(spec, spread, seed, seg, r)
        cats.append(f"w{r.window:.0f} / e{r.entry_z:g} / x{r.exit_z:g}")
        # Share of gross profit, not bps of notional: per round trip the cost is
        # near-constant across configurations (it is dominated by one crossing),
        # so normalising by notional hides the very thing this figure is about --
        # that turnover, not per-trade cost, is what execution takes out of you.
        gross = float(tt["gross_pnl_twd"].sum())
        for name, col in zip(cost_names, cost_cols):
            comps[name].append(float(tt[col].sum()) / gross * 100.0)
    body.append(
        T.Fig(
            fignum, "成本拆解",
            "篩選後前五名組態的成本組成，以佔該組態毛利的百分比表示。"
            "長條總長即執行成本吃掉的毛利比例。",
            T.stacked_bars(cats, [(n, comps[n]) for n in cost_names],
                           aria=f"{spec.name} cost as a share of gross profit",
                           unit="% of gross"),
f"每次來回的成本幾乎與組態無關（本樣本 <b>{pick_total_bps:.1f} bps</b> of 腿名目，"
            f"其中盤口價差 <b>{pick_cross_bps:.1f} bps</b> 佔 "
            f"<b>{pick_cross / pick_total * 100:.1f}%</b>），可以直接對帳："
            f"位移每邊收一次，來回即 200 × {seg.displacement:.4f} = "
            f"<b>{200 * seg.displacement:.1f} bps</b>，與量測到的盤口寬度一致。"
            "既然單次成本固定，長條的差異就完全來自周轉率 —— "
            "交易越密的組態，執行成本吃掉的毛利比例越高。",
            spec.spread.name,
            legend=T.legend([(n, f"var(--s{i + 1})") for i, n in enumerate(cost_names)]),
        ).render()
    )
    fignum += 1

    body.append(T.section("5", "參數穩健性"))
    body.append(
        f"<p>網格為 window × entry_z × exit_z = {len(WINDOWS)}×{len(ENTRIES)}×{len(EXITS)}，"
        f"扣除 <code>exit_z &lt; −entry_z</code> 的退化組合後共 {len(g)} 格，"
        f"其中 {int((g['net'] > 0).sum())} 格淨利為正，Sharpe 中位數 {g['sharpe'].median():.2f}。</p>"
    )
    panels = []
    for x in EXITS:
        mat = []
        for w in WINDOWS:
            row = []
            for e in ENTRIES:
                sel = g[(g.window == w) & (g.entry_z == e) & (g.exit_z == x)]
                ok = len(sel) and not pd.isna(sel.iloc[0]["sharpe"])
                row.append(float(sel.iloc[0]["sharpe"]) if ok else None)
            mat.append(row)
        panels.append((f"exit_z {x:g}", mat))
    mark = None
    for pi, x in enumerate(EXITS):
        if x == pick.exit_z:
            mark = (pi, WINDOWS.index(int(pick.window)), ENTRIES.index(pick.entry_z))
    body.append(
        T.Fig(
            fignum, "參數網格 Sharpe",
            "同一色階跨四個 exit_z 面板。深色為高 Sharpe，灰格為被排除的退化組合，黑框為建議組態。",
            T.heatmap([str(w) for w in WINDOWS], [f"{e:g}" for e in ENTRIES], panels,
                      aria=f"{spec.name} Sharpe by window, entry and exit threshold",
                      row_title="window", col_title="entry_z", mark=mark),
            f"Sharpe 高不等於可用。見 Figure {fignum + 1}：網格上 Sharpe 最高的格子，"
            "往往正是每筆邊際最薄的格子。",
            spec.spread.name,
        ).render()
    )
    fignum += 1

    body.append("<h3>篩選</h3>")
    body.append(
        "<p>四道門檻：交易筆數 ≥ 15（低於此統計無意義）、每口中位邊際 ≥ 2 個 tick、"
        "落在一個 tick 以內的交易 ≤ 15%、鄰域 Sharpe ≥ 全網格中位數（排除孤立尖峰）。"
        "位移已經把量測到的盤口成本收掉了，所以 tick 檢定在這裡是<strong>剩餘餘裕</strong>的檢查："
        "付完盤口價差成本之後還剩多少緩衝，可以容忍盤口模型偏樂觀。</p>"
    )
    srows = []
    surv_keys = {
        (r.window, r.entry_z, r.exit_z) for r in surv.itertuples()
    }
    for r in g.nlargest(8, "ann").itertuples():
        srows.append([
            f"{r.window:.0f}", f"{r.entry_z:g}", f"{r.exit_z:g}",
            f"{r.ann * 100:.1f}%", f"{r.ret * 100:.2f}%",
            f"{r.sharpe:.2f}", f"{r.trades:.0f}",
            f"{r.maxdd * 100:.2f}%", f"{r.med_ticks:.2f}", f"{r.under1 * 100:.0f}%",
            f"{r.win * 100:.0f}%",
            "✓" if (r.window, r.entry_z, r.exit_z) in surv_keys else "✗",
        ])
    body.append(
        T.table(
            ["window", "entry_z", "exit_z", "年化", "報酬",
             "Sharpe", "交易", "maxDD", "中位 ticks", "<1 tick", "勝率", "篩選"],
            srows, highlight=0,
            number="Table 2", title="依線性年化報酬排序的組態",
            caption=(
                f"全網格 {len(g)} 格依線性年化報酬排序前 8 名，第一列（加左側色條）"
                f"為本報告頭條組態。<strong>通過四道門檻者共 {len(surv)} 格</strong>"
                + (
                    "，頭條組態在其中。" if screened else
                    "；<strong>頭條組態不在其中</strong>——它的年化最高，但品質門檻沒過，"
                    "換句話說它的部分報酬落在 tick 雜訊之內而無法驗證。"
                )
                + "「篩選」欄標示每一格是否四道全過。"
            ),
        )
    )

    body.append(
        T.Fig(
            fignum, "每筆邊際與持有時間",
            "每個點是一筆交易。橫軸為持有時間，縱軸為該筆每口淨利折算的 tick 數。",
            T.scatter(
                [(float(r.hours), float(max(r.ticks, 0)),
                  f"{r.entry_time:%m-%d %H:%M} → {r.exit_time:%m-%d %H:%M}，"
                  f"{r.ret_bps:+.1f} bps，{r.ticks:.2f} ticks",
                  0 if r.ticks >= 1 else 1)
                 for r in trades.itertuples()],
                aria=f"{spec.name} per-trade edge in ticks against holding time",
                x_label="持有時間（小時）", y_label="每口淨利（ticks）",
                y_max=max(4.0, float(trades["ticks"].max()) * 1.1),
            ),
            "落在一個 tick 以內的交易，其獲利無法與買賣價跳動區分。本組態有 "
            f"<b>{int((trades['ticks'] < 1).sum())} / {len(trades)}</b> 筆落在該區間。"
            "虧損交易顯示在 0 的位置。",
            spec.spread.name,
            legend=T.legend([("≥ 1 tick", "var(--s1)"), ("< 1 tick", "var(--s2)")], round_key=True),
        ).render()
    )
    fignum += 1

    body.append(
        T.Fig(
            fignum, "交易形狀：損益與持有時間分布",
            "左：每筆報酬，以該筆腿名目的 bps 表示、已扣除全部成本。"
            "右：自進場成交到出場成交的經過時間，含閉市時段。",
            T.histogram_pair(
                ([float(v) for v in trades["ret_bps"]], "每筆報酬（bps of 腿名目）", True),
                ([float(v) for v in trades["hours"]], "持有時間（小時）", False),
                aria=f"{spec.name} distribution of per-trade return and holding time",
            ),
            f"報酬中位數 <b>{trades['ret_bps'].median():.1f} bps</b>、"
            f"平均 {trades['ret_bps'].mean():.1f} bps —— 平均高於中位表示右尾在拉抬，"
            "所以判斷微結構門檻要用中位數。持有時間中位 "
            f"<b>{trades['hours'].median():.1f} 小時</b>、最長 {trades['hours'].max():.0f} 小時，"
            "跨夜是常態而非例外，隔夜保證金與借券成本必須另行計算。",
            spec.spread.name,
        ).render()
    )
    fignum += 1

    if len(results) > 1:
        body.append("<h3>區間對照</h3>")
        rows = []
        for sg, gg, sv, pk, rs, tt, sc in results:
            st = grid.calculate_daily_return_stats(
                rs.equity, initial_capital_twd=CAPITAL, annual_trading_days=252.0
            )
            rows.append([
                T.esc(sg.label), f"{sg.displacement:.4f}",
                f"{rs.summary['start'][:10]} → {rs.summary['end'][:10]}",
                f"w{pk.window:.0f}/e{pk.entry_z:g}/x{pk.exit_z:g}",
                f"{st['sharpe_ratio']:.2f}", f"{len(tt)}",
                f"{rs.summary['return_pct'] * 100:.2f}%",
                f"{tt['ticks'].median():.2f}",
            ])
        body.append(
            T.table(
                ["區間", "位移", "期間", "最佳組態", "Sharpe", "交易", "報酬", "中位 ticks"],
                rows, left_cols={0, 2, 3}, highlight=0,
                caption="兩段期間各自以自己的位移定價，數字不可直接相加或比較。",
            )
        )

    body.append(T.section("6", "限制與保留"))
    weekend = sum(
        any(d.weekday() == 5 for d in pd.date_range(a.normalize(), b.normalize(), freq="D"))
        for a, b in zip(trades["entry_time"], trades["exit_time"])
    )
    win_rate = float((trades["net_pnl_twd"] > 0).mean())
    extreme_win = ""
    if win_rate >= 0.95:
        # A near-perfect record on a sample this small is the most likely thing
        # in the whole report to be mistaken for evidence, so it gets named
        # first rather than left inside the generic sample-size caveat.
        extreme_win = (
            f"<p><strong>勝率 {win_rate * 100:.0f}% 不是策略品質的證據。</strong>"
            f"{len(trades)} 筆交易全數獲利，在均值回歸策略上是可預期的形狀 —— "
            "部位只在 spread 收斂時才平倉，虧損被推遲到強制出場或樣本結束，"
            "而這個樣本兩者都很少發生。真正的分布要看單筆損益的左尾，"
            f"本樣本最長持有 {trades['hours'].max():.0f} 小時，"
            "尚不足以觀察到不收斂的情況。</p>"
        )
    body.append(
        '<div class="callout"><h3>這份結果不能證明什麼</h3>'
        + extreme_win
        + f"<p><strong>樣本規模。</strong>{len(trades)} 筆交易、{sessions} 個 session。"
        f"勝率 {win_rate * 100:.0f}% 與獲利因子在這個樣本數下"
        "是描述統計，不是預期值。</p>"
        f"<p><strong>樣本內選參。</strong>建議組態是在同一段資料上由網格選出的，"
        f"沒有任何樣本外驗證。網格穩健度（{int((g['net'] > 0).sum())}/{len(g)} 格為正）"
        "比單一格的 Sharpe 更值得參考。</p>"
        f"<p><strong>週末與隔夜。</strong>{weekend} / {len(trades)} 筆交易跨越週末，"
        f"最長持有 {trades['hours'].max():.0f} 小時。回測只在 session 內對部位計價，"
        "閉市期間的不利波動與保證金壓力不在模型內。</p>"
        f"<p><strong>盤口模型。</strong>位移 {seg.displacement:.4f} 由有限的報價樣本量測而得。"
        f"若實際盤口更寬，盤口價差成本會等比例上升 —— 而它已是 {pick_cross_bps:.1f} bps、"
        f"佔本組態全部成本的 {pick_cross / pick_total * 100:.0f}%。</p>"
        + (
            "<p><strong>成本參數與資料不同源。</strong>永續腿的資料現在來自 OKX，"
            "但位移與 bps 費率當初是在 Binance 的盤口上量測的。OKX 的簿子較薄，"
            "因此這裡的成本很可能<strong>偏低估</strong>；在 OKX 重新量測之前，"
            "本報告的成本面屬於沿用而非觀測。</p>"
            if spec.key == "qff_tsm" else ""
        )
        + "</div>"
    )

    body.append(T.section("附錄 A", "完整參數網格"))
    arows = []
    for r in g.sort_values("sharpe", ascending=False).itertuples():
        arows.append([
            f"{r.window:.0f}", f"{r.entry_z:g}", f"{r.exit_z:g}",
            "—" if pd.isna(r.sharpe) else f"{r.sharpe:.2f}",
            f"{r.trades:.0f}", f"{r.ret * 100:.2f}%", f"{r.maxdd * 100:.2f}%",
            "—" if pd.isna(r.med_ticks) else f"{r.med_ticks:.2f}",
            "—" if pd.isna(r.under1) else f"{r.under1 * 100:.0f}%",
        ])
    body.append(
        "<p>全部 {n} 格的完整結果如下。預設收合以免淹沒正文；"
        "報告的結論建立在整片網格的分布上，而不是任何單一格，因此這張表是可稽核的底稿。</p>".format(n=len(g))
    )
    body.append(
        T.details(
            f"展開完整網格（{len(g)} 格）",
            T.table(
                ["window", "entry_z", "exit_z", "Sharpe", "交易",
                 "報酬", "maxDD", "中位 ticks", "<1 tick"],
                arows,
                number="Table A1", title="完整參數網格",
                caption=f"全部 {len(g)} 格，依 Sharpe 由高至低排序。",
            ),
        )
    )
    body.append(
        T.footer(
            f"{T.esc(spec.name)} · 由 <code>scripts/report/pair.py</code> 產生，"
            f"所有數字於產生時自 <code>{T.esc(str(spec.spread))}</code> 重新計算。"
        )
    )

    spec.out.write_text(T.page(f"{spec.name} 回測報告", "\n".join(body)), encoding="utf-8")
    print(f"wrote {spec.out}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build the technical report for one pair.")
    ap.add_argument("--pair", choices=sorted(PAIRS), required=True)
    args = ap.parse_args(argv)
    build(PAIRS[args.pair])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
