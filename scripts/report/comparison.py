"""Build the head-to-head comparison report for the two pairs.

Scores both pairs on their common window with each pair's own measured book
width, then asks the question the per-pair reports cannot: is the difference
between them larger than the noise in a sample this small?

    python scripts/report/comparison.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import engine as backtest  # noqa: E402
from backtest import grid  # noqa: E402
from features import zscore as zscore_calc  # noqa: E402
from lib import paths  # noqa: E402
from report import pair  # noqa: E402
from report import theme as T  # noqa: E402

OUT = paths.report("pairs_comparison")
# The QFF tick change on 2026-07-05 makes a single displacement meaningless
# across it, so the common window starts after the break.
COMMON_START = "2026-07-06"
BOOTSTRAP = 20000
SEED = 20260816


def daily(equity: pd.DataFrame) -> pd.Series:
    eq = equity[["timestamp", "equity"]].copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True).dt.tz_convert("Asia/Taipei")
    day = (eq["timestamp"] - pd.Timedelta(hours=6)).dt.date
    close = eq.groupby(day)["equity"].last()
    return close.diff().dropna() / pair.CAPITAL


def sharpe(r: pd.Series) -> float:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252.0))


def evaluate(spec: pair.PairSpec) -> dict:
    """Grid, screen and pick on the common window, then keep trade detail."""
    spread, seed = pair.load_frames(spec)
    seg = pair.Segment(
        "common", COMMON_START, None, spec.segments[0].displacement,
        spec.segments[0].note,
    )
    g = pair.run_grid(spec, spread, seed, seg)
    g["neighbour"] = pair.neighbour_sharpe(g)
    surv = pair.screen(g)
    pick = (surv if len(surv) else g).nlargest(1, "sharpe").iloc[0]
    res, trades = pair.trade_detail(spec, spread, seed, seg, pick)
    s = res.summary
    days = (pd.Timestamp(s["end"]) - pd.Timestamp(s["start"])).total_seconds() / 86400.0
    weekend = sum(
        any(d.weekday() == 5 for d in pd.date_range(a.normalize(), b.normalize(), freq="D"))
        for a, b in zip(trades["entry_time"], trades["exit_time"])
    )
    r = daily(res.equity)
    notional = float(trades["actual_leg_notional_twd"].sum())
    return dict(
        cost_bps=float(trades["total_fee_twd"].sum() / notional * 10000.0),
        cross_bps=float(trades["crossing_cost_twd"].sum() / notional * 10000.0),
        gross_x=float(trades["actual_leg_notional_twd"].median() * 2 / pair.CAPITAL),
        ann=float(s["return_pct"]) * 365.0 / days,
        spec=spec, grid=g, surv=surv, pick=pick, res=res, trades=trades,
        summary=s, daily=r, sharpe=sharpe(r), days=days, weekend=weekend,
        disp=seg.displacement,
        fee_gross=float(trades["total_fee_twd"].sum() / trades["gross_pnl_twd"].sum()),
        cross_share=float(trades["crossing_cost_twd"].sum() / trades["total_fee_twd"].sum()),
    )


def bootstrap_gap(a: pd.Series, b: pd.Series) -> dict:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(joined))
    gaps = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        pick = rng.choice(idx, size=len(idx), replace=True)
        gaps[i] = sharpe(joined["a"].iloc[pick]) - sharpe(joined["b"].iloc[pick])
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return dict(
        days=len(joined), point=sharpe(joined["a"]) - sharpe(joined["b"]),
        lo=float(lo), hi=float(hi), p_b=float((gaps < 0).mean()),
        corr=float(joined["a"].corr(joined["b"])), draws=gaps,
    )


def main(argv: list[str]) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    a = evaluate(pair.PAIRS["ccf_umc"])
    b = evaluate(pair.PAIRS["qff_tsm"])
    bs = bootstrap_gap(a["daily"], b["daily"])
    joined = pd.concat(
        [a["daily"].rename("a"), b["daily"].rename("b")], axis=1
    ).fillna(0.0)
    blend = 0.5 * joined["a"] + 0.5 * joined["b"]

    an, bn = a["spec"].name, b["spec"].name
    body = [
        T.doc_header(
            "Pairs · head-to-head",
            "CCF / UMC 對 QFF / TSM 對照報告",
            "兩個配對在同一段共同視窗、各自以自己量測的盤口寬度定價後的正面比較。"
            "皆為 no leverage，結果以資本百分比與腿名目 bps 表示。"
            "核心問題不是誰的數字大，而是差距是否大於這個樣本的雜訊。",
            [
                ("共同視窗", f"{COMMON_START} → {a['summary']['end'][:10]}"),
                ("重疊 session", f"{bs['days']}"),
                ("Bootstrap", f"{BOOTSTRAP:,} 次重抽樣"),
            ],
        )
    ]

    verdict = (
        "點估計偏向 " + an + "，但統計上不足以宣稱勝出。"
        if bs["lo"] < 0 < bs["hi"]
        else "差距在此樣本下具統計顯著性。"
    )
    body.append(
        '<div class="abstract"><h3>摘要</h3>'
        f"<p><strong>{verdict}</strong>在 {bs['days']} 個重疊 session 上，"
        f"Sharpe 差距為 <strong>{bs['point']:+.2f}</strong>，"
        f"其 95% bootstrap 信賴區間為 <strong>[{bs['lo']:+.2f}, {bs['hi']:+.2f}]</strong>，"
        f"而 {bn} 在 <strong>{bs['p_b'] * 100:.1f}%</strong> 的重抽樣中勝出。"
        "五週的樣本分不開這兩個策略。</p>"
        f"<p>真正撐得住的是<strong>網格穩健度</strong>：{an} 在 "
        f"{int((a['grid']['net'] > 0).sum())}/{len(a['grid'])} 格淨利為正、Sharpe 中位 "
        f"{a['grid']['sharpe'].median():.2f}；{bn} 為 "
        f"{int((b['grid']['net'] > 0).sum())}/{len(b['grid'])} 格、中位 "
        f"{b['grid']['sharpe'].median():.2f}。這個比較建立在每邊上百格的分布上，"
        "而非單一組態，因此比任何單格 Sharpe 都穩固。</p>"
        f"<p>兩者日報酬相關僅 <strong>{bs['corr']:+.3f}</strong>，"
        f"50/50 混合的 Sharpe 為 <strong>{sharpe(blend):.2f}</strong>。"
        f"{bn} 的價值在於分散，而不是單獨取代。</p></div>"
    )

    body.append(T.section("1", "共同視窗上的正面比較"))
    body.append(
        f"<p>兩邊都在 {COMMON_START} 之後評分。這個起點不是任意的："
        "QFF 的最小跳動單位在 2026-07-05 由 5 TWD 降為 1 TWD，"
        "盤口價差成本一次降掉八成，單一位移參數跨不過那個斷點。"
        "每個配對各自跑完整網格、套用同一組篩選門檻，再取各自的最佳組態比較。</p>"
    )

    def row(label, x, y, note=""):
        return [label, x, y, note]

    rows = [
        row("最佳組態",
            f"w{a['pick'].window:.0f} / e{a['pick'].entry_z:g} / x{a['pick'].exit_z:g}",
            f"w{b['pick'].window:.0f} / e{b['pick'].entry_z:g} / x{b['pick'].exit_z:g}",
            "各自網格篩選後的第一名"),
        row("單邊位移", f"{a['disp']:.4f}", f"{b['disp']:.4f}", "spread 單位，各自量測"),
        row("期間", f"{a['days']:.0f} 日曆日", f"{b['days']:.0f} 日曆日", ""),
        row("交易筆數", f"{len(a['trades'])}", f"{len(b['trades'])}", ""),
        row("總報酬", f"{a['summary']['return_pct'] * 100:.2f}%",
            f"{b['summary']['return_pct'] * 100:.2f}%", "資本百分比，no leverage"),
        row("線性年化", f"{a['ann'] * 100:.1f}%", f"{b['ann'] * 100:.1f}%",
            "外插而非預測，樣本僅五週"),
        row("名目 / 資本", f"{a['gross_x']:.2f}x", f"{b['gross_x']:.2f}x",
            "兩腿等名目反向，淨曝險約為零"),
        row("Sharpe", f"{a['sharpe']:.2f}", f"{b['sharpe']:.2f}", "252 日年化，日報酬"),
        row("最大回撤", f"{a['summary']['max_drawdown_pct'] * 100:.2f}%",
            f"{b['summary']['max_drawdown_pct'] * 100:.2f}%", ""),
        row("勝率", f"{(a['trades'].net_pnl_twd > 0).mean() * 100:.0f}%",
            f"{(b['trades'].net_pnl_twd > 0).mean() * 100:.0f}%", "樣本過小，描述性數字"),
        row("每筆報酬中位", f"{a['trades'].ret_bps.median():.1f} bps",
            f"{b['trades'].ret_bps.median():.1f} bps", "腿名目 bps，兩者接近"),
        row("全部成本（來回）", f"{a['cost_bps']:.1f} bps", f"{b['cost_bps']:.1f} bps", "腿名目"),
        row("其中盤口價差", f"{a['cross_bps']:.1f} bps", f"{b['cross_bps']:.1f} bps",
            f"佔 {a['cross_share'] * 100:.0f}% / {b['cross_share'] * 100:.0f}%，CCF 的盤口主導其成本"),
        row("成本 / 毛利", f"{a['fee_gross'] * 100:.1f}%", f"{b['fee_gross'] * 100:.1f}%",
            "接近相同，所以差距來自 edge 而非成本"),
        row("每口中位邊際", f"{a['trades'].ticks.median():.2f} ticks",
            f"{b['trades'].ticks.median():.2f} ticks", "QFF 遠離微結構雜訊"),
        row("跨週末部位", f"{a['weekend']} / {len(a['trades'])}",
            f"{b['weekend']} / {len(b['trades'])}", "回測看不見閉市期間的風險"),
        row("在場時間", f"{a['summary']['exposure_ratio'] * 100:.1f}%",
            f"{b['summary']['exposure_ratio'] * 100:.1f}%", ""),
    ]
    body.append(
        T.table(
            ["項目", an, bn, "說明"], rows, left_cols={0, 3},
            number="Table 1", title="共同視窗上的正面比較",
            caption="兩邊各自以自己量測的盤口寬度定價，因此淨利可比，但成本組成不可互換解讀。",
        )
    )

    body.append(T.section("2", "差距有多少是雜訊"))
    body.append(
        f"<p>把 {bs['days']} 個重疊 session 的日報酬整段重抽樣 {BOOTSTRAP:,} 次，"
        "每次重算兩邊的 Sharpe 再取差，得到差距本身的分布。"
        "這是本報告唯一能回答「這個排名可信嗎」的統計。</p>"
    )
    lo_i, hi_i = np.percentile(bs["draws"], [2.5, 97.5])
    body.append(
        T.Fig(
            1, "Sharpe 差距的 bootstrap 分布",
            f"{an} 減 {bn} 的 Sharpe 差。虛線為 95% 區間，實線為零。",
            T.histogram(
                [float(v) for v in bs["draws"]],
                aria="bootstrap distribution of the Sharpe gap between the two pairs",
                x_label=f"Sharpe 差（{an} − {bn}）", bins=44, zero_line=True,
                height=260,
            ),
            f"分布橫跨零：區間 <b>[{lo_i:+.2f}, {hi_i:+.2f}]</b>，"
            f"落在零以下的比例為 <b>{bs['p_b'] * 100:.1f}%</b>。"
            "點估計是實數，但它與零之間的距離小於樣本雜訊，因此不足以據此選邊。",
            "兩配對的日報酬序列",
        ).render()
    )

    body.append(T.section("3", "網格穩健度：撐得住的那個比較"))
    body.append(
        "<p>單一組態的 Sharpe 受選參與運氣影響很大，整片網格的分布則不然。"
        "同一組 window × entry_z × exit_z 掃描套在兩個配對上，問的是"
        "「這個 edge 是否在參數選擇下普遍存在」。</p>"
    )
    grows = []
    for label, ga, gb in [
        ("淨利為正的格數", f"{int((a['grid'].net > 0).sum())}/{len(a['grid'])}",
         f"{int((b['grid'].net > 0).sum())}/{len(b['grid'])}"),
        ("Sharpe 中位數", f"{a['grid'].sharpe.median():.2f}", f"{b['grid'].sharpe.median():.2f}"),
        ("Sharpe > 3 的格數", f"{int((a['grid'].sharpe > 3).sum())}/{len(a['grid'])}",
         f"{int((b['grid'].sharpe > 3).sum())}/{len(b['grid'])}"),
        ("通過篩選的格數", f"{len(a['surv'])}/{len(a['grid'])}", f"{len(b['surv'])}/{len(b['grid'])}"),
        ("報酬中位數", f"{a['grid'].ret.median() * 100:.2f}%", f"{b['grid'].ret.median() * 100:.2f}%"),
    ]:
        grows.append([label, ga, gb])
    body.append(
        T.table(
            ["指標", an, bn], grows, left_cols={0},
            number="Table 2", title="網格分布比較",
            caption="篩選門檻與各配對報告相同：交易 ≥ 15 筆、每口中位邊際 ≥ 2 ticks、"
                    "一個 tick 以內 ≤ 15%、鄰域 Sharpe ≥ 該網格中位數。",
        )
    )

    body.append(T.section("4", "分散價值"))
    dd_a = (joined["a"].cumsum() - joined["a"].cumsum().cummax()).min()
    dd_b = (joined["b"].cumsum() - joined["b"].cumsum().cummax()).min()
    dd_x = (blend.cumsum() - blend.cumsum().cummax()).min()
    body.append(
        f"<p>兩者日報酬相關 <strong>{bs['corr']:+.3f}</strong>，"
        "接近無關。等權混合後：</p>"
    )
    body.append(
        T.table(
            ["組合", "Sharpe", "區間內最大回撤"],
            [
                [an, f"{sharpe(joined['a']):.2f}", f"{dd_a * 100:.2f}%"],
                [bn, f"{sharpe(joined['b']):.2f}", f"{dd_b * 100:.2f}%"],
                ["50 / 50 混合", f"{sharpe(blend):.2f}", f"{dd_x * 100:.2f}%"],
            ],
            left_cols={0}, highlight=2,
            number="Table 3", title="等權混合",
            caption="以重疊 session 的日報酬計算，缺漏日以 0 補。"
                    f"混合的 Sharpe {'高於' if sharpe(blend) > sharpe(joined['a']) else '未超過'}單押 {an}。",
        )
    )

    body.append(T.section("5", "保留"))
    body.append(
        '<div class="callout"><h3>這份比較不能回答什麼</h3>'
        f"<p><strong>樣本。</strong>{bs['days']} 個重疊 session、"
        f"{len(a['trades'])} 與 {len(b['trades'])} 筆交易。任何排名都在雜訊帶內。</p>"
        "<p><strong>風險不對稱。</strong>兩者的在場時間與跨週末部位數差距明顯，"
        "而回測只在 session 內對部位計價，閉市期間的不利波動與保證金壓力不在模型內。"
        "報酬相同不代表風險相同。</p>"
        "<p><strong>皆為樣本內。</strong>兩邊的組態都是在同一段資料上選出的，"
        "沒有任何樣本外驗證。網格穩健度是本報告中最不受此影響的指標。</p></div>"
    )
    body.append(
        T.footer(
            "由 <code>scripts/report/comparison.py</code> 產生，"
            "所有數字於產生時重新計算。"
        )
    )

    OUT.write_text(T.page("配對策略對照報告", "\n".join(body)), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
