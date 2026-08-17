"""Shared visual language for the backtest reports.

One module so every report speaks the same language: the page shell, the figure
wrapper, the tables, and the SVG chart primitives all live here and both
generators import them. Previously each report carried its own CSS and hand-rolled
its own SVG, which is why no two of them looked alike.

Design: a quant research-note skeleton (clean sans, generous whitespace, slate
ink, one restrained accent, tables as first-class content) with data-journalism
chart discipline (title left-aligned above the plot, direct labels instead of
legends where possible, hairline horizontal gridlines only, a source line under
every figure).

Charts are inline SVG with no external libraries, so a report is a single file
that renders offline, in light or dark, and prints.
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass

# Institutional research-note palette: deep navy-slate ink, one restrained
# accent, everything else neutral. Series slots are NOT four competing hues --
# slot 1 is the accent and carries the thing that matters, slots 2-4 are steps
# of slate that recede. That is the whole point: on the cost chart, crossing
# cost is the finding and the commissions are context, so hue marks the finding
# instead of spreading attention across four equals. Identity still never rests
# on color alone -- every chart using more than one step ships a legend, direct
# labels and per-mark tooltips.
HEAT_STEPS = 7


def esc(value: object) -> str:
    return html.escape(str(value))


def fmt(value: float | None, digits: int = 2, comma: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    spec = f",.{digits}f" if comma else f".{digits}f"
    return format(value, spec)


def pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.{digits}f}%"


# --------------------------------------------------------------------------- #
# Page shell                                                                    #
# --------------------------------------------------------------------------- #
STYLE = """
/* Latin letters and digits set in Times New Roman, CJK left on the sans face.
   The unicode-range confines the local() serif to Latin, digits, punctuation
   and the arrows/maths this report uses, so every CJK codepoint falls through
   to the next family in the stack rather than being rendered by whatever CJK
   glyphs the serif happens to carry. */
@font-face{font-family:"ReportSerif";
 src:local("Times New Roman"),local("Times"),local("Liberation Serif"),local("Tinos");
 unicode-range:U+0020-007E,U+00A0-00FF,U+2010-2027,U+2030-205E,U+2070-209F,
 U+20A0-20BF,U+2190-21FF,U+2200-22FF,U+2264-2265;}
:root{color-scheme:light;
 --page:#eef1f4;--surface:#ffffff;--ink:#16202e;--ink2:#4c5a6b;--muted:#8b97a5;
 --grid:#e4e8ec;--rule:#ccd4dc;--accent:#1f4e79;--accent-soft:#e9eff5;
 --s1:#1f4e79;--s2:#6b7c8f;--s3:#97a4b2;--s4:#c2cbd4;
 --good:#1f6f4a;--warn:#a8720f;--crit:#9e3232;
 --h0:#eaf0f5;--h1:#d2e0ea;--h2:#b1c8dc;--h3:#8aabc9;--h4:#5f8bb2;--h5:#3a6c98;--h6:#1f4e79;
 --hi0:#16202e;--hi1:#16202e;--hi2:#16202e;--hi3:#16202e;--hi4:#ffffff;--hi5:#ffffff;--hi6:#ffffff;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --page:#0b1016;--surface:#141c25;--ink:#e9eef3;--ink2:#a9b6c3;--muted:#78848f;
 --grid:#222c37;--rule:#2f3a46;--accent:#6ea8d8;--accent-soft:#18242f;
 --s1:#6ea8d8;--s2:#7e8d9c;--s3:#5c6a78;--s4:#3f4a56;
 --good:#4fa87a;--warn:#d3a03c;--crit:#d06767;
 --h0:#18242f;--h1:#1f3345;--h2:#2a475f;--h3:#3a6188;--h4:#4d7fac;--h5:#6ea8d8;--h6:#9cc7e8;
 --hi0:#e9eef3;--hi1:#e9eef3;--hi2:#e9eef3;--hi3:#ffffff;--hi4:#0b1016;--hi5:#0b1016;--hi6:#0b1016;}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0b1016;--surface:#141c25;--ink:#e9eef3;--ink2:#a9b6c3;--muted:#78848f;
 --grid:#222c37;--rule:#2f3a46;--accent:#6ea8d8;--accent-soft:#18242f;
 --s1:#6ea8d8;--s2:#7e8d9c;--s3:#5c6a78;--s4:#3f4a56;
 --good:#4fa87a;--warn:#d3a03c;--crit:#d06767;
 --h0:#18242f;--h1:#1f3345;--h2:#2a475f;--h3:#3a6188;--h4:#4d7fac;--h5:#6ea8d8;--h6:#9cc7e8;
 --hi0:#e9eef3;--hi1:#e9eef3;--hi2:#e9eef3;--hi3:#ffffff;--hi4:#0b1016;--hi5:#0b1016;--hi6:#0b1016;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
 font:16px/1.75 "ReportSerif",system-ui,-apple-system,"Segoe UI","Noto Sans TC",
 "Microsoft JhengHei",sans-serif;
 -webkit-font-smoothing:antialiased}
/* The tracked uppercase micro-labels stay on the sans: letterspaced serif caps
   read as decoration, and the contrast against the serif body is what makes
   the section furniture legible as furniture. */
.kicker,.fignum,.tnum,thead th,.abstract h3,.callout h3,.card .k{
 font-family:system-ui,-apple-system,"Segoe UI","Noto Sans TC","Microsoft JhengHei",sans-serif}
main{max-width:1000px;margin:0 auto;padding:0 24px 96px}
header.doc{padding:56px 0 26px;border-bottom:2px solid var(--ink);margin-bottom:8px}
.kicker{font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;
 color:var(--accent);font-weight:650;margin-bottom:12px}
h1{font-size:33px;line-height:1.2;margin:0 0 12px;letter-spacing:-.022em;font-weight:660}
.standfirst{font-size:17px;color:var(--ink2);margin:0 0 18px;max-width:64ch;line-height:1.6}
.meta{display:flex;flex-wrap:wrap;gap:8px 26px;font-size:12.5px;color:var(--muted);
 font-variant-numeric:tabular-nums}
.meta b{color:var(--ink2);font-weight:600}
h2{font-size:20px;margin:52px 0 6px;letter-spacing:-.015em;font-weight:640;
 padding-bottom:8px;border-bottom:1px solid var(--rule)}
h2 .num{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:12px}
h3{font-size:15.5px;margin:30px 0 6px;color:var(--ink);font-weight:640}
p{margin:0 0 15px;color:var(--ink2);max-width:74ch}
p.lead{color:var(--ink);font-size:16.5px}
strong{color:var(--ink);font-weight:640}
code{background:var(--accent-soft);color:var(--accent);padding:1.5px 6px;
 border-radius:4px;font-size:12.5px;font-family:ui-monospace,"SF Mono",Menlo,monospace}
a{color:var(--accent)}
.abstract{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);
 border-radius:3px;padding:22px 26px;margin:26px 0 8px}
.abstract h3{margin:0 0 8px;font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;
 color:var(--accent)}
.abstract p{margin:0 0 11px}.abstract p:last-child{margin:0}
.abstract p.ptr{font-size:13px;color:var(--muted);padding-top:9px;border-top:1px solid var(--grid)}
.callout{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--warn);
 border-radius:3px;padding:18px 24px;margin:22px 0}
.callout h3{margin:0 0 8px;font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--warn)}
.callout p:last-child{margin:0}
/* Three headline figures, so three columns -- a fixed count rather than
   auto-fit, which leaves an orphan cell whose background reads as a broken
   tile when the card count is not a multiple of the fitted column count. */
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
 background:var(--rule);border:1px solid var(--rule);border-radius:3px;overflow:hidden;margin:26px 0}
@media(max-width:640px){.cards{grid-template-columns:1fr}}
.card{padding:19px 21px}
.card .v{font-size:29px}
.card{background:var(--surface);padding:15px 17px}
.card .k{font-size:11.5px;color:var(--muted);margin-bottom:5px;letter-spacing:.02em}
.card .v{font-size:24px;font-weight:650;letter-spacing:-.02em;line-height:1.15}
.card .s{font-size:11.5px;color:var(--muted);margin-top:3px;font-variant-numeric:tabular-nums}
.card .v.pos{color:var(--good)}.card .v.neg{color:var(--crit)}
figure{margin:34px 0;background:var(--surface);border:1px solid var(--rule);
 border-radius:3px;padding:22px 24px 16px}
.fignum{font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);
 font-weight:650;margin-bottom:5px}
.figtitle{font-size:16px;font-weight:640;color:var(--ink);margin-bottom:3px;letter-spacing:-.012em}
.figsub{font-size:13px;color:var(--ink2);margin-bottom:16px;max-width:72ch}
figcaption{font-size:12px;color:var(--muted);margin-top:13px;padding-top:11px;
 border-top:1px solid var(--grid);max-width:78ch;line-height:1.6}
figcaption b{color:var(--ink2);font-weight:600}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin:0 0 12px}
.key{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:6px;vertical-align:0}
.key.round{border-radius:50%}
/* Tables are first-class content here, not a fallback for a missing chart:
   they get the same surface, border and vertical rhythm as a figure. */
.tbl{background:var(--surface);border:1px solid var(--rule);border-radius:3px;
 padding:20px 22px 12px;margin:30px 0}
.tbl .tnum{font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--accent);font-weight:650;margin-bottom:5px}
.tbl .ttitle{font-size:15.5px;font-weight:640;color:var(--ink);margin-bottom:14px;
 letter-spacing:-.012em}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
caption{text-align:left;font-size:12.5px;color:var(--muted);padding:10px 0 0;
 caption-side:bottom;line-height:1.6}
th,td{padding:9px 14px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
thead th{color:var(--muted);font-weight:620;border-bottom:1px solid var(--ink2);
 font-size:11px;letter-spacing:.07em;text-transform:uppercase;padding-bottom:7px}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:none}
tbody th,td.l,th.l{text-align:left;font-weight:500;color:var(--ink2);white-space:normal}
tbody tr:hover{background:var(--accent-soft)}
tr.hi{background:var(--accent-soft)}
tr.hi td,tr.hi th{font-weight:650;color:var(--ink);box-shadow:inset 3px 0 0 var(--accent)}
details{margin:30px 0}
details>summary{cursor:pointer;font-size:13.5px;color:var(--accent);font-weight:600;
 padding:11px 16px;background:var(--surface);border:1px solid var(--rule);border-radius:3px;
 list-style:none}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸ ";color:var(--muted)}
details[open]>summary::before{content:"▾ "}
details[open]>summary{border-radius:3px 3px 0 0;border-bottom:none}
details .tbl{margin:0;border-radius:0 0 3px 3px}
td.note{text-align:left;color:var(--muted);font-size:12px;white-space:normal}
.grid1{stroke:var(--grid);stroke-width:1}
.axis{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.axist{fill:var(--ink2);font-size:12px}
.dlabel{fill:var(--ink);font-size:11.5px;font-weight:640}
.mid{text-anchor:middle}.end{text-anchor:end}.start{text-anchor:start}
.ring{stroke:var(--surface);stroke-width:2}
footer{margin-top:64px;padding-top:18px;border-top:1px solid var(--rule);
 font-size:12px;color:var(--muted)}
@media print{
 body{background:#fff;font-size:10.5pt}
 main{max-width:none;padding:0}
 figure,.cards,.abstract,.callout{break-inside:avoid;border-color:#bbb}
 h2{break-after:avoid}figcaption{break-before:avoid}
 tbody tr:hover{background:none}
}
@media(max-width:640px){h1{font-size:26px}main{padding:0 15px 60px}.card .v{font-size:20px}}
"""


def page(title: str, body: str) -> str:
    return f"<title>{esc(title)}</title>\n<style>{STYLE}</style>\n<main>\n{body}\n</main>\n"


def doc_header(kicker: str, title: str, standfirst: str, meta: list[tuple[str, str]]) -> str:
    items = "".join(f"<span><b>{esc(k)}</b> {esc(v)}</span>" for k, v in meta)
    return (
        f'<header class="doc"><div class="kicker">{esc(kicker)}</div>'
        f"<h1>{esc(title)}</h1><p class=\"standfirst\">{esc(standfirst)}</p>"
        f'<div class="meta">{items}</div></header>'
    )


def section(number: str, title: str) -> str:
    return f'<h2><span class="num">{esc(number)}</span>{esc(title)}</h2>'


def cards(items: list[tuple[str, str, str, str]]) -> str:
    """(label, value, sub, tone) where tone is '', 'pos' or 'neg'."""
    body = "".join(
        f'<div class="card"><div class="k">{esc(k)}</div>'
        f'<div class="v {tone}">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
        for k, v, s, tone in items
    )
    return f'<div class="cards">{body}</div>'


@dataclass
class Fig:
    number: int
    title: str
    subtitle: str
    svg: str
    caption: str
    source: str
    legend: str = ""

    def render(self) -> str:
        cap = (
            f"<figcaption>{self.caption} "
            f"<b>Source:</b> {esc(self.source)}</figcaption>"
        )
        return (
            f'<figure><div class="fignum">Figure {self.number}</div>'
            f'<div class="figtitle">{esc(self.title)}</div>'
            f'<div class="figsub">{esc(self.subtitle)}</div>'
            f"{self.legend}{self.svg}{cap}</figure>"
        )


def legend(entries: list[tuple[str, str]], round_key: bool = False) -> str:
    shape = "key round" if round_key else "key"
    body = "".join(
        f'<span><span class="{shape}" style="background:{c}"></span>{esc(label)}</span>'
        for label, c in entries
    )
    return f'<div class="legend">{body}</div>'


def table(
    headers: list[str],
    rows: list[list[str]],
    *,
    left_cols: set[int] | None = None,
    highlight: int | None = None,
    caption: str = "",
    number: str = "",
    title: str = "",
    bare: bool = False,
) -> str:
    left = left_cols or set()
    head = "".join(
        f'<th class="l">{esc(h)}</th>' if i in left else f"<th>{esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for r, row in enumerate(rows):
        cls = ' class="hi"' if highlight == r else ""
        tds = "".join(
            f'<td class="l">{c}</td>' if i in left else f"<td>{c}</td>"
            for i, c in enumerate(row)
        )
        body.append(f"<tr{cls}>{tds}</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    core = (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>{cap}"
    )
    if bare:
        return core
    head_bits = ""
    if number:
        head_bits += f'<div class="tnum">{esc(number)}</div>'
    if title:
        head_bits += f'<div class="ttitle">{esc(title)}</div>'
    return f'<div class="tbl">{head_bits}{core}</div>'


def details(summary: str, inner: str) -> str:
    return f"<details><summary>{esc(summary)}</summary>{inner}</details>"


# --------------------------------------------------------------------------- #
# Chart primitives                                                              #
# --------------------------------------------------------------------------- #
def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / count
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 0.5:
        out.append(round(v, 10))
        v += step
    return out


def _svg(w: int, h: int, label: str, body: str) -> str:
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="display:block" '
        f'role="img" aria-label="{esc(label)}">{body}</svg>'
    )


def equity_drawdown(
    labels: list[str],
    cum_return_pct: list[float],
    *,
    aria: str,
    height: int = 340,
) -> str:
    """Cumulative return over a drawdown panel, both in percent.

    Everything here is scale-free on purpose: an equity axis in currency only
    describes the capital base that was typed into the backtest, so two runs at
    different sizing look different while being the same strategy.
    """
    W, H = 940, height
    ml, mr, mt = 74, 92, 12
    split, gap, mb = int((height - 60) * 0.66), 26, 40
    ph2 = H - mb - mt - split - gap
    pw = W - ml - mr
    equity = cum_return_pct
    n = len(equity)
    peak, dd = -1e18, []
    for v in equity:
        peak = max(peak, v)
        dd.append(v - peak)
    lo, hi = min(equity + [0.0]), max(equity + [0.0])
    pad = (hi - lo) * 0.10 or 1.0
    lo, hi = lo - pad, hi + pad
    ddlo = min(dd + [0.0])

    def sx(i: int) -> float:
        return ml + (i / max(n - 1, 1)) * pw

    def sy(v: float) -> float:
        return mt + split - (v - lo) / (hi - lo) * split

    def sy2(v: float) -> float:
        return mt + split + gap + (v / (ddlo or -1)) * ph2

    p = []
    for t in _nice_ticks(lo, hi, 4):
        if not lo <= t <= hi:
            continue
        y = sy(t)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" class="grid1"/>')
        p.append(f'<text x="{ml-11}" y="{y+4:.1f}" class="axis end">{t:+.0f}%</text>')
    for t in _nice_ticks(ddlo, 0.0, 3):
        if not ddlo <= t <= 0:
            continue
        y = sy2(t)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" class="grid1"/>')
        p.append(f'<text x="{ml-11}" y="{y+4:.1f}" class="axis end">{t:.1f}%</text>')

    area = " ".join(f"{sx(i):.1f},{sy2(v):.1f}" for i, v in enumerate(dd))
    p.append(
        f'<polygon points="{ml},{sy2(0):.1f} {area} {ml+pw},{sy2(0):.1f}" '
        f'fill="var(--s1)" fill-opacity="0.10"/>'
    )
    p.append(
        f'<polyline points="{area}" fill="none" stroke="var(--s1)" stroke-width="2" '
        f'stroke-linejoin="round"/>'
    )
    line = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(equity))
    p.append(
        f'<polyline points="{line}" fill="none" stroke="var(--s1)" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    p.append(
        f'<line x1="{ml}" y1="{sy(0.0):.1f}" x2="{ml+pw}" y2="{sy(0.0):.1f}" '
        f'stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>'
    )
    p.append(f'<circle cx="{sx(n-1):.1f}" cy="{sy(equity[-1]):.1f}" r="4.5" fill="var(--s1)" class="ring"/>')
    p.append(
        f'<text x="{sx(n-1)+11:.1f}" y="{sy(equity[-1])+4:.1f}" class="dlabel">'
        f"{equity[-1]:+.2f}%</text>"
    )
    trough = dd.index(min(dd))
    ty = sy2(dd[trough])
    p.append(f'<circle cx="{sx(trough):.1f}" cy="{ty:.1f}" r="4.5" fill="var(--s1)" class="ring"/>')
    # Sit the label above the trough marker: below it would land on the date row.
    p.append(
        f'<text x="{min(sx(trough)+11, ml+pw-42):.1f}" y="{ty-9:.1f}" '
        f'class="dlabel">{dd[trough]:.2f}%</text>'
    )
    step = max(1, n // 6)
    for i in range(0, n, step):
        p.append(f'<text x="{sx(i):.1f}" y="{H-14}" class="axis mid">{esc(labels[i])}</text>')
    p.append(f'<text x="{ml}" y="{mt-1}" class="axist">累積報酬（% of capital）</text>')
    p.append(f'<text x="{ml}" y="{mt+split+gap-6}" class="axist">回撤（% of capital）</text>')
    return _svg(W, H, aria, "".join(p))


def heatmap(
    rows: list[str],
    cols: list[str],
    panels: list[tuple[str, list[list[float | None]]]],
    *,
    aria: str,
    row_title: str,
    col_title: str,
    mark: tuple[int, int, int] | None = None,
) -> str:
    """Sequential heat panels sharing one scale. mark = (panel, row, col)."""
    vals = [v for _, g in panels for r in g for v in r if v is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    cw, ch, gap_x = 52, 30, 26
    ml, mt, mb = 66, 44, 34
    pw = len(cols) * cw
    W = ml + len(panels) * (pw + gap_x)
    H = mt + len(rows) * ch + mb
    p = []
    for pi, (ptitle, g) in enumerate(panels):
        x0 = ml + pi * (pw + gap_x)
        p.append(f'<text x="{x0}" y="{mt-24}" class="axist">{esc(ptitle)}</text>')
        for ci, c in enumerate(cols):
            p.append(
                f'<text x="{x0+ci*cw+cw/2:.0f}" y="{mt-8}" class="axis mid">{esc(c)}</text>'
            )
        for ri in range(len(rows)):
            for ci in range(len(cols)):
                v = g[ri][ci]
                x, y = x0 + ci * cw, mt + ri * ch
                if v is None:
                    p.append(
                        f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" '
                        f'fill="var(--grid)" opacity="0.5"/>'
                    )
                    continue
                t = 0.0 if hi == lo else (v - lo) / (hi - lo)
                idx = min(int(t * (HEAT_STEPS - 1) + 0.5), HEAT_STEPS - 1)
                # Fill and its ink are paired CSS variables so the ramp inverts
                # for dark mode without the label ever losing contrast against
                # its own cell.
                p.append(
                    f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" rx="2" '
                    f'fill="var(--h{idx})"><title>{esc(rows[ri])} / {esc(cols[ci])}: '
                    f"{v:.2f}</title></rect>"
                )
                p.append(
                    f'<text x="{x+cw/2:.0f}" y="{y+ch/2+4:.0f}" class="mid" '
                    f'style="font-size:10.5px;fill:var(--hi{idx});'
                    f'font-variant-numeric:tabular-nums">{v:.1f}</text>'
                )
                if mark == (pi, ri, ci):
                    p.append(
                        f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" rx="2" '
                        f'fill="none" stroke="var(--ink)" stroke-width="2.5"/>'
                    )
        if pi == 0:
            for ri, r in enumerate(rows):
                p.append(
                    f'<text x="{ml-11}" y="{mt+ri*ch+ch/2+4:.0f}" class="axis end">{esc(r)}</text>'
                )
    p.append(f'<text x="6" y="{mt-24}" class="axist">{esc(row_title)}</text>')
    p.append(f'<text x="{ml}" y="{H-10}" class="axis start">{esc(col_title)} &#8594;</text>')
    return _svg(W, H, aria, "".join(p))


def stacked_bars(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    *,
    aria: str,
    unit: str = "TWD",
    height: int = 250,
) -> str:
    W, H = 940, height
    # Left margin sized to the longest category label so a config name is never
    # clipped; right margin holds the total at the bar end.
    ml = max(112, int(max((len(c) for c in categories), default=0) * 6.6) + 18)
    mr, mt, mb = 118, 16, 40
    pw, ph = W - ml - mr, H - mt - mb
    totals = [sum(s[1][i] for s in series) for i in range(len(categories))]
    hi = max(totals) if totals else 1.0
    band = ph / max(len(categories), 1)
    bh = min(24, band - 14)
    p = []
    for t in _nice_ticks(0, hi, 4):
        x = ml + t / hi * pw
        p.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" class="grid1"/>')
        p.append(f'<text x="{x:.1f}" y="{H-14}" class="axis mid">{t:g}</text>')
    for i, cat in enumerate(categories):
        y = mt + i * band + (band - bh) / 2
        p.append(f'<text x="{ml-12}" y="{y+bh/2+4:.1f}" class="axis end">{esc(cat)}</text>')
        cursor = 0.0
        for si, (name, vals) in enumerate(series):
            v = vals[i]
            if v <= 0:
                continue
            x0 = ml + cursor / hi * pw
            w = v / hi * pw
            # 2px surface gap between touching segments, and only the final
            # segment gets the rounded data-end.
            last = si == len(series) - 1 or sum(s[1][i] for s in series[si + 1 :]) <= 0
            rx = 4 if last else 0
            p.append(
                f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(w-2,0.5):.1f}" height="{bh:.1f}" '
                f'rx="{rx}" fill="var(--s{si+1})"><title>{esc(cat)} — {esc(name)}: '
                f"{v:,.1f} {esc(unit)} ({v/totals[i]*100:.1f}%)</title></rect>"
            )
            cursor += v
        p.append(
            f'<text x="{ml + totals[i]/hi*pw + 11:.1f}" y="{y+bh/2+4:.1f}" class="dlabel">'
            f"{totals[i]:,.1f}</text>"
        )
    return _svg(W, H, aria, "".join(p))


def scatter(
    points: list[tuple[float, float, str, int]],
    *,
    aria: str,
    x_label: str,
    y_label: str,
    x_max: float | None = None,
    y_max: float | None = None,
    vline: tuple[float, str] | None = None,
    height: int = 330,
) -> str:
    """points = (x, y, tooltip, series index 0..2)."""
    W, H = 940, height
    ml, mr, mt, mb = 62, 24, 14, 46
    pw, ph = W - ml - mr, H - mt - mb
    xm = x_max or (max(p[0] for p in points) * 1.06 if points else 1)
    ym = y_max or (max(p[1] for p in points) * 1.10 if points else 1)

    def sx(v: float) -> float:
        return ml + min(v, xm) / xm * pw

    def sy(v: float) -> float:
        return mt + ph - min(v, ym) / ym * ph

    p = []
    for t in _nice_ticks(0, ym, 4):
        y = sy(t)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" class="grid1"/>')
        p.append(f'<text x="{ml-11}" y="{y+4:.1f}" class="axis end">{t:g}</text>')
    for t in _nice_ticks(0, xm, 6):
        x = sx(t)
        p.append(f'<text x="{x:.1f}" y="{H-22}" class="axis mid">{t:g}</text>')
    if vline:
        v, lab = vline
        p.append(
            f'<line x1="{sx(v):.1f}" y1="{mt}" x2="{sx(v):.1f}" y2="{mt+ph}" '
            f'stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<text x="{sx(v)+7:.1f}" y="{mt+13}" class="axis start">{esc(lab)}</text>'
        )
    for x, y, tip, si in points:
        p.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="var(--s{si+1})" '
            f'class="ring"><title>{esc(tip)}</title></circle>'
        )
    p.append(f'<text x="{ml+pw/2:.0f}" y="{H-4}" class="axist mid">{esc(x_label)}</text>')
    p.append(
        f'<text x="14" y="{mt+ph/2:.0f}" class="axist mid" '
        f'transform="rotate(-90 14 {mt+ph/2:.0f})">{esc(y_label)}</text>'
    )
    return _svg(W, H, aria, "".join(p))


def histogram(
    values: list[float],
    *,
    aria: str,
    x_label: str,
    bins: int = 18,
    series: int = 0,
    zero_line: bool = True,
    unit: str = "",
    height: int = 250,
    width: int = 940,
) -> str:
    W, H = width, height
    ml, mr, mt, mb = 56, 24, 14, 46
    pw, ph = W - ml - mr, H - mt - mb
    if not values:
        return _svg(W, H, aria, "")
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / width), bins - 1)] += 1
    cmax = max(counts) or 1

    def sx(v: float) -> float:
        return ml + (v - lo) / (hi - lo) * pw

    p = []
    for t in _nice_ticks(0, cmax, 4):
        y = mt + ph - t / cmax * ph
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" class="grid1"/>')
        p.append(f'<text x="{ml-11}" y="{y+4:.1f}" class="axis end">{t:g}</text>')
    bw = pw / bins
    for i, c in enumerate(counts):
        if not c:
            continue
        h = c / cmax * ph
        x = ml + i * bw
        p.append(
            f'<rect x="{x+1:.1f}" y="{mt+ph-h:.1f}" width="{max(bw-2,1):.1f}" '
            f'height="{h:.1f}" rx="3" fill="var(--s{series+1})">'
            f"<title>{lo+i*width:,.1f} to {lo+(i+1)*width:,.1f} {esc(unit)}: "
            f"{c} trades</title></rect>"
        )
    if zero_line and lo < 0 < hi:
        p.append(
            f'<line x1="{sx(0):.1f}" y1="{mt}" x2="{sx(0):.1f}" y2="{mt+ph}" '
            f'stroke="var(--ink2)" stroke-width="1"/>'
            f'<text x="{sx(0)+6:.1f}" y="{mt+13}" class="axis start">0</text>'
        )
    span = hi - lo
    digits = 0 if span >= 20 else (1 if span >= 2 else 2)
    for t in _nice_ticks(lo, hi, 6):
        if lo <= t <= hi:
            p.append(
                f'<text x="{sx(t):.1f}" y="{H-22}" class="axis mid">{t:,.{digits}f}</text>'
            )
    p.append(f'<text x="{ml+pw/2:.0f}" y="{H-4}" class="axist mid">{esc(x_label)}</text>')
    return _svg(W, H, aria, "".join(p))


def histogram_pair(
    left: tuple[list[float], str, bool],
    right: tuple[list[float], str, bool],
    *,
    aria: str,
    bins: int = 16,
    height: int = 250,
) -> str:
    """Two distributions side by side as one exhibit.

    They answer one question together -- what shape do the trades have -- so a
    single figure with two panels beats two figures the reader has to hold in
    mind at once. Each panel keeps its own x scale; the y axes are independent
    because the counts are of the same trades either way.
    """
    W = 940
    gap = 54
    pw = (W - gap) // 2
    out = [f'<div style="display:flex;gap:{gap}px;flex-wrap:wrap">']
    for values, x_label, zero in (left, right):
        svg = histogram(
            values, aria=f"{aria} — {x_label}", x_label=x_label, bins=bins,
            zero_line=zero, height=height, width=pw,
            series=0 if zero else 1,
        )
        out.append(f'<div style="flex:1 1 {pw//2}px;min-width:300px">{svg}</div>')
    out.append("</div>")
    return "".join(out)


def footer(text: str) -> str:
    return f"<footer>{text}</footer>"


def trade_anatomy(
    *,
    spread: list[float],
    mean: list[float],
    std: list[float],
    z_mid: list[float],
    z_short: list[float],
    z_long: list[float],
    labels: list[str],
    entry_z: float,
    exit_z: float,
    direction: str,
    events: dict[str, int],
    aria: str,
) -> str:
    """Two stacked panels over one real trade: the spread with its band, and
    the three z series the engine actually decides on.

    The reason this figure exists is that the spread is not one line. The engine
    scores each direction against the side of the book that order would have to
    cross, so there are three series -- the mid, and one displaced each way --
    and every threshold in the report is applied to the displaced pair, not the
    mid. That is very hard to convey in prose and immediate in a picture.

    events maps 'entry_signal' / 'entry_fill' / 'exit_signal' / 'exit_fill' to
    bar indices within the supplied window.
    """
    W = 940
    ml, mr, mt = 62, 118, 26
    hA, gap, hB, mb = 190, 46, 150, 40
    H = mt + hA + gap + hB + mb
    pw = W - ml - mr
    n = len(spread)

    band_hi = [m + entry_z * s for m, s in zip(mean, std)]
    band_lo = [m - entry_z * s for m, s in zip(mean, std)]
    lo_a = min(min(spread), min(band_lo))
    hi_a = max(max(spread), max(band_hi))
    pad = (hi_a - lo_a) * 0.12 or 1.0
    lo_a, hi_a = lo_a - pad, hi_a + pad
    zs = z_mid + z_short + z_long + [entry_z, -entry_z, exit_z, -exit_z]
    lo_b, hi_b = min(zs), max(zs)
    padb = (hi_b - lo_b) * 0.12 or 1.0
    lo_b, hi_b = lo_b - padb, hi_b + padb

    def sx(i: float) -> float:
        return ml + (i / max(n - 1, 1)) * pw

    def syA(v: float) -> float:
        return mt + hA - (v - lo_a) / (hi_a - lo_a) * hA

    def syB(v: float) -> float:
        return mt + hA + gap + hB - (v - lo_b) / (hi_b - lo_b) * hB

    p: list[str] = []

    def poly(vals, sy, cls, extra=""):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        p.append(f'<polyline points="{pts}" fill="none" class="{cls}" {extra}/>')

    # holding period wash, drawn first so every line sits on top of it
    a, b = events["entry_fill"], events["exit_fill"]
    p.append(
        f'<rect x="{sx(a):.1f}" y="{mt}" width="{sx(b)-sx(a):.1f}" '
        f'height="{hA}" fill="var(--accent)" opacity="0.05"/>'
        f'<rect x="{sx(a):.1f}" y="{mt+hA+gap}" width="{sx(b)-sx(a):.1f}" '
        f'height="{hB}" fill="var(--accent)" opacity="0.05"/>'
    )

    for t in _nice_ticks(lo_a, hi_a, 3):
        if lo_a <= t <= hi_a:
            y = syA(t)
            p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" class="grid1"/>')
            p.append(f'<text x="{ml-10}" y="{y+4:.1f}" class="axis end">{t:.2f}</text>')
    for t in _nice_ticks(lo_b, hi_b, 4):
        if lo_b <= t <= hi_b:
            y = syB(t)
            p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" class="grid1"/>')
            p.append(f'<text x="{ml-10}" y="{y+4:.1f}" class="axis end">{t:+.1f}</text>')

    # panel A: entry band, mean, spread
    band_pts = (
        " ".join(f"{sx(i):.1f},{syA(v):.1f}" for i, v in enumerate(band_hi))
        + " "
        + " ".join(f"{sx(i):.1f},{syA(v):.1f}" for i, v in reversed(list(enumerate(band_lo))))
    )
    p.append(f'<polygon points="{band_pts}" fill="var(--s3)" opacity="0.13"/>')
    poly(band_hi, syA, "", 'stroke="var(--s3)" stroke-width="1.5" stroke-dasharray="4 3"')
    poly(band_lo, syA, "", 'stroke="var(--s3)" stroke-width="1.5" stroke-dasharray="4 3"')
    poly(mean, syA, "", 'stroke="var(--s2)" stroke-width="1.5" stroke-dasharray="5 3"')
    for vals, txt in ((band_hi, "+entry"), (mean, "mean"), (band_lo, "−entry")):
        p.append(
            f'<text x="{ml+pw+7}" y="{syA(vals[-1])+4:.1f}" class="axis start">{esc(txt)}</text>'
        )
    poly(spread, syA, "", 'stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"')

    # panel B: the displaced pair with the gap between them washed in, then mid
    gap_pts = (
        " ".join(f"{sx(i):.1f},{syB(v):.1f}" for i, v in enumerate(z_long))
        + " "
        + " ".join(
            f"{sx(i):.1f},{syB(v):.1f}" for i, v in reversed(list(enumerate(z_short)))
        )
    )
    p.append(f'<polygon points="{gap_pts}" fill="var(--s1)" opacity="0.10"/>')
    for lab, series in (("z_long", z_long), ("z_short", z_short)):
        poly(series, syB, "", 'stroke="var(--s3)" stroke-width="1.5"')
    poly(z_mid, syB, "", 'stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"')

    for lvl, txt in ((entry_z, f"+entry {entry_z:g}"), (-entry_z, f"−entry {-entry_z:g}")):
        if lo_b <= lvl <= hi_b:
            y = syB(lvl)
            p.append(
                f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                f'stroke="var(--s2)" stroke-width="1" stroke-dasharray="4 3"/>'
                f'<text x="{ml+pw+7}" y="{y+4:.1f}" class="axis start">{esc(txt)}</text>'
            )
    for lvl, txt in ((exit_z, f"+exit {exit_z:g}"), (-exit_z, f"−exit {-exit_z:g}")):
        if lo_b <= lvl <= hi_b:
            y = syB(lvl)
            p.append(
                f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                f'stroke="var(--muted)" stroke-width="1" stroke-dasharray="2 3"/>'
                f'<text x="{ml+pw+7}" y="{y+4:.1f}" class="axis start">{esc(txt)}</text>'
            )

    # Event guides. Signal and fill are one bar apart, so a label on each
    # overprints; instead each pair gets one label and a shaded one-bar sliver
    # that shows the lag without needing two pieces of text.
    short_side = direction == "short_tsm_long_qff"
    entry_series = z_short if short_side else z_long
    exit_series = z_long if short_side else z_short
    pairs = [
        ("entry_signal", "entry_fill", "進場", entry_series),
        ("exit_signal", "exit_fill", "出場", exit_series),
    ]
    for sig_key, fill_key, lab, series in pairs:
        i, j = events[sig_key], events[fill_key]
        x0, x1 = sx(i), sx(j)
        p.append(
            f'<rect x="{x0:.1f}" y="{mt}" width="{max(x1-x0,2):.1f}" '
            f'height="{hA+gap+hB}" fill="var(--s2)" opacity="0.16"/>'
        )
        for x in (x0, x1):
            p.append(
                f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+hA+gap+hB}" '
                f'stroke="var(--ink2)" stroke-width="1" stroke-dasharray="2 4" opacity="0.75"/>'
            )
        p.append(
            f'<text x="{(x0+x1)/2:.1f}" y="{mt-10}" class="axis mid" '
            f'style="fill:var(--ink);font-weight:640">{esc(lab)}</text>'
        )
        for idx in (i, j):
            p.append(
                f'<circle cx="{sx(idx):.1f}" cy="{syA(spread[idx]):.1f}" r="4" '
                f'fill="var(--s1)" class="ring"/>'
                f'<circle cx="{sx(idx):.1f}" cy="{syB(series[idx]):.1f}" r="4" '
                f'fill="var(--s1)" class="ring"/>'
            )

    p.append(f'<text x="{ml}" y="{mt+12}" class="axist">Spread（% 尺度）</text>')
    p.append(f'<text x="{ml}" y="{mt+hA+gap-8}" class="axist">z-score</text>')
    step = max(1, n // 5)
    for i in range(0, n, step):
        p.append(f'<text x="{sx(i):.1f}" y="{H-14}" class="axis mid">{esc(labels[i])}</text>')
    return _svg(W, H, aria, "".join(p))
