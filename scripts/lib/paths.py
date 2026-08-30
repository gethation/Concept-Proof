"""Where everything lives on disk.

One module owns the layout so a path appears once instead of once per script.
The layout separates files by *lifetime*, which is what the old flat
data/processed could not express:

    data/raw/      exchange downloads, byte-for-byte as received
    data/bars/     the canonical bar series -- one file per symbol+interval,
                   append-merged forever, expensive or impossible to re-fetch
    data/features/ spread and z-score, a pure function of bars + parameters
    data/runs/     one directory per backtest or grid, each self-describing
    reports/       generated HTML, at the repo root because it is a deliverable

Bar filenames carry no date and no vintage suffix. A file like
``binance_tsmusdtp_1m_taipei_0812.csv`` reads as "the 08-12 capture", which is
how the project ended up with several vintages of one series and a report bound
to whichever one a command line happened to name.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
BARS = DATA / "bars"
FEATURES = DATA / "features"
RUNS = DATA / "runs"
REPORTS = REPO_ROOT / "reports"

# --- canonical bar series -------------------------------------------------

TAIFEX = BARS / "taifex"
NYSE = BARS / "nyse"
BINANCE = BARS / "binance"
BITOPRO = BARS / "bitopro"
OKX = BARS / "okx"
FXIDC = BARS / "fxidc"
IB = BARS / "ib"

QFF1_1M = TAIFEX / "qff1_1m.csv"
QFF1_15M = TAIFEX / "qff1_15m.csv"
CCF1_1M = TAIFEX / "ccf1_1m.csv"
CCF1_5M = TAIFEX / "ccf1_5m.csv"
CCF1_15M = TAIFEX / "ccf1_15m.csv"
CCF1_1H = TAIFEX / "ccf1_1h.csv"

UMC_1M = NYSE / "umc_1m.csv"
UMC_5M = NYSE / "umc_5m.csv"
UMC_15M = NYSE / "umc_15m.csv"
UMC_1H = NYSE / "umc_1h.csv"

TSMUSDTP_1M = BINANCE / "tsmusdtp_1m.csv"
USDTTWD_1M = BITOPRO / "usdttwd_1m.csv"

OKX_TSMUSDTP_1M = OKX / "tsmusdtp_1m.csv"
OKX_TSMUSDTP_15M = OKX / "tsmusdtp_15m.csv"
USDTTWD_15M = BITOPRO / "usdttwd_15m.csv"
# Re-anchored onto the TAIFEX QFF 15m grid (night bars at :25/:40/:55/:10).
# The exchange files above are :00/:15/:30/:45 and cannot serve a QFF index.
OKX_TSMUSDTP_15M_QFFGRID = OKX / "tsmusdtp_15m_qffgrid.csv"
USDTTWD_15M_QFFGRID = BITOPRO / "usdttwd_15m_qffgrid.csv"

# FX_IDC splice inputs, finest interval first (see lib.fx.build_fx_series).
FX_IDC_SPLICE: list[tuple[int, Path]] = [
    (5, FXIDC / "usdtwd_5m.csv"),
    (15, FXIDC / "usdtwd_15m.csv"),
    (60, FXIDC / "usdtwd_1h.csv"),
]


def feature(pair: str, name: str) -> Path:
    """A derived series for one pair, e.g. feature('ccf_umc', 'spread_1m')."""
    return FEATURES / pair / f"{name}.csv"


def run_dir(tag: str) -> Path:
    """The directory holding one backtest or grid run's outputs."""
    return RUNS / tag


def report(name: str) -> Path:
    """A generated HTML report at the repo root."""
    return REPORTS / f"{name}.html"
