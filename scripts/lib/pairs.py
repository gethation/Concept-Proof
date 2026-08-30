"""The pair registry: what each spread configuration is made of.

Four spread scripts used to encode this as module-level constants, one file per
pair-and-interval. The differences between them were never the *maths* -- the
fair-value and spread formulas are identical everywhere -- but the market
structure: which leg defines the index, whether the other leg is required to
cover it or aligned as-of, and where the FX rate comes from.

So there are two alignment strategies and a table of configurations, rather than
one file per configuration.

Some fields exist only to preserve the behaviour of the script a configuration
replaced; those are marked. `fx_session_filter` is the one to look at: the 1m
CCF/UMC script defined a hard FX-staleness cap and then never applied it, so it
accepts sessions the 5m/15m script drops. That is preserved here rather than
silently fixed, because changing it changes published spread files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lib import paths

# TAIFEX single-stock futures are quoted per share; the US ADR represents this
# many underlying shares. Both live pairs happen to use 5.
ADR_SHARE_RATIO = 5.0


@dataclass(frozen=True)
class TaifexGridSpec:
    """The index is TAIFEX's own session clock; every other leg must cover it.

    Used where the non-TAIFEX legs trade continuously (Binance perp, a crypto FX
    pair), so a missing minute means broken data rather than a closed market --
    hence `assert_external_complete` rather than an as-of alignment.
    """

    pair: str
    interval_minutes: int
    tw_leg: Path
    us_leg: Path
    fx_leg: Path
    share_ratio: float = ADR_SHARE_RATIO
    default_out: str = "spread_1m"


@dataclass(frozen=True)
class UsRthSpec:
    """The index is the US leg's own RTH bars; the TAIFEX leg is aligned as-of.

    Used where both legs are exchange-traded equities: NYSE RTH is the only
    window in which the pair can actually be traded together, and TAIFEX trades
    far longer, so TAIFEX is the leg that bends onto the grid.
    """

    pair: str
    interval_minutes: int
    tw_leg: Path
    us_leg: Path
    fx_splice: list[tuple[int, Path]] = field(
        default_factory=lambda: list(paths.FX_IDC_SPLICE)
    )
    share_ratio: float = ADR_SHARE_RATIO

    # Sessions with fewer native TAIFEX bars inside the US window are dropped
    # entirely: 30 native minutes out of ~390, or a single bar on a coarse grid.
    min_tw_bars_per_session: int = 1

    # Drop whole sessions whose FX staleness exceeds the hard cap.
    # False on the 1m configuration ONLY because the script it replaces defined
    # the cap and never used it. Turning it on changes the 1m spread file.
    fx_session_filter: bool = True

    # Whether FX coverage narrows the date range, or only the two equity legs
    # do. False on 1m for the same reason: the script it replaces ignored FX
    # when picking start/end.
    range_includes_fx: bool = True

    # TAIFEX night trade is thin, so a stale close warns; a wildly stale one
    # means the session filter let a non-trading day through, and raises.
    tw_staleness_warn_minutes: float = 0.0  # 0 -> 4 x interval
    tw_staleness_max_minutes: float = 24 * 60.0

    weekend_policy: str = "flat"

    # The 5m/15m script asserted its own mask columns; the 1m one did not.
    validate_masks: bool = True

    # 1m writes a synthetic OHLCV FX file for the backtest's open-fill lookup;
    # the coarser grids write an as-of FX frame with a staleness column instead.
    fx_output: str = "asof"  # "asof" | "ohlcv"

    # 15m only: TAIFEX night bars start ~10min after the UMC grid point, so an
    # honest fill price needs the next TAIFEX bar's open, not the as-of close.
    write_delayed_open: bool = False

    default_out: str = "spread"

    @property
    def staleness_warn(self) -> float:
        return self.tw_staleness_warn_minutes or 4.0 * self.interval_minutes


SPECS: dict[tuple[str, str], TaifexGridSpec | UsRthSpec] = {
    ("qff_tsm", "1m"): TaifexGridSpec(
        pair="qff_tsm",
        interval_minutes=1,
        tw_leg=paths.QFF1_1M,
        # OKX, not Binance. The Binance perp archive starts 2026-07-07 and the
        # OKX one 2026-04-01, and the TAIFEX leg now reaches 2025-12-31, so the
        # US leg had become the binding constraint on a pair whose Taiwan side
        # had just been extended by six months. This also brings the 1m and 15m
        # configurations onto the same venue, which they were not before.
        #
        # It is a different book, not a relabelled one: executable_displacement
        # was measured against Binance, so the cost side of any run using this
        # leg is inherited rather than observed until it is re-measured on OKX.
        us_leg=paths.OKX_TSMUSDTP_1M,
        fx_leg=paths.USDTTWD_1M,
        default_out="spread_1m",
    ),
    ("qff_tsm", "15m"): TaifexGridSpec(
        pair="qff_tsm",
        interval_minutes=15,
        tw_leg=paths.QFF1_15M,
        # OKX, not the Binance perp the 1m configuration uses. This grid exists
        # to reach BACK, and its depth comes from tvdatafeed's QFF1! series --
        # TAIFEX time-and-sales publishes 30 trading days, so the 1m pipeline
        # cannot go before mid-July however it is run. The US leg follows the
        # source the original 15m study used so the two are comparable; a
        # Binance-legged 15m series would be a third thing again.
        # The QFF-grid copies, not the plain 15m files. build_taifex_session_index
        # anchors night bars at :25/:40/:55/:10 while the exchange files are
        # :00/:15/:30/:45, so the plain files miss every night stamp and the
        # build dies on the first one. These are the series the shipped
        # spread_15m.csv was actually built from.
        us_leg=paths.OKX_TSMUSDTP_15M_QFFGRID,
        fx_leg=paths.USDTTWD_15M_QFFGRID,
        default_out="spread_15m",
    ),
    ("ccf_umc", "1m"): UsRthSpec(
        pair="ccf_umc",
        interval_minutes=1,
        tw_leg=paths.CCF1_1M,
        us_leg=paths.UMC_1M,
        min_tw_bars_per_session=30,
        fx_session_filter=False,
        range_includes_fx=False,
        tw_staleness_warn_minutes=15.0,
        tw_staleness_max_minutes=240.0,
        validate_masks=False,
        fx_output="ohlcv",
        default_out="spread_1m",
    ),
    ("ccf_umc", "5m"): UsRthSpec(
        pair="ccf_umc",
        interval_minutes=5,
        tw_leg=paths.CCF1_5M,
        us_leg=paths.UMC_5M,
        default_out="spread_5m",
    ),
    # Hourly exists for one reason: TradingView's anonymous endpoint serves a
    # fixed ~5-6k bars whatever the interval, so the calendar span you get is
    # bars x interval. CCF1! reaches 18 days at 1m and 602 at 1h, and TAIFEX's
    # own time-and-sales archive is 30 trading days deep, so an hourly series
    # is the only route to more than a year of this pair that exists at all.
    #
    # Every flag below mirrors the 1m entry, deliberately: this configuration's
    # first job is a controlled comparison against 1m over their shared period,
    # and any second difference would confound it. Only the two staleness
    # bounds move, because the TAIFEX leg is aligned as-of and an hourly grid
    # makes an hour of staleness structural rather than a warning sign.
    ("ccf_umc", "1h"): UsRthSpec(
        pair="ccf_umc",
        interval_minutes=60,
        tw_leg=paths.CCF1_1H,
        us_leg=paths.UMC_1H,
        # Every hour of the US session must have its own native CCF bar. Not the
        # 1m entry's 7.7% equivalent, because the two thresholds guard different
        # failures. At 1m a normal session prints hundreds of native bars and 30
        # merely excludes the pathological. At 1h seven IS the maximum, and one
        # missing hour is not a thin patch -- it means CCF's night session had
        # not opened, so the as-of close reaches back to the 13:25 day close and
        # the spread is built on a price eight hours old. Measured: that breaches
        # the 240-minute staleness cap outright, at 525 minutes.
        #
        # THIS FILTER SELECTS, AND THE SELECTION IS NOT RANDOM. What survives is
        # the subset of days CCF happened to trade actively all night, which is a
        # biased sample and must not be read as "the same strategy, more data".
        #
        # 6, not 7. The window runs [first US bar, last US bar + 60min), and CCF
        # anchors its night hours at :25. Under US standard time that window is
        # [22:30, 05:30), which can hold at most six :25 bars (23:25..04:25) --
        # 22:25 starts before the window and no 05:25 bar exists. A threshold of
        # 7 was therefore unreachable every winter: all 129 standard-time
        # sessions on file scored 0%, against 53.7% of the 283 DST sessions, so
        # the series that exists to reach back a year was silently summer-only
        # and the "33% of 2025 complete" figure above was largely this artifact.
        min_tw_bars_per_session=6,
        fx_session_filter=False,
        range_includes_fx=False,
        tw_staleness_warn_minutes=60.0,
        tw_staleness_max_minutes=240.0,
        validate_masks=False,
        fx_output="ohlcv",
        default_out="spread_1h",
    ),
    ("ccf_umc", "15m"): UsRthSpec(
        pair="ccf_umc",
        interval_minutes=15,
        tw_leg=paths.CCF1_15M,
        us_leg=paths.UMC_15M,
        write_delayed_open=True,
        default_out="spread_15m",
    ),
}

PAIRS = sorted({pair for pair, _ in SPECS})
INTERVALS = sorted({interval for _, interval in SPECS})


def get_spec(pair: str, interval: str) -> TaifexGridSpec | UsRthSpec:
    try:
        return SPECS[(pair, interval)]
    except KeyError:
        available = ", ".join(f"{p}/{i}" for p, i in sorted(SPECS))
        raise SystemExit(
            f"No spread configuration for {pair}/{interval}. Available: {available}"
        ) from None
