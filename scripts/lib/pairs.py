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
        us_leg=paths.TSMUSDTP_1M,
        fx_leg=paths.USDTTWD_1M,
        default_out="spread_1m",
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
