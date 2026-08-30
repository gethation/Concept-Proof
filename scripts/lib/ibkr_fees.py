"""What IBKR actually charges for the US leg.

The backtest priced this leg at a flat 2.5 bps because a bps model is all a
single number can express. IBKR charges per SHARE, with a per-order minimum and
a cap, plus regulatory fees levied on the SELLER only.

WHICH WAY THE FLAT MODEL ERRS DEPENDS ON PRICE. Commission is per share and does
not move with price; a bps charge scales with it. The two cross at roughly
$23.2 an ADR: above that the flat 2.5 bps over-charges, below it the flat model
charges LESS than IBKR does. "2.5 bps is conservative" is therefore not a
property of the model, it is a property of the price at the time it was checked.

Measured against four real UMC fills in Project Lux's store, 2026-08-10 to
08-21, with UMC at $18.30-$19.49 -- below the crossover, so the flat model
under-charges:

    buy  396 @ 19.2095    $1.98    2.60 bps    flat model 2.50 bps
    sell 396 @ 19.4850    $2.26    2.93 bps    flat model 2.50 bps
    sell 394 @ 18.4639    $2.24    3.08 bps    flat model 2.50 bps
    buy  404 @ 18.2995    $2.02    2.73 bps    flat model 2.50 bps

The buys are pure commission and reproduce to the cent at $0.005 a share. The
sells carry SEC Section 31 and FINRA TAF on top, which is why a model that
averaged the two sides would understate every exit of a long and overstate every
entry of one.

RATES ARE NOT CONSTANTS OF NATURE. The SEC fee rate is reset annually and the
FINRA TAF changes by rule filing. Both are dated below and both must be
confirmed against IBKR's current schedule before this model is used to justify a
live decision.

TWO THINGS THIS DELIBERATELY DOES NOT MODEL, both of which make it optimistic:

  *Borrow.* A short US leg pays a stock-loan fee for every day it is held. That
  makes the strategy directionally asymmetric -- long US / short TW borrows
  nothing, short US / long TW pays daily -- and roughly half the backtest's
  trades are the second kind. The engine has no per-day cost dimension at all,
  so adding borrow here would charge it at the wrong time.

  *Whole ADRs.* The engine sizes in fractional shares (`leg_notional / price`),
  while a real order is an integer number of ADRs. Rounding here would leave the
  fee disagreeing with the PnL that the same fractional position produced.
  Rounding is a sizing change, not a fee change, and belongs with sizing.

Kept deliberately parallel to Project Lux's `integrations/ibkr/fees.py` so the
two can be diffed. The constants below are that file's, unchanged.
"""
from __future__ import annotations

# IBKR US equities, Fixed tier. Recorded 2026-07-25; confirm before live use.
COMMISSION_PER_SHARE_USD = 0.005
COMMISSION_MINIMUM_USD = 1.00
# The cap stops the per-share charge exceeding the trade itself on very
# low-priced stock. At $1 a share the per-share rate alone is 0.5%.
COMMISSION_MAX_FRACTION_OF_VALUE = 0.01

# Regulatory fees. SELL SIDE ONLY -- both are levied on the seller.
# SEC Section 31 fee, per dollar of proceeds. Reset annually by the SEC.
SEC_FEE_PER_USD = 0.0000278
# FINRA Trading Activity Fee, per share sold, capped per trade.
FINRA_TAF_PER_SHARE_USD = 0.000166
FINRA_TAF_MAX_USD = 8.30

RATES_AS_OF = "2026-07-25"

BUY = "buy"
SELL = "sell"


def commission_usd(adrs: float, price_usd: float) -> float:
    """Per-share commission, floored at the order minimum and capped at 1%."""
    per_share = abs(adrs) * COMMISSION_PER_SHARE_USD
    charged = max(per_share, COMMISSION_MINIMUM_USD)
    value = abs(adrs) * float(price_usd)
    if value > 0:
        charged = min(charged, value * COMMISSION_MAX_FRACTION_OF_VALUE)
    return charged


def trade_cost_usd(*, adrs: float, price_usd: float, side: str) -> dict[str, float]:
    """Commission plus regulatory fees for one side, in USD.

    Regulatory fees are charged to the SELLER only, so a buy and a sell of the
    same size are not the same cost.
    """
    if side not in (BUY, SELL):
        raise RuntimeError(f"side must be {BUY!r} or {SELL!r}, got {side!r}")
    adrs = abs(float(adrs))
    commission = commission_usd(adrs, price_usd)
    if side == SELL:
        proceeds = adrs * float(price_usd)
        sec_fee = proceeds * SEC_FEE_PER_USD
        taf = min(adrs * FINRA_TAF_PER_SHARE_USD, FINRA_TAF_MAX_USD)
    else:
        sec_fee = 0.0
        taf = 0.0
    return {
        "commission_usd": commission,
        "sec_fee_usd": sec_fee,
        "finra_taf_usd": taf,
        "total_usd": commission + sec_fee + taf,
    }


def us_leg_fee_twd(
    *,
    shares: float,
    price_twd_per_share: float,
    side: str,
    usdtwd: float,
    share_ratio: float,
) -> float:
    """The engine's units in, TWD out.

    The engine holds the US leg in UNDERLYING SHARES priced in TWD, because that
    is the form the spread is defined on. IBKR trades and charges per ADR in
    USD. Both conversions happen here, in one place, so neither the engine nor
    the fee model has to carry the other's units.
    """
    if share_ratio <= 0:
        raise RuntimeError("share_ratio must be positive")
    if not (usdtwd > 0):
        raise RuntimeError(
            "the IBKR fee model needs a positive USDTWD rate to convert a "
            f"per-share USD commission into TWD, got {usdtwd!r}"
        )
    adrs = abs(float(shares)) / float(share_ratio)
    price_usd = float(price_twd_per_share) * float(share_ratio) / float(usdtwd)
    return trade_cost_usd(adrs=adrs, price_usd=price_usd, side=side)["total_usd"] * float(usdtwd)


__all__ = [
    "BUY",
    "COMMISSION_MAX_FRACTION_OF_VALUE",
    "COMMISSION_MINIMUM_USD",
    "COMMISSION_PER_SHARE_USD",
    "FINRA_TAF_MAX_USD",
    "FINRA_TAF_PER_SHARE_USD",
    "RATES_AS_OF",
    "SEC_FEE_PER_USD",
    "SELL",
    "commission_usd",
    "trade_cost_usd",
    "us_leg_fee_twd",
]
