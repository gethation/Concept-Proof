"""Scale-in (tranche) variant of the pair backtest.

The single-entry engine commits the whole leg notional at the first bar past
entry_z, which is also the *worst* place to be full size when the spread keeps
running instead of reverting -- the one-sided problem. This engine splits the
leg budget into n equal tranches: tranche 1 enters exactly like the single
engine, and each further tranche is added only after the spread has moved
another fixed distance *against* the position.

Two design decisions carry the experiment:

**The add spacing is frozen at entry, not measured in live z.** A one-sided run
drags the rolling mean with it and inflates the rolling std, so "one more z"
gets further away exactly when the spread is running -- the same goalpost drift
frozen_mean_exit and drift_bail exist to defeat. The knob is therefore
``add_spacing_k``, in multiples of the ENTRY-TIME rolling spread std: tranche
m fills once the adverse excursion from tranche 1's fill exceeds
``(m-1) x add_spacing_k x std_ref``. Scale-free (std_ref is in spread units,
spread units are percent of leg notional) and immune to drift.

**The spacing is a spread DIFFERENCE, so executable displacement cancels.**
The engine refuses frozen_mean_exit and drift_bail under a displacement
because they read raw spread LEVELS against undisplaced references. The add
rule compares two spreads on the same executable side of the book, and the
displacement is a constant in spread units, so it drops out of the
difference -- the rule is identical at mid and at the executable price. Entry
and exit thresholds keep the displaced z treatment, and every tranche fill
pays its own crossing cost, exactly as in the single engine.

Exit modes:

    basket      all tranches close together on the single engine's z-score
                exit rule (plus the same forced closes). The pure "does
                averaging the entry help" test.
    breakeven   once >= 2 tranches are on, additionally close everything when
                the spread has reverted to the notional-weighted average entry
                spread by more than the round-trip cost floor plus
                ``be_offset_k x std_ref``. Takes the escape the averaging
                bought instead of holding out for the full reversion. The
                z-score exit stays active as a backstop; single-tranche
                baskets never see the breakeven rule.

The cost floor (``be_cost_floor``) is in spread units and belongs in the rule
because "breakeven" at mid is a loss in reality: each tranche's round trip
crosses the book twice (2 x displacement) and pays commissions and tax. With
be_offset_k = 0 the rule closes at actual net breakeven, not apparent.

Exits close the whole basket as ONE order, so exit costs are computed on the
aggregate position and allocated to tranches pro rata by notional -- charging
the IBKR per-order minimum once per tranche would overstate the cost of the
very mechanism under test.

Deliberately unsupported (raise, not ignore): qff_lots, frozen_mean_exit,
drift_bail_c, persistence_k > 1, and the entry gates. They are orthogonal to
the question and each would need its own reconciliation with multi-tranche
state; the experiment keeps the surface at exactly the headline configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import (  # noqa: E402
    BacktestParams,
    FRIDAY_SESSION_END,
    LONG_TSM_SHORT_QFF,
    SHORT_TSM_LONG_QFF,
    TIME_STOP,
    add_trading_masks,
    compute_next_indices,
    direction_sign,
    displacement_at,
    entry_direction,
    executable_zscores,
    fill_costs,
    find_spread_stat_columns,
    format_timestamp,
    format_timestamp_column,
    minutes_between,
    require_spread_std_for_displacement,
    should_exit,
    size_position_for_direction,
    us_side,
    validate_params,
)
from lib import paths  # noqa: E402


def default_be_cost_floor(base: BacktestParams) -> float:
    """Round-trip cost of one tranche, in spread units.

    Two crossings plus the commissions and tax the crossings do not include.
    The old default was 2 x displacement alone, so a basket closing exactly at
    the trigger recovered the book gap and nothing else -- roughly 650 TWD of
    fees per 2-contract CCF basket, about 11 bps of leg notional, booked as a
    loss on every exit the rule called "breakeven". The docstring above already
    said commissions and tax belonged here; now they are.

    Priced on the params' own nominal leg notional, so the floor is one number
    for the run rather than something that moves with each basket's fill.
    """
    notional = base.leg_notional_twd
    if notional <= 0:
        return 2.0 * base.executable_displacement
    contracts = max(1.0, round(notional / (base.qff_contract_multiplier * 100.0)))
    # Futures leg: flat fee per contract per side, plus the exchange tax.
    futures = 2.0 * contracts * base.qff_fee_per_contract_twd
    tax = 2.0 * notional * base.qff_tax_rate
    # US leg: the bps model both ways is the right order of magnitude for the
    # floor even under the per-share model, which crosses it near $23/ADR.
    us_leg = 2.0 * notional * base.tsm_fee_bps / 10_000.0
    fee_spread_units = (futures + tax + us_leg) / notional * 100.0
    return 2.0 * base.executable_displacement + fee_spread_units


BREAKEVEN_EXIT = "breakeven_exit"
ZSCORE_EXIT = "zscore_exit"
END_OF_DATA = "end_of_data"

EXIT_MODES = ("basket", "breakeven")


@dataclass(frozen=True)
class ScaleInParams:
    base: BacktestParams  # leg_notional_twd is the TOTAL leg budget
    n_tranches: int = 1
    add_spacing_k: float = 1.0  # entry-time std multiples between adds
    exit_mode: str = "basket"
    be_offset_k: float = 0.0  # margin beyond the cost floor, entry-time stds
    be_cost_floor: float = 0.0  # round-trip cost in spread units; see docstring


@dataclass
class ScaleInResult:
    equity: pd.DataFrame
    tranches: pd.DataFrame
    baskets: pd.DataFrame
    summary: dict[str, Any]


def validate_scale_in_params(params: ScaleInParams) -> None:
    validate_params(params.base)
    if params.n_tranches < 1:
        raise RuntimeError("n_tranches must be at least 1")
    if params.n_tranches > 1 and params.add_spacing_k <= 0:
        raise RuntimeError("add_spacing_k must be positive when n_tranches > 1")
    if params.exit_mode not in EXIT_MODES:
        raise RuntimeError(f"exit_mode must be one of {EXIT_MODES}")
    if params.be_offset_k < 0 or params.be_cost_floor < 0:
        raise RuntimeError("be_offset_k and be_cost_floor must be non-negative")
    unsupported = {
        "qff_lots": params.base.qff_lots != 0,
        "frozen_mean_exit": params.base.frozen_mean_exit,
        "drift_bail_c": params.base.drift_bail_c > 0,
        "persistence_k": params.base.persistence_k > 1,
        "max_entry_vol_ratio": params.base.max_entry_vol_ratio > 0,
        "max_entry_adr_share": params.base.max_entry_adr_share > 0,
        "max_entry_qff_vol_surprise": params.base.max_entry_qff_vol_surprise > 0,
    }
    bad = [name for name, is_set in unsupported.items() if is_set]
    if bad:
        raise RuntimeError(
            f"scale_in does not support these base params (unset them): {bad}"
        )


def tranche_params(params: ScaleInParams) -> BacktestParams:
    """Sizing params for ONE tranche: the total budget split n ways."""
    return replace(
        params.base,
        leg_notional_twd=params.base.leg_notional_twd / params.n_tranches,
    )


def run_scale_in(frame: pd.DataFrame, params: ScaleInParams) -> ScaleInResult:
    validate_scale_in_params(params)
    data = add_trading_masks(frame)
    n_rows = len(data)
    per_tranche = tranche_params(params)

    timestamps = pd.DatetimeIndex(data["timestamp"])
    qff = data["qff_close_filled"].to_numpy(dtype=float)
    tsm = data["tsm_twd_fair"].to_numpy(dtype=float)
    qff_entry = (
        data["qff_entry_open_filled"].to_numpy(dtype=float)
        if "qff_entry_open_filled" in data.columns
        else qff
    )
    tsm_entry = (
        data["tsm_twd_fair_open"].to_numpy(dtype=float)
        if "tsm_twd_fair_open" in data.columns
        else tsm
    )
    usdtwd = (
        data["usdttwd_close"].to_numpy(dtype=float)
        if "usdttwd_close" in data.columns
        else np.full(n_rows, np.nan)
    )
    usdtwd_entry = (
        data["usdttwd_open"].to_numpy(dtype=float)
        if "usdttwd_open" in data.columns
        else usdtwd
    )
    zscore = data["spread_zscore"].to_numpy(dtype=float)
    zvalid = data["zscore_valid"].to_numpy(dtype=bool)
    mean_col, std_col = find_spread_stat_columns(list(data.columns))
    if "spread" not in data.columns:
        raise RuntimeError("scale_in needs the raw spread column for add spacing")
    spread_level = data["spread"].to_numpy(dtype=float)
    if std_col is None:
        raise RuntimeError(
            "scale_in needs a rolling spread std column (spread_std*) to freeze "
            "the add spacing at entry"
        )
    # No all-ones fallback. Synthesising a std would reinterpret displacement
    # from spread units into raw z and rescale the add spacing at the same
    # time, both silently.
    spread_std = data[std_col].to_numpy(dtype=float)
    entry_allowed = data["entry_allowed"].to_numpy(dtype=bool)
    close_allowed = data["close_allowed"].to_numpy(dtype=bool)
    friday_session_end = data["friday_session_end_force_close"].to_numpy(dtype=bool)

    # The per-bar displacement, derived exactly as engine.run_backtest derives
    # it: the frame's column when it carries one (a tick-regime schedule), then
    # scaled by ref/price. Feeding the raw scalar here while fill_costs applied
    # the scaled value split the two halves of one correction -- thresholds
    # shifted by one number, crossings charged at another.
    base_disp = (
        data["executable_displacement"].to_numpy(dtype=float)
        if "executable_displacement" in data.columns
        else np.full(n_rows, params.base.executable_displacement)
    )
    bar_disp = displacement_at(qff, params.base, base_disp)
    require_spread_std_for_displacement(
        std_col, params.base, bool(np.any(bar_disp > 0.0))
    )
    z_short, z_long = executable_zscores(zscore, spread_std, zvalid, bar_disp)
    entry_observation = entry_allowed & zvalid
    close_observation = close_allowed & zvalid
    next_entry_fill = compute_next_indices(entry_allowed)
    next_close_fill = compute_next_indices(close_allowed)

    # --- basket state ---------------------------------------------------
    direction: str | None = None
    tranches: list[dict[str, Any]] = []  # filled tranches of the open basket
    ref_spread = np.nan  # spread at tranche 1's fill bar
    std_ref = np.nan  # rolling std at tranche 1's signal bar
    basket_entry_time: pd.Timestamp | None = None
    basket_mae_spread = 0.0  # worst adverse excursion, spread units
    basket_id = 0
    # one pending action at a time: ("entry"|"add"|"exit", signal_idx, fill_idx)
    pending_kind: str | None = None
    pending_signal_idx = -1
    pending_fill_idx = -1
    pending_exit_reason = ""

    realized_pnl = 0.0
    tranche_rows: list[dict[str, Any]] = []
    basket_rows: list[dict[str, Any]] = []

    tranche_count_out = np.zeros(n_rows, dtype=np.int64)
    qff_contracts_out = np.zeros(n_rows, dtype=np.int64)
    gross_notional_out = np.zeros(n_rows)
    realized_out = np.zeros(n_rows)
    unrealized_out = np.zeros(n_rows)
    equity_out = np.zeros(n_rows)

    def clear_pending() -> None:
        nonlocal pending_kind, pending_signal_idx, pending_fill_idx
        nonlocal pending_exit_reason
        pending_kind = None
        pending_signal_idx = -1
        pending_fill_idx = -1
        pending_exit_reason = ""

    def vwa_entry_spread() -> float:
        weights = np.array([t["actual_leg_notional_twd"] for t in tranches])
        spreads = np.array([t["entry_spread"] for t in tranches])
        return float((weights * spreads).sum() / weights.sum())

    def fill_tranche(index: int, signal_idx: int) -> bool:
        """Fill tranche len(tranches)+1 at this bar. False if sizing rounds
        to zero contracts (the tranche budget cannot buy one contract)."""
        nonlocal ref_spread, std_ref, basket_entry_time, basket_mae_spread
        nonlocal realized_pnl
        assert direction is not None
        sizing = size_position_for_direction(
            direction, tsm_entry[index], qff_entry[index], per_tranche
        )
        if sizing is None:
            return False
        entry_costs = fill_costs(
            tsm_units=sizing.tsm_units,
            tsm_price=tsm_entry[index],
            qff_contracts=sizing.qff_contracts,
            qff_price=qff_entry[index],
            params=per_tranche,
            tsm_side=us_side(sizing.tsm_units),
            usdtwd=usdtwd_entry[index],
            displacement=bar_disp[index],
        )
        realized_pnl -= entry_costs["total_fee_twd"]
        m = len(tranches) + 1
        if m == 1:
            ref_spread = spread_level[index]
            std_ref = spread_std[signal_idx]
            basket_entry_time = timestamps[index]
            basket_mae_spread = 0.0
        tranches.append(
            {
                "basket_id": basket_id,
                "tranche": m,
                "direction": direction,
                "entry_signal_idx": signal_idx,
                "entry_signal_time": timestamps[signal_idx],
                "entry_signal_zscore": zscore[signal_idx],
                "entry_idx": index,
                "entry_time": timestamps[index],
                "entry_spread": spread_level[index],
                # The excursion the add RULE saw (signal bar) and the one the
                # fill actually got, both in entry-time stds. The signal one is
                # what validation checks against the spacing; the fill one is
                # informative -- one bar later the spread may sit anywhere.
                "signal_excursion_k": (
                    0.0
                    if m == 1
                    else float(
                        direction_sign(direction)
                        * (spread_level[signal_idx] - ref_spread)
                        / std_ref
                    )
                ),
                "entry_excursion_k": (
                    0.0
                    if m == 1
                    else float(
                        direction_sign(direction)
                        * (spread_level[index] - ref_spread)
                        / std_ref
                    )
                ),
                "entry_tsm_twd_fair": tsm_entry[index],
                "entry_qff_close": qff_entry[index],
                "entry_usdtwd": usdtwd_entry[index],
                "tsm_units": sizing.tsm_units,
                "qff_units": sizing.qff_units,
                "qff_contracts": sizing.qff_contracts,
                "actual_leg_notional_twd": sizing.actual_leg_notional_twd,
                "entry_fee_twd": entry_costs["total_fee_twd"],
                "entry_crossing_cost_twd": entry_costs["crossing_cost_twd"],
            }
        )
        return True

    def close_basket(
        index: int, exit_reason: str, exit_signal_idx: int, price_type: str
    ) -> None:
        """Close every tranche at this bar as one aggregate order."""
        nonlocal realized_pnl, direction, tranches, basket_id
        nonlocal ref_spread, std_ref, basket_entry_time, basket_mae_spread
        assert direction is not None and tranches
        if price_type == "open":
            exit_tsm, exit_qff, exit_fx = (
                tsm_entry[index], qff_entry[index], usdtwd_entry[index],
            )
        else:
            exit_tsm, exit_qff, exit_fx = tsm[index], qff[index], usdtwd[index]

        total_tsm_units = sum(t["tsm_units"] for t in tranches)
        total_contracts = sum(int(t["qff_contracts"]) for t in tranches)
        total_notional = sum(t["actual_leg_notional_twd"] for t in tranches)
        exit_costs = fill_costs(
            tsm_units=total_tsm_units,
            tsm_price=exit_tsm,
            qff_contracts=total_contracts,
            qff_price=exit_qff,
            params=per_tranche,
            tsm_side=us_side(-total_tsm_units),
            usdtwd=exit_fx,
            displacement=bar_disp[index],
        )

        basket_gross = 0.0
        basket_fees = 0.0
        for t in tranches:
            share = t["actual_leg_notional_twd"] / total_notional
            gross = t["tsm_units"] * (exit_tsm - t["entry_tsm_twd_fair"]) + t[
                "qff_units"
            ] * (exit_qff - t["entry_qff_close"])
            exit_fee = exit_costs["total_fee_twd"] * share
            net = gross - t["entry_fee_twd"] - exit_fee
            tranche_rows.append(
                {
                    **t,
                    "exit_signal_idx": exit_signal_idx,
                    "exit_signal_time": timestamps[exit_signal_idx],
                    "exit_idx": index,
                    "exit_time": timestamps[index],
                    "exit_fill_price_type": price_type,
                    "exit_reason": exit_reason,
                    "exit_spread": spread_level[index],
                    "exit_zscore": zscore[index],
                    "exit_tsm_twd_fair": exit_tsm,
                    "exit_qff_close": exit_qff,
                    "exit_fee_twd": exit_fee,
                    "gross_pnl_twd": gross,
                    "total_fee_twd": t["entry_fee_twd"] + exit_fee,
                    "net_pnl_twd": net,
                    "holding_minutes": minutes_between(
                        t["entry_time"], timestamps[index]
                    ),
                }
            )
            basket_gross += gross
            basket_fees += t["entry_fee_twd"] + exit_fee
        realized_pnl += basket_gross - exit_costs["total_fee_twd"]

        basket_rows.append(
            {
                "basket_id": basket_id,
                "direction": direction,
                "tranches_filled": len(tranches),
                "entry_first_time": tranches[0]["entry_time"],
                "entry_last_time": tranches[-1]["entry_time"],
                "ref_spread": ref_spread,
                "std_ref": std_ref,
                "vwa_entry_spread": vwa_entry_spread(),
                "qff_contracts": total_contracts,
                "gross_notional_twd": total_notional,
                "exit_time": timestamps[index],
                "exit_reason": exit_reason,
                "exit_spread": spread_level[index],
                "mae_spread": basket_mae_spread,
                "mae_k": basket_mae_spread / std_ref if std_ref > 0 else np.nan,
                "gross_pnl_twd": basket_gross,
                "total_fee_twd": basket_fees,
                "net_pnl_twd": basket_gross - basket_fees,
                "holding_minutes": minutes_between(
                    tranches[0]["entry_time"], timestamps[index]
                ),
            }
        )
        basket_id += 1
        direction = None
        tranches = []
        ref_spread = np.nan
        std_ref = np.nan
        basket_entry_time = None
        basket_mae_spread = 0.0
        clear_pending()

    for index in range(n_rows):
        filled_this_bar = False

        if pending_kind in ("entry", "add") and index == pending_fill_idx:
            signal_idx = pending_signal_idx
            clear_pending()
            if not fill_tranche(index, signal_idx):
                if len(tranches) == 0:
                    direction = None
                # A tranche budget that cannot buy one contract now will not
                # buy one later either; stop adding to this basket.
            filled_this_bar = True

        elif pending_kind == "exit" and index == pending_fill_idx:
            reason, signal_idx = pending_exit_reason, pending_signal_idx
            close_basket(index, reason, signal_idx, "open")
            filled_this_bar = True

        if (
            not filled_this_bar
            and tranches
            and friday_session_end[index]
        ):
            close_basket(index, FRIDAY_SESSION_END, index, "close")
            filled_this_bar = True

        if (
            not filled_this_bar
            and tranches
            and pending_kind != "exit"
            and params.base.max_holding_minutes > 0
            and close_allowed[index]
            and basket_entry_time is not None
            and minutes_between(basket_entry_time, timestamps[index])
            >= params.base.max_holding_minutes
        ):
            fill_idx = next_close_fill[index]
            if fill_idx != -1:
                pending_kind = "exit"
                pending_signal_idx = index
                pending_fill_idx = fill_idx
                pending_exit_reason = TIME_STOP

        if not filled_this_bar and direction is None and pending_kind is None:
            if entry_observation[index]:
                candidate = entry_direction(
                    z_short[index],
                    z_long[index],
                    zscore[index],
                    params.base.entry_z,
                    params.base.entry_z_max,
                )
                if candidate is not None:
                    fill_idx = next_entry_fill[index]
                    if fill_idx != -1 and minutes_between(
                        timestamps[index], timestamps[fill_idx]
                    ) <= params.base.max_entry_delay_minutes:
                        direction = candidate
                        pending_kind = "entry"
                        pending_signal_idx = index
                        pending_fill_idx = fill_idx

        # Skip the exit check only on the bar tranche 1 fills -- that is the
        # single engine's own behaviour, and n=1 parity depends on it. Once a
        # basket exists, an add filling (or waiting to fill) must not suppress
        # the exit: adds fire at adverse extremes, which is exactly where the
        # reversal that should close the basket tends to start, so discarding
        # those signals lengthened holds only in the multi-tranche cells --
        # biasing the very comparison this engine exists to make, in a
        # direction n=1 parity could never reveal.
        elif tranches and not (filled_this_bar and len(tranches) == 1):
            adverse = direction_sign(direction) * (spread_level[index] - ref_spread)
            basket_mae_spread = max(basket_mae_spread, adverse)

            if pending_kind != "exit" and close_observation[index]:
                exit_reason: str | None = None
                if (
                    params.exit_mode == "breakeven"
                    and len(tranches) >= 2
                    and std_ref > 0
                ):
                    reverted = -direction_sign(direction) * (
                        spread_level[index] - vwa_entry_spread()
                    )
                    if reverted >= (
                        params.be_cost_floor + params.be_offset_k * std_ref
                    ):
                        exit_reason = BREAKEVEN_EXIT
                if exit_reason is None and should_exit(
                    z_short[index], z_long[index], direction, params.base.exit_z
                ):
                    exit_reason = ZSCORE_EXIT
                if exit_reason is not None:
                    fill_idx = next_close_fill[index]
                    if fill_idx != -1:
                        # An exit outranks an add that has not filled yet:
                        # cancel the add rather than sizing up into a reversal.
                        pending_kind = "exit"
                        pending_signal_idx = index
                        pending_fill_idx = fill_idx
                        pending_exit_reason = exit_reason

            if (
                pending_kind is None
                and len(tranches) < params.n_tranches
                and entry_observation[index]
                and std_ref > 0
                and adverse >= len(tranches) * params.add_spacing_k * std_ref
            ):
                # entry_z_max is a structural-break rejection; it should veto
                # adds for the same reason it vetoes entries.
                if not (
                    params.base.entry_z_max > 0
                    and abs(zscore[index]) > params.base.entry_z_max
                ):
                    fill_idx = next_entry_fill[index]
                    if fill_idx != -1 and minutes_between(
                        timestamps[index], timestamps[fill_idx]
                    ) <= params.base.max_entry_delay_minutes:
                        pending_kind = "add"
                        pending_signal_idx = index
                        pending_fill_idx = fill_idx

        unrealized = sum(
            t["tsm_units"] * (tsm[index] - t["entry_tsm_twd_fair"])
            + t["qff_units"] * (qff[index] - t["entry_qff_close"])
            for t in tranches
        )
        tranche_count_out[index] = len(tranches)
        qff_contracts_out[index] = sum(int(t["qff_contracts"]) for t in tranches)
        gross_notional_out[index] = sum(
            t["actual_leg_notional_twd"] for t in tranches
        )
        realized_out[index] = realized_pnl
        unrealized_out[index] = unrealized
        equity_out[index] = (
            params.base.initial_capital_twd + realized_pnl + unrealized
        )

    if tranches:
        last = n_rows - 1
        close_basket(last, END_OF_DATA, last, "close")
        tranche_count_out[last] = 0
        qff_contracts_out[last] = 0
        gross_notional_out[last] = 0.0
        realized_out[last] = realized_pnl
        unrealized_out[last] = 0.0
        equity_out[last] = params.base.initial_capital_twd + realized_pnl

    equity_curve = pd.DataFrame(
        {
            "timestamp": timestamps,
            "spread": spread_level,
            "spread_zscore": zscore,
            "tranche_count": tranche_count_out,
            "qff_contracts": qff_contracts_out,
            "gross_notional_twd": gross_notional_out,
            "realized_pnl": realized_out,
            "unrealized_pnl": unrealized_out,
            "equity": equity_out,
        }
    )
    equity_curve["running_max_equity"] = equity_curve["equity"].cummax()
    equity_curve["drawdown_twd"] = (
        equity_curve["equity"] - equity_curve["running_max_equity"]
    )
    equity_curve["drawdown_pct"] = np.where(
        equity_curve["running_max_equity"] != 0,
        equity_curve["drawdown_twd"] / equity_curve["running_max_equity"],
        0.0,
    )

    tranches_frame = pd.DataFrame(tranche_rows)
    baskets_frame = pd.DataFrame(basket_rows)
    summary = build_scale_in_summary(equity_curve, tranches_frame, baskets_frame, params)
    validate_scale_in(equity_curve, tranches_frame, baskets_frame, params)
    return ScaleInResult(
        equity=equity_curve,
        tranches=tranches_frame,
        baskets=baskets_frame,
        summary=summary,
    )


def build_scale_in_summary(
    equity: pd.DataFrame,
    tranches: pd.DataFrame,
    baskets: pd.DataFrame,
    params: ScaleInParams,
) -> dict[str, Any]:
    base = params.base
    total_pnl = float(equity["equity"].iloc[-1] - base.initial_capital_twd)
    n_baskets = int(len(baskets))
    fill_hist = (
        baskets["tranches_filled"].value_counts().sort_index().to_dict()
        if n_baskets
        else {}
    )
    wins = int((baskets["net_pnl_twd"] > 0).sum()) if n_baskets else 0
    exit_reasons = (
        baskets["exit_reason"].value_counts().to_dict() if n_baskets else {}
    )
    exposure_bars = int((equity["tranche_count"] > 0).sum())
    return {
        "parameters": {
            "engine": "scale_in",
            "n_tranches": params.n_tranches,
            "add_spacing_k": params.add_spacing_k,
            "exit_mode": params.exit_mode,
            "be_offset_k": params.be_offset_k,
            "be_cost_floor": params.be_cost_floor,
            "entry_z": base.entry_z,
            "entry_z_max": base.entry_z_max,
            "exit_z": base.exit_z,
            "leg_notional_twd_total": base.leg_notional_twd,
            "leg_notional_twd_per_tranche": base.leg_notional_twd
            / params.n_tranches,
            "initial_capital_twd": base.initial_capital_twd,
            "max_entry_delay_minutes": base.max_entry_delay_minutes,
            "max_holding_minutes": base.max_holding_minutes,
            "tsm_fee_bps": base.tsm_fee_bps,
            "tsm_fee_model": base.tsm_fee_model,
            "tsm_share_ratio": base.tsm_share_ratio,
            "qff_fee_per_contract_twd": base.qff_fee_per_contract_twd,
            "qff_tax_rate": base.qff_tax_rate,
            "qff_contract_multiplier": base.qff_contract_multiplier,
            "executable_displacement": base.executable_displacement,
            # Both change what a run costs, and both were missing, so a run
            # varying either was unreproducible from its own manifest.
            "displacement_ref_price": base.displacement_ref_price,
            "qff_fee_bps": base.qff_fee_bps,
        },
        "rows": int(len(equity)),
        "start": format_timestamp(equity["timestamp"].iloc[0]),
        "end": format_timestamp(equity["timestamp"].iloc[-1]),
        "baskets": n_baskets,
        "tranche_fills": int(len(tranches)),
        "tranches_filled_histogram": {str(k): int(v) for k, v in fill_hist.items()},
        "exit_reasons": {str(k): int(v) for k, v in exit_reasons.items()},
        "winning_baskets": wins,
        "win_rate": float(wins / n_baskets) if n_baskets else 0.0,
        "total_pnl_twd": total_pnl,
        "total_fee_twd": float(baskets["total_fee_twd"].sum()) if n_baskets else 0.0,
        "return_pct": float(total_pnl / base.initial_capital_twd),
        "max_drawdown_twd": float(equity["drawdown_twd"].min()),
        "max_drawdown_pct": float(equity["drawdown_pct"].min()),
        "worst_basket_twd": float(baskets["net_pnl_twd"].min()) if n_baskets else 0.0,
        "best_basket_twd": float(baskets["net_pnl_twd"].max()) if n_baskets else 0.0,
        "median_basket_twd": (
            float(baskets["net_pnl_twd"].median()) if n_baskets else 0.0
        ),
        "max_gross_notional_twd": float(equity["gross_notional_twd"].max()),
        "exposure_bars": exposure_bars,
        "exposure_ratio": float(exposure_bars / len(equity)),
        "final_equity_twd": float(equity["equity"].iloc[-1]),
    }


def validate_scale_in(
    equity: pd.DataFrame,
    tranches: pd.DataFrame,
    baskets: pd.DataFrame,
    params: ScaleInParams,
) -> None:
    base = params.base
    identity = (
        base.initial_capital_twd
        + equity["realized_pnl"]
        + equity["unrealized_pnl"]
        - equity["equity"]
    ).abs()
    if identity.max() > 1e-7:
        raise RuntimeError("Equity accounting identity failed")
    if baskets.empty:
        return
    if abs(
        float(baskets["net_pnl_twd"].sum())
        - float(equity["equity"].iloc[-1] - base.initial_capital_twd)
    ) > 1e-6:
        raise RuntimeError("Basket net PnL does not match final equity PnL")
    if abs(
        float(tranches["net_pnl_twd"].sum()) - float(baskets["net_pnl_twd"].sum())
    ) > 1e-6:
        raise RuntimeError("Tranche PnL does not aggregate to basket PnL")
    per_basket = tranches.groupby("basket_id").size()
    if not per_basket.le(params.n_tranches).all():
        raise RuntimeError("A basket filled more tranches than n_tranches")
    # Every add must respect the frozen spacing at its SIGNAL bar -- the bar
    # the rule fired on. The fill bar is one bar later and free to sit
    # anywhere, so it proves nothing about the rule.
    for row in tranches.itertuples():
        if row.tranche == 1:
            continue
        b = baskets.loc[baskets["basket_id"] == row.basket_id].iloc[0]
        signal_needed = (row.tranche - 1) * params.add_spacing_k
        if row.signal_excursion_k < signal_needed - 1e-9:
            raise RuntimeError(
                f"Add {row.tranche} of basket {row.basket_id} fired at "
                f"{row.signal_excursion_k:.3f}k, under the required "
                f"{signal_needed:.3f}k"
            )
        if b["std_ref"] <= 0:
            raise RuntimeError("Basket with adds has no positive std_ref")


def write_scale_in_outputs(result: ScaleInResult, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    equity = result.equity.copy()
    equity["timestamp"] = format_timestamp_column(equity["timestamp"])
    equity.to_csv(run_dir / "equity.csv", index=False)
    for name, frame in (("tranches", result.tranches), ("baskets", result.baskets)):
        out = frame.copy()
        for column in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[column]):
                out[column] = format_timestamp_column(out[column])
        out.to_csv(run_dir / f"{name}.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, default=str), encoding="utf-8"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale-in pair backtest. Defaults are the CCF/UMC headline "
        "configuration (w2500 / entry 1.0 / exit 0.25 / displacement 0.2317)."
    )
    parser.add_argument("--spread-path", type=Path,
                        default=paths.feature("ccf_umc", "spread_1m"))
    parser.add_argument("--window", type=int, default=2500)
    parser.add_argument("--entry-z", type=float, default=1.0)
    parser.add_argument("--exit-z", type=float, default=0.25)
    parser.add_argument("--displacement", type=float, default=0.2317)
    parser.add_argument("--n-tranches", type=int, default=3)
    parser.add_argument("--add-spacing-k", type=float, default=1.0)
    parser.add_argument("--exit-mode", choices=list(EXIT_MODES), default="basket")
    parser.add_argument("--be-offset-k", type=float, default=0.0)
    parser.add_argument("--be-cost-floor", type=float, default=-1.0,
                        help="Round-trip cost floor for the breakeven exit, "
                        "spread units. Negative (default) = 2 x displacement.")
    parser.add_argument("--leg-notional-twd", type=float, default=1_000_000.0)
    parser.add_argument("--initial-capital-twd", type=float, default=2_000_000.0)
    parser.add_argument("--tsm-fee-model", default="ibkr", choices=["bps", "ibkr"])
    parser.add_argument("--tsm-fee-bps", type=float, default=2.5)
    parser.add_argument("--qff-fee-per-contract-twd", type=float, default=88.0)
    parser.add_argument("--qff-tax-rate", type=float, default=2e-5)
    parser.add_argument("--qff-contract-multiplier", type=float, default=2000.0)
    parser.add_argument("--run-tag", default="scale_in_scratch")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    from features import zscore as zscore_calc

    spread = zscore_calc.read_spread_frame(args.spread_path)
    frame = zscore_calc.calculate_zscore(spread, args.window)
    base = BacktestParams(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        leg_notional_twd=args.leg_notional_twd,
        initial_capital_twd=args.initial_capital_twd,
        max_entry_delay_minutes=15,
        tsm_fee_bps=args.tsm_fee_bps,
        tsm_fee_model=args.tsm_fee_model,
        qff_fee_per_contract_twd=args.qff_fee_per_contract_twd,
        qff_tax_rate=args.qff_tax_rate,
        qff_contract_multiplier=args.qff_contract_multiplier,
        executable_displacement=args.displacement,
    )
    be_cost_floor = (
        default_be_cost_floor(base) if args.be_cost_floor < 0 else args.be_cost_floor
    )
    params = ScaleInParams(
        base=base,
        n_tranches=args.n_tranches,
        add_spacing_k=args.add_spacing_k,
        exit_mode=args.exit_mode,
        be_offset_k=args.be_offset_k,
        be_cost_floor=be_cost_floor,
    )
    result = run_scale_in(frame, params)
    run_dir = paths.run_dir(args.run_tag)
    write_scale_in_outputs(result, run_dir)
    s = result.summary
    print(
        f"baskets {s['baskets']} (fills {s['tranches_filled_histogram']}), "
        f"win rate {s['win_rate']:.0%}, net {s['total_pnl_twd']:,.0f} TWD "
        f"({s['return_pct']:.2%}), maxDD {s['max_drawdown_pct']:.2%}, "
        f"worst basket {s['worst_basket_twd']:,.0f} TWD"
    )
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
