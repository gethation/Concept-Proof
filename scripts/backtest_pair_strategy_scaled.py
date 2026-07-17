"""Two-level scaled (pyramided) QFF/TSM pairs backtest.

Position mechanism (level1 < level2, each unit = full leg notional):
  - |z| > level1  -> open unit 1 (direction per sign of z)
  - unit 1 open and |z| > level2 (same direction) -> open unit 2
  - unit 2 closes when z crosses back through level1
  - unit 1 closes when z crosses through 0
  - unit 2 re-arms after closing if |z| re-crosses level2 while unit 1 is open
  - --level2 0 disables unit 2, which degenerates the engine to the base
    single-position machine with entry_z=level1 / exit_z=0.0 (verified exactly
    by the regression self-test).

The single-position engine in backtest_pair_strategy_1m.py is imported for all
shared mechanics (data loading, session masks, sizing, fees, per-unit trade
closing). Fill conventions are identical: entries and z-exits fill at the next
allowed bar's open; Friday session-end and end-of-data close at the same bar's
close. Accounting statement order matches the base engine so the level2=0 run
is float-identical to the base run.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_pair_strategy_1m as base  # noqa: E402

FLAT = base.FLAT
ENTRY_PENDING = base.ENTRY_PENDING
EXIT_PENDING = base.EXIT_PENDING
OPEN = "open"
FRIDAY_SESSION_END = base.FRIDAY_SESSION_END

DEFAULT_INPUT_PATH = Path("data/processed/qff_tsm_spread_zscore_15m_w33.csv")
DEFAULT_QFF_OHLCV_PATH = Path("data/processed/qff1_15m_taipei_tv.csv")
DEFAULT_TSM_OHLCV_PATH = Path(
    "data/processed/okx_tsmusdtp_15m_taipei_qff_session.csv"
)
DEFAULT_USDTTWD_OHLCV_PATH = Path(
    "data/processed/bitopro_usdttwd_15m_taipei_qff_session.csv"
)
DEFAULT_EQUITY_PATH = Path(
    "data/processed/qff_tsm_pair_backtest_equity_scaled_15m.csv"
)
DEFAULT_TRADES_PATH = Path(
    "data/processed/qff_tsm_pair_backtest_trades_scaled_15m.csv"
)
DEFAULT_SUMMARY_PATH = Path(
    "data/processed/qff_tsm_pair_backtest_summary_scaled_15m.json"
)


@dataclass(frozen=True)
class ScaledParams:
    level1: float
    level2: float
    leg_notional_twd: float
    initial_capital_twd: float
    max_entry_delay_minutes: int
    tsm_fee_bps: float
    qff_fee_per_contract_twd: float
    qff_tax_rate: float
    qff_contract_multiplier: float


@dataclass
class UnitSlot:
    unit_level: int
    state: str = FLAT
    direction: str | None = None
    candidate_idx: int = -1
    candidate_zscore: float = np.nan
    entry_fill_idx: int = -1
    exit_signal_idx: int = -1
    exit_fill_idx: int = -1
    exit_reason_pending: str = ""
    open_trade: dict[str, Any] | None = None
    filled_this_bar: bool = False


def validate_scaled_params(params: ScaledParams) -> None:
    if params.level1 <= 0:
        raise RuntimeError("level1 must be positive")
    if params.level2 != 0 and params.level2 <= params.level1:
        raise RuntimeError("level2 must be 0 (disabled) or greater than level1")
    if params.leg_notional_twd <= 0:
        raise RuntimeError("leg_notional_twd must be positive")
    if params.initial_capital_twd <= 0:
        raise RuntimeError("initial_capital_twd must be positive")
    if params.max_entry_delay_minutes < 0:
        raise RuntimeError("max_entry_delay_minutes must be non-negative")
    if params.tsm_fee_bps < 0:
        raise RuntimeError("tsm_fee_bps must be non-negative")
    if params.qff_fee_per_contract_twd < 0:
        raise RuntimeError("qff_fee_per_contract_twd must be non-negative")
    if params.qff_tax_rate < 0:
        raise RuntimeError("qff_tax_rate must be non-negative")
    if params.qff_contract_multiplier <= 0:
        raise RuntimeError("qff_contract_multiplier must be positive")


def should_exit_level(zscore: float, direction: str, exit_level: float) -> bool:
    """Unit 2 exits at exit_level=level1, unit 1 at 0.0. Signed same-side
    coordinates; identical to base.should_exit(z, d, 0.0) when exit_level=0."""
    return base.direction_sign(direction) * zscore < exit_level


def unit_exit_level(unit_level: int, params: ScaledParams) -> float:
    return params.level1 if unit_level == 2 else 0.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the QFF/TSM pairs strategy with a two-level scaled "
            "position book (unit 1 at level1, unit 2 added at level2, "
            "layered exits at level1 and 0)."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--qff-ohlcv", type=Path, default=DEFAULT_QFF_OHLCV_PATH)
    parser.add_argument("--tsm-ohlcv", type=Path, default=DEFAULT_TSM_OHLCV_PATH)
    parser.add_argument(
        "--usdttwd-ohlcv", type=Path, default=DEFAULT_USDTTWD_OHLCV_PATH
    )
    parser.add_argument("--equity-out", type=Path, default=DEFAULT_EQUITY_PATH)
    parser.add_argument("--trades-out", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--level1", type=float, default=1.0)
    parser.add_argument(
        "--level2",
        type=float,
        default=2.0,
        help="Second entry level. 0 disables unit 2 (single-unit mode).",
    )
    parser.add_argument("--leg-notional-twd", type=float, default=1_000_000.0)
    parser.add_argument("--initial-capital-twd", type=float, default=4_000_000.0)
    parser.add_argument("--max-entry-delay-minutes", type=int, default=30)
    parser.add_argument("--tsm-fee-bps", type=float, default=base.DEFAULT_TSM_FEE_BPS)
    parser.add_argument(
        "--qff-fee-per-contract-twd",
        type=float,
        default=base.DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
    )
    parser.add_argument(
        "--qff-tax-rate", type=float, default=base.DEFAULT_QFF_TAX_RATE
    )
    parser.add_argument(
        "--qff-contract-multiplier",
        type=float,
        default=base.DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )
    parser.add_argument("--skip-self-test", action="store_true")
    return parser.parse_args(argv)


def reset_unit(unit: UnitSlot) -> None:
    unit.state = FLAT
    unit.direction = None
    unit.candidate_idx = -1
    unit.candidate_zscore = np.nan
    unit.entry_fill_idx = -1
    unit.exit_signal_idx = -1
    unit.exit_fill_idx = -1
    unit.exit_reason_pending = ""
    unit.open_trade = None


def run_scaled_backtest(
    frame: pd.DataFrame, params: ScaledParams
) -> base.BacktestResult:
    validate_scaled_params(params)
    data = base.add_trading_masks(frame)
    n_rows = len(data)

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
    qff_entry_open_was_filled = (
        data["qff_entry_open_was_filled"].to_numpy(dtype=bool)
        if "qff_entry_open_was_filled" in data.columns
        else np.zeros(n_rows, dtype=bool)
    )
    zscore = data["spread_zscore"].to_numpy(dtype=float)
    zvalid = data["zscore_valid"].to_numpy(dtype=bool)
    entry_allowed = data["entry_allowed"].to_numpy(dtype=bool)
    close_allowed = data["close_allowed"].to_numpy(dtype=bool)
    friday_night = data["friday_night_close_only"].to_numpy(dtype=bool)
    weekend_session_close_only = data["weekend_session_close_only"].to_numpy(
        dtype=bool
    )
    friday_session_end = data["friday_session_end_force_close"].to_numpy(dtype=bool)

    entry_observation = entry_allowed & zvalid
    close_observation = close_allowed & zvalid
    next_entry_fill = base.compute_next_indices(entry_allowed)
    next_close_fill = base.compute_next_indices(close_allowed)

    unit1 = UnitSlot(1)
    unit2 = UnitSlot(2)
    realized_pnl = 0.0
    realized_fee_twd = 0.0
    trades: list[dict[str, Any]] = []
    cycle_id = 0
    unit2_fills_this_cycle = 0
    unit2_rearm_count = 0
    cancelled_unit2_fills = 0
    bars_at_two_units = 0

    state_out: list[str] = []
    position_out: list[str] = []
    tsm_units_out = np.zeros(n_rows)
    qff_units_out = np.zeros(n_rows)
    qff_contracts_out = np.zeros(n_rows, dtype=np.int64)
    actual_leg_notional_out = np.zeros(n_rows)
    units_open_out = np.zeros(n_rows, dtype=np.int64)
    unit2_state_out: list[str] = []
    realized_out = np.zeros(n_rows)
    realized_fee_out = np.zeros(n_rows)
    unrealized_out = np.zeros(n_rows)
    equity_out = np.zeros(n_rows)

    def close_unit(
        unit: UnitSlot,
        index: int,
        exit_signal_idx: int,
        exit_reason: str,
        exit_fill_price_type: str,
    ) -> None:
        nonlocal realized_pnl, realized_fee_twd
        if unit.open_trade is None:
            raise RuntimeError("close_unit called without an open trade")
        exit_tsm = tsm[index] if exit_fill_price_type == "close" else tsm_entry[index]
        exit_qff = qff[index] if exit_fill_price_type == "close" else qff_entry[index]
        realized_pnl += base.close_open_trade(
            trades,
            unit.open_trade,
            exit_idx=index,
            exit_time=timestamps[index],
            exit_tsm=exit_tsm,
            exit_qff=exit_qff,
            exit_zscore=zscore[index],
            exit_signal_idx=exit_signal_idx,
            exit_signal_time=timestamps[exit_signal_idx],
            exit_signal_zscore=zscore[exit_signal_idx],
            exit_reason=exit_reason,
            exit_fill_price_type=exit_fill_price_type,
            params=params,
        )
        realized_fee_twd += trades[-1]["exit_fee_twd"]
        reset_unit(unit)
        unit.filled_this_bar = True

    for index in range(n_rows):
        unit1.filled_this_bar = False
        unit2.filled_this_bar = False

        # ---- Phase 1a: exit fills, unit 2 first (LIFO trade-row order).
        for unit in (unit2, unit1):
            if unit.state == EXIT_PENDING and index == unit.exit_fill_idx:
                close_unit(
                    unit,
                    index,
                    exit_signal_idx=unit.exit_signal_idx,
                    exit_reason=unit.exit_reason_pending,
                    exit_fill_price_type="open",
                )

        # ---- Phase 1b: entry fills, unit 1 first.
        for unit in (unit1, unit2):
            if unit.state == ENTRY_PENDING and index == unit.entry_fill_idx:
                if unit is unit2 and unit1.state != OPEN:
                    # Orphan guard: unit 1 left the book while unit 2's fill
                    # was pending (only reachable with very long entry delays).
                    reset_unit(unit)
                    unit.filled_this_bar = True
                    cancelled_unit2_fills += 1
                    continue
                direction = unit.direction
                if direction is None:
                    raise RuntimeError("Entry pending without a direction")
                entry_tsm = tsm_entry[index]
                entry_qff = qff_entry[index]
                sizing = base.size_position_for_direction(
                    direction, entry_tsm, entry_qff, params
                )
                if sizing is None:
                    reset_unit(unit)
                    unit.filled_this_bar = True
                    continue
                entry_costs = base.fill_costs(
                    tsm_units=sizing.tsm_units,
                    tsm_price=entry_tsm,
                    qff_contracts=sizing.qff_contracts,
                    qff_price=entry_qff,
                    params=params,
                )
                realized_pnl -= entry_costs["total_fee_twd"]
                realized_fee_twd += entry_costs["total_fee_twd"]
                if unit is unit1:
                    cycle_id += 1
                    unit2_fills_this_cycle = 0
                else:
                    unit2_fills_this_cycle += 1
                    if unit2_fills_this_cycle > 1:
                        unit2_rearm_count += 1
                unit.open_trade = {
                    "entry_signal_idx": unit.candidate_idx,
                    "entry_signal_time": timestamps[unit.candidate_idx],
                    "entry_signal_zscore": unit.candidate_zscore,
                    "entry_idx": index,
                    "entry_time": timestamps[index],
                    "entry_fill_price_type": "open",
                    "entry_delay_minutes": base.minutes_between(
                        timestamps[unit.candidate_idx], timestamps[index]
                    ),
                    "entry_fill_zscore": zscore[index],
                    "direction": direction,
                    "entry_tsm_twd_fair": entry_tsm,
                    "entry_tsm_twd_fair_open": entry_tsm,
                    "entry_qff_close": entry_qff,
                    "entry_qff_open_filled": entry_qff,
                    "entry_qff_open_was_filled": bool(
                        qff_entry_open_was_filled[index]
                    ),
                    "tsm_units": sizing.tsm_units,
                    "qff_units": sizing.qff_units,
                    "qff_contracts": sizing.qff_contracts,
                    "raw_qff_contracts": sizing.raw_qff_contracts,
                    "leg_notional_twd": params.leg_notional_twd,
                    "actual_leg_notional_twd": sizing.actual_leg_notional_twd,
                    "qff_contract_multiplier": params.qff_contract_multiplier,
                    "entry_tsm_fee_twd": entry_costs["tsm_fee_twd"],
                    "entry_qff_fee_twd": entry_costs["qff_fee_twd"],
                    "entry_qff_tax_twd": entry_costs["qff_tax_twd"],
                    "entry_fee_twd": entry_costs["total_fee_twd"],
                    "unit_level": unit.unit_level,
                    "cycle_id": cycle_id,
                }
                unit.state = OPEN
                unit.candidate_idx = -1
                unit.candidate_zscore = np.nan
                unit.entry_fill_idx = -1
                unit.filled_this_bar = True

        # ---- Phase 2: Friday session-end force close (same-bar close).
        if friday_session_end[index]:
            for unit in (unit2, unit1):
                if (
                    not unit.filled_this_bar
                    and unit.open_trade is not None
                    and unit.state in (OPEN, EXIT_PENDING)
                ):
                    close_unit(
                        unit,
                        index,
                        exit_signal_idx=index,
                        exit_reason=FRIDAY_SESSION_END,
                        exit_fill_price_type="close",
                    )

        # ---- Phase 3: exit signal detection (independent per unit).
        for unit in (unit2, unit1):
            if (
                unit.state == OPEN
                and not unit.filled_this_bar
                and close_observation[index]
                and unit.direction is not None
                and should_exit_level(
                    zscore[index],
                    unit.direction,
                    unit_exit_level(unit.unit_level, params),
                )
            ):
                fill_idx = next_close_fill[index]
                if fill_idx != -1:
                    unit.exit_signal_idx = index
                    unit.exit_fill_idx = fill_idx
                    unit.exit_reason_pending = "zscore_exit"
                    unit.state = EXIT_PENDING

        # ---- Phase 4: entry signal detection.
        if (
            unit1.state == FLAT
            and not unit1.filled_this_bar
            and entry_observation[index]
        ):
            direction = base.entry_direction(zscore[index], params.level1)
            if direction is not None:
                fill_idx = next_entry_fill[index]
                if fill_idx != -1 and base.minutes_between(
                    timestamps[index], timestamps[fill_idx]
                ) <= params.max_entry_delay_minutes:
                    unit1.state = ENTRY_PENDING
                    unit1.direction = direction
                    unit1.candidate_idx = index
                    unit1.candidate_zscore = zscore[index]
                    unit1.entry_fill_idx = fill_idx
        if (
            params.level2 > 0
            and unit2.state == FLAT
            and not unit2.filled_this_bar
            and unit1.state == OPEN
            and entry_observation[index]
        ):
            direction = base.entry_direction(zscore[index], params.level2)
            if direction is not None and direction == unit1.direction:
                fill_idx = next_entry_fill[index]
                if fill_idx != -1 and base.minutes_between(
                    timestamps[index], timestamps[fill_idx]
                ) <= params.max_entry_delay_minutes:
                    unit2.state = ENTRY_PENDING
                    unit2.direction = direction
                    unit2.candidate_idx = index
                    unit2.candidate_zscore = zscore[index]
                    unit2.entry_fill_idx = fill_idx

        # ---- Phase 5: equity mark.
        unrealized = 0.0
        tsm_units_sum = 0.0
        qff_units_sum = 0.0
        qff_contracts_sum = 0
        notional_sum = 0.0
        units_open = 0
        for unit in (unit1, unit2):
            trade = unit.open_trade
            if trade is not None and unit.state in (OPEN, EXIT_PENDING):
                unrealized += trade["tsm_units"] * (
                    tsm[index] - trade["entry_tsm_twd_fair"]
                ) + trade["qff_units"] * (qff[index] - trade["entry_qff_close"])
                tsm_units_sum += trade["tsm_units"]
                qff_units_sum += trade["qff_units"]
                qff_contracts_sum += int(trade["qff_contracts"])
                notional_sum += trade["actual_leg_notional_twd"]
                units_open += 1
        if units_open == 2:
            bars_at_two_units += 1
        equity = params.initial_capital_twd + realized_pnl + unrealized

        if unit1.state == OPEN:
            state_str = unit1.direction or FLAT
        else:
            state_str = unit1.state
        state_out.append(state_str)
        position_out.append(
            (unit1.direction or FLAT)
            if unit1.state in (OPEN, EXIT_PENDING)
            else FLAT
        )
        unit2_state_out.append(
            (unit2.direction or FLAT) if unit2.state == OPEN else unit2.state
        )
        tsm_units_out[index] = tsm_units_sum
        qff_units_out[index] = qff_units_sum
        qff_contracts_out[index] = qff_contracts_sum
        actual_leg_notional_out[index] = notional_sum
        units_open_out[index] = units_open
        realized_out[index] = realized_pnl
        realized_fee_out[index] = realized_fee_twd
        unrealized_out[index] = unrealized
        equity_out[index] = equity

    # ---- End-of-data close (unit 2 first, same-bar close).
    last_idx = n_rows - 1
    any_forced = False
    for unit in (unit2, unit1):
        if unit.open_trade is not None and unit.state in (OPEN, EXIT_PENDING):
            close_unit(
                unit,
                last_idx,
                exit_signal_idx=(
                    unit.exit_signal_idx if unit.exit_signal_idx != -1 else last_idx
                ),
                exit_reason="end_of_data",
                exit_fill_price_type="close",
            )
            any_forced = True
    if any_forced:
        state_out[-1] = base.FORCED_CLOSED
        position_out[-1] = FLAT
        unit2_state_out[-1] = FLAT
        tsm_units_out[-1] = 0.0
        qff_units_out[-1] = 0.0
        qff_contracts_out[-1] = 0
        actual_leg_notional_out[-1] = 0.0
        units_open_out[-1] = 0
        realized_out[-1] = realized_pnl
        realized_fee_out[-1] = realized_fee_twd
        unrealized_out[-1] = 0.0
        equity_out[-1] = params.initial_capital_twd + realized_pnl

    equity_curve = pd.DataFrame(
        {
            "timestamp": timestamps,
            "state": state_out,
            "position": position_out,
            "spread_zscore": zscore,
            "zscore_valid": zvalid,
            "entry_allowed": entry_allowed,
            "close_allowed": close_allowed,
            "friday_night_close_only": friday_night,
            "weekend_session_close_only": weekend_session_close_only,
            "friday_session_end_force_close": friday_session_end,
            "qff_close_filled": qff,
            "tsm_twd_fair": tsm,
            "qff_entry_open_filled": qff_entry,
            "qff_entry_open_was_filled": qff_entry_open_was_filled,
            "tsm_twd_fair_open": tsm_entry,
            "tsm_units": tsm_units_out,
            "qff_units": qff_units_out,
            "qff_contracts": qff_contracts_out,
            "actual_leg_notional_twd": actual_leg_notional_out,
            "realized_pnl": realized_out,
            "realized_fee_twd": realized_fee_out,
            "unrealized_pnl": unrealized_out,
            "equity": equity_out,
            "units_open": units_open_out,
            "unit2_state": unit2_state_out,
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

    trades_frame = pd.DataFrame(trades)
    summary = build_scaled_summary(
        data=data,
        equity=equity_curve,
        trades=trades_frame,
        params=params,
        unit2_rearm_count=unit2_rearm_count,
        cancelled_unit2_fills=cancelled_unit2_fills,
        bars_at_two_units=bars_at_two_units,
    )
    validate_scaled_backtest(
        data=data,
        equity=equity_curve,
        trades=trades_frame,
        params=params,
    )
    return base.BacktestResult(
        equity=equity_curve, trades=trades_frame, summary=summary
    )


def build_scaled_summary(
    data: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    params: ScaledParams,
    unit2_rearm_count: int,
    cancelled_unit2_fills: int,
    bars_at_two_units: int,
) -> dict[str, Any]:
    total_pnl = float(equity["equity"].iloc[-1] - params.initial_capital_twd)
    trade_count = int(len(trades))
    wins = int((trades["total_pnl"] > 0).sum()) if trade_count else 0
    gross_profit = (
        float(trades.loc[trades["total_pnl"] > 0, "total_pnl"].sum())
        if trade_count
        else 0.0
    )
    gross_loss = (
        float(trades.loc[trades["total_pnl"] < 0, "total_pnl"].sum())
        if trade_count
        else 0.0
    )
    exposure_minutes = int(trades["holding_minutes"].sum()) if trade_count else 0
    elapsed_minutes = base.minutes_between(
        pd.Timestamp(equity["timestamp"].iloc[0]),
        pd.Timestamp(equity["timestamp"].iloc[-1]),
    )
    unit1_count = (
        int((trades["unit_level"] == 1).sum()) if trade_count else 0
    )
    unit2_count = (
        int((trades["unit_level"] == 2).sum()) if trade_count else 0
    )
    unit2_trailing_exits = 0
    if unit2_count:
        u1 = trades[trades["unit_level"] == 1][["cycle_id", "exit_idx"]]
        u2 = trades[trades["unit_level"] == 2][["cycle_id", "exit_idx"]]
        joined = u2.merge(u1, on="cycle_id", suffixes=("_u2", "_u1"))
        unit2_trailing_exits = int(
            (joined["exit_idx_u2"] > joined["exit_idx_u1"]).sum()
        )
    return {
        "fee_defaults_as_of": base.FEE_DEFAULTS_AS_OF,
        "engine": "two_level_scaled",
        "trade_count_semantics": "per_unit_round_trip",
        "parameters": {
            "level1": params.level1,
            "level2": params.level2,
            "unit2_exit_level": params.level1,
            "unit1_exit_level": 0.0,
            "leg_notional_twd_per_unit": params.leg_notional_twd,
            "initial_capital_twd": params.initial_capital_twd,
            "max_entry_delay_minutes": params.max_entry_delay_minutes,
            "tsm_fee_bps": params.tsm_fee_bps,
            "qff_fee_per_contract_twd": params.qff_fee_per_contract_twd,
            "qff_tax_rate": params.qff_tax_rate,
            "qff_contract_multiplier": params.qff_contract_multiplier,
            "entry_fill_price": "next_entry_allowed_open",
            "exit_fill_price": "signal_exit_next_close_allowed_open",
            "forced_exit_fill_price": "close",
        },
        "rows": int(len(equity)),
        "start": base.format_timestamp(pd.Timestamp(equity["timestamp"].iloc[0])),
        "end": base.format_timestamp(pd.Timestamp(equity["timestamp"].iloc[-1])),
        "entry_allowed_minutes": int(equity["entry_allowed"].sum()),
        "close_allowed_minutes": int(equity["close_allowed"].sum()),
        "trade_count": trade_count,
        "unit1_trade_count": unit1_count,
        "unit2_trade_count": unit2_count,
        "unit2_rearm_count": int(unit2_rearm_count),
        "unit2_trailing_exits": unit2_trailing_exits,
        "cancelled_unit2_fills": int(cancelled_unit2_fills),
        "bars_at_two_units": int(bars_at_two_units),
        "max_units_open": int(equity["units_open"].max()) if len(equity) else 0,
        "friday_session_forced_exits": (
            int((trades["exit_reason"] == FRIDAY_SESSION_END).sum())
            if trade_count
            else 0
        ),
        "winning_trades": wins,
        "losing_trades": trade_count - wins,
        "win_rate": wins / trade_count if trade_count else 0.0,
        "total_pnl_twd": total_pnl,
        "gross_pnl_twd": (
            float(trades["gross_pnl_twd"].sum()) if trade_count else 0.0
        ),
        "net_pnl_twd": (
            float(trades["net_pnl_twd"].sum()) if trade_count else 0.0
        ),
        "total_fee_twd": (
            float(trades["total_fee_twd"].sum()) if trade_count else 0.0
        ),
        "return_pct": total_pnl / params.initial_capital_twd,
        "gross_profit_twd": gross_profit,
        "gross_loss_twd": gross_loss,
        "profit_factor": (
            gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")
        ),
        "avg_trade_pnl_twd": (
            float(trades["total_pnl"].mean()) if trade_count else 0.0
        ),
        "max_drawdown_twd": float(equity["drawdown_twd"].min()),
        "max_drawdown_pct": float(equity["drawdown_pct"].min()),
        "elapsed_minutes": elapsed_minutes,
        "exposure_minutes": exposure_minutes,
        "final_equity_twd": float(equity["equity"].iloc[-1]),
    }


def validate_scaled_backtest(
    data: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    params: ScaledParams,
) -> None:
    tolerance = 1e-6

    identity_gap = (
        params.initial_capital_twd
        + equity["realized_pnl"]
        + equity["unrealized_pnl"]
        - equity["equity"]
    ).abs()
    if (identity_gap > tolerance).any():
        raise RuntimeError("Equity accounting identity violated")

    if trades.empty:
        return

    entry_allowed = data["entry_allowed"].to_numpy(dtype=bool)
    close_allowed = data["close_allowed"].to_numpy(dtype=bool)
    friday_session_end = data["friday_session_end_force_close"].to_numpy(dtype=bool)
    zscore = data["spread_zscore"].to_numpy(dtype=float)
    next_entry_fill = base.compute_next_indices(entry_allowed)
    next_close_fill = base.compute_next_indices(close_allowed)
    timestamps = pd.DatetimeIndex(data["timestamp"])
    last_idx = len(data) - 1

    total_net = float(trades["net_pnl_twd"].sum())
    final_pnl = float(equity["equity"].iloc[-1]) - params.initial_capital_twd
    if abs(total_net - final_pnl) > 1e-3:
        raise RuntimeError("Sum of trade net PnL does not match final equity")
    total_fee = float(trades["total_fee_twd"].sum())
    if abs(total_fee - float(equity["realized_fee_twd"].iloc[-1])) > 1e-3:
        raise RuntimeError("Sum of trade fees does not match realized fees")

    if params.level2 == 0 and (trades["unit_level"] == 2).any():
        raise RuntimeError("Unit 2 trades exist although level2 is disabled")

    for _, trade in trades.iterrows():
        unit_level = int(trade["unit_level"])
        entry_signal_idx = int(trade["entry_signal_idx"])
        entry_idx = int(trade["entry_idx"])
        exit_signal_idx = int(trade["exit_signal_idx"])
        exit_idx = int(trade["exit_idx"])
        direction = str(trade["direction"])
        exit_reason = str(trade["exit_reason"])
        entry_level = params.level2 if unit_level == 2 else params.level1
        exit_level = params.level1 if unit_level == 2 else 0.0

        if not entry_allowed[entry_idx]:
            raise RuntimeError("Trade entry filled on a non entry-allowed bar")
        if next_entry_fill[entry_signal_idx] != entry_idx:
            raise RuntimeError("Trade entry fill is not the next allowed bar")
        delay = base.minutes_between(
            timestamps[entry_signal_idx], timestamps[entry_idx]
        )
        if delay > params.max_entry_delay_minutes:
            raise RuntimeError("Trade entry delay exceeds the maximum")
        if not base.direction_still_valid(
            zscore[entry_signal_idx], direction, entry_level
        ):
            raise RuntimeError(
                f"Unit {unit_level} entry signal below its entry level"
            )

        if exit_reason == "zscore_exit":
            if next_close_fill[exit_signal_idx] != exit_idx:
                raise RuntimeError("Z-score exit fill is not the next allowed bar")
            if trade["exit_fill_price_type"] != "open":
                raise RuntimeError("Z-score exit must fill at open")
            if not should_exit_level(
                zscore[exit_signal_idx], direction, exit_level
            ):
                raise RuntimeError(
                    f"Unit {unit_level} z-exit signal does not satisfy its level"
                )
        elif exit_reason == FRIDAY_SESSION_END:
            if exit_signal_idx != exit_idx:
                raise RuntimeError("Friday exit signal and fill must coincide")
            if not friday_session_end[exit_idx]:
                raise RuntimeError("Friday exit on a non force-close bar")
            if trade["exit_fill_price_type"] != "close":
                raise RuntimeError("Friday exit must fill at close")
        elif exit_reason == "end_of_data":
            if exit_idx != last_idx:
                raise RuntimeError("End-of-data exit not on the last bar")
        else:
            raise RuntimeError(f"Unknown exit reason: {exit_reason}")

        raw_expected = params.leg_notional_twd / (
            trade["entry_qff_open_filled"] * params.qff_contract_multiplier
        )
        expected_contracts = base.round_half_up_nonnegative(raw_expected)
        if abs(int(trade["qff_contracts"])) != expected_contracts:
            raise RuntimeError("QFF contract rounding mismatch")
        expected_notional = (
            expected_contracts
            * params.qff_contract_multiplier
            * trade["entry_qff_open_filled"]
        )
        if abs(trade["actual_leg_notional_twd"] - expected_notional) > tolerance:
            raise RuntimeError("Actual leg notional mismatch")
        expected_tsm_units = expected_notional / trade["entry_tsm_twd_fair_open"]
        if abs(abs(trade["tsm_units"]) - expected_tsm_units) > tolerance:
            raise RuntimeError("TSM units mismatch")
        sgn = base.direction_sign(direction)
        if sgn * trade["tsm_units"] > 0:
            raise RuntimeError("TSM units sign mismatch")
        if sgn * trade["qff_units"] < 0:
            raise RuntimeError("QFF units sign mismatch")

        expected_tsm_pnl = trade["tsm_units"] * (
            trade["exit_tsm_twd_fair"] - trade["entry_tsm_twd_fair"]
        )
        expected_qff_pnl = trade["qff_units"] * (
            trade["exit_qff_close"] - trade["entry_qff_close"]
        )
        if abs(trade["tsm_pnl"] - expected_tsm_pnl) > 1e-4:
            raise RuntimeError("TSM PnL mismatch")
        if abs(trade["qff_pnl"] - expected_qff_pnl) > 1e-4:
            raise RuntimeError("QFF PnL mismatch")
        if (
            abs(
                trade["net_pnl_twd"]
                - (trade["gross_pnl_twd"] - trade["total_fee_twd"])
            )
            > 1e-4
        ):
            raise RuntimeError("Net PnL reconciliation failed")

    # Book invariants from reconstructed open intervals.
    for level in (1, 2):
        level_trades = trades[trades["unit_level"] == level].sort_values(
            "entry_idx"
        )
        prev_exit = -1
        for _, trade in level_trades.iterrows():
            if int(trade["entry_idx"]) < prev_exit:
                raise RuntimeError(
                    f"Overlapping unit-{level} trades in the book"
                )
            prev_exit = int(trade["exit_idx"])

    unit1_trades = trades[trades["unit_level"] == 1]
    for _, u2 in trades[trades["unit_level"] == 2].iterrows():
        parent = unit1_trades[unit1_trades["cycle_id"] == u2["cycle_id"]]
        if len(parent) != 1:
            raise RuntimeError("Unit-2 trade without exactly one parent cycle")
        p = parent.iloc[0]
        if p["direction"] != u2["direction"]:
            raise RuntimeError("Unit-2 direction differs from its cycle parent")
        if not (
            int(p["entry_idx"]) <= int(u2["entry_idx"]) < int(p["exit_idx"])
        ):
            raise RuntimeError("Unit-2 entry not inside its cycle's open interval")
        # Unit 2 normally closes no later than unit 1 (its exit condition is
        # implied by unit 1's). The one exception is the stale-fill race:
        # unit 2's entry fill lands on/after the bar unit 1's exit signal
        # fires, so its own exit detection starts one bar late and it briefly
        # trails the closed core.
        if int(u2["exit_idx"]) > int(p["exit_idx"]) and int(
            u2["entry_idx"]
        ) < int(p["exit_signal_idx"]):
            raise RuntimeError(
                "Unit-2 exit trails its cycle without a stale-fill race"
            )

    units_open = np.zeros(len(data), dtype=np.int64)
    for _, trade in trades.iterrows():
        units_open[int(trade["entry_idx"]) : int(trade["exit_idx"])] += 1
    # The exit bar itself still marks the position (fills reset after marking
    # in the base convention: exit fill bar shows zero units).
    if (units_open > 2).any():
        raise RuntimeError("More than two units open at once")


def run_self_tests() -> None:
    params = ScaledParams(
        level1=2.0,
        level2=3.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=base.DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=base.DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=base.DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=base.DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )

    # (a) Ladder round trip: add at level2, peel at level1, close at 0.
    ladder = base.make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=8, freq="min", tz=base.TAIPEI_TZ),
        zscores=[2.1, 2.2, 3.2, 3.1, 1.5, 1.4, -0.1, 0.3],
        tsm_start=100.0,
        qff_start=100.0,
    )
    result = run_scaled_backtest(ladder, params)
    if len(result.trades) != 2:
        raise RuntimeError("Self-test (a) failed: expected two unit trades")
    u2, u1 = result.trades.iloc[0], result.trades.iloc[1]
    if int(u2["unit_level"]) != 2 or int(u1["unit_level"]) != 1:
        raise RuntimeError("Self-test (a) failed: LIFO trade-row order wrong")
    if (
        int(u1["entry_signal_idx"]) != 0
        or int(u1["entry_idx"]) != 1
        or int(u2["entry_signal_idx"]) != 2
        or int(u2["entry_idx"]) != 3
    ):
        raise RuntimeError("Self-test (a) failed: entry timing wrong")
    if (
        int(u2["exit_signal_idx"]) != 4
        or int(u2["exit_idx"]) != 5
        or int(u1["exit_signal_idx"]) != 6
        or int(u1["exit_idx"]) != 7
    ):
        raise RuntimeError("Self-test (a) failed: layered exit timing wrong")
    if int(u1["cycle_id"]) != int(u2["cycle_id"]):
        raise RuntimeError("Self-test (a) failed: cycle ids differ")
    if int(result.summary["max_units_open"]) != 2:
        raise RuntimeError("Self-test (a) failed: max_units_open should be 2")

    # (b) Straight-through move: unit 2 arms on unit 1's fill bar; a joint
    # reversal closes both on the same bar (unit 2 row first).
    straight = base.make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=6, freq="min", tz=base.TAIPEI_TZ),
        zscores=[3.5, 3.4, 3.3, -0.1, -0.2, 0.1],
        tsm_start=100.0,
        qff_start=100.0,
    )
    result = run_scaled_backtest(straight, params)
    if len(result.trades) != 2:
        raise RuntimeError("Self-test (b) failed: expected two unit trades")
    u2, u1 = result.trades.iloc[0], result.trades.iloc[1]
    if int(u2["unit_level"]) != 2:
        raise RuntimeError("Self-test (b) failed: unit 2 must close first")
    if int(u1["entry_idx"]) != 1 or int(u2["entry_idx"]) != 2:
        raise RuntimeError("Self-test (b) failed: cascade entry timing wrong")
    if int(u2["exit_idx"]) != 4 or int(u1["exit_idx"]) != 4:
        raise RuntimeError("Self-test (b) failed: joint exit bar wrong")

    # (c) Unit-2 re-arm inside one unit-1 cycle.
    rearm = base.make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=12, freq="min", tz=base.TAIPEI_TZ),
        zscores=[2.5, 2.5, 3.2, 3.1, 1.8, 1.9, 3.3, 3.2, 1.5, -0.1, 0.1, 0.2],
        tsm_start=100.0,
        qff_start=100.0,
    )
    result = run_scaled_backtest(rearm, params)
    if len(result.trades) != 3:
        raise RuntimeError("Self-test (c) failed: expected three unit trades")
    if int(result.summary["unit2_trade_count"]) != 2:
        raise RuntimeError("Self-test (c) failed: expected two unit-2 trades")
    if int(result.summary["unit2_rearm_count"]) != 1:
        raise RuntimeError("Self-test (c) failed: expected one unit-2 re-arm")

    # (d) Friday session-end force close with both units open.
    friday_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-12 13:41", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 13:42", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 13:43", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:25", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:26", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:27", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-15 08:45", tz=base.TAIPEI_TZ),
        ]
    )
    friday = base.make_synthetic_frame(
        friday_times,
        zscores=[2.1, 3.5, 3.4, 2.5, 2.6, 2.7, 0.0],
        tsm_start=100.0,
        qff_start=100.0,
    )
    result = run_scaled_backtest(friday, params)
    if len(result.trades) != 2:
        raise RuntimeError("Self-test (d) failed: expected two forced closes")
    if not (result.trades["exit_reason"] == FRIDAY_SESSION_END).all():
        raise RuntimeError("Self-test (d) failed: exit reason must be Friday")
    if not (result.trades["exit_idx"] == 5).all():
        raise RuntimeError("Self-test (d) failed: force close bar wrong")
    if int(result.trades.iloc[0]["unit_level"]) != 2:
        raise RuntimeError("Self-test (d) failed: unit 2 must close first")

    # (e) Regression equivalence: level2=0 must reproduce the base engine
    # with entry_z=level1 / exit_z=0.0 exactly.
    single_params = ScaledParams(
        level1=2.0,
        level2=0.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=base.DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=base.DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=base.DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=base.DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )
    base_params = base.BacktestParams(
        entry_z=2.0,
        exit_z=0.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=base.DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=base.DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=base.DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=base.DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )
    equivalence_frames = {
        "whipsaw": base.make_synthetic_frame(
            pd.date_range(
                "2026-06-08 08:45", periods=14, freq="min", tz=base.TAIPEI_TZ
            ),
            zscores=[
                2.1, 1.3, -2.1, -2.2, -2.3, 0.5, 0.6, 2.4, 2.2, -0.4,
                -2.6, -2.4, 0.3, 0.1,
            ],
            tsm_start=100.0,
            qff_start=100.0,
        ),
        "friday_force": base.make_synthetic_frame(
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2026-06-12 13:41", tz=base.TAIPEI_TZ),
                    pd.Timestamp("2026-06-12 13:42", tz=base.TAIPEI_TZ),
                    pd.Timestamp("2026-06-12 13:43", tz=base.TAIPEI_TZ),
                    pd.Timestamp("2026-06-12 17:25", tz=base.TAIPEI_TZ),
                    pd.Timestamp("2026-06-12 17:26", tz=base.TAIPEI_TZ),
                    pd.Timestamp("2026-06-12 17:27", tz=base.TAIPEI_TZ),
                    pd.Timestamp("2026-06-15 08:45", tz=base.TAIPEI_TZ),
                ]
            ),
            zscores=[2.1, 2.1, 2.1, 1.1, 1.2, 1.3, 0.0],
            tsm_start=100.0,
            qff_start=100.0,
        ),
    }
    for name, eq_frame in equivalence_frames.items():
        scaled_result = run_scaled_backtest(eq_frame, single_params)
        base_result = base.run_backtest(eq_frame, base_params)
        scaled_trades = scaled_result.trades.drop(
            columns=["unit_level", "cycle_id"]
        )
        pd.testing.assert_frame_equal(
            scaled_trades.reset_index(drop=True),
            base_result.trades.reset_index(drop=True),
            check_exact=True,
        )
        shared = [c for c in base_result.equity.columns]
        pd.testing.assert_frame_equal(
            scaled_result.equity[shared].reset_index(drop=True),
            base_result.equity.reset_index(drop=True),
            check_exact=True,
        )
        if name and scaled_result.summary["unit2_trade_count"] != 0:
            raise RuntimeError("Self-test (e) failed: unit 2 traded while disabled")

    # (f) Orphan guard: unit 2's delayed fill lands after unit 1 has left.
    orphan_params = ScaledParams(
        level1=2.0,
        level2=3.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=5000,
        tsm_fee_bps=base.DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=base.DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=base.DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=base.DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )
    orphan_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-12 13:43", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 13:44", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 13:45", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:25", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:26", tz=base.TAIPEI_TZ),
            pd.Timestamp("2026-06-15 08:45", tz=base.TAIPEI_TZ),
        ]
    )
    orphan = base.make_synthetic_frame(
        orphan_times,
        zscores=[2.1, 2.5, 3.5, -0.5, -0.4, 0.0],
        tsm_start=100.0,
        qff_start=100.0,
    )
    result = run_scaled_backtest(orphan, orphan_params)
    if len(result.trades) != 1:
        raise RuntimeError("Self-test (f) failed: expected one unit-1 trade")
    if int(result.summary["cancelled_unit2_fills"]) != 1:
        raise RuntimeError("Self-test (f) failed: orphan fill was not cancelled")

    # (h) Stale-fill race: unit 2's entry fills on the reversal bar, unit 1
    # exits first, unit 2 trails by one bar (validator must accept this).
    race_params = ScaledParams(
        level1=1.0,
        level2=2.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=base.DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=base.DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=base.DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=base.DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )
    race = base.make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=10, freq="min", tz=base.TAIPEI_TZ),
        zscores=[1.2, 1.1, 2.5, -0.5, -0.4, 1.5, 1.4, 0.9, -0.1, 0.1],
        tsm_start=100.0,
        qff_start=100.0,
    )
    result = run_scaled_backtest(race, race_params)
    if len(result.trades) != 3:
        raise RuntimeError("Self-test (h) failed: expected three trades")
    u2_rows = result.trades[result.trades["unit_level"] == 2]
    if len(u2_rows) != 1:
        raise RuntimeError("Self-test (h) failed: expected one unit-2 trade")
    u2_row = u2_rows.iloc[0]
    parent = result.trades[
        (result.trades["unit_level"] == 1)
        & (result.trades["cycle_id"] == u2_row["cycle_id"])
    ].iloc[0]
    if not int(u2_row["exit_idx"]) > int(parent["exit_idx"]):
        raise RuntimeError("Self-test (h) failed: race case was not exercised")
    if int(result.summary["unit2_trailing_exits"]) != 1:
        raise RuntimeError("Self-test (h) failed: trailing exit not counted")

    # (g) Parameter validation.
    for bad_level2 in (1.5, 2.0):
        try:
            validate_scaled_params(
                ScaledParams(
                    level1=2.0,
                    level2=bad_level2,
                    leg_notional_twd=1_000_000.0,
                    initial_capital_twd=2_000_000.0,
                    max_entry_delay_minutes=15,
                    tsm_fee_bps=0.0,
                    qff_fee_per_contract_twd=0.0,
                    qff_tax_rate=0.0,
                    qff_contract_multiplier=100.0,
                )
            )
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Self-test (g) failed: invalid level2 was accepted"
            )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_self_test:
        run_self_tests()
        print("Self-tests passed")

    params = ScaledParams(
        level1=args.level1,
        level2=args.level2,
        leg_notional_twd=args.leg_notional_twd,
        initial_capital_twd=args.initial_capital_twd,
        max_entry_delay_minutes=args.max_entry_delay_minutes,
        tsm_fee_bps=args.tsm_fee_bps,
        qff_fee_per_contract_twd=args.qff_fee_per_contract_twd,
        qff_tax_rate=args.qff_tax_rate,
        qff_contract_multiplier=args.qff_contract_multiplier,
    )
    frame = base.read_input_frame(
        args.input,
        qff_ohlcv_path=args.qff_ohlcv,
        tsm_ohlcv_path=args.tsm_ohlcv,
        usdttwd_ohlcv_path=args.usdttwd_ohlcv,
    )
    result = run_scaled_backtest(frame, params)
    base.write_outputs(result, args.equity_out, args.trades_out, args.summary_out)

    print(f"Wrote equity curve to {args.equity_out}")
    print(f"Wrote trades to {args.trades_out}")
    print(f"Wrote summary to {args.summary_out}")
    print(
        "Summary: "
        f"trades={result.summary['trade_count']} "
        f"(u1={result.summary['unit1_trade_count']}, "
        f"u2={result.summary['unit2_trade_count']}, "
        f"rearms={result.summary['unit2_rearm_count']}), "
        f"total_pnl={result.summary['total_pnl_twd']:.2f}, "
        f"return={result.summary['return_pct']:.4%}, "
        f"max_dd={result.summary['max_drawdown_twd']:.2f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
