from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_INPUT_PATH = Path("data/processed/qff_tsm_spread_zscore_1m_taipei.csv")
DEFAULT_EQUITY_PATH = Path("data/processed/qff_tsm_pair_backtest_equity_1m.csv")
DEFAULT_TRADES_PATH = Path("data/processed/qff_tsm_pair_backtest_trades.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/qff_tsm_pair_backtest_summary.json")

SHORT_TSM_LONG_QFF = "short_tsm_long_qff"
LONG_TSM_SHORT_QFF = "long_tsm_short_qff"
FLAT = "flat"
ENTRY_PENDING = "entry_pending_fill"
EXIT_PENDING = "exit_pending_fill"
FORCED_CLOSED = "forced_closed_end_of_data"

DAY_START_MINUTE = 8 * 60 + 45
DAY_END_MINUTE = 13 * 60 + 45
NIGHT_START_MINUTE = 17 * 60 + 25
NIGHT_END_MINUTE = 5 * 60


@dataclass(frozen=True)
class BacktestParams:
    entry_z: float
    exit_z: float
    leg_notional_twd: float
    initial_capital_twd: float
    max_entry_delay_minutes: int


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, Any]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest a simple QFF/TSM pairs strategy using next "
            "entry-allowed-minute close fills with a maximum entry delay."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--equity-out", type=Path, default=DEFAULT_EQUITY_PATH)
    parser.add_argument("--trades-out", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.0)
    parser.add_argument("--leg-notional-twd", type=float, default=1_000_000.0)
    parser.add_argument("--initial-capital-twd", type=float, default=2_000_000.0)
    parser.add_argument("--max-entry-delay-minutes", type=int, default=15)
    parser.add_argument("--skip-self-test", action="store_true")
    return parser.parse_args(argv)


def read_input_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    frame = pd.read_csv(path)
    required = {
        "timestamp",
        "qff_close",
        "qff_close_filled",
        "tsm_twd_fair",
        "spread_zscore",
        "zscore_valid",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise RuntimeError(f"Input CSV has no rows: {path}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    for column in ["qff_close", "qff_close_filled", "tsm_twd_fair", "spread_zscore"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["qff_close_filled"].isna().any():
        raise RuntimeError("qff_close_filled contains missing values")
    if frame["tsm_twd_fair"].isna().any():
        raise RuntimeError("tsm_twd_fair contains missing values")

    frame["zscore_valid"] = parse_bool_series(frame["zscore_valid"])
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    expected = pd.date_range(timestamps[0], timestamps[-1], freq="min")
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise RuntimeError("Input timestamps must be unique and sorted")
    if len(timestamps) != len(expected) or not timestamps.equals(expected):
        raise RuntimeError("Input must be a continuous 1m series")
    return frame


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes"])


def add_trading_masks(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    ts = pd.DatetimeIndex(output["timestamp"])
    minute = ts.hour * 60 + ts.minute

    day_clock = (minute >= DAY_START_MINUTE) & (minute <= DAY_END_MINUTE)
    night_clock = (minute >= NIGHT_START_MINUTE) | (minute <= NIGHT_END_MINUTE)

    local_day = ts.normalize()
    session_start = pd.Series(local_day, index=output.index)
    session_start.loc[minute <= NIGHT_END_MINUTE] = (
        session_start.loc[minute <= NIGHT_END_MINUTE] - pd.Timedelta(days=1)
    )

    has_qff_trade = output["qff_close"].notna().to_numpy()
    day_active = set(local_day[day_clock & has_qff_trade])
    night_active = set(session_start.loc[night_clock & has_qff_trade])

    day_allowed = day_clock & pd.Series(local_day, index=output.index).isin(day_active).to_numpy()
    night_allowed = night_clock & session_start.isin(night_active).to_numpy()
    close_allowed = day_allowed | night_allowed

    friday_night = night_allowed & session_start.dt.weekday.eq(4).to_numpy()
    entry_allowed = close_allowed & ~friday_night

    output["close_allowed"] = close_allowed
    output["entry_allowed"] = entry_allowed
    output["friday_night_close_only"] = close_allowed & ~entry_allowed
    return output


def compute_next_indices(mask: np.ndarray) -> np.ndarray:
    next_indices = np.full(len(mask), -1, dtype=np.int64)
    next_seen = -1
    for index in range(len(mask) - 1, -1, -1):
        next_indices[index] = next_seen
        if mask[index]:
            next_seen = index
    return next_indices


def entry_direction(zscore: float, entry_z: float) -> str | None:
    if zscore > entry_z:
        return SHORT_TSM_LONG_QFF
    if zscore < -entry_z:
        return LONG_TSM_SHORT_QFF
    return None


def direction_still_valid(zscore: float, direction: str, entry_z: float) -> bool:
    if direction == SHORT_TSM_LONG_QFF:
        return zscore > entry_z
    if direction == LONG_TSM_SHORT_QFF:
        return zscore < -entry_z
    raise ValueError(f"Unknown direction: {direction}")


def should_exit(zscore: float, direction: str, exit_z: float) -> bool:
    if direction == SHORT_TSM_LONG_QFF:
        return zscore < -exit_z
    if direction == LONG_TSM_SHORT_QFF:
        return zscore > exit_z
    raise ValueError(f"Unknown direction: {direction}")


def units_for_direction(
    direction: str, tsm_price: float, qff_price: float, leg_notional_twd: float
) -> tuple[float, float]:
    tsm_units = leg_notional_twd / tsm_price
    qff_units = leg_notional_twd / qff_price
    if direction == SHORT_TSM_LONG_QFF:
        return -tsm_units, qff_units
    if direction == LONG_TSM_SHORT_QFF:
        return tsm_units, -qff_units
    raise ValueError(f"Unknown direction: {direction}")


def minutes_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start).total_seconds() // 60)


def run_backtest(frame: pd.DataFrame, params: BacktestParams) -> BacktestResult:
    data = add_trading_masks(frame)
    n_rows = len(data)

    timestamps = pd.DatetimeIndex(data["timestamp"])
    qff = data["qff_close_filled"].to_numpy(dtype=float)
    tsm = data["tsm_twd_fair"].to_numpy(dtype=float)
    zscore = data["spread_zscore"].to_numpy(dtype=float)
    zvalid = data["zscore_valid"].to_numpy(dtype=bool)
    entry_allowed = data["entry_allowed"].to_numpy(dtype=bool)
    close_allowed = data["close_allowed"].to_numpy(dtype=bool)
    friday_night = data["friday_night_close_only"].to_numpy(dtype=bool)

    entry_observation = entry_allowed & zvalid
    close_observation = close_allowed & zvalid
    next_entry_fill = compute_next_indices(entry_allowed)
    next_close_fill = compute_next_indices(close_allowed)

    state = FLAT
    position_direction: str | None = None
    candidate_direction: str | None = None
    candidate_idx = -1
    candidate_zscore = np.nan
    entry_fill_idx = -1
    exit_signal_idx = -1
    exit_fill_idx = -1

    entry_tsm = np.nan
    entry_qff = np.nan
    entry_zscore = np.nan
    tsm_units = 0.0
    qff_units = 0.0
    realized_pnl = 0.0
    open_trade: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []

    state_out: list[str] = []
    position_out: list[str] = []
    tsm_units_out = np.zeros(n_rows)
    qff_units_out = np.zeros(n_rows)
    realized_out = np.zeros(n_rows)
    unrealized_out = np.zeros(n_rows)
    equity_out = np.zeros(n_rows)

    for index in range(n_rows):
        filled_this_bar = False

        if state == ENTRY_PENDING and index == entry_fill_idx:
            position_direction = candidate_direction
            if position_direction is None:
                raise RuntimeError("Entry pending without a direction")
            entry_tsm = tsm[index]
            entry_qff = qff[index]
            entry_zscore = zscore[index]
            tsm_units, qff_units = units_for_direction(
                position_direction, entry_tsm, entry_qff, params.leg_notional_twd
            )
            open_trade = {
                "entry_signal_idx": candidate_idx,
                "entry_signal_time": timestamps[candidate_idx],
                "entry_signal_zscore": candidate_zscore,
                "entry_idx": index,
                "entry_time": timestamps[index],
                "entry_delay_minutes": minutes_between(
                    timestamps[candidate_idx], timestamps[index]
                ),
                "entry_fill_zscore": entry_zscore,
                "direction": position_direction,
                "entry_tsm_twd_fair": entry_tsm,
                "entry_qff_close": entry_qff,
                "tsm_units": tsm_units,
                "qff_units": qff_units,
                "leg_notional_twd": params.leg_notional_twd,
            }
            state = position_direction
            candidate_direction = None
            candidate_idx = -1
            candidate_zscore = np.nan
            entry_fill_idx = -1
            filled_this_bar = True

        elif state == EXIT_PENDING and index == exit_fill_idx:
            if open_trade is None or position_direction is None:
                raise RuntimeError("Exit pending without an open trade")
            realized_pnl += close_open_trade(
                trades,
                open_trade,
                exit_idx=index,
                exit_time=timestamps[index],
                exit_tsm=tsm[index],
                exit_qff=qff[index],
                exit_zscore=zscore[index],
                exit_signal_idx=exit_signal_idx,
                exit_signal_time=timestamps[exit_signal_idx],
                exit_signal_zscore=zscore[exit_signal_idx],
                exit_reason="zscore_exit",
            )
            state = FLAT
            position_direction = None
            open_trade = None
            tsm_units = 0.0
            qff_units = 0.0
            entry_tsm = np.nan
            entry_qff = np.nan
            entry_zscore = np.nan
            exit_signal_idx = -1
            exit_fill_idx = -1
            filled_this_bar = True

        if not filled_this_bar:
            if state == FLAT and entry_observation[index]:
                direction = entry_direction(zscore[index], params.entry_z)
                if direction is not None:
                    fill_idx = next_entry_fill[index]
                    if fill_idx != -1 and minutes_between(
                        timestamps[index], timestamps[fill_idx]
                    ) <= params.max_entry_delay_minutes:
                        state = ENTRY_PENDING
                        candidate_direction = direction
                        candidate_idx = index
                        candidate_zscore = zscore[index]
                        entry_fill_idx = fill_idx

            elif position_direction is not None and state == position_direction:
                if close_observation[index] and should_exit(
                    zscore[index], position_direction, params.exit_z
                ):
                    fill_idx = next_close_fill[index]
                    if fill_idx != -1:
                        exit_signal_idx = index
                        exit_fill_idx = fill_idx
                        state = EXIT_PENDING

        unrealized = 0.0
        if position_direction is not None:
            unrealized = tsm_units * (tsm[index] - entry_tsm) + qff_units * (
                qff[index] - entry_qff
            )
        equity = params.initial_capital_twd + realized_pnl + unrealized

        state_out.append(state)
        position_out.append(position_direction or FLAT)
        tsm_units_out[index] = tsm_units
        qff_units_out[index] = qff_units
        realized_out[index] = realized_pnl
        unrealized_out[index] = unrealized
        equity_out[index] = equity

    if position_direction is not None and open_trade is not None:
        last_idx = n_rows - 1
        realized_pnl += close_open_trade(
            trades,
            open_trade,
            exit_idx=last_idx,
            exit_time=timestamps[last_idx],
            exit_tsm=tsm[last_idx],
            exit_qff=qff[last_idx],
            exit_zscore=zscore[last_idx],
            exit_signal_idx=exit_signal_idx if exit_signal_idx != -1 else last_idx,
            exit_signal_time=(
                timestamps[exit_signal_idx] if exit_signal_idx != -1 else timestamps[last_idx]
            ),
            exit_signal_zscore=(
                zscore[exit_signal_idx] if exit_signal_idx != -1 else zscore[last_idx]
            ),
            exit_reason="end_of_data",
        )
        state_out[-1] = FORCED_CLOSED
        position_out[-1] = FLAT
        tsm_units_out[-1] = 0.0
        qff_units_out[-1] = 0.0
        realized_out[-1] = realized_pnl
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
            "qff_close_filled": qff,
            "tsm_twd_fair": tsm,
            "tsm_units": tsm_units_out,
            "qff_units": qff_units_out,
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

    trades_frame = pd.DataFrame(trades)
    summary = build_summary(
        data=data,
        equity=equity_curve,
        trades=trades_frame,
        params=params,
    )
    validate_backtest(
        data=data,
        equity=equity_curve,
        trades=trades_frame,
        params=params,
    )
    return BacktestResult(equity=equity_curve, trades=trades_frame, summary=summary)


def close_open_trade(
    trades: list[dict[str, Any]],
    open_trade: dict[str, Any],
    exit_idx: int,
    exit_time: pd.Timestamp,
    exit_tsm: float,
    exit_qff: float,
    exit_zscore: float,
    exit_signal_idx: int,
    exit_signal_time: pd.Timestamp,
    exit_signal_zscore: float,
    exit_reason: str,
) -> float:
    tsm_pnl = open_trade["tsm_units"] * (exit_tsm - open_trade["entry_tsm_twd_fair"])
    qff_pnl = open_trade["qff_units"] * (exit_qff - open_trade["entry_qff_close"])
    total_pnl = tsm_pnl + qff_pnl
    trade = {
        **open_trade,
        "exit_signal_idx": exit_signal_idx,
        "exit_signal_time": exit_signal_time,
        "exit_signal_zscore": exit_signal_zscore,
        "exit_idx": exit_idx,
        "exit_time": exit_time,
        "exit_fill_zscore": exit_zscore,
        "exit_tsm_twd_fair": exit_tsm,
        "exit_qff_close": exit_qff,
        "tsm_pnl": tsm_pnl,
        "qff_pnl": qff_pnl,
        "total_pnl": total_pnl,
        "exit_reason": exit_reason,
        "holding_minutes": exit_idx - open_trade["entry_idx"],
    }
    trades.append(trade)
    return total_pnl


def build_summary(
    data: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    params: BacktestParams,
) -> dict[str, Any]:
    total_pnl = float(equity["equity"].iloc[-1] - params.initial_capital_twd)
    trade_count = int(len(trades))
    wins = int((trades["total_pnl"] > 0).sum()) if trade_count else 0
    gross_profit = float(trades.loc[trades["total_pnl"] > 0, "total_pnl"].sum()) if trade_count else 0.0
    gross_loss = float(trades.loc[trades["total_pnl"] < 0, "total_pnl"].sum()) if trade_count else 0.0
    exposure_minutes = int((equity["position"] != FLAT).sum())
    return {
        "parameters": {
            "entry_z": params.entry_z,
            "exit_z": params.exit_z,
            "leg_notional_twd": params.leg_notional_twd,
            "initial_capital_twd": params.initial_capital_twd,
            "max_entry_delay_minutes": params.max_entry_delay_minutes,
        },
        "rows": int(len(equity)),
        "start": format_timestamp(equity["timestamp"].iloc[0]),
        "end": format_timestamp(equity["timestamp"].iloc[-1]),
        "entry_allowed_minutes": int(data["entry_allowed"].sum()),
        "close_allowed_minutes": int(data["close_allowed"].sum()),
        "friday_night_close_only_minutes": int(data["friday_night_close_only"].sum()),
        "trade_count": trade_count,
        "winning_trades": wins,
        "losing_trades": int((trades["total_pnl"] < 0).sum()) if trade_count else 0,
        "win_rate": float(wins / trade_count) if trade_count else 0.0,
        "total_pnl_twd": total_pnl,
        "return_pct": float(total_pnl / params.initial_capital_twd),
        "gross_profit_twd": gross_profit,
        "gross_loss_twd": gross_loss,
        "profit_factor": (
            float(gross_profit / abs(gross_loss)) if gross_loss != 0 else None
        ),
        "avg_trade_pnl_twd": float(trades["total_pnl"].mean()) if trade_count else 0.0,
        "max_drawdown_twd": float(equity["drawdown_twd"].min()),
        "max_drawdown_pct": float(equity["drawdown_pct"].min()),
        "exposure_minutes": exposure_minutes,
        "exposure_ratio": float(exposure_minutes / len(equity)),
        "final_equity_twd": float(equity["equity"].iloc[-1]),
    }


def validate_backtest(
    data: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    params: BacktestParams,
) -> None:
    identity = (
        params.initial_capital_twd
        + equity["realized_pnl"]
        + equity["unrealized_pnl"]
        - equity["equity"]
    ).abs()
    if identity.max() > 1e-7:
        raise RuntimeError("Equity accounting identity failed")

    if trades.empty:
        return

    entry_allowed = data["entry_allowed"].to_numpy(dtype=bool)
    close_allowed = data["close_allowed"].to_numpy(dtype=bool)
    zvalid = data["zscore_valid"].to_numpy(dtype=bool)
    zscore = data["spread_zscore"].to_numpy(dtype=float)
    entry_obs = entry_allowed & zvalid
    next_entry_fill = compute_next_indices(entry_allowed)
    next_close_fill = compute_next_indices(close_allowed)

    for _, trade in trades.iterrows():
        entry_signal_idx = int(trade["entry_signal_idx"])
        entry_idx = int(trade["entry_idx"])
        exit_signal_idx = int(trade["exit_signal_idx"])
        exit_idx = int(trade["exit_idx"])
        direction = str(trade["direction"])

        if not entry_obs[entry_signal_idx]:
            raise RuntimeError(
                f"Entry signal where entry observation is false: {entry_signal_idx}"
            )
        if not entry_allowed[entry_idx]:
            raise RuntimeError(f"Trade entered where entry_allowed is false: {entry_idx}")
        if str(trade["exit_reason"]) != "end_of_data" and not close_allowed[exit_idx]:
            raise RuntimeError(f"Trade exited where close_allowed is false: {exit_idx}")
        if next_entry_fill[entry_signal_idx] != entry_idx:
            raise RuntimeError("Entry fill is not the next entry-allowed minute")
        if int(trade["entry_delay_minutes"]) > params.max_entry_delay_minutes:
            raise RuntimeError("Entry delay exceeded max_entry_delay_minutes")
        if not direction_still_valid(zscore[entry_signal_idx], direction, params.entry_z):
            raise RuntimeError("Entry signal z-score does not match trade direction")
        if str(trade["exit_reason"]) != "end_of_data":
            if next_close_fill[exit_signal_idx] != exit_idx:
                raise RuntimeError("Exit fill is not the next close-allowed minute")
            if not should_exit(zscore[exit_signal_idx], direction, params.exit_z):
                raise RuntimeError("Exit signal z-score does not match exit rule")

        expected_tsm_pnl = trade["tsm_units"] * (
            trade["exit_tsm_twd_fair"] - trade["entry_tsm_twd_fair"]
        )
        expected_qff_pnl = trade["qff_units"] * (
            trade["exit_qff_close"] - trade["entry_qff_close"]
        )
        if abs(expected_tsm_pnl - trade["tsm_pnl"]) > 1e-7:
            raise RuntimeError("TSM leg PnL validation failed")
        if abs(expected_qff_pnl - trade["qff_pnl"]) > 1e-7:
            raise RuntimeError("QFF leg PnL validation failed")
        if abs(expected_tsm_pnl + expected_qff_pnl - trade["total_pnl"]) > 1e-7:
            raise RuntimeError("Total trade PnL validation failed")


def run_self_tests() -> None:
    params = BacktestParams(
        entry_z=2.0,
        exit_z=0.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
    )

    stale = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=8, freq="min", tz=TAIPEI_TZ),
        zscores=[2.1, 1.3, -2.1, -2.2, -2.3, 0.5, 0.6, 0.7],
        tsm_start=100.0,
        qff_start=100.0,
    )
    stale_result = run_backtest(stale, params)
    if len(stale_result.trades) != 2:
        raise RuntimeError("Self-test failed: next-bar entry should create two trades")
    stale_trade = stale_result.trades.iloc[0]
    if stale_trade["direction"] != SHORT_TSM_LONG_QFF:
        raise RuntimeError("Self-test failed: first trade direction is wrong")
    if stale_trade["entry_signal_idx"] != 0 or stale_trade["entry_idx"] != 1:
        raise RuntimeError("Self-test failed: next entry-allowed fill timing is wrong")
    if stale_trade["entry_fill_zscore"] >= params.entry_z:
        raise RuntimeError("Self-test failed: stale fill case was not exercised")

    delay_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-08 13:45", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-08 17:25", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-08 17:26", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-08 17:27", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-08 17:28", tz=TAIPEI_TZ),
        ]
    )
    delay = make_synthetic_frame(
        delay_times,
        zscores=[2.1, 1.0, -2.2, 0.2, 0.3],
        tsm_start=100.0,
        qff_start=100.0,
    )
    delay_result = run_backtest(delay, params)
    if len(delay_result.trades) != 1:
        raise RuntimeError("Self-test failed: delayed signal should be cancelled")
    delay_trade = delay_result.trades.iloc[0]
    if delay_trade["entry_signal_idx"] != 2 or delay_trade["entry_idx"] != 3:
        raise RuntimeError("Self-test failed: wrong trade after delay cancellation")

    friday_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-12 13:41", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 13:42", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 13:43", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:25", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:26", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:27", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:28", tz=TAIPEI_TZ),
        ]
    )
    friday = make_synthetic_frame(
        friday_times,
        zscores=[2.1, 2.2, 2.3, -0.1, -0.2, 2.5, 2.6],
        tsm_start=100.0,
        qff_start=100.0,
    )
    friday_result = run_backtest(friday, params)
    if len(friday_result.trades) != 1:
        raise RuntimeError("Self-test failed: Friday night should not open a new trade")
    friday_trade = friday_result.trades.iloc[0]
    if friday_trade["entry_idx"] != 1 or friday_trade["exit_idx"] != 4:
        raise RuntimeError("Self-test failed: Friday night close-only timing is wrong")
    if friday_result.equity["entry_allowed"].iloc[5]:
        raise RuntimeError("Self-test failed: Friday night entry should be disallowed")


def make_synthetic_frame(
    timestamps: pd.DatetimeIndex,
    zscores: list[float],
    tsm_start: float,
    qff_start: float,
) -> pd.DataFrame:
    if len(timestamps) != len(zscores):
        raise ValueError("timestamps and zscores must have the same length")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "qff_close": np.full(len(timestamps), qff_start),
            "qff_close_filled": np.full(len(timestamps), qff_start),
            "tsm_twd_fair": tsm_start + np.arange(len(timestamps), dtype=float),
            "spread_zscore": zscores,
            "zscore_valid": True,
        }
    )


def format_timestamp(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M:%S%z").replace("+0800", "+08:00")


def format_timestamp_column(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
        r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True
    )


def write_outputs(result: BacktestResult, equity_path: Path, trades_path: Path, summary_path: Path) -> None:
    equity_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    equity = result.equity.copy()
    equity["timestamp"] = format_timestamp_column(equity["timestamp"])
    equity.to_csv(equity_path, index=False)

    trades = result.trades.copy()
    timestamp_columns = [
        "entry_signal_time",
        "entry_time",
        "exit_signal_time",
        "exit_time",
    ]
    for column in timestamp_columns:
        if column in trades.columns and not trades.empty:
            trades[column] = format_timestamp_column(pd.to_datetime(trades[column]))
    trades.to_csv(trades_path, index=False)

    summary_path.write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_self_test:
        run_self_tests()
        print("Self-tests passed")

    params = BacktestParams(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        leg_notional_twd=args.leg_notional_twd,
        initial_capital_twd=args.initial_capital_twd,
        max_entry_delay_minutes=args.max_entry_delay_minutes,
    )
    frame = read_input_frame(args.input)
    result = run_backtest(frame, params)
    write_outputs(result, args.equity_out, args.trades_out, args.summary_out)

    print(f"Wrote equity curve to {args.equity_out}")
    print(f"Wrote trades to {args.trades_out}")
    print(f"Wrote summary to {args.summary_out}")
    print(
        "Summary: "
        f"trades={result.summary['trade_count']}, "
        f"total_pnl={result.summary['total_pnl_twd']:.2f}, "
        f"return={result.summary['return_pct']:.4%}, "
        f"max_dd={result.summary['max_drawdown_twd']:.2f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
