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
DEFAULT_INPUT_PATH = Path(
    "data/processed/qff_tsm_spread_zscore_1m_taipei_qff_session.csv"
)
DEFAULT_QFF_OHLCV_PATH = Path("data/processed/qff1_1m.csv")
DEFAULT_TSM_OHLCV_PATH = Path("data/processed/binance_tsmusdtp_1m_taipei.csv")
DEFAULT_USDTTWD_OHLCV_PATH = Path("data/processed/bitopro_usdttwd_1m_taipei.csv")
DEFAULT_EQUITY_PATH = Path(
    "data/processed/qff_tsm_pair_backtest_equity_1m_qff_session.csv"
)
DEFAULT_TRADES_PATH = Path("data/processed/qff_tsm_pair_backtest_trades_qff_session.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/qff_tsm_pair_backtest_summary_qff_session.json")
FEE_DEFAULTS_AS_OF = "2026-06-17"
DEFAULT_TSM_FEE_BPS = 5.0
DEFAULT_QFF_FEE_PER_CONTRACT_TWD = 5.0
DEFAULT_QFF_TAX_RATE = 0.00002
DEFAULT_QFF_CONTRACT_MULTIPLIER = 100.0

SHORT_TSM_LONG_QFF = "short_tsm_long_qff"
LONG_TSM_SHORT_QFF = "long_tsm_short_qff"
FLAT = "flat"
ENTRY_PENDING = "entry_pending_fill"
EXIT_PENDING = "exit_pending_fill"
FORCED_CLOSED = "forced_closed_end_of_data"
FRIDAY_SESSION_CLOSED = "forced_closed_friday_session_end"
FRIDAY_SESSION_END = "friday_session_end"
TIME_STOP = "time_stop"

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
    tsm_fee_bps: float
    qff_fee_per_contract_twd: float
    qff_tax_rate: float
    qff_contract_multiplier: float
    max_holding_minutes: int = 0
    entry_z_max: float = 0.0
    max_entry_vol_ratio: float = 0.0
    persistence_k: int = 1


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class PositionSizing:
    tsm_units: float
    qff_units: float
    qff_contracts: int
    raw_qff_contracts: float
    actual_leg_notional_twd: float


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest a simple QFF/TSM pairs strategy using next "
            "entry-allowed-minute open fills with a maximum entry delay."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--qff-ohlcv", type=Path, default=DEFAULT_QFF_OHLCV_PATH)
    parser.add_argument("--tsm-ohlcv", type=Path, default=DEFAULT_TSM_OHLCV_PATH)
    parser.add_argument("--usdttwd-ohlcv", type=Path, default=DEFAULT_USDTTWD_OHLCV_PATH)
    parser.add_argument("--equity-out", type=Path, default=DEFAULT_EQUITY_PATH)
    parser.add_argument("--trades-out", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument(
        "--entry-z-max",
        type=float,
        default=0.0,
        help=(
            "Upper |z| bound for entries. Only enter when entry_z < |z| <= "
            "entry_z_max. 0 disables the upper bound (no band cap)."
        ),
    )
    parser.add_argument(
        "--max-entry-vol-ratio",
        type=float,
        default=0.0,
        help=(
            "Skip entries when the conditional-vol gate column entry_vol_ratio "
            "exceeds this value (EWMA spread-change vol over its trailing "
            "baseline). 0 disables the gate. Requires the entry_vol_ratio "
            "column (see scripts/add_entry_vol_ratio.py)."
        ),
    )
    parser.add_argument(
        "--persistence-k",
        type=int,
        default=1,
        help=(
            "Require |z| > entry_z in the same direction for k consecutive "
            "entry-eligible observations before an entry signal fires. 1 "
            "disables the persistence filter."
        ),
    )
    parser.add_argument("--exit-z", type=float, default=0.0)
    parser.add_argument("--leg-notional-twd", type=float, default=1_000_000.0)
    parser.add_argument("--initial-capital-twd", type=float, default=2_000_000.0)
    parser.add_argument("--max-entry-delay-minutes", type=int, default=15)
    parser.add_argument(
        "--max-holding-minutes",
        type=int,
        default=0,
        help=(
            "Force-close a position at the next close-allowed bar once it has "
            "been held at least this many minutes. 0 disables the time stop."
        ),
    )
    parser.add_argument(
        "--tsm-fee-bps",
        type=float,
        default=DEFAULT_TSM_FEE_BPS,
        help=(
            "TSMUSDT.P one-way trading fee in bps. Default is Binance "
            "TradFi Perps taker-style cost, without BNB discount."
        ),
    )
    parser.add_argument(
        "--qff-fee-per-contract-twd",
        type=float,
        default=DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        help=(
            "QFF one-way fixed fee in TWD per contract. Default is TAIFEX "
            "single stock futures transaction plus clearing baseline."
        ),
    )
    parser.add_argument(
        "--qff-tax-rate",
        type=float,
        default=DEFAULT_QFF_TAX_RATE,
        help="QFF futures transaction tax rate applied per fill.",
    )
    parser.add_argument(
        "--qff-contract-multiplier",
        type=float,
        default=DEFAULT_QFF_CONTRACT_MULTIPLIER,
        help="QFF contract multiplier. Default is 100 shares per contract.",
    )
    parser.add_argument("--skip-self-test", action="store_true")
    return parser.parse_args(argv)


def read_open_series(path: Path, name: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"OHLCV CSV does not exist: {path}")

    frame = pd.read_csv(path)
    missing = {"timestamp", "open"}.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise RuntimeError(f"OHLCV CSV has no rows: {path}")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(TAIPEI_TZ)
    open_price = pd.to_numeric(frame["open"], errors="coerce")
    if open_price.isna().any():
        bad_rows = frame.loc[open_price.isna(), ["timestamp", "open"]].head(10)
        raise RuntimeError(f"{path} has invalid open values:\n{bad_rows}")

    series = pd.Series(
        open_price.to_numpy(), index=pd.DatetimeIndex(timestamps), name=name
    )
    if series.index.has_duplicates:
        duplicates = series.index[series.index.duplicated()].unique()
        raise RuntimeError(
            f"{path} has duplicate timestamps, first examples: {list(duplicates[:5])}"
        )
    return series.sort_index()


def add_entry_open_prices(
    frame: pd.DataFrame,
    qff_ohlcv_path: Path,
    tsm_ohlcv_path: Path,
    usdttwd_ohlcv_path: Path,
) -> pd.DataFrame:
    output = frame.copy()
    full_index = pd.DatetimeIndex(output["timestamp"])

    qff_open = read_open_series(qff_ohlcv_path, "qff_open").reindex(full_index)
    tsm_open = read_open_series(tsm_ohlcv_path, "tsm_open").reindex(full_index)
    usdttwd_open = read_open_series(usdttwd_ohlcv_path, "usdttwd_open").reindex(
        full_index
    )

    external_missing = {
        "tsm_open": int(tsm_open.isna().sum()),
        "usdttwd_open": int(usdttwd_open.isna().sum()),
    }
    bad_external = {key: value for key, value in external_missing.items() if value}
    if bad_external:
        raise RuntimeError(f"External open series has missing values: {bad_external}")

    qff_close_filled = pd.Series(
        output["qff_close_filled"].to_numpy(), index=full_index
    )
    output["qff_entry_open"] = qff_open.to_numpy()
    output["qff_entry_open_filled"] = qff_open.fillna(qff_close_filled).to_numpy()
    output["qff_entry_open_was_filled"] = qff_open.isna().to_numpy()
    output["tsm_open"] = tsm_open.to_numpy()
    output["usdttwd_open"] = usdttwd_open.to_numpy()
    output["tsm_twd_fair_open"] = (
        output["tsm_open"] * output["usdttwd_open"] / 5
    )
    return output


def read_input_frame(
    path: Path,
    qff_ohlcv_path: Path,
    tsm_ohlcv_path: Path,
    usdttwd_ohlcv_path: Path,
) -> pd.DataFrame:
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
    if "entry_vol_ratio" in frame.columns:
        frame["entry_vol_ratio"] = pd.to_numeric(
            frame["entry_vol_ratio"], errors="coerce"
        )
    else:
        frame["entry_vol_ratio"] = 1.0
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise RuntimeError("Input timestamps must be unique and sorted")
    frame = add_entry_open_prices(
        frame,
        qff_ohlcv_path=qff_ohlcv_path,
        tsm_ohlcv_path=tsm_ohlcv_path,
        usdttwd_ohlcv_path=usdttwd_ohlcv_path,
    )
    if frame["qff_entry_open_filled"].isna().any():
        raise RuntimeError("qff_entry_open_filled contains missing values")
    if frame["tsm_twd_fair_open"].isna().any():
        raise RuntimeError("tsm_twd_fair_open contains missing values")
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

    session_kind = np.where(night_clock, "N", "D")
    session_key = pd.Series(session_kind, index=output.index).str.cat(
        session_start.dt.strftime("%Y-%m-%d"), sep=":"
    )
    friday_night = night_allowed & session_start.dt.weekday.eq(4).to_numpy()
    week_end_force_close = compute_week_end_force_close(close_allowed, ts)
    weekend_session_close_only = compute_session_containing_marker(
        close_allowed, session_key, week_end_force_close
    )
    close_only = friday_night | weekend_session_close_only
    entry_allowed = close_allowed & ~close_only

    output["close_allowed"] = close_allowed
    output["entry_allowed"] = entry_allowed
    output["friday_night_close_only"] = friday_night
    output["weekend_session_close_only"] = weekend_session_close_only
    output["friday_session_end_force_close"] = week_end_force_close
    return output


def compute_week_end_force_close(
    close_allowed: np.ndarray, timestamps: pd.DatetimeIndex
) -> np.ndarray:
    force_close = np.zeros(len(close_allowed), dtype=bool)
    close_indices = np.flatnonzero(close_allowed)
    if len(close_indices) < 2:
        return force_close

    for current_idx, next_idx in zip(close_indices[:-1], close_indices[1:]):
        current_iso = timestamps[current_idx].isocalendar()
        next_iso = timestamps[next_idx].isocalendar()
        if (current_iso.year, current_iso.week) != (next_iso.year, next_iso.week):
            force_close[current_idx] = True
    return force_close


def compute_session_containing_marker(
    close_allowed: np.ndarray, session_key: pd.Series, marker: np.ndarray
) -> np.ndarray:
    marker_keys = set(session_key.loc[marker].dropna())
    if not marker_keys:
        return np.zeros(len(close_allowed), dtype=bool)
    return close_allowed & session_key.isin(marker_keys).to_numpy()


def compute_next_indices(mask: np.ndarray) -> np.ndarray:
    next_indices = np.full(len(mask), -1, dtype=np.int64)
    next_seen = -1
    for index in range(len(mask) - 1, -1, -1):
        next_indices[index] = next_seen
        if mask[index]:
            next_seen = index
    return next_indices


def entry_direction(
    zscore: float, entry_z: float, entry_z_max: float = 0.0
) -> str | None:
    if entry_z_max > 0.0 and abs(zscore) > entry_z_max:
        return None
    if zscore > entry_z:
        return SHORT_TSM_LONG_QFF
    if zscore < -entry_z:
        return LONG_TSM_SHORT_QFF
    return None


def direction_still_valid(
    zscore: float, direction: str, entry_z: float, entry_z_max: float = 0.0
) -> bool:
    if entry_z_max > 0.0 and abs(zscore) > entry_z_max:
        return False
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


def compute_persistence_count(
    entry_observation: np.ndarray, zscore: np.ndarray, entry_z: float
) -> np.ndarray:
    """For each bar, the run length of consecutive entry-eligible observations
    (skipping non-eligible bars) whose z-score has stayed beyond entry_z in the
    same direction. Used by the persistence filter to require k consecutive
    over-threshold observations before an entry signal fires."""
    counts = np.zeros(len(zscore), dtype=np.int64)
    streak = 0
    last_dir = 0
    for index in range(len(zscore)):
        if not entry_observation[index]:
            continue
        if zscore[index] > entry_z:
            direction = 1
        elif zscore[index] < -entry_z:
            direction = -1
        else:
            direction = 0
        if direction == 0:
            streak = 0
            last_dir = 0
        elif direction == last_dir:
            streak += 1
        else:
            streak = 1
            last_dir = direction
        counts[index] = streak
    return counts


def round_half_up_nonnegative(value: float) -> int:
    if value < 0:
        raise ValueError(f"Expected a non-negative value, got {value}")
    return int(np.floor(value + 0.5))


def size_position_for_direction(
    direction: str, tsm_price: float, qff_price: float, params: BacktestParams
) -> PositionSizing | None:
    raw_qff_contracts = params.leg_notional_twd / (
        qff_price * params.qff_contract_multiplier
    )
    qff_contract_count = round_half_up_nonnegative(raw_qff_contracts)
    if qff_contract_count == 0:
        return None

    actual_leg_notional_twd = (
        qff_contract_count * params.qff_contract_multiplier * qff_price
    )
    tsm_units = actual_leg_notional_twd / tsm_price
    qff_units = qff_contract_count * params.qff_contract_multiplier
    if direction == SHORT_TSM_LONG_QFF:
        return PositionSizing(
            tsm_units=-tsm_units,
            qff_units=qff_units,
            qff_contracts=qff_contract_count,
            raw_qff_contracts=raw_qff_contracts,
            actual_leg_notional_twd=actual_leg_notional_twd,
        )
    if direction == LONG_TSM_SHORT_QFF:
        return PositionSizing(
            tsm_units=tsm_units,
            qff_units=-qff_units,
            qff_contracts=-qff_contract_count,
            raw_qff_contracts=raw_qff_contracts,
            actual_leg_notional_twd=actual_leg_notional_twd,
        )
    raise ValueError(f"Unknown direction: {direction}")


def minutes_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start).total_seconds() // 60)


def validate_params(params: BacktestParams) -> None:
    if params.leg_notional_twd <= 0:
        raise RuntimeError("leg_notional_twd must be positive")
    if params.initial_capital_twd <= 0:
        raise RuntimeError("initial_capital_twd must be positive")
    if params.max_entry_delay_minutes < 0:
        raise RuntimeError("max_entry_delay_minutes must be non-negative")
    if params.max_holding_minutes < 0:
        raise RuntimeError("max_holding_minutes must be non-negative")
    if params.entry_z_max < 0:
        raise RuntimeError("entry_z_max must be non-negative")
    if params.entry_z_max > 0 and params.entry_z_max <= params.entry_z:
        raise RuntimeError("entry_z_max must be greater than entry_z when set")
    if params.max_entry_vol_ratio < 0:
        raise RuntimeError("max_entry_vol_ratio must be non-negative")
    if params.persistence_k < 1:
        raise RuntimeError("persistence_k must be at least 1")
    if params.tsm_fee_bps < 0:
        raise RuntimeError("tsm_fee_bps must be non-negative")
    if params.qff_fee_per_contract_twd < 0:
        raise RuntimeError("qff_fee_per_contract_twd must be non-negative")
    if params.qff_tax_rate < 0:
        raise RuntimeError("qff_tax_rate must be non-negative")
    if params.qff_contract_multiplier <= 0:
        raise RuntimeError("qff_contract_multiplier must be positive")


def fill_costs(
    *,
    tsm_units: float,
    tsm_price: float,
    qff_contracts: int,
    qff_price: float,
    params: BacktestParams,
) -> dict[str, float]:
    tsm_fee_twd = abs(tsm_units) * tsm_price * params.tsm_fee_bps / 10000.0
    qff_fee_twd = abs(qff_contracts) * params.qff_fee_per_contract_twd
    qff_tax_per_contract_twd = round_half_up_nonnegative(
        qff_price * params.qff_contract_multiplier * params.qff_tax_rate
    )
    qff_tax_twd = abs(qff_contracts) * qff_tax_per_contract_twd
    total_fee_twd = tsm_fee_twd + qff_fee_twd + qff_tax_twd
    return {
        "tsm_fee_twd": tsm_fee_twd,
        "qff_fee_twd": qff_fee_twd,
        "qff_tax_twd": qff_tax_twd,
        "total_fee_twd": total_fee_twd,
    }


def run_backtest(frame: pd.DataFrame, params: BacktestParams) -> BacktestResult:
    validate_params(params)
    data = add_trading_masks(frame)
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
    entry_vol_ratio = (
        data["entry_vol_ratio"].to_numpy(dtype=float)
        if "entry_vol_ratio" in data.columns
        else np.ones(n_rows)
    )
    entry_allowed = data["entry_allowed"].to_numpy(dtype=bool)
    close_allowed = data["close_allowed"].to_numpy(dtype=bool)
    friday_night = data["friday_night_close_only"].to_numpy(dtype=bool)
    weekend_session_close_only = data["weekend_session_close_only"].to_numpy(dtype=bool)
    friday_session_end = data["friday_session_end_force_close"].to_numpy(dtype=bool)

    entry_observation = entry_allowed & zvalid
    close_observation = close_allowed & zvalid
    next_entry_fill = compute_next_indices(entry_allowed)
    next_close_fill = compute_next_indices(close_allowed)
    persistence_count = compute_persistence_count(
        entry_observation, zscore, params.entry_z
    )

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
    qff_contracts = 0
    actual_leg_notional_twd = 0.0
    realized_pnl = 0.0
    realized_fee_twd = 0.0
    open_trade: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []

    state_out: list[str] = []
    position_out: list[str] = []
    tsm_units_out = np.zeros(n_rows)
    qff_units_out = np.zeros(n_rows)
    qff_contracts_out = np.zeros(n_rows, dtype=np.int64)
    actual_leg_notional_out = np.zeros(n_rows)
    realized_out = np.zeros(n_rows)
    realized_fee_out = np.zeros(n_rows)
    unrealized_out = np.zeros(n_rows)
    equity_out = np.zeros(n_rows)

    for index in range(n_rows):
        filled_this_bar = False

        if state == ENTRY_PENDING and index == entry_fill_idx:
            position_direction = candidate_direction
            if position_direction is None:
                raise RuntimeError("Entry pending without a direction")
            entry_tsm = tsm_entry[index]
            entry_qff = qff_entry[index]
            entry_zscore = zscore[index]
            sizing = size_position_for_direction(
                position_direction, entry_tsm, entry_qff, params
            )
            if sizing is None:
                state = FLAT
                position_direction = None
                candidate_direction = None
                candidate_idx = -1
                candidate_zscore = np.nan
                entry_fill_idx = -1
                filled_this_bar = True
            else:
                tsm_units = sizing.tsm_units
                qff_units = sizing.qff_units
                qff_contracts = sizing.qff_contracts
                actual_leg_notional_twd = sizing.actual_leg_notional_twd
                entry_costs = fill_costs(
                    tsm_units=tsm_units,
                    tsm_price=entry_tsm,
                    qff_contracts=qff_contracts,
                    qff_price=entry_qff,
                    params=params,
                )
                realized_pnl -= entry_costs["total_fee_twd"]
                realized_fee_twd += entry_costs["total_fee_twd"]
                open_trade = {
                    "entry_signal_idx": candidate_idx,
                    "entry_signal_time": timestamps[candidate_idx],
                    "entry_signal_zscore": candidate_zscore,
                    "entry_idx": index,
                    "entry_time": timestamps[index],
                    "entry_fill_price_type": "open",
                    "entry_delay_minutes": minutes_between(
                        timestamps[candidate_idx], timestamps[index]
                    ),
                    "entry_fill_zscore": entry_zscore,
                    "direction": position_direction,
                    "entry_tsm_twd_fair": entry_tsm,
                    "entry_tsm_twd_fair_open": entry_tsm,
                    "entry_qff_close": entry_qff,
                    "entry_qff_open_filled": entry_qff,
                    "entry_qff_open_was_filled": bool(qff_entry_open_was_filled[index]),
                    "tsm_units": tsm_units,
                    "qff_units": qff_units,
                    "qff_contracts": qff_contracts,
                    "raw_qff_contracts": sizing.raw_qff_contracts,
                    "leg_notional_twd": params.leg_notional_twd,
                    "actual_leg_notional_twd": actual_leg_notional_twd,
                    "qff_contract_multiplier": params.qff_contract_multiplier,
                    "entry_tsm_fee_twd": entry_costs["tsm_fee_twd"],
                    "entry_qff_fee_twd": entry_costs["qff_fee_twd"],
                    "entry_qff_tax_twd": entry_costs["qff_tax_twd"],
                    "entry_fee_twd": entry_costs["total_fee_twd"],
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
                params=params,
            )
            realized_fee_twd += trades[-1]["exit_fee_twd"]
            state = FLAT
            position_direction = None
            open_trade = None
            tsm_units = 0.0
            qff_units = 0.0
            qff_contracts = 0
            actual_leg_notional_twd = 0.0
            entry_tsm = np.nan
            entry_qff = np.nan
            entry_zscore = np.nan
            exit_signal_idx = -1
            exit_fill_idx = -1
            filled_this_bar = True

        if (
            not filled_this_bar
            and position_direction is not None
            and open_trade is not None
            and friday_session_end[index]
        ):
            realized_pnl += close_open_trade(
                trades,
                open_trade,
                exit_idx=index,
                exit_time=timestamps[index],
                exit_tsm=tsm[index],
                exit_qff=qff[index],
                exit_zscore=zscore[index],
                exit_signal_idx=index,
                exit_signal_time=timestamps[index],
                exit_signal_zscore=zscore[index],
                exit_reason=FRIDAY_SESSION_END,
                params=params,
            )
            realized_fee_twd += trades[-1]["exit_fee_twd"]
            state = FLAT
            position_direction = None
            open_trade = None
            tsm_units = 0.0
            qff_units = 0.0
            qff_contracts = 0
            actual_leg_notional_twd = 0.0
            entry_tsm = np.nan
            entry_qff = np.nan
            entry_zscore = np.nan
            exit_signal_idx = -1
            exit_fill_idx = -1
            filled_this_bar = True

        if (
            not filled_this_bar
            and position_direction is not None
            and open_trade is not None
            and params.max_holding_minutes > 0
            and close_allowed[index]
            and minutes_between(open_trade["entry_time"], timestamps[index])
            >= params.max_holding_minutes
        ):
            realized_pnl += close_open_trade(
                trades,
                open_trade,
                exit_idx=index,
                exit_time=timestamps[index],
                exit_tsm=tsm[index],
                exit_qff=qff[index],
                exit_zscore=zscore[index],
                exit_signal_idx=index,
                exit_signal_time=timestamps[index],
                exit_signal_zscore=zscore[index],
                exit_reason=TIME_STOP,
                params=params,
            )
            realized_fee_twd += trades[-1]["exit_fee_twd"]
            state = FLAT
            position_direction = None
            open_trade = None
            tsm_units = 0.0
            qff_units = 0.0
            qff_contracts = 0
            actual_leg_notional_twd = 0.0
            entry_tsm = np.nan
            entry_qff = np.nan
            entry_zscore = np.nan
            exit_signal_idx = -1
            exit_fill_idx = -1
            filled_this_bar = True

        if not filled_this_bar:
            if state == FLAT and entry_observation[index]:
                direction = entry_direction(
                    zscore[index], params.entry_z, params.entry_z_max
                )
                if (
                    direction is not None
                    and params.max_entry_vol_ratio > 0.0
                    and not np.isnan(entry_vol_ratio[index])
                    and entry_vol_ratio[index] > params.max_entry_vol_ratio
                ):
                    direction = None
                if (
                    direction is not None
                    and params.persistence_k > 1
                    and persistence_count[index] < params.persistence_k
                ):
                    direction = None
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
        qff_contracts_out[index] = qff_contracts
        actual_leg_notional_out[index] = actual_leg_notional_twd
        realized_out[index] = realized_pnl
        realized_fee_out[index] = realized_fee_twd
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
            params=params,
        )
        realized_fee_twd += trades[-1]["exit_fee_twd"]
        state_out[-1] = FORCED_CLOSED
        position_out[-1] = FLAT
        tsm_units_out[-1] = 0.0
        qff_units_out[-1] = 0.0
        qff_contracts_out[-1] = 0
        actual_leg_notional_out[-1] = 0.0
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
    params: BacktestParams,
) -> float:
    tsm_pnl = open_trade["tsm_units"] * (exit_tsm - open_trade["entry_tsm_twd_fair"])
    qff_pnl = open_trade["qff_units"] * (exit_qff - open_trade["entry_qff_close"])
    gross_pnl = tsm_pnl + qff_pnl
    exit_costs = fill_costs(
        tsm_units=open_trade["tsm_units"],
        tsm_price=exit_tsm,
        qff_contracts=int(open_trade["qff_contracts"]),
        qff_price=exit_qff,
        params=params,
    )
    tsm_fee_twd = open_trade["entry_tsm_fee_twd"] + exit_costs["tsm_fee_twd"]
    qff_fee_twd = open_trade["entry_qff_fee_twd"] + exit_costs["qff_fee_twd"]
    qff_tax_twd = open_trade["entry_qff_tax_twd"] + exit_costs["qff_tax_twd"]
    total_fee_twd = open_trade["entry_fee_twd"] + exit_costs["total_fee_twd"]
    net_pnl = gross_pnl - total_fee_twd
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
        "gross_pnl_twd": gross_pnl,
        "exit_tsm_fee_twd": exit_costs["tsm_fee_twd"],
        "exit_qff_fee_twd": exit_costs["qff_fee_twd"],
        "exit_qff_tax_twd": exit_costs["qff_tax_twd"],
        "exit_fee_twd": exit_costs["total_fee_twd"],
        "tsm_fee_twd": tsm_fee_twd,
        "qff_fee_twd": qff_fee_twd,
        "qff_tax_twd": qff_tax_twd,
        "total_fee_twd": total_fee_twd,
        "net_pnl_twd": net_pnl,
        "total_pnl": net_pnl,
        "exit_reason": exit_reason,
        "holding_minutes": minutes_between(open_trade["entry_time"], exit_time),
    }
    trades.append(trade)
    return gross_pnl - exit_costs["total_fee_twd"]


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
    exposure_minutes = (
        int(trades["holding_minutes"].sum()) if trade_count else 0
    )
    elapsed_minutes = max(
        minutes_between(equity["timestamp"].iloc[0], equity["timestamp"].iloc[-1]), 1
    )
    gross_pnl = float(trades["gross_pnl_twd"].sum()) if trade_count else 0.0
    net_pnl = float(trades["net_pnl_twd"].sum()) if trade_count else 0.0
    total_fee = float(trades["total_fee_twd"].sum()) if trade_count else 0.0
    total_tsm_fee = float(trades["tsm_fee_twd"].sum()) if trade_count else 0.0
    total_qff_fee = float(trades["qff_fee_twd"].sum()) if trade_count else 0.0
    total_qff_tax = float(trades["qff_tax_twd"].sum()) if trade_count else 0.0
    return {
        "fee_defaults_as_of": FEE_DEFAULTS_AS_OF,
        "parameters": {
            "entry_z": params.entry_z,
            "entry_z_max": params.entry_z_max,
            "exit_z": params.exit_z,
            "max_holding_minutes": params.max_holding_minutes,
            "max_entry_vol_ratio": params.max_entry_vol_ratio,
            "persistence_k": params.persistence_k,
            "leg_notional_twd": params.leg_notional_twd,
            "initial_capital_twd": params.initial_capital_twd,
            "max_entry_delay_minutes": params.max_entry_delay_minutes,
            "tsm_fee_bps": params.tsm_fee_bps,
            "qff_fee_per_contract_twd": params.qff_fee_per_contract_twd,
            "qff_tax_rate": params.qff_tax_rate,
            "qff_contract_multiplier": params.qff_contract_multiplier,
            "entry_fill_price": "next_entry_allowed_open",
            "exit_fill_price": "next_close_allowed_close",
        },
        "rows": int(len(equity)),
        "start": format_timestamp(equity["timestamp"].iloc[0]),
        "end": format_timestamp(equity["timestamp"].iloc[-1]),
        "entry_allowed_minutes": int(data["entry_allowed"].sum()),
        "close_allowed_minutes": int(data["close_allowed"].sum()),
        "friday_night_close_only_minutes": int(data["friday_night_close_only"].sum()),
        "weekend_session_close_only_minutes": int(
            data["weekend_session_close_only"].sum()
        ),
        "friday_session_end_force_close_minutes": int(
            data["friday_session_end_force_close"].sum()
        ),
        "trade_count": trade_count,
        "friday_session_forced_exits": (
            int((trades["exit_reason"] == FRIDAY_SESSION_END).sum())
            if trade_count
            else 0
        ),
        "time_stop_exits": (
            int((trades["exit_reason"] == TIME_STOP).sum()) if trade_count else 0
        ),
        "winning_trades": wins,
        "losing_trades": int((trades["total_pnl"] < 0).sum()) if trade_count else 0,
        "win_rate": float(wins / trade_count) if trade_count else 0.0,
        "total_pnl_twd": total_pnl,
        "gross_pnl_twd": gross_pnl,
        "net_pnl_twd": net_pnl,
        "total_fee_twd": total_fee,
        "total_tsm_fee_twd": total_tsm_fee,
        "total_qff_fee_twd": total_qff_fee,
        "total_qff_tax_twd": total_qff_tax,
        "return_pct": float(total_pnl / params.initial_capital_twd),
        "gross_profit_twd": gross_profit,
        "gross_loss_twd": gross_loss,
        "profit_factor": (
            float(gross_profit / abs(gross_loss)) if gross_loss != 0 else None
        ),
        "avg_trade_pnl_twd": float(trades["total_pnl"].mean()) if trade_count else 0.0,
        "max_drawdown_twd": float(equity["drawdown_twd"].min()),
        "max_drawdown_pct": float(equity["drawdown_pct"].min()),
        "elapsed_minutes": elapsed_minutes,
        "exposure_minutes": exposure_minutes,
        "exposure_ratio": float(exposure_minutes / elapsed_minutes),
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
    weekend_session_close_only = data["weekend_session_close_only"].to_numpy(dtype=bool)
    friday_session_end = data["friday_session_end_force_close"].to_numpy(dtype=bool)
    zvalid = data["zscore_valid"].to_numpy(dtype=bool)
    zscore = data["spread_zscore"].to_numpy(dtype=float)
    vol_ratio = (
        data["entry_vol_ratio"].to_numpy(dtype=float)
        if "entry_vol_ratio" in data.columns
        else np.ones(len(data))
    )
    entry_obs = entry_allowed & zvalid
    next_entry_fill = compute_next_indices(entry_allowed)
    next_close_fill = compute_next_indices(close_allowed)

    if np.any(entry_allowed & weekend_session_close_only):
        raise RuntimeError("Weekend close-only session should not allow entries")

    for _, trade in trades.iterrows():
        entry_signal_idx = int(trade["entry_signal_idx"])
        entry_idx = int(trade["entry_idx"])
        exit_signal_idx = int(trade["exit_signal_idx"])
        exit_idx = int(trade["exit_idx"])
        direction = str(trade["direction"])
        qff_contracts = int(trade["qff_contracts"])
        exit_reason = str(trade["exit_reason"])
        expected_holding_minutes = minutes_between(
            trade["entry_time"], trade["exit_time"]
        )

        if not entry_obs[entry_signal_idx]:
            raise RuntimeError(
                f"Entry signal where entry observation is false: {entry_signal_idx}"
            )
        if int(trade["holding_minutes"]) != expected_holding_minutes:
            raise RuntimeError("Holding minutes should use timestamp elapsed time")
        if not entry_allowed[entry_idx]:
            raise RuntimeError(f"Trade entered where entry_allowed is false: {entry_idx}")
        if exit_reason != "end_of_data" and not close_allowed[exit_idx]:
            raise RuntimeError(f"Trade exited where close_allowed is false: {exit_idx}")
        if next_entry_fill[entry_signal_idx] != entry_idx:
            raise RuntimeError("Entry fill is not the next entry-allowed minute")
        if int(trade["entry_delay_minutes"]) > params.max_entry_delay_minutes:
            raise RuntimeError("Entry delay exceeded max_entry_delay_minutes")
        if not direction_still_valid(
            zscore[entry_signal_idx], direction, params.entry_z, params.entry_z_max
        ):
            raise RuntimeError("Entry signal z-score does not match trade direction")
        if params.max_entry_vol_ratio > 0:
            signal_ratio = vol_ratio[entry_signal_idx]
            if not np.isnan(signal_ratio) and signal_ratio > params.max_entry_vol_ratio:
                raise RuntimeError("Entry taken where the vol gate should have blocked it")
        if exit_reason == "zscore_exit":
            if next_close_fill[exit_signal_idx] != exit_idx:
                raise RuntimeError("Exit fill is not the next close-allowed minute")
            if not should_exit(zscore[exit_signal_idx], direction, params.exit_z):
                raise RuntimeError("Exit signal z-score does not match exit rule")
        elif exit_reason == FRIDAY_SESSION_END:
            if exit_signal_idx != exit_idx:
                raise RuntimeError("Friday forced exit should use the exit bar as signal")
            if not friday_session_end[exit_idx]:
                raise RuntimeError("Friday forced exit is not at session end")
        elif exit_reason == TIME_STOP:
            if exit_signal_idx != exit_idx:
                raise RuntimeError("Time stop should use the exit bar as the signal")
            if not close_allowed[exit_idx]:
                raise RuntimeError("Time stop exit is not at a close-allowed bar")
            if int(trade["holding_minutes"]) < params.max_holding_minutes:
                raise RuntimeError("Time stop fired before max_holding_minutes")
        elif exit_reason != "end_of_data":
            raise RuntimeError(f"Unknown exit reason: {exit_reason}")

        expected_contracts = round_half_up_nonnegative(
            params.leg_notional_twd
            / (trade["entry_qff_close"] * params.qff_contract_multiplier)
        )
        if abs(qff_contracts) != expected_contracts:
            raise RuntimeError("QFF contract rounding validation failed")
        expected_qff_units = qff_contracts * params.qff_contract_multiplier
        if abs(expected_qff_units - trade["qff_units"]) > 1e-7:
            raise RuntimeError("QFF units validation failed")
        expected_actual_notional = (
            abs(qff_contracts)
            * params.qff_contract_multiplier
            * trade["entry_qff_close"]
        )
        if abs(expected_actual_notional - trade["actual_leg_notional_twd"]) > 1e-7:
            raise RuntimeError("Actual leg notional validation failed")
        expected_tsm_entry_notional = (
            abs(trade["tsm_units"]) * trade["entry_tsm_twd_fair"]
        )
        if abs(expected_tsm_entry_notional - trade["actual_leg_notional_twd"]) > 1e-7:
            raise RuntimeError("TSM notional does not match rounded QFF notional")
        if direction == SHORT_TSM_LONG_QFF and not (
            trade["tsm_units"] < 0 and qff_contracts > 0
        ):
            raise RuntimeError("Short TSM / long QFF sign validation failed")
        if direction == LONG_TSM_SHORT_QFF and not (
            trade["tsm_units"] > 0 and qff_contracts < 0
        ):
            raise RuntimeError("Long TSM / short QFF sign validation failed")

        expected_tsm_pnl = trade["tsm_units"] * (
            trade["exit_tsm_twd_fair"] - trade["entry_tsm_twd_fair"]
        )
        expected_qff_pnl = trade["qff_units"] * (
            trade["exit_qff_close"] - trade["entry_qff_close"]
        )
        expected_entry_costs = fill_costs(
            tsm_units=trade["tsm_units"],
            tsm_price=trade["entry_tsm_twd_fair"],
            qff_contracts=qff_contracts,
            qff_price=trade["entry_qff_close"],
            params=params,
        )
        expected_exit_costs = fill_costs(
            tsm_units=trade["tsm_units"],
            tsm_price=trade["exit_tsm_twd_fair"],
            qff_contracts=qff_contracts,
            qff_price=trade["exit_qff_close"],
            params=params,
        )
        expected_gross_pnl = expected_tsm_pnl + expected_qff_pnl
        expected_total_fee = (
            expected_entry_costs["total_fee_twd"]
            + expected_exit_costs["total_fee_twd"]
        )
        expected_net_pnl = expected_gross_pnl - expected_total_fee
        if abs(expected_tsm_pnl - trade["tsm_pnl"]) > 1e-7:
            raise RuntimeError("TSM leg PnL validation failed")
        if abs(expected_qff_pnl - trade["qff_pnl"]) > 1e-7:
            raise RuntimeError("QFF leg PnL validation failed")
        if abs(expected_entry_costs["total_fee_twd"] - trade["entry_fee_twd"]) > 1e-7:
            raise RuntimeError("Entry fee validation failed")
        if abs(expected_exit_costs["total_fee_twd"] - trade["exit_fee_twd"]) > 1e-7:
            raise RuntimeError("Exit fee validation failed")
        if abs(expected_gross_pnl - trade["gross_pnl_twd"]) > 1e-7:
            raise RuntimeError("Gross trade PnL validation failed")
        if abs(expected_total_fee - trade["total_fee_twd"]) > 1e-7:
            raise RuntimeError("Total fee validation failed")
        if abs(expected_net_pnl - trade["net_pnl_twd"]) > 1e-7:
            raise RuntimeError("Net trade PnL validation failed")
        if abs(expected_net_pnl - trade["total_pnl"]) > 1e-7:
            raise RuntimeError("Total trade PnL validation failed")

    final_pnl = float(equity["equity"].iloc[-1] - params.initial_capital_twd)
    if abs(float(trades["net_pnl_twd"].sum()) - final_pnl) > 1e-7:
        raise RuntimeError("Trade net PnL does not match final equity PnL")
    if abs(float(trades["total_fee_twd"].sum()) - equity["realized_fee_twd"].iloc[-1]) > 1e-7:
        raise RuntimeError("Realized fee total does not match trade fees")


def run_self_tests() -> None:
    params = BacktestParams(
        entry_z=2.0,
        exit_z=0.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=DEFAULT_QFF_CONTRACT_MULTIPLIER,
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

    open_fill = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=4, freq="min", tz=TAIPEI_TZ),
        zscores=[2.1, 1.0, -0.1, -0.2],
        tsm_start=100.0,
        qff_start=100.0,
    )
    open_fill["tsm_twd_fair_open"] = [100.0, 90.0, 102.0, 103.0]
    open_fill["qff_entry_open_filled"] = [100.0, 80.0, 100.0, 100.0]
    open_fill["qff_entry_open_was_filled"] = False
    open_fill_result = run_backtest(open_fill, params)
    if len(open_fill_result.trades) != 1:
        raise RuntimeError("Self-test failed: open-fill case should create one trade")
    open_fill_trade = open_fill_result.trades.iloc[0]
    if open_fill_trade["entry_fill_price_type"] != "open":
        raise RuntimeError("Self-test failed: entry fill price type should be open")
    if abs(open_fill_trade["entry_tsm_twd_fair"] - 90.0) > 1e-7:
        raise RuntimeError("Self-test failed: entry TSM price should use open")
    if abs(open_fill_trade["entry_qff_open_filled"] - 80.0) > 1e-7:
        raise RuntimeError("Self-test failed: entry QFF price should use open")

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

    friday_force_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-12 13:41", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 13:42", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 13:43", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:25", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:26", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-12 17:27", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-15 08:45", tz=TAIPEI_TZ),
        ]
    )
    friday_force = make_synthetic_frame(
        friday_force_times,
        zscores=[2.1, 2.1, 2.1, 1.1, 1.2, 1.3, 0.0],
        tsm_start=100.0,
        qff_start=100.0,
    )
    friday_force_result = run_backtest(friday_force, params)
    if len(friday_force_result.trades) != 1:
        raise RuntimeError("Self-test failed: Friday forced close should create one trade")
    friday_force_trade = friday_force_result.trades.iloc[0]
    if friday_force_trade["exit_reason"] != FRIDAY_SESSION_END:
        raise RuntimeError("Self-test failed: Friday forced close reason is wrong")
    if friday_force_trade["exit_idx"] != 5:
        raise RuntimeError("Self-test failed: Friday forced close timing is wrong")

    holiday_week_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-18 13:41", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-18 13:42", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-18 13:43", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-18 17:25", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-18 17:26", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-19 05:00", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-22 08:45", tz=TAIPEI_TZ),
        ]
    )
    holiday_week = make_synthetic_frame(
        holiday_week_times,
        zscores=[2.1, 2.1, 2.1, 2.5, 2.6, 1.3, 0.0],
        tsm_start=100.0,
        qff_start=100.0,
    )
    holiday_week_result = run_backtest(holiday_week, params)
    if len(holiday_week_result.trades) != 1:
        raise RuntimeError("Self-test failed: holiday week should create one trade")
    holiday_week_trade = holiday_week_result.trades.iloc[0]
    if holiday_week_trade["exit_reason"] != FRIDAY_SESSION_END:
        raise RuntimeError("Self-test failed: holiday week forced close reason is wrong")
    if holiday_week_trade["exit_idx"] != 5:
        raise RuntimeError("Self-test failed: holiday week forced close timing is wrong")
    if holiday_week_result.equity["entry_allowed"].iloc[3:6].any():
        raise RuntimeError("Self-test failed: week-end session should be close-only")

    fee_case = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=4, freq="min", tz=TAIPEI_TZ),
        zscores=[2.1, 1.0, -0.1, -0.2],
        tsm_start=100.0,
        qff_start=250.0,
    )
    fee_result = run_backtest(fee_case, params)
    if len(fee_result.trades) != 1:
        raise RuntimeError("Self-test failed: fee case should create one trade")
    fee_trade = fee_result.trades.iloc[0]
    if int(fee_trade["qff_contracts"]) != 40:
        raise RuntimeError("Self-test failed: QFF contract rounding is wrong")
    if abs(fee_trade["actual_leg_notional_twd"] - 1_000_000.0) > 1e-7:
        raise RuntimeError("Self-test failed: actual notional should match rounded QFF")
    if abs(abs(fee_trade["tsm_units"]) * fee_trade["entry_tsm_twd_fair"] - 1_000_000.0) > 1e-7:
        raise RuntimeError("Self-test failed: TSM notional should match QFF notional")
    expected_entry_fee = 500.0 + 40 * 5.0 + 40 * 1.0
    if abs(fee_trade["entry_fee_twd"] - expected_entry_fee) > 1e-7:
        raise RuntimeError("Self-test failed: entry fee calculation is wrong")
    if abs(fee_trade["gross_pnl_twd"] - fee_trade["total_fee_twd"] - fee_trade["net_pnl_twd"]) > 1e-7:
        raise RuntimeError("Self-test failed: net PnL should equal gross minus fees")

    tiny_params = BacktestParams(
        entry_z=2.0,
        exit_z=0.0,
        leg_notional_twd=1.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )
    tiny = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=2, freq="min", tz=TAIPEI_TZ),
        zscores=[2.1, 2.1],
        tsm_start=100.0,
        qff_start=250.0,
    )
    tiny_result = run_backtest(tiny, tiny_params)
    if len(tiny_result.trades) != 0:
        raise RuntimeError("Self-test failed: zero-contract entry should be cancelled")

    sparse_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-08 08:45", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-08 08:50", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-08 09:10", tz=TAIPEI_TZ),
            pd.Timestamp("2026-06-08 09:11", tz=TAIPEI_TZ),
        ]
    )
    sparse = make_synthetic_frame(
        sparse_times,
        zscores=[2.1, 1.0, -0.1, -0.2],
        tsm_start=100.0,
        qff_start=100.0,
    )
    sparse_result = run_backtest(sparse, params)
    if len(sparse_result.trades) != 1:
        raise RuntimeError("Self-test failed: sparse timestamp case should trade once")
    sparse_trade = sparse_result.trades.iloc[0]
    if int(sparse_trade["holding_minutes"]) != 21:
        raise RuntimeError("Self-test failed: sparse holding time should use timestamps")
    if sparse_result.summary["exposure_minutes"] != 21:
        raise RuntimeError("Self-test failed: sparse exposure should use timestamps")

    time_stop_params = BacktestParams(
        entry_z=2.0,
        exit_z=0.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=DEFAULT_QFF_CONTRACT_MULTIPLIER,
        max_holding_minutes=3,
    )
    time_stop = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=5, freq="min", tz=TAIPEI_TZ),
        zscores=[2.1, 2.1, 2.1, 2.1, 2.1],
        tsm_start=100.0,
        qff_start=100.0,
    )
    time_stop_result = run_backtest(time_stop, time_stop_params)
    if len(time_stop_result.trades) != 1:
        raise RuntimeError("Self-test failed: time stop should create exactly one trade")
    time_stop_trade = time_stop_result.trades.iloc[0]
    if time_stop_trade["exit_reason"] != TIME_STOP:
        raise RuntimeError("Self-test failed: trade should exit via the time stop")
    if time_stop_trade["entry_idx"] != 1 or time_stop_trade["exit_idx"] != 4:
        raise RuntimeError("Self-test failed: time stop fill timing is wrong")
    if int(time_stop_trade["holding_minutes"]) != 3:
        raise RuntimeError("Self-test failed: time stop holding should equal the cap")
    if time_stop_result.summary["time_stop_exits"] != 1:
        raise RuntimeError("Self-test failed: summary should count the time stop exit")
    if time_stop_result.trades.iloc[0]["exit_signal_zscore"] != 2.1:
        raise RuntimeError("Self-test failed: time stop signal z-score should be the exit bar")

    band_params = BacktestParams(
        entry_z=2.5,
        exit_z=0.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=DEFAULT_QFF_CONTRACT_MULTIPLIER,
        entry_z_max=3.5,
    )
    above_band = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=4, freq="min", tz=TAIPEI_TZ),
        zscores=[4.0, 4.0, 4.0, 4.0],
        tsm_start=100.0,
        qff_start=100.0,
    )
    if len(run_backtest(above_band, band_params).trades) != 0:
        raise RuntimeError("Self-test failed: |z| above entry_z_max should not enter")
    within_band = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=4, freq="min", tz=TAIPEI_TZ),
        zscores=[3.0, 3.0, -0.1, -0.2],
        tsm_start=100.0,
        qff_start=100.0,
    )
    within_result = run_backtest(within_band, band_params)
    if len(within_result.trades) != 1:
        raise RuntimeError("Self-test failed: |z| inside the band should enter")
    if within_result.trades.iloc[0]["direction"] != SHORT_TSM_LONG_QFF:
        raise RuntimeError("Self-test failed: within-band trade direction is wrong")

    vol_gate_params = BacktestParams(
        entry_z=2.0,
        exit_z=0.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=DEFAULT_QFF_CONTRACT_MULTIPLIER,
        max_entry_vol_ratio=1.5,
    )
    high_vol = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=4, freq="min", tz=TAIPEI_TZ),
        zscores=[2.1, 2.1, -0.1, -0.2],
        tsm_start=100.0,
        qff_start=100.0,
        entry_vol_ratio=2.0,
    )
    if len(run_backtest(high_vol, vol_gate_params).trades) != 0:
        raise RuntimeError("Self-test failed: high vol ratio should block the entry")
    low_vol = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=4, freq="min", tz=TAIPEI_TZ),
        zscores=[2.1, 2.1, -0.1, -0.2],
        tsm_start=100.0,
        qff_start=100.0,
        entry_vol_ratio=1.0,
    )
    if len(run_backtest(low_vol, vol_gate_params).trades) != 1:
        raise RuntimeError("Self-test failed: normal vol ratio should allow the entry")

    persistence_frame = make_synthetic_frame(
        pd.date_range("2026-06-08 08:45", periods=6, freq="min", tz=TAIPEI_TZ),
        zscores=[2.1, 1.0, 2.1, 2.1, 2.1, 2.1],
        tsm_start=100.0,
        qff_start=100.0,
    )
    persistence_base = dict(
        entry_z=2.0,
        exit_z=0.0,
        leg_notional_twd=1_000_000.0,
        initial_capital_twd=2_000_000.0,
        max_entry_delay_minutes=15,
        tsm_fee_bps=DEFAULT_TSM_FEE_BPS,
        qff_fee_per_contract_twd=DEFAULT_QFF_FEE_PER_CONTRACT_TWD,
        qff_tax_rate=DEFAULT_QFF_TAX_RATE,
        qff_contract_multiplier=DEFAULT_QFF_CONTRACT_MULTIPLIER,
    )
    persistence_result = run_backtest(
        persistence_frame, BacktestParams(**persistence_base, persistence_k=2)
    )
    if (
        len(persistence_result.trades) != 1
        or persistence_result.trades.iloc[0]["entry_idx"] != 4
    ):
        raise RuntimeError(
            "Self-test failed: persistence k=2 should delay entry to the 2nd consecutive signal"
        )
    no_persistence_result = run_backtest(
        persistence_frame, BacktestParams(**persistence_base)
    )
    if no_persistence_result.trades.iloc[0]["entry_idx"] != 1:
        raise RuntimeError(
            "Self-test failed: persistence k=1 should enter on the first signal"
        )


def make_synthetic_frame(
    timestamps: pd.DatetimeIndex,
    zscores: list[float],
    tsm_start: float,
    qff_start: float,
    entry_vol_ratio: float | list[float] = 1.0,
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
            "entry_vol_ratio": entry_vol_ratio,
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
        tsm_fee_bps=args.tsm_fee_bps,
        qff_fee_per_contract_twd=args.qff_fee_per_contract_twd,
        qff_tax_rate=args.qff_tax_rate,
        qff_contract_multiplier=args.qff_contract_multiplier,
        max_holding_minutes=args.max_holding_minutes,
        entry_z_max=args.entry_z_max,
        max_entry_vol_ratio=args.max_entry_vol_ratio,
        persistence_k=args.persistence_k,
    )
    frame = read_input_frame(
        args.input,
        qff_ohlcv_path=args.qff_ohlcv,
        tsm_ohlcv_path=args.tsm_ohlcv,
        usdttwd_ohlcv_path=args.usdttwd_ohlcv,
    )
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
