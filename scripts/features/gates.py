"""Add causal entry-gate columns to a spread/z-score file.

    python scripts/features/gates.py --gate vol_ratio
    python scripts/features/gates.py --gate vol_ratio --gate adr_share --gate qff_vol_surprise

Replaces add_entry_vol_ratio.py, add_entry_adr_share.py and
add_entry_qff_vol_surprise.py. Those were one template written three times:
read a z-score file, compute a single column from information available at or
before each bar, self-test the causality, write the file back out with the
column appended. Only the middle step ever differed.

Stacking them in one pass is the practical gain. Previously three gates meant
three runs through three intermediate files, each an opportunity to feed the
wrong vintage into the next stage.

Every gate here must be causal: the value at bar t may only use data known by
the close of bar t. Each carries a self-test that asserts exactly that, and the
tests run by default -- pass --skip-self-test to suppress them.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths  # noqa: E402
from lib.timeutil import TAIPEI_TZ  # noqa: E402

# TAIFEX day session, Taipei minutes past midnight.
DAY_START_MIN = 8 * 60 + 45
DAY_END_MIN = 13 * 60 + 45


# --------------------------------------------------------------------------
# gate 1: conditional volatility ratio
# --------------------------------------------------------------------------


def compute_ewma_vol(spread: np.ndarray, is_break: np.ndarray, lam: float) -> np.ndarray:
    """EWMA volatility of 1-bar spread changes. Gap bars (is_break) are skipped
    (variance carried, not updated) so the session jump is not counted."""
    changes = np.diff(spread, prepend=spread[0])
    changes[0] = np.nan
    changes[is_break] = np.nan
    v = np.nanvar(changes)
    vol = np.full(len(spread), np.nan)
    started = False
    for i in range(len(spread)):
        change = changes[i]
        if not np.isnan(change):
            v = lam * v + (1.0 - lam) * change * change
            started = True
        if started:
            vol[i] = np.sqrt(v)
    return vol


def add_vol_ratio(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Elevated conditional volatility marks entries that revert worse.

    The high-vol tercile of |z|>=2 entries showed ~0 forward edge, so the
    backtest skips an entry when this ratio exceeds --max-entry-vol-ratio.
    """
    if not 0.0 < args.lam < 1.0:
        raise RuntimeError("--lambda-ewma must be in (0, 1)")
    if args.baseline_window <= 1:
        raise RuntimeError("--baseline-window must be > 1")

    out = frame.copy()
    timestamps = pd.DatetimeIndex(out["timestamp"])
    gaps = timestamps.to_series().diff().dt.total_seconds().to_numpy() / 60.0
    median_bar = float(np.nanmedian(gaps[1:]))
    is_break = np.zeros(len(out), dtype=bool)
    is_break[1:] = gaps[1:] > 1.5 * median_bar

    spread = pd.to_numeric(out["spread"], errors="coerce").to_numpy(dtype=float)
    ewma_vol = compute_ewma_vol(spread, is_break, args.lam)
    baseline = (
        pd.Series(ewma_vol)
        .rolling(args.baseline_window, min_periods=args.baseline_window)
        .mean()
        .to_numpy()
    )
    out["ewma_vol"] = ewma_vol
    out["vol_baseline"] = baseline
    out["entry_vol_ratio"] = ewma_vol / baseline
    return out


def self_test_vol_ratio() -> None:
    rng = np.random.default_rng(0)
    n = 1200
    calm = np.cumsum(rng.standard_normal(n) * 1.0)
    timestamps = pd.date_range("2026-01-01 09:00", periods=n, freq="min", tz=TAIPEI_TZ)
    args = argparse.Namespace(lam=0.94, baseline_window=200)

    frame = pd.DataFrame({"timestamp": timestamps, "spread": calm})
    tail = add_vol_ratio(frame, args)["entry_vol_ratio"].iloc[300:].dropna()
    if not 0.5 < tail.median() < 1.6:
        raise RuntimeError(f"calm ratio median off: {tail.median():.3f}")

    burst = calm.copy()
    burst[800:830] += np.cumsum(rng.standard_normal(30) * 12.0)
    spiked = add_vol_ratio(
        pd.DataFrame({"timestamp": timestamps, "spread": burst}), args
    )
    if not spiked["entry_vol_ratio"].iloc[800:835].max() > 2.0:
        raise RuntimeError("vol burst did not lift the ratio")
    print("  vol_ratio: calm ratio ~1, vol burst lifts ratio above 2")


# --------------------------------------------------------------------------
# gate 2: QFF volume surprise
# --------------------------------------------------------------------------


def add_qff_vol_surprise(
    frame: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Thin-volume dislocations revert about twice as well as busy ones.

    Thin reads as stale-price microstructure noise -- the best fades. Elevated
    reads as informed 2330/CDF flow propagating into QFF. Day-session only: the
    evidence is day-session-specific and night volume behaves differently, so
    elsewhere the column is NaN and the backtest gate passes.
    """
    ohlcv = args._qff_ohlcv
    volume = ohlcv[["timestamp", "volume"]].copy()
    volume["volume"] = pd.to_numeric(volume["volume"], errors="coerce")
    volume = volume.sort_values("timestamp").reset_index(drop=True)

    timestamps = pd.DatetimeIndex(volume["timestamp"])
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    is_day = (
        (minute_of_day >= DAY_START_MIN)
        & (minute_of_day <= DAY_END_MIN)
        & (timestamps.dayofweek < 5)
    )
    volume["slot"] = timestamps.strftime("%H:%M")
    # trailing same-slot median, excluding the current bar (shift THEN roll)
    volume["slot_median"] = volume.groupby("slot")["volume"].transform(
        lambda s: s.shift(1)
        .rolling(args.slot_window, min_periods=args.slot_window // 2)
        .median()
    )
    surprise = volume["volume"] / volume["slot_median"]
    volume["entry_qff_vol_surprise"] = surprise.where(is_day, np.nan)

    return frame.merge(
        volume[["timestamp", "entry_qff_vol_surprise"]], on="timestamp", how="left"
    )


def self_test_qff_vol_surprise() -> None:
    slots = ["09:00", "09:15", "09:30", "21:30"]
    rows = []
    for day in range(12):
        date = pd.Timestamp("2026-06-01", tz=TAIPEI_TZ) + pd.Timedelta(days=day)
        if date.dayofweek >= 5:
            continue
        for slot in slots:
            hour, minute = map(int, slot.split(":"))
            rows.append(
                {
                    "timestamp": date + pd.Timedelta(hours=hour, minutes=minute),
                    "volume": 100.0,
                }
            )
    ohlcv = pd.DataFrame(rows)
    ohlcv.loc[ohlcv.index[-4], "volume"] = 300.0
    frame = pd.DataFrame({"timestamp": ohlcv["timestamp"]})
    args = argparse.Namespace(slot_window=10, _qff_ohlcv=ohlcv)
    result = add_qff_vol_surprise(frame, args)

    timestamps = pd.DatetimeIndex(result["timestamp"])
    if not result.loc[timestamps.hour == 21, "entry_qff_vol_surprise"].isna().all():
        raise RuntimeError("night bars should be NaN")

    spiked = result["entry_qff_vol_surprise"].iloc[-4]
    if not 2.5 < spiked < 3.5:
        raise RuntimeError(f"spike surprise should be ~3, got {spiked}")
    calm = result["entry_qff_vol_surprise"].iloc[-3]
    if not 0.9 < calm < 1.1:
        raise RuntimeError(f"calm surprise should be ~1, got {calm}")

    # causality: a FUTURE bar's volume must not change the surprise at -4
    future = ohlcv.copy()
    future.loc[future.index[-1], "volume"] = 10_000.0
    peeked = add_qff_vol_surprise(
        frame, argparse.Namespace(slot_window=10, _qff_ohlcv=future)
    )
    if peeked["entry_qff_vol_surprise"].iloc[-4] != spiked:
        raise RuntimeError("future volume leaked into the surprise")
    print("  qff_vol_surprise: calm ~1, spike ~3, night NaN, future bars invisible")


# --------------------------------------------------------------------------
# gate 3: ADR authorship share
# --------------------------------------------------------------------------


def adr_log_close_asof(
    timestamps: pd.DatetimeIndex, adr: pd.DataFrame, bar_minutes: int
) -> np.ndarray:
    """Last ADR log close whose bar has COMPLETED by each timestamp."""
    completion = (
        pd.DatetimeIndex(adr["timestamp"]) + pd.Timedelta(minutes=bar_minutes)
    ).asi8
    log_close = np.log(
        pd.to_numeric(adr["close"], errors="coerce").to_numpy(dtype=float)
    )
    index = np.searchsorted(completion, timestamps.asi8, side="right") - 1
    out = np.full(len(timestamps), np.nan)
    valid = index >= 0
    out[valid] = log_close[index[valid]]
    return out


def add_adr_share(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """How much of the current z run-up the NYSE ADR authored.

    Entries whose run-up was majority-authored by the ADR -- the perp pegged to
    an informed ADR move -- are the only net-negative entry subgroup in the
    sample. Outside US hours the ADR price is carried, contributing zero.
    """
    out = frame.copy()
    spread = pd.to_numeric(out["spread"], errors="coerce").to_numpy(dtype=float)
    zscore = pd.to_numeric(out["spread_zscore"], errors="coerce").to_numpy(dtype=float)
    adr_log = adr_log_close_asof(
        pd.DatetimeIndex(out["timestamp"]), args._adr, args.adr_bar_minutes
    )

    n = len(out)
    share = np.full(n, np.nan)
    for i in range(n):
        z = zscore[i]
        if np.isnan(z) or abs(z) < args.runup_z:
            continue
        sign = 1.0 if z > 0 else -1.0
        j = i - 1
        while j > 0 and i - j < args.max_lookback and abs(zscore[j]) >= args.runup_z:
            j -= 1
        if j < 0:
            continue
        total = sign * (spread[i] - spread[j])
        if not np.isfinite(total) or total <= 0:
            continue
        if np.isnan(adr_log[i]) or np.isnan(adr_log[j]):
            continue
        share[i] = sign * 100.0 * (adr_log[i] - adr_log[j]) / total

    out["entry_adr_share"] = share
    return out


def self_test_adr_share() -> None:
    timestamps = pd.date_range(
        "2026-06-08 21:30", periods=12, freq="15min", tz=TAIPEI_TZ
    )
    zscores = [0.5, 0.5, 1.5, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.1, 3.2, 3.3]
    spread = np.linspace(10.0, 13.0, 12)
    frame = pd.DataFrame(
        {"timestamp": timestamps, "spread": spread, "spread_zscore": zscores}
    )

    def run(adr):
        args = argparse.Namespace(
            runup_z=1.0, max_lookback=32, adr_bar_minutes=15, _adr=adr
        )
        return add_adr_share(frame, args)

    driven = run(pd.DataFrame({"timestamp": timestamps, "close": 100.0 * np.exp(spread / 100.0)}))
    got = driven["entry_adr_share"].iloc[-1]
    if not 0.95 < got < 1.05:
        raise RuntimeError(f"ADR-driven share should be ~1, got {got:.3f}")

    flat_adr = pd.DataFrame({"timestamp": timestamps, "close": np.full(12, 100.0)})
    flat = run(flat_adr)["entry_adr_share"].iloc[-1]
    if abs(flat) >= 1e-9:
        raise RuntimeError(f"flat-ADR share should be 0, got {flat:.3f}")

    # causality: an ADR bar starting AT the evaluation bar has not completed yet
    peek_adr = pd.concat(
        [flat_adr, pd.DataFrame({"timestamp": [timestamps[-1]], "close": [500.0]})],
        ignore_index=True,
    )
    peeked = run(peek_adr)["entry_adr_share"].iloc[-1]
    if abs(peeked - flat) >= 1e-12:
        raise RuntimeError(
            f"incomplete ADR bar leaked into the share ({flat:.6f} -> {peeked:.6f})"
        )

    if not driven["entry_adr_share"].iloc[:2].isna().all():
        raise RuntimeError("|z| below runup_z should have NaN share")
    print("  adr_share: ADR-driven ~1, flat ~0, incomplete bars invisible")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    column: str
    apply: Callable[[pd.DataFrame, argparse.Namespace], pd.DataFrame]
    self_test: Callable[[], None]
    needs: tuple[str, ...] = ()  # extra input frames to load


GATES: dict[str, Gate] = {
    "vol_ratio": Gate("entry_vol_ratio", add_vol_ratio, self_test_vol_ratio),
    "qff_vol_surprise": Gate(
        "entry_qff_vol_surprise",
        add_qff_vol_surprise,
        self_test_qff_vol_surprise,
        needs=("qff_ohlcv",),
    ),
    "adr_share": Gate(
        "entry_adr_share", add_adr_share, self_test_adr_share, needs=("adr",)
    ),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gate", action="append", choices=sorted(GATES), required=True,
        help="Repeatable; gates stack onto one output file.",
    )
    parser.add_argument(
        "--input", type=Path, default=paths.feature("qff_tsm", "zscore_15m_w33")
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-self-test", action="store_true")
    # vol_ratio
    parser.add_argument("--lambda-ewma", type=float, default=0.94, dest="lam")
    parser.add_argument("--baseline-window", type=int, default=200)
    # qff_vol_surprise
    parser.add_argument("--qff-ohlcv", type=Path, default=paths.QFF1_15M)
    parser.add_argument("--slot-window", type=int, default=10)
    # adr_share
    parser.add_argument("--adr-path", type=Path, default=paths.BARS / "factors" / "nyse_tsm_15m.csv")
    parser.add_argument("--runup-z", type=float, default=1.0)
    parser.add_argument("--max-lookback", type=int, default=32)
    parser.add_argument("--adr-bar-minutes", type=int, default=15)
    return parser.parse_args(argv)


def read_timestamped(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    return frame


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    wanted = list(dict.fromkeys(args.gate))  # de-duplicate, keep order

    if not args.skip_self_test:
        print("Self-tests:")
        for name in wanted:
            GATES[name].self_test()

    frame = read_timestamped(args.input, "input")
    if "spread" not in frame.columns:
        raise RuntimeError(f"{args.input} has no 'spread' column")

    needed = {need for name in wanted for need in GATES[name].needs}
    if "qff_ohlcv" in needed:
        args._qff_ohlcv = read_timestamped(args.qff_ohlcv, "QFF OHLCV")
    if "adr" in needed:
        args._adr = read_timestamped(args.adr_path, "ADR bars")

    for name in wanted:
        frame = GATES[name].apply(frame, args)

    out = args.out or args.input.with_name(
        args.input.stem + "_" + "_".join(wanted) + ".csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    print(f"\nWrote {len(frame):,} rows to {out}")
    for name in wanted:
        values = frame[GATES[name].column].dropna()
        if values.empty:
            print(f"{GATES[name].column}: no defined values")
            continue
        print(
            f"{GATES[name].column}: p50={values.quantile(0.5):.3f}  "
            f"p67={values.quantile(0.67):.3f}  p80={values.quantile(0.8):.3f}  "
            f"p90={values.quantile(0.9):.3f}  max={values.max():.3f}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
