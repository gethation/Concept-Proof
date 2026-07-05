"""Add a causal ADR-authorship gate column to a spread/z-score CSV.

The 2026-07-05 anchored attribution found that entries whose z run-up was
majority-authored by the NYSE TSM ADR (the perp pegged to an informed ADR
move) are the only net-NEGATIVE entry subgroup in the sample. To gate on
that *causally*, for every bar t we measure how much of the current run-up
(from the last bar with |z| below --runup-z, capped at --max-lookback bars)
is explained by the ADR:

    total_t          = sign(z_t) * (spread_t - spread_j)      (run-up size)
    adr_contrib_t    = sign(z_t) * 100 * (ln ADR_t - ln ADR_j)
    entry_adr_share_t = adr_contrib_t / total_t

ADR closes are taken as-of bar COMPLETION time (bar start + bar interval),
so only fully closed ADR bars are visible at t. Outside US hours the ADR
price is simply carried, contributing zero. The backtest skips an entry when
entry_adr_share exceeds a threshold (--max-entry-adr-share).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAIPEI_TZ = "Asia/Taipei"
DEFAULT_INPUT = Path("data/processed/qff_tsm_spread_zscore_15m_w33.csv")
DEFAULT_ADR = Path("data/processed/factor_nyse_tsm_15m_taipei.csv")
DEFAULT_OUT = Path("data/processed/qff_tsm_spread_zscore_15m_w33_adr.csv")
DEFAULT_RUNUP_Z = 1.0
DEFAULT_MAX_LOOKBACK = 32
DEFAULT_ADR_BAR_MINUTES = 15


def adr_log_close_asof(
    timestamps: pd.DatetimeIndex,
    adr: pd.DataFrame,
    bar_minutes: int,
) -> np.ndarray:
    """Last ADR log close whose bar has COMPLETED by each timestamp."""
    completion = (
        pd.DatetimeIndex(adr["timestamp"]) + pd.Timedelta(minutes=bar_minutes)
    ).asi8
    lnc = np.log(pd.to_numeric(adr["close"], errors="coerce").to_numpy(dtype=float))
    idx = np.searchsorted(completion, timestamps.asi8, side="right") - 1
    out = np.full(len(timestamps), np.nan)
    valid = idx >= 0
    out[valid] = lnc[idx[valid]]
    return out


def add_adr_share(
    frame: pd.DataFrame,
    adr: pd.DataFrame,
    runup_z: float,
    max_lookback: int,
    adr_bar_minutes: int,
) -> pd.DataFrame:
    out = frame.copy()
    spread = pd.to_numeric(out["spread"], errors="coerce").to_numpy(dtype=float)
    zscore = pd.to_numeric(out["spread_zscore"], errors="coerce").to_numpy(dtype=float)
    adr_lnc = adr_log_close_asof(
        pd.DatetimeIndex(out["timestamp"]), adr, adr_bar_minutes
    )

    n = len(out)
    share = np.full(n, np.nan)
    for i in range(n):
        z = zscore[i]
        if np.isnan(z) or abs(z) < runup_z:
            continue
        sgn = 1.0 if z > 0 else -1.0
        j = i - 1
        while j > 0 and i - j < max_lookback and abs(zscore[j]) >= runup_z:
            j -= 1
        if j < 0:
            continue
        total = sgn * (spread[i] - spread[j])
        if not np.isfinite(total) or total <= 0:
            continue
        if np.isnan(adr_lnc[i]) or np.isnan(adr_lnc[j]):
            continue
        share[i] = sgn * 100.0 * (adr_lnc[i] - adr_lnc[j]) / total

    out["entry_adr_share"] = share
    return out


def run_self_test() -> None:
    ts = pd.date_range("2026-06-08 21:30", periods=12, freq="15min", tz=TAIPEI_TZ)
    zscores = [0.5, 0.5, 1.5, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.1, 3.2, 3.3]
    spread = np.linspace(10.0, 13.0, 12)

    # 1) run-up driven entirely by the ADR -> share ~= 1
    adr_driven = pd.DataFrame(
        {"timestamp": ts, "close": 100.0 * np.exp(spread / 100.0)}
    )
    frame = pd.DataFrame({"timestamp": ts, "spread": spread, "spread_zscore": zscores})
    res = add_adr_share(frame, adr_driven, DEFAULT_RUNUP_Z, DEFAULT_MAX_LOOKBACK, 15)
    got = res["entry_adr_share"].iloc[-1]
    if not (0.95 < got < 1.05):
        raise RuntimeError(f"Self-test failed: ADR-driven share should be ~1, got {got:.3f}")

    # 2) flat ADR -> share ~= 0
    adr_flat = pd.DataFrame({"timestamp": ts, "close": np.full(12, 100.0)})
    res_flat = add_adr_share(frame, adr_flat, DEFAULT_RUNUP_Z, DEFAULT_MAX_LOOKBACK, 15)
    got_flat = res_flat["entry_adr_share"].iloc[-1]
    if not abs(got_flat) < 1e-9:
        raise RuntimeError(f"Self-test failed: flat-ADR share should be 0, got {got_flat:.3f}")

    # 3) causality: an ADR bar starting AT the evaluation bar has not completed
    #    yet and must not change the share
    adr_peek = pd.concat(
        [adr_flat, pd.DataFrame({"timestamp": [ts[-1]], "close": [500.0]})],
        ignore_index=True,
    )
    res_peek = add_adr_share(frame, adr_peek, DEFAULT_RUNUP_Z, DEFAULT_MAX_LOOKBACK, 15)
    got_peek = res_peek["entry_adr_share"].iloc[-1]
    if not abs(got_peek - got_flat) < 1e-12:
        raise RuntimeError(
            f"Self-test failed: incomplete ADR bar leaked into the share "
            f"({got_flat:.6f} -> {got_peek:.6f})"
        )

    # 4) bars below the run-up threshold carry no share
    if not res["entry_adr_share"].iloc[:2].isna().all():
        raise RuntimeError("Self-test failed: |z| below runup_z should have NaN share")

    print("Self-test passed: ADR-driven ~1, flat ~0, incomplete bars invisible")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--adr-ohlcv", type=Path, default=DEFAULT_ADR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--runup-z", type=float, default=DEFAULT_RUNUP_Z)
    p.add_argument("--max-lookback", type=int, default=DEFAULT_MAX_LOOKBACK)
    p.add_argument("--adr-bar-minutes", type=int, default=DEFAULT_ADR_BAR_MINUTES)
    p.add_argument("--skip-self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_self_test:
        run_self_test()
    if args.runup_z <= 0:
        raise RuntimeError("--runup-z must be positive")
    if args.max_lookback <= 1:
        raise RuntimeError("--max-lookback must be > 1")

    frame = pd.read_csv(args.input)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    for column in ("spread", "spread_zscore"):
        if column not in frame.columns:
            raise RuntimeError(f"{args.input} has no '{column}' column")

    adr = pd.read_csv(args.adr_ohlcv)
    adr["timestamp"] = pd.to_datetime(adr["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    adr = adr.sort_values("timestamp").reset_index(drop=True)

    out = add_adr_share(
        frame, adr, args.runup_z, args.max_lookback, args.adr_bar_minutes
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    valid = out["entry_adr_share"].dropna()
    print(f"Wrote {len(out):,} rows to {args.out}")
    print(
        f"entry_adr_share ({len(valid):,} run-up bars): "
        f"p10={valid.quantile(0.1):+.3f}  p50={valid.quantile(0.5):+.3f}  "
        f"p90={valid.quantile(0.9):+.3f}  frac>0.5={100 * (valid > 0.5).mean():.1f}%"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
