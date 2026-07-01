"""Add a causal conditional-volatility gate column to a spread/z-score CSV.

The earlier analysis showed that entries taken when the spread's conditional
volatility is elevated revert worse (high-vol tercile of |z|>=2 entries had
~0 forward edge). To gate on that *causally* (no look-ahead), we compute:

    ewma_vol_t   = sqrt(EWMA of squared 1-bar spread changes), lambda=0.94
    baseline_t   = trailing rolling mean of ewma_vol over `baseline_window` bars
    entry_vol_ratio_t = ewma_vol_t / baseline_t

Both pieces use only information up to t. The backtest skips an entry when
entry_vol_ratio exceeds a threshold (--max-entry-vol-ratio). Session-break bars
(timestamp gaps) are excluded from the change series so the overnight jump does
not pollute the volatility estimate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAIPEI_TZ = "Asia/Taipei"
DEFAULT_INPUT = Path("data/processed/qff_tsm_spread_zscore_15m_w33.csv")
DEFAULT_OUT = Path("data/processed/qff_tsm_spread_zscore_15m_w33_vol.csv")
DEFAULT_LAMBDA = 0.94
DEFAULT_BASELINE_WINDOW = 200


def compute_ewma_vol(spread: np.ndarray, is_break: np.ndarray, lam: float) -> np.ndarray:
    """EWMA volatility of 1-bar spread changes. Gap bars (is_break) are skipped
    (variance carried, not updated) so the session jump is not counted."""
    changes = np.diff(spread, prepend=spread[0])  # change[i] = spread[i]-spread[i-1]
    changes[0] = np.nan
    changes[is_break] = np.nan
    seed = np.nanvar(changes)
    vol = np.full(len(spread), np.nan)
    v = seed
    started = False
    for i in range(len(spread)):
        c = changes[i]
        if not np.isnan(c):
            v = lam * v + (1.0 - lam) * c * c
            started = True
        if started:
            vol[i] = np.sqrt(v)
    return vol


def add_vol_ratio(
    frame: pd.DataFrame, lam: float, baseline_window: int
) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.DatetimeIndex(out["timestamp"])
    dt_min = ts.to_series().diff().dt.total_seconds().to_numpy() / 60.0
    median_bar = float(np.nanmedian(dt_min[1:]))
    is_break = np.zeros(len(out), dtype=bool)
    is_break[1:] = dt_min[1:] > 1.5 * median_bar
    spread = pd.to_numeric(out["spread"], errors="coerce").to_numpy(dtype=float)

    ewma_vol = compute_ewma_vol(spread, is_break, lam)
    baseline = (
        pd.Series(ewma_vol).rolling(baseline_window, min_periods=baseline_window).mean()
    )
    ratio = ewma_vol / baseline.to_numpy()
    out["ewma_vol"] = ewma_vol
    out["vol_baseline"] = baseline.to_numpy()
    out["entry_vol_ratio"] = ratio
    return out


def run_self_test() -> None:
    # Constant-vol random walk -> ratio hovers near 1; an injected vol burst -> ratio >> 1.
    rng = np.random.default_rng(0)
    n = 1200
    calm = np.cumsum(rng.standard_normal(n) * 1.0)
    ts = pd.date_range("2026-01-01 09:00", periods=n, freq="min", tz=TAIPEI_TZ)
    frame = pd.DataFrame({"timestamp": ts, "spread": calm})
    res = add_vol_ratio(frame, DEFAULT_LAMBDA, 200)
    tail_ratio = res["entry_vol_ratio"].iloc[300:].dropna()
    if not (0.5 < tail_ratio.median() < 1.6):
        raise RuntimeError(f"Self-test failed: calm ratio median off: {tail_ratio.median():.3f}")
    burst = calm.copy()
    burst[800:830] += np.cumsum(rng.standard_normal(30) * 12.0)  # volatility spike
    frame_b = pd.DataFrame({"timestamp": ts, "spread": burst})
    res_b = add_vol_ratio(frame_b, DEFAULT_LAMBDA, 200)
    if not (res_b["entry_vol_ratio"].iloc[800:835].max() > 2.0):
        raise RuntimeError("Self-test failed: vol burst did not lift the ratio")
    print("Self-test passed: calm ratio ~1, vol burst lifts ratio above 2")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--lambda-ewma", type=float, default=DEFAULT_LAMBDA, dest="lam")
    p.add_argument("--baseline-window", type=int, default=DEFAULT_BASELINE_WINDOW)
    p.add_argument("--skip-self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_self_test:
        run_self_test()
    if not (0.0 < args.lam < 1.0):
        raise RuntimeError("--lambda-ewma must be in (0, 1)")
    if args.baseline_window <= 1:
        raise RuntimeError("--baseline-window must be > 1")

    frame = pd.read_csv(args.input)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    if "spread" not in frame.columns:
        raise RuntimeError(f"{args.input} has no 'spread' column")
    out = add_vol_ratio(frame, args.lam, args.baseline_window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    valid = out["entry_vol_ratio"].dropna()
    print(f"Wrote {len(out):,} rows to {args.out}")
    print(
        "entry_vol_ratio: "
        f"p50={valid.quantile(0.5):.3f}  p67={valid.quantile(0.67):.3f}  "
        f"p80={valid.quantile(0.8):.3f}  p90={valid.quantile(0.9):.3f}  max={valid.max():.3f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
