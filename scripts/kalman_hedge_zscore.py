"""Kalman-filter dynamic hedge ratio + dynamic mean for the QFF/TSM pair.

The current spread fixes hedge ratio = 1 (dollar-neutral in fair-value space)
and normalises with a fixed rolling window, even though the log-log hedge is
~0.81 and the basis drifts. This filter jointly estimates a time-varying hedge
ratio beta_t and intercept alpha_t (the dynamic mean), giving a built-in,
regime-adaptive z-score.

State-space (Chan-style pairs Kalman, log-log). To keep the z-score sign
consistent with the existing strategy (high z => TSM rich vs QFF => short TSM /
long QFF) we regress TSM on QFF:

    y_t = log(tsm_twd_fair),  g_t = log(qff_close_filled)
    state  x_t = [beta_t, alpha_t],  x_t = x_{t-1} + w_t,  w_t ~ N(0, W)
    obs    y_t = [g_t, 1] . x_t + v_t,  v_t ~ N(0, Ve)

The forecast error e_t (= dynamic spread, ~zero mean since alpha_t absorbs the
level) and its variance Q_t give kalman_z = e_t / sqrt(Q_t).

W = q_ratio * Ve * I. q_ratio is the state/observation signal-to-noise ratio;
its default (~0.0037) is calibrated so the intercept's effective memory is about
`--seed-window` (33) bars -- i.e. the same lookback as the current best MA
window -- so this isolates "adaptive beta + mean" vs "fixed beta=1 + 33-bar
rolling mean" at one timescale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAIPEI_TZ = "Asia/Taipei"
BAR_MINUTES = 15
DEFAULT_INPUT = Path("data/processed/qff_tsm_spread_15m_taipei_qff_session.csv")
DEFAULT_OUT = Path("data/processed/qff_tsm_kalman_zscore_15m.csv")
DEFAULT_SEED_WINDOW = 33
DEFAULT_Q_RATIO = 0.0037  # ~33-bar effective memory on the level (see module docstring)


def kalman_filter(
    y: np.ndarray, g: np.ndarray, seed_window: int, q_ratio: float, ve_override: float | None
) -> dict[str, np.ndarray]:
    n = len(y)
    if n <= seed_window + 5:
        raise RuntimeError(f"Need more than {seed_window + 5} rows, got {n}")

    # OLS seed on the first seed_window bars: y ~ beta*g + alpha
    gs = g[:seed_window]
    ys = y[:seed_window]
    amat = np.vstack([gs, np.ones_like(gs)]).T
    coef, *_ = np.linalg.lstsq(amat, ys, rcond=None)
    resid = ys - amat @ coef
    ve = float(np.var(resid, ddof=2)) if ve_override is None else ve_override
    if ve <= 0:
        ve = 1e-8

    w = q_ratio * ve * np.eye(2)
    x = coef.astype(float).copy()              # [beta, alpha]
    p = np.eye(2) * ve * 10.0                   # mildly diffuse prior

    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    e_out = np.full(n, np.nan)
    q_out = np.full(n, np.nan)
    z_out = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for t in range(seed_window, n):
        # predict (random walk)
        p_pred = p + w
        h = np.array([g[t], 1.0])
        y_pred = h @ x
        e = y[t] - y_pred
        q = float(h @ p_pred @ h + ve)
        k = (p_pred @ h) / q
        x = x + k * e
        p = p_pred - np.outer(k, h @ p_pred)

        beta[t] = x[0]
        alpha[t] = x[1]
        e_out[t] = e
        q_out[t] = q
        z_out[t] = e / np.sqrt(q)
        valid[t] = True

    return {
        "beta": beta, "alpha": alpha, "spread": e_out,
        "q": q_out, "zscore": z_out, "valid": valid,
    }


def ou_diagnostics(spread: np.ndarray, timestamps: pd.Series, label: str) -> str:
    dt = timestamps.diff().dt.total_seconds().to_numpy()[1:] / 60.0
    consec = np.isclose(dt, BAR_MINUTES)
    s = spread[:-1]
    dy = np.diff(spread)
    mask = consec & ~np.isnan(s) & ~np.isnan(dy)
    x = s[mask]
    d = dy[mask]
    a, b = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, d, rcond=None)[0]
    hl = np.log(2) / (-b) * BAR_MINUTES if b < 0 else np.inf
    # ADF-ish t-stat on the lagged-level coefficient (5 lags)
    dyf = np.diff(spread)
    ylag = spread[:-1]
    rows = []
    for t in range(5, len(dyf)):
        if not consec[t] or not all(consec[t - i] for i in range(6)):
            continue
        if np.isnan(ylag[t]) or any(np.isnan(dyf[t - i]) for i in range(6)):
            continue
        rows.append((dyf[t], [1.0, ylag[t]] + [dyf[t - i] for i in range(1, 6)]))
    yv = np.array([r[0] for r in rows])
    xv = np.array([r[1] for r in rows])
    beta_adf, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    res = yv - xv @ beta_adf
    s2 = (res @ res) / (len(yv) - xv.shape[1])
    se = np.sqrt(s2 * np.diag(np.linalg.inv(xv.T @ xv)))
    t_adf = beta_adf[1] / se[1]
    return f"{label}: ADF t={t_adf:.2f}  half-life={hl:.0f}min  (n={len(yv)})"


def run_self_test() -> None:
    rng = np.random.default_rng(3)
    n = 2400
    g = np.cumsum(rng.standard_normal(n) * 0.01) + 7.0          # log regressor random walk
    beta_true = np.where(np.arange(n) < 1200, 1.20, 1.50)
    alpha_true = 0.5
    y = beta_true * g + alpha_true + rng.standard_normal(n) * 0.01
    out = kalman_filter(y, g, seed_window=33, q_ratio=0.02, ve_override=None)
    beta = out["beta"]
    if abs(np.nanmean(beta[600:1200]) - 1.20) > 0.1:
        raise RuntimeError(f"Self-test failed: beta did not converge to 1.20 ({np.nanmean(beta[600:1200]):.3f})")
    if abs(np.nanmean(beta[2200:2400]) - 1.50) > 0.15:
        raise RuntimeError(f"Self-test failed: beta did not track the step to 1.50 ({np.nanmean(beta[2200:2400]):.3f})")
    # z scale must be order 1 in a steady (constant-beta) window; the beta step
    # at t=1200 deliberately produces a large transient, so exclude it.
    z_steady = out["zscore"][600:1100]
    if not (0.6 < np.nanstd(z_steady) < 1.8):
        raise RuntimeError(f"Self-test failed: steady kalman z std off ({np.nanstd(z_steady):.3f})")
    print("Self-test passed: beta converges to 1.20, tracks step to 1.50, steady z ~ unit scale")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed-window", type=int, default=DEFAULT_SEED_WINDOW)
    p.add_argument("--q-ratio", type=float, default=DEFAULT_Q_RATIO)
    p.add_argument("--ve", type=float, default=None, help="Observation noise var; default from seed OLS residual.")
    p.add_argument("--skip-self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_self_test:
        run_self_test()

    frame = pd.read_csv(args.input)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(TAIPEI_TZ)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    for col in ["qff_close_filled", "tsm_twd_fair"]:
        if col not in frame.columns:
            raise RuntimeError(f"{args.input} missing required column {col}")
    qff = pd.to_numeric(frame["qff_close_filled"], errors="coerce").to_numpy(dtype=float)
    tsm = pd.to_numeric(frame["tsm_twd_fair"], errors="coerce").to_numpy(dtype=float)
    y = np.log(tsm)
    g = np.log(qff)

    out = kalman_filter(y, g, args.seed_window, args.q_ratio, args.ve)

    result = pd.DataFrame({
        "timestamp": frame["timestamp"],
        "qff_close": frame.get("qff_close", frame["qff_close_filled"]),
        "qff_close_filled": frame["qff_close_filled"],
        "tsm_twd_fair": frame["tsm_twd_fair"],
        "spread": out["spread"],            # Kalman log residual (dynamic spread)
        "spread_zscore": out["zscore"],
        "zscore_valid": out["valid"],
        "beta_t": out["beta"],
        "alpha_t": out["alpha"],
        "Q_t": out["q"],
    })
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
        r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    zv = out["zscore"][out["valid"]]
    beta_v = out["beta"][out["valid"]]
    print(f"Wrote {len(result):,} rows to {args.out}")
    print(f"beta_t: median={np.median(beta_v):.3f}  min={np.min(beta_v):.3f}  max={np.max(beta_v):.3f}  (OLS log-log ~1.23)")
    print(f"kalman z: std={np.nanstd(zv):.3f}  p5={np.nanpercentile(zv,5):.2f}  p95={np.nanpercentile(zv,95):.2f}  max|z|={np.nanmax(np.abs(zv)):.2f}")
    print(ou_diagnostics(out["spread"], frame["timestamp"], "kalman residual"))
    if "spread" in frame.columns:
        print(ou_diagnostics(pd.to_numeric(frame["spread"], errors="coerce").to_numpy(float),
                             frame["timestamp"], "current spread "))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
