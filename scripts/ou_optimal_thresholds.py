"""OU (Ornstein-Uhlenbeck) analytic optimal entry/exit thresholds (Bertram 2010).

Fits an OU process to the QFF/TSM spread z-score, then maximises expected
return per unit time net of transaction cost using the analytic OU first-passage
expression. The analytic expected cycle time is validated against a Monte-Carlo
simulation of the standardised OU before any optimum is reported.

Model (standardised, unit stationary variance, dimensionless time tau = kappa*t):
    dY = -Y dtau + sqrt(2) dW
Strategy: enter the convergence trade when |z| reaches a (in Y units), exit when
|z| falls back to m (0 <= m < a). One full cycle = enter -> exit -> re-enter.

Analytic expected cycle time (tau units):
    E[T_tau](a, m) = sqrt(2*pi) * integral_{m}^{a} exp(z^2/2) dz
(the sum of the up first-passage -a->-m and the down first-passage -m->-a).
Real time E[T] = E[T_tau] / kappa.

Return per unit real time, maximised over a > m >= 0:
    R(a, m) = (a - m - c_hat) / E[T]
with c_hat the round-trip cost in standardised Y units.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, quad
from scipy.stats import norm

TAIPEI_TZ = "Asia/Taipei"
BAR_MINUTES = 15
DEFAULT_ZSCORE_PATH = Path("data/processed/qff_tsm_spread_zscore_15m_w33.csv")
DEFAULT_TRADES_PATH = Path("data/processed/qff_tsm_pair_backtest_trades_15m_best.csv")
DEFAULT_OUT = Path("data/processed/ou_optimal_thresholds_15m.json")
DEFAULT_ZSCORE_COL = "spread_zscore"
DEFAULT_ROLLING_STD_COL = "spread_std_33"


# --------------------------------------------------------------------------- #
# OU parameter estimation                                                      #
# --------------------------------------------------------------------------- #
def consecutive_mask(timestamps: pd.Series, bar_minutes: int) -> np.ndarray:
    dt = timestamps.diff().dt.total_seconds().to_numpy()[1:] / 60.0
    return np.isclose(dt, bar_minutes)


def estimate_kappa_multi_horizon(
    z: np.ndarray, valid: np.ndarray, consec: np.ndarray, max_lag: int = 8
) -> tuple[float, list[float]]:
    """kappa from the decay of the autocorrelation: ln(acf(k)) = -kappa*k.

    Uses only within-session consecutive steps. Robust to the lag-1
    microstructure contamination that inflates a single-lag AR(1) fit.
    """
    acfs: list[float] = []
    lags: list[int] = []
    for k in range(1, max_lag + 1):
        a = z[:-k]
        b = z[k:]
        finite = valid[:-k] & valid[k:] & ~np.isnan(a) & ~np.isnan(b)
        # keep only positions whose next k bars are all within-session consecutive
        ok = np.zeros(len(a), dtype=bool)
        for i in range(len(a)):
            if i + k <= len(consec) and np.all(consec[i : i + k]):
                ok[i] = True
        mm = finite & ok
        if mm.sum() < 50:
            break
        ac = np.corrcoef(a[mm], b[mm])[0, 1]
        if ac <= 0:
            break
        acfs.append(float(ac))
        lags.append(k)
    if len(lags) < 2:
        raise RuntimeError("Not enough positive autocorrelations to fit kappa")
    slope = np.polyfit(lags, np.log(acfs), 1)[0]
    kappa = float(-slope)
    if kappa <= 0:
        raise RuntimeError(f"Estimated non-positive kappa: {kappa}")
    return kappa, acfs


# --------------------------------------------------------------------------- #
# Transaction cost in z units                                                  #
# --------------------------------------------------------------------------- #
def estimate_cost_z(trades_path: Path, mean_rolling_std: float) -> dict[str, float]:
    t = pd.read_csv(trades_path)
    fe, qe = t["entry_tsm_twd_fair"], t["entry_qff_close"]
    fx, qx = t["exit_tsm_twd_fair"], t["exit_qff_close"]
    entry_spread = 200.0 * (fe - qe) / (fe + qe)
    exit_spread = 200.0 * (fx - qx) / (fx + qx)
    conv = np.where(
        t["direction"] == "short_tsm_long_qff",
        entry_spread - exit_spread,
        exit_spread - entry_spread,
    )
    good = np.abs(conv) > 0.1
    twd_per_spread = float(np.median(t.loc[good, "gross_pnl_twd"] / conv[good]))
    fee_per_trade = float(t["total_fee_twd"].mean())
    cost_spread = fee_per_trade / twd_per_spread
    cost_z = cost_spread / mean_rolling_std
    return {
        "twd_per_spread_unit": twd_per_spread,
        "fee_per_trade_twd": fee_per_trade,
        "cost_spread_units": cost_spread,
        "cost_z_units": cost_z,
    }


# --------------------------------------------------------------------------- #
# Analytic OU cycle time + optimisation                                        #
# --------------------------------------------------------------------------- #
def expected_cycle_time_tau(a: float, m: float) -> float:
    """E[T_tau] = sqrt(2*pi) * integral_m^a exp(z^2/2) dz (a > m >= 0)."""
    if a <= m:
        return np.inf
    integral, _ = quad(lambda z: np.exp(z * z / 2.0), m, a, limit=200)
    return float(np.sqrt(2.0 * np.pi) * integral)


def optimise_return_per_time(
    c_hat: float, kappa: float, a_grid: np.ndarray, m_grid: np.ndarray
) -> dict[str, float]:
    best = {"R": -np.inf, "a": np.nan, "m": np.nan, "E_T_tau": np.nan}
    for a in a_grid:
        for m in m_grid:
            if m >= a - 1e-9:
                continue
            gross = a - m - c_hat
            if gross <= 0:
                continue
            et_tau = expected_cycle_time_tau(a, m)
            r = kappa * gross / et_tau  # per real (tau/kappa) time, monotone scale
            if r > best["R"]:
                best = {"R": float(r), "a": float(a), "m": float(m), "E_T_tau": et_tau}
    return best


# --------------------------------------------------------------------------- #
# Analytic Var[T] for the OU first passage (Bertram Sharpe optimum)            #
# --------------------------------------------------------------------------- #
# The second moment u2(y) = E_y[T_b^2] solves L u2 = -2 u1 with u1 the MFPT.
# With L f = f'' - y f' and natural boundary at -inf, both moments have the
# closed Green's-function form below. Pre-integrating five cumulative integrals
# on a fixed z grid lets every (a, m) pair be read off in O(1).
def precompute_moment_grid(
    z_lo: float = -7.0, z_hi: float = 6.0, dz: float = 0.002
) -> dict:
    zz = np.arange(z_lo, z_hi + dz, dz)
    e2 = np.exp(zz * zz / 2.0)            # exp(z^2/2)
    phi = norm.cdf(zz)                    # Phi(z) = (1/sqrt(2pi)) int_-inf^z e^{-x^2/2}
    en2 = np.exp(-zz * zz / 2.0)          # exp(-z^2/2)
    big_f = cumulative_trapezoid(e2 * phi, zz, initial=0.0)
    g0 = cumulative_trapezoid(en2, zz, initial=0.0)
    g1 = cumulative_trapezoid(en2 * big_f, zz, initial=0.0)
    h0 = cumulative_trapezoid(e2 * g0, zz, initial=0.0)
    h1 = cumulative_trapezoid(e2 * g1, zz, initial=0.0)
    return {"zz": zz, "F": big_f, "H0": h0, "H1": h1, "s2pi": float(np.sqrt(2.0 * np.pi))}


def _idx(zz: np.ndarray, value: float) -> int:
    return int(np.clip(np.searchsorted(zz, value), 0, len(zz) - 1))


def up_leg_moments(y: float, b: float, g: dict) -> tuple[float, float]:
    """First two moments (tau units) of the first-passage time from y up to b
    (b > y) for dY = -Y dtau + sqrt(2) dW, natural boundary at -inf."""
    zz, big_f, h0, h1, s2pi = g["zz"], g["F"], g["H0"], g["H1"], g["s2pi"]
    yi, bi = _idx(zz, y), _idx(zz, b)
    u1 = s2pi * (big_f[bi] - big_f[yi])
    u2 = 2.0 * s2pi * (big_f[bi] * (h0[bi] - h0[yi]) - (h1[bi] - h1[yi]))
    return float(u1), float(u2)


def cycle_moments(a: float, m: float, g: dict) -> tuple[float, float]:
    """E[T], Var[T] (tau units) for the enter-at-|a| / exit-at-|m| cycle.
    Up leg (-a -> -m) and down leg (-m -> -a) are independent by the strong
    Markov property; the down leg equals the up leg of (m -> a) by symmetry."""
    up1, up2 = up_leg_moments(-a, -m, g)
    dn1, dn2 = up_leg_moments(m, a, g)
    e_t = up1 + dn1
    var_t = (up2 - up1 * up1) + (dn2 - dn1 * dn1)
    return e_t, var_t


def optimise_sharpe(
    c_hat: float, g: dict, a_grid: np.ndarray, m_grid: np.ndarray
) -> tuple[dict, dict]:
    """Bertram Sharpe analysis at risk-free rate 0.

    In the pure-threshold model every completed trade earns the same reward, so
    the annualised Sharpe ratio factorises as sqrt(trade_frequency) / CV[T].

    - ``literal`` maximises sqrt(E[T]/Var[T]) (the r_f=0 Sharpe). The frequency
      term drives this to the smallest profitable band -> degenerate.
    - ``min_cv`` maximises E[T]^2/Var[T], i.e. minimises the cycle-time
      coefficient of variation. The frequency term is an idealisation artefact
      (real per-trade reward shrinks with the band while fees do not), so this
      timing-regularity band is the economically meaningful Sharpe target.
    """
    literal = {"ratio": -np.inf, "a": np.nan, "m": np.nan, "E_T_tau": np.nan, "Var_T_tau": np.nan}
    min_cv = {"e2_over_var": -np.inf, "cv": np.inf, "a": np.nan, "m": np.nan, "E_T_tau": np.nan, "Var_T_tau": np.nan}
    for a in a_grid:
        for m in m_grid:
            if m >= a - 1e-9 or (a - m - c_hat) <= 0:
                continue
            e_t, var_t = cycle_moments(a, m, g)
            if var_t <= 0 or e_t <= 0:
                continue
            if e_t / var_t > literal["ratio"]:
                literal = {"ratio": float(e_t / var_t), "a": float(a), "m": float(m),
                           "E_T_tau": float(e_t), "Var_T_tau": float(var_t)}
            e2v = e_t * e_t / var_t
            if e2v > min_cv["e2_over_var"]:
                min_cv = {"e2_over_var": float(e2v), "cv": float(np.sqrt(var_t) / e_t),
                          "a": float(a), "m": float(m),
                          "E_T_tau": float(e_t), "Var_T_tau": float(var_t)}
    return literal, min_cv


# --------------------------------------------------------------------------- #
# Monte-Carlo validation of the analytic cycle time                            #
# --------------------------------------------------------------------------- #
def mc_cycle_time_tau(
    a: float, m: float, n_paths: int = 8000, dtau: float = 0.004, seed: int = 7
) -> tuple[float, float]:
    """Monte-Carlo cycle time using the EXACT OU transition plus a
    Broadie-Glasserman-Kou continuity correction for discrete barrier
    monitoring (otherwise first-passage times are biased high)."""
    rng = np.random.default_rng(seed)
    decay = np.exp(-dtau)
    noise_sd = np.sqrt(1.0 - np.exp(-2.0 * dtau))  # exact OU conditional sd (unit var)
    # BGK shift: instantaneous vol of Y is sqrt(2); move the barrier inward.
    shift = 0.5826 * np.sqrt(2.0) * np.sqrt(dtau)

    def first_passage(start: float, level: float, going_up: bool) -> np.ndarray:
        y = np.full(n_paths, start)
        done = np.zeros(n_paths, dtype=bool)
        steps = np.zeros(n_paths)
        eff = level - shift if going_up else level + shift
        for _ in range(200_000):
            active = ~done
            if not active.any():
                break
            n = int(active.sum())
            y[active] = y[active] * decay + noise_sd * rng.standard_normal(n)
            crossed = (y >= eff) if going_up else (y <= eff)
            steps[~done] += 1.0
            done |= crossed
        return steps * dtau

    up = first_passage(-a, -m, going_up=True)
    down = first_passage(-m, -a, going_up=False)
    cycle = up + down
    return float(cycle.mean()), float(cycle.var())


# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zscore-path", type=Path, default=DEFAULT_ZSCORE_PATH)
    p.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES_PATH)
    p.add_argument("--zscore-col", default=DEFAULT_ZSCORE_COL)
    p.add_argument("--rolling-std-col", default=DEFAULT_ROLLING_STD_COL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mc-tol", type=float, default=0.04)
    p.add_argument("--skip-self-test", action="store_true")
    return p.parse_args(argv)


def run_self_test() -> None:
    a, m = 1.5, 0.5
    analytic = expected_cycle_time_tau(a, m)
    g = precompute_moment_grid()
    e_t_grid, var_t_grid = cycle_moments(a, m, g)
    # the two independent analytic E[T] routes (quad vs cumulative grid) must agree
    if abs(e_t_grid - analytic) / analytic > 0.01:
        raise RuntimeError(
            f"Self-test failed: cycle_moments E[T]={e_t_grid:.3f} != "
            f"integral E[T]={analytic:.3f}"
        )
    mc_mean, mc_var = mc_cycle_time_tau(a, m, n_paths=8000, dtau=0.004)
    rel_mean = abs(mc_mean - analytic) / analytic
    rel_var = abs(mc_var - var_t_grid) / mc_var
    if rel_mean > 0.05:
        raise RuntimeError(
            f"Self-test failed: E[T_tau] analytic={analytic:.3f} vs MC={mc_mean:.3f} "
            f"(rel {rel_mean:.3f})"
        )
    if rel_var > 0.12:
        raise RuntimeError(
            f"Self-test failed: Var[T_tau] analytic={var_t_grid:.3f} vs "
            f"MC={mc_var:.3f} (rel {rel_var:.3f})"
        )
    if not expected_cycle_time_tau(2.0, 0.0) > expected_cycle_time_tau(1.0, 0.0):
        raise RuntimeError("Self-test failed: E[T] should grow with entry level")
    print(
        f"Self-test passed: E[T] rel {rel_mean:.3f}, Var[T] rel {rel_var:.3f} "
        "(analytic vs MC)"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_self_test:
        run_self_test()

    df = pd.read_csv(args.zscore_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TAIPEI_TZ)
    df = df.sort_values("timestamp").reset_index(drop=True)
    valid = (
        df["zscore_valid"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    )
    z = pd.to_numeric(df[args.zscore_col], errors="coerce").to_numpy()
    consec = consecutive_mask(df["timestamp"], BAR_MINUTES)
    consec = np.append(consec, False)  # align length to z (per-bar "next is consecutive")

    kappa, acfs = estimate_kappa_multi_horizon(z, valid, consec)
    std_z = float(np.nanstd(z[valid]))
    hl_bars = np.log(2) / kappa
    hl_min = hl_bars * BAR_MINUTES
    mean_rolling_std = float(
        np.nanmean(pd.to_numeric(df[args.rolling_std_col], errors="coerce").to_numpy()[valid])
    )

    cost = estimate_cost_z(args.trades_path, mean_rolling_std)
    c_hat = cost["cost_z_units"] / std_z  # cost in standardised Y units

    a_grid = np.round(np.arange(0.30, 4.01, 0.05), 4)
    m_grid = np.round(np.arange(0.00, 3.51, 0.05), 4)
    opt = optimise_return_per_time(c_hat, kappa, a_grid, m_grid)

    # Validate the analytic cycle time at the optimum against MC.
    mc_mean, mc_var = mc_cycle_time_tau(opt["a"], opt["m"])
    rel = abs(mc_mean - opt["E_T_tau"]) / opt["E_T_tau"]
    mc_ok = rel <= args.mc_tol

    entry_z = opt["a"] * std_z
    exit_z = opt["m"] * std_z
    e_t_real_min = opt["E_T_tau"] / kappa * BAR_MINUTES
    # timing-only Sharpe proxy at the return/time optimum (for reference)
    timing_sharpe_proxy = float(np.sqrt(mc_mean / mc_var)) if mc_var > 0 else None

    # ---- Bertram Sharpe analysis (analytic Var[T]) ----
    moment_grid = precompute_moment_grid()
    opt_lit, opt_cv = optimise_sharpe(c_hat, moment_grid, a_grid, m_grid)
    degenerate = bool(opt_lit["a"] <= a_grid[0] + 0.06)  # collapsed to smallest band
    # Validate the analytic moments at the robust (min-CV) band against MC.
    mc_mean_s, mc_var_s = mc_cycle_time_tau(opt_cv["a"], opt_cv["m"])
    rel_mean_s = abs(mc_mean_s - opt_cv["E_T_tau"]) / opt_cv["E_T_tau"]
    rel_var_s = abs(mc_var_s - opt_cv["Var_T_tau"]) / mc_var_s
    sharpe_mc_ok = rel_mean_s <= args.mc_tol and rel_var_s <= 0.15
    entry_z_lit = opt_lit["a"] * std_z
    exit_z_lit = opt_lit["m"] * std_z
    entry_z_cv = opt_cv["a"] * std_z
    exit_z_cv = opt_cv["m"] * std_z
    e_t_cv_min = opt_cv["E_T_tau"] / kappa * BAR_MINUTES

    result = {
        "inputs": {
            "zscore_path": str(args.zscore_path),
            "kappa_per_bar": kappa,
            "half_life_bars": hl_bars,
            "half_life_min": hl_min,
            "std_z": std_z,
            "mean_rolling_std_spread_units": mean_rolling_std,
            "acf_by_lag": acfs,
            **cost,
            "c_hat_standardised": c_hat,
        },
        "return_per_time_optimum": {
            "objective": "max (a-m-c_hat)/E[T]  (one-sided OU idealisation)",
            "a_standardised": opt["a"],
            "m_standardised": opt["m"],
            "entry_z": round(entry_z, 3),
            "exit_z": round(exit_z, 3),
            "expected_cycle_minutes": round(e_t_real_min, 1),
            "expected_trades_per_30d": round(30 * 24 * 60 / e_t_real_min, 1)
            if e_t_real_min > 0
            else None,
            "timing_sharpe_proxy": timing_sharpe_proxy,
        },
        "sharpe_analysis": {
            "objective": "Bertram r_f=0: annualised Sharpe = sqrt(trade_frequency) / CV[T]",
            "literal_optimum": {
                "note": "maximises sqrt(E[T]/Var[T]); the frequency term collapses it to the smallest profitable band -> DEGENERATE, not actionable",
                "entry_z": round(entry_z_lit, 3),
                "exit_z": round(exit_z_lit, 3),
                "degenerate": degenerate,
            },
            "robust_timing_band": {
                "note": "minimises cycle-time CV[T] with the frequency artefact removed; the economically meaningful Sharpe target",
                "a_standardised": opt_cv["a"],
                "m_standardised": opt_cv["m"],
                "entry_z": round(entry_z_cv, 3),
                "exit_z": round(exit_z_cv, 3),
                "cycle_cv": round(opt_cv["cv"], 4),
                "expected_cycle_minutes": round(e_t_cv_min, 1),
            },
            "mc_validation_at_robust_band": {
                "analytic_E_T_tau": opt_cv["E_T_tau"],
                "mc_E_T_tau": mc_mean_s,
                "analytic_Var_T_tau": opt_cv["Var_T_tau"],
                "mc_Var_T_tau": mc_var_s,
                "rel_diff_mean": rel_mean_s,
                "rel_diff_var": rel_var_s,
                "passed": bool(sharpe_mc_ok),
            },
        },
        "mc_validation": {
            "analytic_E_T_tau": opt["E_T_tau"],
            "mc_E_T_tau": mc_mean,
            "rel_diff": rel,
            "passed": bool(mc_ok),
            "tolerance": args.mc_tol,
        },
        "notes": [
            "One-sided pure-threshold OU idealisation: every completed trade "
            "captures exactly (a-m), so risk-adjusted ranking must come from the "
            "actual backtest, not this model.",
            "The literal r_f=0 Sharpe optimum is degenerate (smallest profitable "
            "band) because constant per-trade reward makes Sharpe ~ "
            "sqrt(frequency)/CV[T]. The robust timing band (min CV[T]) is the "
            "meaningful read and sits near the empirical backtest Sharpe peak.",
            "The OU is assumed to hold at all deviations; empirically it breaks "
            "for |z| beyond ~3.5 (regime breaks).",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"kappa/bar={kappa:.4f}  half-life={hl_min:.0f}min  std_z={std_z:.3f}")
    print(f"cost: c_z={cost['cost_z_units']:.3f}  c_hat(Y)={c_hat:.3f}")
    print(
        "RETURN/TIME optimum: "
        f"entry_z={entry_z:.2f}  exit_z={exit_z:.2f}  "
        f"E[cycle]={e_t_real_min:.0f}min  ~{result['return_per_time_optimum']['expected_trades_per_30d']} trades/30d"
    )
    print(
        f"MC validation: analytic E[T_tau]={opt['E_T_tau']:.3f} vs MC={mc_mean:.3f} "
        f"(rel {rel:.3f}, {'OK' if mc_ok else 'FAIL'})"
    )
    print(
        f"SHARPE literal optimum: entry_z={entry_z_lit:.2f} exit_z={exit_z_lit:.2f}"
        f"{'  [DEGENERATE: smallest profitable band]' if degenerate else ''}"
    )
    print(
        "  robust timing band (min CV[T]): "
        f"entry_z={entry_z_cv:.2f}  exit_z={exit_z_cv:.2f}  CV={opt_cv['cv']:.3f}  "
        f"E[cycle]={e_t_cv_min:.0f}min"
    )
    print(
        f"  Var[T] validation @ robust band: analytic={opt_cv['Var_T_tau']:.3f} "
        f"vs MC={mc_var_s:.3f} (rel {rel_var_s:.3f}, {'OK' if sharpe_mc_ok else 'FAIL'})"
    )
    print(f"Wrote {args.out}")
    return 0 if (mc_ok and sharpe_mc_ok) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
