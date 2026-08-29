"""What moves the CCF/UMC spread: a leg-level driver decomposition.

The spread is ``200*(fair - ccf)/(fair + ccf)`` with ``fair = UMC * USDTWD / 5``,
which is ``100*ln(fair/ccf)`` to first order. Every spread change is therefore,
up to a curvature term of order ``(spread/100)^2``, an exact sum of three leg
log-returns:

    d(spread) ~= 100*(dln UMC + dln USDTWD - dln CCF)

so "what drives the spread" turns into measurable questions:

  1. Beta shares: how much of d(spread) variance does each leg's return
     account for, ``cov(component, ds)/var(ds)``, summing to ~1?
  2. Staleness: how much of the minute-level variance happens while CCF simply
     has not printed -- a mechanical artifact where ds IS the fair-value move?
  3. Error correction: when the spread is stretched, does CCF move toward the
     fair value or the fair value toward CCF? Two regressions of leg returns on
     the lagged spread deviation, Newey-West errors, plus the implied half-life.
  4. Lead-lag: cross-correlation of CCF returns against fair-value returns at
     +/- k minutes.
  5. Session horizon: is the spread moved overnight (a window containing the
     whole Taiwan day session plus the US overnight gap) or intraday during US
     RTH -- and which leg carries the overnight move?

Runs on the trade-price 1m spread (2026-01 onward) and repeats the minute-level
blocks on the book-mid spread (Lux, 2026-08-07 onward) where the staleness
artifact does not exist.

Usage:
    python scripts/studies/spread_drivers.py

Outputs land in data/runs/spread_drivers/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import paths  # noqa: E402
from lib.sessions import us_session_day  # noqa: E402

SPREAD_1M = paths.DATA / "features" / "ccf_umc" / "spread_1m.csv"
SPREAD_1M_MID = paths.DATA / "features" / "ccf_umc" / "spread_1m_mid.csv"
OUT_DIR = paths.DATA / "runs" / "spread_drivers"

# The strategy's own window (w3900) so the deviation the error-correction
# regression sees is the one the backtest trades on.
ROLLING_MEAN_MINUTES = 3900
SESSION_ROLLING_MEAN = 10  # sessions, ~ the same horizon
NW_LAGS = 30
LEADLAG_MAX = 10


# --------------------------------------------------------------------------
# small numerics
# --------------------------------------------------------------------------


def nw_ols(y: np.ndarray, x: np.ndarray, lags: int) -> dict:
    """OLS of y on [1, x] with Newey-West (Bartlett) standard errors."""
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xe[lag:].T @ Xe[:-lag]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {
        "slope": float(beta[1]),
        "slope_se": float(se[1]),
        "slope_t": float(beta[1] / se[1]),
        "intercept": float(beta[0]),
        "n": int(len(y)),
    }


def beta_shares(ds: pd.Series, components: dict[str, pd.Series]) -> dict:
    """cov(component, ds) / var(ds) for each component; they sum to ~1."""
    mask = ds.notna()
    for c in components.values():
        mask &= c.notna()
    d = ds[mask]
    var = float(d.var())
    out = {
        name: float(np.cov(comp[mask], d)[0, 1] / var)
        for name, comp in components.items()
    }
    out["_n"] = int(mask.sum())
    out["_ds_var"] = var
    return out


def variance_terms(components: dict[str, pd.Series], ds: pd.Series) -> dict:
    """Full var/cov expansion of ds = sum(components), as shares of var(ds)."""
    mask = ds.notna()
    for c in components.values():
        mask &= c.notna()
    var = float(ds[mask].var())
    names = list(components)
    out = {}
    for i, a in enumerate(names):
        out[f"var({a})"] = float(components[a][mask].var() / var)
        for b in names[i + 1:]:
            cov = float(np.cov(components[a][mask], components[b][mask])[0, 1])
            out[f"2cov({a},{b})"] = 2.0 * cov / var
    return out


# --------------------------------------------------------------------------
# loading and per-minute frame
# --------------------------------------------------------------------------


def load_spread(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["session"] = us_session_day(pd.DatetimeIndex(df["timestamp"]))
    return df


def add_minute_components(df: pd.DataFrame) -> pd.DataFrame:
    """Within-session leg returns in spread points, and d(spread)."""
    g = df.groupby("session")
    df = df.copy()
    df["ds"] = g["spread"].diff()
    df["c_us"] = 100.0 * np.log(df["tsm_close"]).groupby(df["session"]).diff()
    df["c_fx"] = 100.0 * np.log(df["usdttwd_close"]).groupby(df["session"]).diff()
    df["c_tw"] = -100.0 * np.log(df["qff_close_filled"]).groupby(df["session"]).diff()
    df["c_fair"] = df["c_us"] + df["c_fx"]
    df["resid"] = df["ds"] - (df["c_us"] + df["c_fx"] + df["c_tw"])
    return df


def minute_block(df: pd.DataFrame, label: str) -> dict:
    """Beta shares, variance expansion and the staleness split, one dataset."""
    valid = df["ds"].notna()
    ds = df.loc[valid, "ds"]
    comp = {k: df.loc[valid, k] for k in ("c_us", "c_fx", "c_tw")}

    out = {
        "label": label,
        "rows": int(valid.sum()),
        "sessions": int(df.loc[valid, "session"].nunique()),
        "max_abs_identity_residual": float(df["resid"].abs().max()),
        "ds_std": float(ds.std()),
        "beta_shares": beta_shares(df["ds"], {k: df[k] for k in comp}),
        "variance_terms": variance_terms({k: df[k] for k in comp}, df["ds"]),
    }

    # Staleness split. On a stale minute CCF printed nothing, its return is 0
    # by construction, and ds is mechanically the fair-value move.
    stale = df["qff_was_filled"].astype(bool) & valid
    fresh = ~df["qff_was_filled"].astype(bool) & valid
    ss_total = float((ds**2).sum())
    out["stale_split"] = {
        "stale_minute_fraction": float(stale.sum() / valid.sum()),
        "ds_var_stale": float(df.loc[stale, "ds"].var()),
        "ds_var_fresh": float(df.loc[fresh, "ds"].var()),
        "share_of_ds_sumsq_on_stale_minutes": float(
            (df.loc[stale, "ds"] ** 2).sum() / ss_total
        ),
        "beta_shares_fresh_only": beta_shares(
            df["ds"].where(fresh), {k: df[k] for k in ("c_us", "c_fx", "c_tw")}
        ),
    }
    return out


# --------------------------------------------------------------------------
# lead-lag and error correction (minute level)
# --------------------------------------------------------------------------


def leadlag_table(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """corr(r_tw_t, c_fair_{t-k}): k>0 means the fair value leads CCF."""
    r_tw = -df["c_tw"]  # CCF's own return, positive sign
    rows = []
    session = df["session"]
    for k in range(-max_lag, max_lag + 1):
        shifted = df["c_fair"].shift(k)
        same = session == session.shift(k)
        mask = r_tw.notna() & shifted.notna() & same
        rows.append(
            {
                "lag_minutes": k,
                "corr": float(np.corrcoef(r_tw[mask], shifted[mask])[0, 1]),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def error_correction(df: pd.DataFrame, sample_start: str | None = None) -> dict:
    """Who closes the gap: leg returns regressed on the lagged deviation.

    Deviation is spread minus its 3900-minute rolling mean -- the strategy's
    own window -- computed over the concatenated session index. The regression
    sample is within-session minutes with a full window behind them;
    ``sample_start`` restricts the regression sample (not the window) so a
    long series can be compared like-for-like against a short one.
    """
    dev = df["spread"] - df["spread"].rolling(
        ROLLING_MEAN_MINUTES, min_periods=ROLLING_MEAN_MINUTES
    ).mean()
    dev_lag = dev.shift(1)
    same_session = df["session"] == df["session"].shift(1)

    r_tw = -df["c_tw"]
    mask = dev_lag.notna() & r_tw.notna() & df["c_fair"].notna() & same_session
    if sample_start is not None:
        mask &= df["timestamp"] >= pd.Timestamp(sample_start)
    x = dev_lag[mask].to_numpy()

    reg_tw = nw_ols(r_tw[mask].to_numpy(), x, NW_LAGS)
    reg_fair = nw_ols(df.loc[mask, "c_fair"].to_numpy(), x, NW_LAGS)
    reg_ds = nw_ols(df.loc[mask, "ds"].to_numpy(), x, NW_LAGS)

    kappa = reg_ds["slope"]  # ds_t = kappa * dev_{t-1}; kappa < 0 reverts
    half_life = float(np.log(2) / -kappa) if kappa < 0 else float("inf")
    lam_tw, lam_fair = reg_tw["slope"], reg_fair["slope"]
    denom = lam_tw - lam_fair
    return {
        "deviation_window_minutes": ROLLING_MEAN_MINUTES,
        "nw_lags": NW_LAGS,
        "ccf_return_on_lagged_dev": reg_tw,
        "fair_return_on_lagged_dev": reg_fair,
        "ds_on_lagged_dev": reg_ds,
        "half_life_minutes": half_life,
        # Fraction of the gap-closing done by CCF moving toward fair. 1.0 means
        # CCF does all the adjusting; 0.0 means the fair value does.
        "ccf_adjustment_share": float(lam_tw / denom) if denom != 0 else float("nan"),
    }


# --------------------------------------------------------------------------
# session level
# --------------------------------------------------------------------------


def session_frame(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("session")
    out = pd.DataFrame(
        {
            "s_open": g["spread"].first(),
            "s_close": g["spread"].last(),
            "us_open": g["tsm_close"].first(),
            "us_close": g["tsm_close"].last(),
            "fx_open": g["usdttwd_close"].first(),
            "fx_close": g["usdttwd_close"].last(),
            "tw_open": g["qff_close_filled"].first(),
            "tw_close": g["qff_close_filled"].last(),
            "bars": g["spread"].size(),
        }
    )
    out["ds_cc"] = out["s_close"] - out["s_close"].shift(1)
    out["ds_overnight"] = out["s_open"] - out["s_close"].shift(1)
    out["ds_intraday"] = out["s_close"] - out["s_open"]

    for leg, col in (("us", "us"), ("fx", "fx"), ("tw", "tw")):
        sign = -1.0 if leg == "tw" else 1.0
        out[f"c_{leg}_on"] = sign * 100.0 * np.log(
            out[f"{col}_open"] / out[f"{col}_close"].shift(1)
        )
        out[f"c_{leg}_id"] = sign * 100.0 * np.log(
            out[f"{col}_close"] / out[f"{col}_open"]
        )
        out[f"c_{leg}_cc"] = out[f"c_{leg}_on"] + out[f"c_{leg}_id"]
    return out


def session_block(sess: pd.DataFrame) -> dict:
    ds = sess["ds_cc"]
    var_cc = float(ds.var())
    cov_on_id = float(
        np.cov(
            sess["ds_overnight"][ds.notna()], sess["ds_intraday"][ds.notna()]
        )[0, 1]
    )
    out = {
        "sessions": int(ds.notna().sum()),
        "ds_cc_std": float(ds.std()),
        "beta_shares_close_to_close": beta_shares(
            ds, {k: sess[f"c_{k}_cc"] for k in ("us", "fx", "tw")}
        ),
        "overnight_vs_intraday": {
            "var_share_overnight": float(sess["ds_overnight"].var() / var_cc),
            "var_share_intraday": float(sess["ds_intraday"].var() / var_cc),
            "var_share_2cov": 2.0 * cov_on_id / var_cc,
            "beta_shares_overnight_legs": beta_shares(
                sess["ds_overnight"],
                {k: sess[f"c_{k}_on"] for k in ("us", "fx", "tw")},
            ),
            "beta_shares_intraday_legs": beta_shares(
                sess["ds_intraday"],
                {k: sess[f"c_{k}_id"] for k in ("us", "fx", "tw")},
            ),
        },
    }

    # The two stock legs share one company, so their session returns are near
    # collinear and beta shares blow past 100%. Report the raw ingredients too:
    # per-leg volatility and the leg-return correlation.
    r_us = sess["c_us_cc"]
    r_tw = -sess["c_tw_cc"]
    both = r_us.notna() & r_tw.notna()
    out["leg_stats_close_to_close"] = {
        "std_us_pts": float(sess["c_us_cc"].std()),
        "std_fx_pts": float(sess["c_fx_cc"].std()),
        "std_tw_pts": float(sess["c_tw_cc"].std()),
        "corr_us_tw_returns": float(
            np.corrcoef(r_us[both], r_tw[both])[0, 1]
        ),
    }

    # Which window does the mean reversion: overnight or intraday moves
    # regressed on the lagged deviation from a 10-session rolling mean.
    dev = sess["s_close"] - sess["s_close"].rolling(
        SESSION_ROLLING_MEAN, min_periods=SESSION_ROLLING_MEAN
    ).mean()
    dev_lag = dev.shift(1)
    for name in ("ds_overnight", "ds_intraday", "ds_cc"):
        mask = dev_lag.notna() & sess[name].notna()
        out[f"{name}_on_lagged_dev"] = nw_ols(
            sess.loc[mask, name].to_numpy(), dev_lag[mask].to_numpy(), lags=5
        )
    kappa = out["ds_cc_on_lagged_dev"]["slope"]
    out["session_half_life"] = float(np.log(2) / -kappa) if kappa < 0 else float("inf")

    # Who does the overnight reverting: the overnight ds splits leg-by-leg as
    # ds_on = (c_us_on + c_fx_on) + c_tw_on, so the two slopes below sum to the
    # overnight slope above. c_tw_on is CCF repricing through the Taiwan day
    # session; the fair part is the ADR's overnight gap plus FX.
    fair_on = sess["c_us_on"] + sess["c_fx_on"]
    mask = dev_lag.notna() & fair_on.notna() & sess["c_tw_on"].notna()
    x = dev_lag[mask].to_numpy()
    reg_fair_on = nw_ols(fair_on[mask].to_numpy(), x, lags=5)
    reg_tw_on = nw_ols(sess.loc[mask, "c_tw_on"].to_numpy(), x, lags=5)
    total = reg_fair_on["slope"] + reg_tw_on["slope"]
    out["overnight_reversion_by_leg"] = {
        "fair_gap_on_lagged_dev": reg_fair_on,
        "ccf_daysession_on_lagged_dev": reg_tw_on,
        "ccf_share_of_overnight_reversion": (
            float(reg_tw_on["slope"] / total) if total != 0 else float("nan")
        ),
    }
    return out


# --------------------------------------------------------------------------
# time-of-day profile
# --------------------------------------------------------------------------


def tod_profile(df: pd.DataFrame) -> pd.DataFrame:
    """|ds| and staleness by 15-minute bucket since session open."""
    minute_in_session = df.groupby("session").cumcount()
    bucket = (minute_in_session // 15) * 15
    valid = df["ds"].notna()
    out = (
        pd.DataFrame(
            {
                "bucket_min": bucket[valid],
                "abs_ds": df.loc[valid, "ds"].abs(),
                "stale": df.loc[valid, "qff_was_filled"].astype(bool),
                "ds_sq": df.loc[valid, "ds"] ** 2,
            }
        )
        .groupby("bucket_min")
        .agg(
            mean_abs_ds=("abs_ds", "mean"),
            ds_var=("ds_sq", "mean"),
            stale_share=("stale", "mean"),
            n=("abs_ds", "size"),
        )
        .reset_index()
    )
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = add_minute_components(load_spread(SPREAD_1M))
    result = {
        "parameters": {
            "spread_file": str(SPREAD_1M),
            "mid_file": str(SPREAD_1M_MID),
            "rolling_mean_minutes": ROLLING_MEAN_MINUTES,
            "session_rolling_mean_sessions": SESSION_ROLLING_MEAN,
            "nw_lags": NW_LAGS,
            "leadlag_max_minutes": LEADLAG_MAX,
            "period": [
                str(df["timestamp"].iloc[0]),
                str(df["timestamp"].iloc[-1]),
            ],
        },
        "minute_trade_price": minute_block(df, "trade-price 1m"),
        "error_correction_trade_price": error_correction(df),
    }

    sess = session_frame(df)
    result["session_level"] = session_block(sess)

    leadlag = leadlag_table(df, LEADLAG_MAX)
    leadlag.to_csv(OUT_DIR / "leadlag_trade_price.csv", index=False)
    tod = tod_profile(df)
    tod.to_csv(OUT_DIR / "tod_profile.csv", index=False)
    sess.reset_index(names="session").to_csv(OUT_DIR / "sessions.csv", index=False)

    # Book-mid robustness set: no trade-print staleness, short sample.
    if SPREAD_1M_MID.exists():
        mid = add_minute_components(load_spread(SPREAD_1M_MID))
        result["minute_book_mid"] = minute_block(mid, "book-mid 1m (Lux)")
        result["error_correction_book_mid"] = error_correction(mid)
        # Same August window on trade prices, so the mid-vs-trade comparison is
        # not confounded by the period.
        result["error_correction_trade_price_aug"] = error_correction(
            df, sample_start=str(mid["timestamp"].iloc[0])
        )
        leadlag_table(mid, LEADLAG_MAX).to_csv(
            OUT_DIR / "leadlag_book_mid.csv", index=False
        )

    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(result, fh, indent=2)

    # ---- human-readable digest ------------------------------------------
    def pct(x: float) -> str:
        return f"{100 * x:6.1f}%"

    mb = result["minute_trade_price"]
    print(f"\n=== {mb['label']}: {mb['rows']:,} minutes, {mb['sessions']} sessions ===")
    print(f"identity residual max |.|: {mb['max_abs_identity_residual']:.2e}")
    print(f"d(spread) std: {mb['ds_std']:.4f} pts/min")
    print("beta shares of var(ds):  "
          + "  ".join(f"{k}={pct(v)}" for k, v in mb["beta_shares"].items()
                      if not k.startswith("_")))
    st = mb["stale_split"]
    print(f"stale CCF minutes: {pct(st['stale_minute_fraction'])} of minutes, "
          f"carrying {pct(st['share_of_ds_sumsq_on_stale_minutes'])} of sum ds^2")
    print("fresh-only beta shares:  "
          + "  ".join(f"{k}={pct(v)}" for k, v in
                      st["beta_shares_fresh_only"].items()
                      if not k.startswith("_")))

    ec = result["error_correction_trade_price"]
    print(f"\nerror correction (dev = spread - {ROLLING_MEAN_MINUTES}m mean):")
    print(f"  CCF leg:  slope {ec['ccf_return_on_lagged_dev']['slope']:+.5f} "
          f"(t={ec['ccf_return_on_lagged_dev']['slope_t']:+.1f})")
    print(f"  fair leg: slope {ec['fair_return_on_lagged_dev']['slope']:+.5f} "
          f"(t={ec['fair_return_on_lagged_dev']['slope_t']:+.1f})")
    print(f"  half-life {ec['half_life_minutes']:.0f} min; "
          f"CCF does {pct(ec['ccf_adjustment_share'])} of the adjusting")

    sl = result["session_level"]
    print(f"\n=== session level: {sl['sessions']} close-to-close changes ===")
    print(f"ds std {sl['ds_cc_std']:.4f} pts/session")
    print("beta shares c2c:  "
          + "  ".join(f"{k}={pct(v)}" for k, v in
                      sl["beta_shares_close_to_close"].items()
                      if not k.startswith("_")))
    oi = sl["overnight_vs_intraday"]
    print(f"variance split: overnight {pct(oi['var_share_overnight'])}, "
          f"intraday {pct(oi['var_share_intraday'])}, "
          f"2cov {pct(oi['var_share_2cov'])}")
    print("overnight legs:  "
          + "  ".join(f"{k}={pct(v)}" for k, v in
                      oi["beta_shares_overnight_legs"].items()
                      if not k.startswith("_")))
    print("intraday legs:   "
          + "  ".join(f"{k}={pct(v)}" for k, v in
                      oi["beta_shares_intraday_legs"].items()
                      if not k.startswith("_")))
    ls = sl["leg_stats_close_to_close"]
    print(f"leg stds (pts/session): us {ls['std_us_pts']:.2f}  "
          f"fx {ls['std_fx_pts']:.2f}  tw {ls['std_tw_pts']:.2f}  "
          f"corr(r_us, r_tw)={ls['corr_us_tw_returns']:.3f}")
    print(f"reversion: overnight slope "
          f"{sl['ds_overnight_on_lagged_dev']['slope']:+.3f} "
          f"(t={sl['ds_overnight_on_lagged_dev']['slope_t']:+.1f}), intraday "
          f"{sl['ds_intraday_on_lagged_dev']['slope']:+.3f} "
          f"(t={sl['ds_intraday_on_lagged_dev']['slope_t']:+.1f}); "
          f"session half-life {sl['session_half_life']:.1f}")
    orl = sl["overnight_reversion_by_leg"]
    print(f"overnight reversion by leg: CCF day-session slope "
          f"{orl['ccf_daysession_on_lagged_dev']['slope']:+.3f} "
          f"(t={orl['ccf_daysession_on_lagged_dev']['slope_t']:+.1f}), "
          f"ADR gap+FX slope {orl['fair_gap_on_lagged_dev']['slope']:+.3f} "
          f"(t={orl['fair_gap_on_lagged_dev']['slope_t']:+.1f}); CCF does "
          f"{pct(orl['ccf_share_of_overnight_reversion'])}")

    if "minute_book_mid" in result:
        mm = result["minute_book_mid"]
        print(f"\n=== {mm['label']}: {mm['rows']:,} minutes ===")
        print("beta shares of var(ds):  "
              + "  ".join(f"{k}={pct(v)}" for k, v in mm["beta_shares"].items()
                          if not k.startswith("_")))
        em = result["error_correction_book_mid"]
        ea = result["error_correction_trade_price_aug"]
        print(f"error correction: CCF share {pct(em['ccf_adjustment_share'])}, "
              f"half-life {em['half_life_minutes']:.0f} min "
              f"(n={em['ds_on_lagged_dev']['n']:,})")
        print(f"trade-price, same window: CCF share "
              f"{pct(ea['ccf_adjustment_share'])}, "
              f"half-life {ea['half_life_minutes']:.0f} min "
              f"(n={ea['ds_on_lagged_dev']['n']:,})")

    ll = leadlag
    lead = ll[ll["lag_minutes"] > 0]["corr"].idxmax()
    print(f"\nlead-lag (trade price): peak corr(r_ccf_t, r_fair_(t-k)) at "
          f"k={ll.loc[lead, 'lag_minutes']:.0f}m, corr={ll.loc[lead, 'corr']:.3f}; "
          f"k=0 corr={ll.loc[ll['lag_minutes'] == 0, 'corr'].iloc[0]:.3f}")
    print(f"\nWrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
