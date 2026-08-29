"""Split the CCF/UMC spread into its two layers against the 2303 home shares.

The spread is two wedges glued together, and 2303 (UMC's TWSE listing) is the
benchmark that separates them. With everything in log points:

    premium_d = 100*ln(UMC_usclose * FX_usclose / 5 / 2303_close)   ADR premium
    basis_d   = 100*ln(CCF@13:29   / 2303_close)                    futures basis
    night_d   = 100*ln(CCF@13:29   / CCF_usclose)                   CCF 13:30 -> US close
    spread_usclose_d  =  premium_d - basis_d + night_d              (identity)

basis is synchronous (CCF day session and the TWSE closing auction overlap);
premium is the standard monitor convention (ADR close vs same-calendar-day TWSE
close). The identity is exact up to the spread formula's curvature.

Questions this answers, in order:

  1. Which layer carries the LEVEL drift of the spread (its -5..+4 wander)?
  2. Which layer does the overnight mean reversion? Regressing each layer's
     next-session change on the lagged spread deviation decomposes the
     session-level error-correction slope exactly into layer channels --
     the leg-level version of this was unidentifiable (leg returns are 0.98
     correlated), but 2303 nets the common factor out.
  3. Contract rolls (contract_month changes) and ex-dividend steps: how much
     of the basis series' movement is calendar mechanics rather than signal.
  4. Authorship: when the US session's intraday move widens the deviation, is
     it authored by the ADR/FX side or by CCF's own night prints -- and does
     the next Taiwan day close US-authored deviations faster? (QFF/TSM's
     2026-07-05 attribution study found exactly that pattern for its pair.)

Usage:
    python scripts/studies/spread_layers.py

Outputs land in data/runs/spread_layers/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import paths  # noqa: E402
from studies.spread_drivers import (  # noqa: E402
    SESSION_ROLLING_MEAN,
    add_minute_components,
    load_spread,
    nw_ols,
    session_frame,
    SPREAD_1M,
)

TWSE_2303_1D = paths.BARS / "twse" / "2303_1d.csv"
OUT_DIR = paths.DATA / "runs" / "spread_layers"

# CCF minute bar whose close is synchronous with the 13:30 TWSE closing
# auction: the bar stamped 13:29 covers 13:29-13:30.
TW_MATCH_TIME = "13:29"


# --------------------------------------------------------------------------
# building the layer table
# --------------------------------------------------------------------------


def load_2303() -> pd.DataFrame:
    df = pd.read_csv(TWSE_2303_1D)
    ts = pd.to_datetime(df["timestamp"])
    return pd.DataFrame(
        {"date": ts.dt.tz_localize(None).dt.normalize(), "twse_close": df["close"]}
    )


def load_ccf_1330() -> pd.DataFrame:
    """CCF's last 1m bar at or before 13:29 each day session, plus its
    contract month (for roll detection)."""
    ccf = pd.read_csv(paths.CCF1_1M, usecols=["timestamp", "close", "contract_month"])
    ts = pd.to_datetime(ccf["timestamp"])
    day = ccf[(ts.dt.time >= pd.Timestamp("09:00").time())
              & (ts.dt.time <= pd.Timestamp(f"{TW_MATCH_TIME}").time())].copy()
    day["date"] = pd.to_datetime(day["timestamp"]).dt.normalize()
    last = day.groupby("date").last().reset_index()
    return pd.DataFrame(
        {
            "date": last["date"].to_numpy(),
            "ccf_1330": last["close"].to_numpy(),
            "contract_month": last["contract_month"].to_numpy(),
            "ccf_1330_time": pd.to_datetime(last["timestamp"]).dt.time.to_numpy(),
        }
    )


def build_layers() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    minutes = add_minute_components(load_spread(SPREAD_1M))
    sess = session_frame(minutes).reset_index(names="session")
    sess["date"] = pd.to_datetime(sess["session"]).dt.tz_localize(None)

    twse = load_2303()
    ccf_day = load_ccf_1330()
    ccf_day["date"] = ccf_day["date"].dt.tz_localize(None)

    merged = sess.merge(twse, on="date", how="left").merge(
        ccf_day, on="date", how="left"
    )
    missing = merged["twse_close"].isna() | merged["ccf_1330"].isna()
    dropped = merged.loc[missing, "date"].dt.date.tolist()

    m = merged[~missing].copy()
    m["premium"] = 100.0 * np.log(
        m["us_close"] * m["fx_close"] / 5.0 / m["twse_close"]
    )
    m["basis"] = 100.0 * np.log(m["ccf_1330"] / m["twse_close"])
    m["night"] = 100.0 * np.log(m["ccf_1330"] / m["tw_close"])
    m["spread_log"] = m["premium"] - m["basis"] + m["night"]
    m["identity_resid"] = m["s_close"] - m["spread_log"]

    # The synchronous ADR-premium layer: ADR vs 2303 at the US close, with
    # 2303 marked to that instant THROUGH its own futures (CCF's 13:30 -> US
    # close move). premium and night each carry the 14.5h timing wedge, but
    # their sum nets it out: spread = adr_layer - basis, both terms
    # synchronous.
    m["adr_layer"] = m["s_close"] + m["basis"]

    # contract_month is NaN on rows sourced from tvdatafeed (only the TAIFEX
    # time-and-sales rows carry it), so forward-fill before detecting rolls.
    cm = m["contract_month"].ffill()
    m["roll_day"] = cm.ne(cm.shift(1))
    m.loc[m.index[0], "roll_day"] = False

    for col in ("premium", "basis", "night", "s_close", "adr_layer"):
        m[f"d_{col}"] = m[col] - m[col].shift(1)
    m["twse_ret"] = 100.0 * np.log(m["twse_close"] / m["twse_close"].shift(1))
    m["ccf_1330_ret"] = 100.0 * np.log(m["ccf_1330"] / m["ccf_1330"].shift(1))

    # Taiwan's +/-10% daily price limit: when 2303 is limit-locked the ADR
    # keeps trading, so premium measured against the locked price is
    # mechanical. Diffs here can span calendar gaps (holidays), so this only
    # flags true one-day limit moves approximately; 9.0 log pts ~ 9.4%.
    m["limit_day"] = m["twse_ret"].abs() >= 9.0

    return m, sess, {"dropped_dates": [str(d) for d in dropped]}


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------


def level_block(m: pd.DataFrame) -> dict:
    def stats(s: pd.Series) -> dict:
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    month = m["date"].dt.strftime("%Y-%m")
    monthly = (
        m.groupby(month)[["s_close", "adr_layer", "basis", "premium", "night"]]
        .mean()
        .round(3)
    )

    n = len(m)
    head, tail = m.iloc[:20], m.iloc[-20:]
    drift = {
        col: {
            "first20_mean": float(head[col].mean()),
            "last20_mean": float(tail[col].mean()),
            "drift": float(tail[col].mean() - head[col].mean()),
        }
        for col in ("s_close", "adr_layer", "basis")
    }

    # spread = adr_layer - basis, both synchronous: beta shares of the
    # session-level spread change.
    ds = m["d_s_close"]
    valid = ds.notna()
    var = float(ds[valid].var())
    change_shares = {
        "adr_layer": float(
            np.cov(m.loc[valid, "d_adr_layer"], ds[valid])[0, 1] / var
        ),
        "basis": -float(np.cov(m.loc[valid, "d_basis"], ds[valid])[0, 1] / var),
    }

    return {
        "sessions": int(n),
        "max_abs_identity_resid": float(m["identity_resid"].abs().max()),
        "levels": {
            c: stats(m[c])
            for c in ("s_close", "adr_layer", "basis", "premium", "night")
        },
        "monthly_means": monthly.to_dict(orient="index"),
        "first20_vs_last20": drift,
        "session_change_beta_shares": change_shares,
        "corr_dadr_dbasis": float(
            m.loc[valid, "d_adr_layer"].corr(m.loc[valid, "d_basis"])
        ),
        "limit_days": int(m["limit_day"].sum()),
    }


def reversion_by_layer(m: pd.DataFrame, exclude: pd.Series | None = None) -> dict:
    """Decompose the session error-correction slope into layer channels.

    dev_d is the spread's deviation at US close d. The next session's spread
    change is d_adr_layer - d_basis by identity, so the two slopes below sum
    (with the basis sign flipped) to the total reversion slope. ``exclude``
    drops sessions from the regression sample (the deviation itself is still
    computed on the full series).
    """
    dev = m["s_close"] - m["s_close"].rolling(
        SESSION_ROLLING_MEAN, min_periods=SESSION_ROLLING_MEAN
    ).mean()
    dev_lag = dev.shift(1)

    out = {}
    mask = dev_lag.notna()
    if exclude is not None:
        mask &= ~exclude
    for col in ("d_s_close", "d_adr_layer", "d_basis"):
        rows = mask & m[col].notna()
        out[col + "_on_lagged_dev"] = nw_ols(
            m.loc[rows, col].to_numpy(), dev_lag[rows].to_numpy(), lags=5
        )
    total = out["d_s_close_on_lagged_dev"]["slope"]
    out["share_adr_layer"] = out["d_adr_layer_on_lagged_dev"]["slope"] / total
    out["share_basis"] = -out["d_basis_on_lagged_dev"]["slope"] / total
    return out


def calendar_block(m: pd.DataFrame) -> dict:
    roll = m["roll_day"].astype(bool) & m["d_basis"].notna()
    other = ~m["roll_day"].astype(bool) & m["d_basis"].notna()

    biggest = m.loc[m["d_basis"].notna()].copy()
    biggest["abs_d_basis"] = biggest["d_basis"].abs()
    top = biggest.nlargest(10, "abs_d_basis")[
        ["date", "d_basis", "d_premium", "twse_ret", "ccf_1330_ret", "roll_day"]
    ]
    top["date"] = top["date"].dt.date.astype(str)

    # Ex-dividend: 2303 went ex NT$2.608 on 2026-07-08. TAIFEX adjusts stock
    # futures contracts for cash dividends (the dividend moves between the
    # two margin accounts and the contract value is unchanged), so unlike most
    # markets the futures carries NO dividend discount and the ex-day should
    # NOT step the basis. Report that day so the claim is checkable.
    exday = m[m["date"] == pd.Timestamp("2026-07-08")]
    exdiv = (
        {
            "twse_ret": float(exday["twse_ret"].iloc[0]),
            "ccf_1330_ret": float(exday["ccf_1330_ret"].iloc[0]),
            "d_basis": float(exday["d_basis"].iloc[0]),
            "d_s_close": float(exday["d_s_close"].iloc[0]),
        }
        if not exday.empty
        else None
    )

    return {
        "roll_days": int(roll.sum()),
        "mean_abs_d_basis_roll": float(m.loc[roll, "d_basis"].abs().mean()),
        "mean_abs_d_basis_other": float(m.loc[other, "d_basis"].abs().mean()),
        "mean_d_basis_roll": float(m.loc[roll, "d_basis"].mean()),
        "top10_abs_d_basis_days": top.to_dict(orient="records"),
        "ex_dividend_2026_07_08": exdiv,
    }


def authorship_block(sess: pd.DataFrame) -> dict:
    """Does the Taiwan day close US-authored deviations faster?

    Authorship of session d's US-hours move: |c_us_id + c_fx_id| vs |c_tw_id|.
    Reversion measured as the NEXT overnight spread change against the
    deviation at session d's close.
    """
    s = sess.copy()
    dev = s["s_close"] - s["s_close"].rolling(
        SESSION_ROLLING_MEAN, min_periods=SESSION_ROLLING_MEAN
    ).mean()
    c_fair_id = s["c_us_id"] + s["c_fx_id"]
    us_share = c_fair_id.abs() / (c_fair_id.abs() + s["c_tw_id"].abs())
    us_authored = (us_share > 0.5).astype(float)

    ds_on_next = s["ds_overnight"].shift(-1)
    base = dev.notna() & ds_on_next.notna() & us_share.notna()

    def group_fit(mask: pd.Series) -> dict:
        rows = base & mask
        fit = nw_ols(
            ds_on_next[rows].to_numpy(), dev[rows].to_numpy(), lags=5
        )
        fit["mean_us_share"] = float(us_share[rows].mean())
        return fit

    out = {
        "all": group_fit(pd.Series(True, index=s.index)),
        "us_authored": group_fit(us_authored == 1.0),
        "ccf_authored": group_fit(us_authored == 0.0),
    }

    # Interaction regression: ds_on_next = a + b*dev + c*(dev x us_authored)
    rows = base
    X = np.column_stack(
        [
            np.ones(int(rows.sum())),
            dev[rows].to_numpy(),
            (dev[rows] * us_authored[rows]).to_numpy(),
        ]
    )
    y = ds_on_next[rows].to_numpy()
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for lag in range(1, 6):
        w = 1.0 - lag / 6.0
        G = Xe[lag:].T @ Xe[:-lag]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    out["interaction"] = {
        "slope_dev": float(beta[1]),
        "slope_dev_t": float(beta[1] / se[1]),
        "slope_dev_x_us_authored": float(beta[2]),
        "slope_dev_x_us_authored_t": float(beta[2] / se[2]),
        "n": int(rows.sum()),
    }

    # The widening-conditioned version: sessions whose intraday move pushed
    # the spread AWAY from the mean by a meaningful amount.
    widened = (
        (np.sign(s["ds_intraday"]) == np.sign(dev))
        & (s["ds_intraday"].abs() >= s["ds_intraday"].abs().median())
    )
    out["widening_sessions_only"] = {
        "us_authored": group_fit(widened & (us_authored == 1.0)),
        "ccf_authored": group_fit(widened & (us_authored == 0.0)),
    }
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m, sess, meta = build_layers()

    result = {
        "parameters": {
            "spread_file": str(SPREAD_1M),
            "twse_file": str(TWSE_2303_1D),
            "ccf_match_time": TW_MATCH_TIME,
            "session_rolling_mean_sessions": SESSION_ROLLING_MEAN,
            **meta,
        },
        "levels": level_block(m),
        "reversion_by_layer": reversion_by_layer(m),
        # The July crash (2026-07-15..07-29: +9.5% then four ~-10% limit
        # sessions in 2303) is one episode; check the reversion result is not
        # only that fortnight.
        "reversion_by_layer_ex_crash": reversion_by_layer(
            m,
            exclude=(m["date"] >= pd.Timestamp("2026-07-15"))
            & (m["date"] <= pd.Timestamp("2026-07-29")),
        ),
        "calendar": calendar_block(m),
        "authorship": authorship_block(sess),
    }

    out_csv = m[
        [
            "date", "s_close", "adr_layer", "premium", "basis", "night",
            "identity_resid", "twse_close", "ccf_1330", "contract_month",
            "roll_day", "limit_day",
            "d_s_close", "d_adr_layer", "d_premium", "d_basis", "d_night",
            "twse_ret", "ccf_1330_ret",
        ]
    ].copy()
    out_csv["date"] = out_csv["date"].dt.date.astype(str)
    out_csv.to_csv(OUT_DIR / "layers.csv", index=False)
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(result, fh, indent=2)

    # ---- digest ----------------------------------------------------------
    lv = result["levels"]
    print(f"\n=== layers: {lv['sessions']} joined sessions "
          f"(dropped {len(meta['dropped_dates'])}: {meta['dropped_dates']}) ===")
    print(f"identity |resid| max {lv['max_abs_identity_resid']:.4f}; "
          f"limit days {lv['limit_days']}")
    print(f"{'':10s}{'mean':>8s}{'std':>8s}{'min':>8s}{'max':>8s}")
    for c in ("s_close", "adr_layer", "basis", "premium", "night"):
        st = lv["levels"][c]
        print(f"{c:10s}{st['mean']:8.3f}{st['std']:8.3f}"
              f"{st['min']:8.3f}{st['max']:8.3f}")
    print("\nmonthly means:")
    print(f"{'month':8s}{'spread':>8s}{'adr_lyr':>8s}{'basis':>8s}")
    for k, v in lv["monthly_means"].items():
        print(f"{k:8s}{v['s_close']:8.2f}{v['adr_layer']:8.2f}{v['basis']:8.2f}")
    dr = lv["first20_vs_last20"]
    print("\ndrift first20 -> last20: "
          + "  ".join(f"{c}={dr[c]['drift']:+.2f}" for c in dr))
    print("session-change beta shares: "
          + "  ".join(f"{k}={100 * v:.1f}%"
                      for k, v in lv["session_change_beta_shares"].items())
          + f"  corr(dA,dB)={lv['corr_dadr_dbasis']:+.3f}")

    for key, title in (
        ("reversion_by_layer", "overnight reversion decomposed by layer"),
        ("reversion_by_layer_ex_crash", "same, excluding 07-15..07-29 crash"),
    ):
        rv = result[key]
        print(f"\n=== {title} ===")
        for col in ("d_s_close", "d_adr_layer", "d_basis"):
            f = rv[f"{col}_on_lagged_dev"]
            print(f"{col:12s} slope {f['slope']:+.3f} (t={f['slope_t']:+.1f}, "
                  f"n={f['n']})")
        print(f"shares: adr_layer {100 * rv['share_adr_layer']:.0f}%  "
              f"basis {100 * rv['share_basis']:.0f}%")

    ca = result["calendar"]
    print(f"\n=== calendar mechanics ===")
    print(f"roll days: {ca['roll_days']}, mean|d_basis| roll "
          f"{ca['mean_abs_d_basis_roll']:.3f} vs other {ca['mean_abs_d_basis_other']:.3f}")
    ex = ca["ex_dividend_2026_07_08"]
    if ex:
        print(f"ex-div 2026-07-08 (NT$2.608, ~1.7%): 2303 {ex['twse_ret']:+.2f}, "
              f"CCF {ex['ccf_1330_ret']:+.2f}, d_basis {ex['d_basis']:+.2f} "
              f"-> no basis step (TAIFEX contract adjustment)")

    au = result["authorship"]
    print(f"\n=== authorship: next-overnight reversion of the close deviation ===")
    for k in ("all", "us_authored", "ccf_authored"):
        f = au[k]
        print(f"{k:14s} slope {f['slope']:+.3f} (t={f['slope_t']:+.1f}, "
              f"n={f['n']}, mean us_share {f['mean_us_share']:.2f})")
    it = au["interaction"]
    print(f"interaction dev x us_authored: {it['slope_dev_x_us_authored']:+.3f} "
          f"(t={it['slope_dev_x_us_authored_t']:+.1f})")
    for k, f in au["widening_sessions_only"].items():
        print(f"widening & {k:13s} slope {f['slope']:+.3f} "
              f"(t={f['slope_t']:+.1f}, n={f['n']})")

    print(f"\nWrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
