from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost.core import XGBoostError


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_INPUT_PATH = Path("data/processed/qff_tsm_spread_15m_taipei_qff_session.csv")
DEFAULT_PREDICTIONS_PATH = Path(
    "data/processed/qff_tsm_xgboost_spread_change_15m_predictions.csv"
)
DEFAULT_METRICS_CSV_PATH = Path(
    "data/processed/qff_tsm_xgboost_spread_change_15m_metrics.csv"
)
DEFAULT_METRICS_JSON_PATH = Path(
    "data/processed/qff_tsm_xgboost_spread_change_15m_metrics.json"
)
DEFAULT_FEATURE_IMPORTANCE_PATH = Path(
    "data/processed/qff_tsm_xgboost_spread_change_15m_feature_importance.csv"
)
DEFAULT_QUANTILES_PATH = Path(
    "data/processed/qff_tsm_xgboost_spread_change_15m_quantiles.csv"
)
DEFAULT_MODEL_DIR = Path("data/processed/qff_tsm_xgboost_spread_change_15m_models")
DEFAULT_REPORT_PATH = Path("qff_tsm_xgboost_spread_change_15m_report.html")

DEFAULT_HORIZONS = [1, 2, 4, 8, 16, 26]
DEFAULT_TRAIN_ROWS = 3700
DEFAULT_TEST_ROWS = 1000
DEFAULT_OBJECTIVE = "reg:pseudohubererror"
FALLBACK_OBJECTIVE = "reg:squarederror"
RANDOM_STATE = 42

DAY_START_MINUTE = 8 * 60 + 45
DAY_END_MINUTE = 13 * 60 + 45
NIGHT_START_MINUTE = 17 * 60 + 25
NIGHT_END_MINUTE = 5 * 60
BAR_MINUTES = 15

FEATURE_COLUMNS = [
    "spread",
    "spread_zscore_67",
    "spread_change_1",
    "spread_change_4",
    "spread_change_16",
    "spread_vol_16",
    "spread_vol_67",
    "tsm_fair_return_4",
    "tsm_fair_return_16",
    "qff_return_4",
    "qff_return_16",
    "is_day_session",
    "session_progress",
    "qff_was_filled",
]


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All horizons must be positive")
    return values


def format_int_list(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train conservative XGBoost regressors to predict future QFF/TSM "
            "15m spread change."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--horizons",
        type=parse_int_list,
        default=DEFAULT_HORIZONS,
        help=f"Comma-separated N-bar horizons. Default: {format_int_list(DEFAULT_HORIZONS)}",
    )
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--metrics-csv-out", type=Path, default=DEFAULT_METRICS_CSV_PATH)
    parser.add_argument("--metrics-json-out", type=Path, default=DEFAULT_METRICS_JSON_PATH)
    parser.add_argument(
        "--feature-importance-out",
        type=Path,
        default=DEFAULT_FEATURE_IMPORTANCE_PATH,
    )
    parser.add_argument("--quantiles-out", type=Path, default=DEFAULT_QUANTILES_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    return parser.parse_args(argv)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes"])


def read_input_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    frame = pd.read_csv(path)
    required = {
        "timestamp",
        "qff_close_filled",
        "qff_was_filled",
        "tsm_twd_fair",
        "spread",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise RuntimeError(f"Input CSV has no rows: {path}")

    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True).dt.tz_convert(
        TAIPEI_TZ
    )
    for column in ["qff_close_filled", "tsm_twd_fair", "spread"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["qff_was_filled"] = parse_bool_series(output["qff_was_filled"]).astype(int)

    invalid = output[["qff_close_filled", "tsm_twd_fair", "spread"]].isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        raise RuntimeError(f"Input has invalid numeric values:\n{invalid}")

    timestamps = pd.DatetimeIndex(output["timestamp"])
    if timestamps.has_duplicates:
        duplicates = timestamps[timestamps.duplicated()].unique()
        raise RuntimeError(
            f"Input has duplicate timestamps, first examples: {list(duplicates[:5])}"
        )
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Input timestamps must be sorted")

    return output


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    spread = output["spread"]
    spread_roll = spread.rolling(window=67, min_periods=67)
    spread_mean = spread_roll.mean()
    spread_std = spread_roll.std(ddof=0)
    output["spread_zscore_67"] = (spread - spread_mean) / spread_std.replace(0, np.nan)

    for lag in [1, 4, 16]:
        output[f"spread_change_{lag}"] = spread.diff(lag)

    spread_1bar_change = spread.diff()
    for window in [16, 67]:
        output[f"spread_vol_{window}"] = spread_1bar_change.rolling(
            window=window, min_periods=window
        ).std(ddof=0)

    for lag in [4, 16]:
        output[f"tsm_fair_return_{lag}"] = output["tsm_twd_fair"].pct_change(lag)
        output[f"qff_return_{lag}"] = output["qff_close_filled"].pct_change(lag)

    timestamps = pd.DatetimeIndex(output["timestamp"])
    minute = timestamps.hour * 60 + timestamps.minute
    is_day = (minute >= DAY_START_MINUTE) & (minute < DAY_END_MINUTE)
    is_night = (minute >= NIGHT_START_MINUTE) | (minute < NIGHT_END_MINUTE)
    if not (is_day | is_night).all():
        bad_rows = output.loc[~(is_day | is_night), ["timestamp"]].head(10)
        raise RuntimeError(f"Input contains non-session timestamps:\n{bad_rows}")

    day_elapsed = minute - DAY_START_MINUTE
    night_elapsed = np.where(
        minute >= NIGHT_START_MINUTE,
        minute - NIGHT_START_MINUTE,
        minute + 24 * 60 - NIGHT_START_MINUTE,
    )
    day_denominator = DAY_END_MINUTE - DAY_START_MINUTE - BAR_MINUTES
    night_denominator = 24 * 60 - NIGHT_START_MINUTE + NIGHT_END_MINUTE - BAR_MINUTES
    output["is_day_session"] = is_day.astype(int)
    output["session_progress"] = np.where(
        is_day,
        day_elapsed / day_denominator,
        night_elapsed / night_denominator,
    )
    output["session_progress"] = output["session_progress"].clip(0.0, 1.0)
    return output


def add_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    output = frame.copy()
    output["target"] = output["spread"].shift(-horizon) - output["spread"]
    output["target_timestamp"] = output["timestamp"].shift(-horizon)
    output["target_spread"] = output["spread"].shift(-horizon)
    output["target_row_index"] = np.arange(len(output)) + horizon
    return output


def split_horizon_frame(
    frame: pd.DataFrame, horizon: int, train_rows: int, test_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_rows <= 0 or test_rows <= 0:
        raise ValueError("--train-rows and --test-rows must be positive")
    test_start = train_rows
    test_end_exclusive = train_rows + test_rows
    if len(frame) < test_end_exclusive + horizon:
        raise RuntimeError(
            f"Need at least {test_end_exclusive + horizon} rows for "
            f"train_rows={train_rows}, test_rows={test_rows}, horizon={horizon}; "
            f"input has {len(frame)} rows"
        )

    data = add_targets(frame, horizon)
    feature_valid = data[FEATURE_COLUMNS].notna().all(axis=1)
    target_valid = data["target"].notna() & data["target_timestamp"].notna()
    row_index = np.arange(len(data))

    train_mask = (
        (row_index < train_rows)
        & (data["target_row_index"] < train_rows)
        & feature_valid
        & target_valid
    )
    test_mask = (
        (row_index >= test_start)
        & (row_index < test_end_exclusive)
        & (data["target_row_index"] < len(data))
        & feature_valid
        & target_valid
    )

    test_feature_count = int(((row_index >= test_start) & (row_index < test_end_exclusive)).sum())
    train = data.loc[train_mask].copy()
    test = data.loc[test_mask].copy()
    if len(test) != test_feature_count:
        raise RuntimeError(
            f"Expected {test_feature_count} usable test rows for horizon={horizon}, "
            f"got {len(test)}"
        )
    if train.empty or test.empty:
        raise RuntimeError(
            f"Train/test split produced empty data for horizon={horizon}: "
            f"train={len(train)}, test={len(test)}"
        )
    if int(train["target_row_index"].max()) >= train_rows:
        raise RuntimeError(f"Train target leaks into test for horizon={horizon}")
    if int(test.index.min()) != test_start or int(test.index.max()) != test_end_exclusive - 1:
        raise RuntimeError(f"Test feature rows are not exactly fixed for horizon={horizon}")
    return train, test


def make_model(args: argparse.Namespace, objective: str) -> XGBRegressor:
    return XGBRegressor(
        objective=objective,
        eval_metric="mae",
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def fit_model(args: argparse.Namespace, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[XGBRegressor, str]:
    model = make_model(args, DEFAULT_OBJECTIVE)
    try:
        model.fit(x_train, y_train)
        return model, DEFAULT_OBJECTIVE
    except XGBoostError as exc:
        print(
            f"WARNING: {DEFAULT_OBJECTIVE} failed ({exc}); "
            f"falling back to {FALLBACK_OBJECTIVE}",
            file=sys.stderr,
        )
        model = make_model(args, FALLBACK_OBJECTIVE)
        model.fit(x_train, y_train)
        return model, FALLBACK_OBJECTIVE


def finite_or_none(value: float | int | np.floating | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0:
        return None
    numerator = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - numerator / denominator


def correlation(y_true: pd.Series, y_pred: pd.Series, method: str) -> float | None:
    value = y_true.corr(y_pred, method=method)
    return finite_or_none(value)  # type: ignore[return-value]


def calculate_metrics(
    *,
    horizon: int,
    split: str,
    data: pd.DataFrame,
    y_pred: np.ndarray,
    objective: str,
) -> dict[str, Any]:
    y_true = data["target"].to_numpy(dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    direction_accuracy = float((np.sign(y_pred) == np.sign(y_true)).mean())
    zsign = np.sign(data["spread_zscore_67"].to_numpy(dtype=float))
    actual_convergence = -zsign * y_true
    expected_convergence = -zsign * y_pred
    high_threshold = np.quantile(expected_convergence, 0.9)
    high_mask = expected_convergence >= high_threshold

    return {
        "horizon": horizon,
        "split": split,
        "objective": objective,
        "rows": int(len(data)),
        "timestamp_start": data["timestamp"].iloc[0],
        "timestamp_end": data["timestamp"].iloc[-1],
        "target_timestamp_start": data["target_timestamp"].iloc[0],
        "target_timestamp_end": data["target_timestamp"].iloc[-1],
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": r2_score_manual(y_true, y_pred),
        "median_absolute_error": float(np.median(abs_error)),
        "pearson_ic": correlation(data["target"], pd.Series(y_pred, index=data.index), "pearson"),
        "spearman_ic": correlation(data["target"], pd.Series(y_pred, index=data.index), "spearman"),
        "directional_accuracy": direction_accuracy,
        "actual_convergence_mean": float(actual_convergence.mean()),
        "expected_convergence_mean": float(expected_convergence.mean()),
        "actual_convergence_hit_rate": float((actual_convergence > 0).mean()),
        "high_convergence_rows": int(high_mask.sum()),
        "high_convergence_actual_mean": float(actual_convergence[high_mask].mean()),
        "high_convergence_hit_rate": float((actual_convergence[high_mask] > 0).mean()),
    }


def prediction_rows(
    *, horizon: int, split: str, data: pd.DataFrame, y_pred: np.ndarray
) -> pd.DataFrame:
    zsign = np.sign(data["spread_zscore_67"].to_numpy(dtype=float))
    output = pd.DataFrame(
        {
            "horizon": horizon,
            "split": split,
            "row_index": data.index.to_numpy(),
            "target_row_index": data["target_row_index"].astype(int).to_numpy(),
            "timestamp": data["timestamp"].to_numpy(),
            "target_timestamp": data["target_timestamp"].to_numpy(),
            "spread": data["spread"].to_numpy(),
            "spread_zscore_67": data["spread_zscore_67"].to_numpy(),
            "target_spread": data["target_spread"].to_numpy(),
            "y_true": data["target"].to_numpy(),
            "y_pred": y_pred,
            "expected_convergence": -zsign * y_pred,
            "actual_convergence": -zsign * data["target"].to_numpy(dtype=float),
            "is_day_session": data["is_day_session"].astype(int).to_numpy(),
            "session_progress": data["session_progress"].to_numpy(),
            "qff_was_filled": data["qff_was_filled"].astype(int).to_numpy(),
        }
    )
    return output


def quantile_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (horizon, split), group in predictions.groupby(["horizon", "split"], sort=True):
        data = group.copy()
        quantile = pd.qcut(
            data["y_pred"], q=bins, labels=False, duplicates="drop"
        )
        data["prediction_quantile"] = quantile.astype("Int64") + 1
        table = (
            data.groupby("prediction_quantile", dropna=True)
            .agg(
                row_count=("y_true", "size"),
                y_pred_min=("y_pred", "min"),
                y_pred_max=("y_pred", "max"),
                y_pred_mean=("y_pred", "mean"),
                y_true_mean=("y_true", "mean"),
                y_true_median=("y_true", "median"),
                y_true_std=("y_true", "std"),
                directional_accuracy=(
                    "y_true",
                    lambda actual: float(
                        (
                            np.sign(data.loc[actual.index, "y_pred"].to_numpy())
                            == np.sign(actual.to_numpy())
                        ).mean()
                    ),
                ),
                expected_convergence_mean=("expected_convergence", "mean"),
                actual_convergence_mean=("actual_convergence", "mean"),
                actual_convergence_hit_rate=(
                    "actual_convergence",
                    lambda values: float((values > 0).mean()),
                ),
            )
            .reset_index()
        )
        table.insert(0, "split", split)
        table.insert(0, "horizon", horizon)
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def feature_importance_rows(
    model: XGBRegressor, horizon: int, objective: str
) -> list[dict[str, Any]]:
    booster = model.get_booster()
    gains = booster.get_score(importance_type="gain")
    weights = booster.get_score(importance_type="weight")
    covers = booster.get_score(importance_type="cover")
    rows = []
    for feature in FEATURE_COLUMNS:
        rows.append(
            {
                "horizon": horizon,
                "objective": objective,
                "feature": feature,
                "gain": float(gains.get(feature, 0.0)),
                "weight": float(weights.get(feature, 0.0)),
                "cover": float(covers.get(feature, 0.0)),
            }
        )
    return rows


def format_timestamp_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ["timestamp", "target_timestamp", "timestamp_start", "timestamp_end", "target_timestamp_start", "target_timestamp_end"]:
        if column in output.columns:
            timestamps = pd.to_datetime(output[column], utc=True).dt.tz_convert(TAIPEI_TZ)
            formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")
            output[column] = formatted.str.replace(
                r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True
            )
    return output


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S%z")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def render_table(frame: pd.DataFrame, float_digits: int = 6, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                if math.isnan(value):
                    text = ""
                else:
                    text = f"{value:.{float_digits}f}"
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_report(
    *,
    metrics: pd.DataFrame,
    quantiles: pd.DataFrame,
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    test_metrics = metrics[metrics["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values(
        ["spearman_ic", "directional_accuracy"], ascending=[False, False]
    )
    top_importance = (
        importance.sort_values(["horizon", "gain"], ascending=[True, False])
        .groupby("horizon", as_index=False)
        .head(8)
        .reset_index(drop=True)
    )
    test_quantiles = quantiles[quantiles["split"] == "test"].copy()

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>QFF/TSM XGBoost 15m Spread Change Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-bottom: 8px; }}
table {{ border-collapse: collapse; margin: 12px 0 28px; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th {{ background: #f4f4f4; position: sticky; top: 0; }}
td:first-child, th:first-child {{ text-align: left; }}
.note {{ color: #555; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>QFF/TSM XGBoost 15m Spread Change Report</h1>
<p class="note">Target: future_spread_change_N = spread[t+N] - spread[t]. Split: first 3,700 feature rows for train, next 1,000 for test, tail rows as target buffer only.</p>
<h2>Test Metrics Ranked By Spearman IC</h2>
{render_table(format_timestamp_columns(test_metrics), float_digits=6)}
<h2>All Metrics</h2>
{render_table(format_timestamp_columns(metrics), float_digits=6)}
<h2>Test Prediction Quantiles</h2>
{render_table(test_quantiles, float_digits=6)}
<h2>Top Feature Importance By Horizon</h2>
{render_table(top_importance, float_digits=6)}
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.train_rows <= 0:
        raise ValueError("--train-rows must be positive")
    if args.test_rows <= 0:
        raise ValueError("--test-rows must be positive")
    if args.n_estimators <= 0:
        raise ValueError("--n-estimators must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.max_depth <= 0:
        raise ValueError("--max-depth must be positive")
    for name in ["subsample", "colsample_bytree"]:
        value = getattr(args, name.replace("-", "_"), None)
        if value is not None and not (0 < value <= 1):
            raise ValueError(f"--{name} must be in (0, 1]")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    validate_args(args)
    base = add_features(read_input_frame(args.input))

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    all_importance: list[dict[str, Any]] = []
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        train, test = split_horizon_frame(
            base,
            horizon=horizon,
            train_rows=args.train_rows,
            test_rows=args.test_rows,
        )
        x_train = train[FEATURE_COLUMNS]
        y_train = train["target"]
        x_test = test[FEATURE_COLUMNS]
        y_test = test["target"]
        model, objective = fit_model(args, x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        model_path = args.model_dir / f"horizon_{horizon}.json"
        model.save_model(model_path)

        all_predictions.append(
            prediction_rows(
                horizon=horizon, split="train", data=train, y_pred=train_pred
            )
        )
        all_predictions.append(
            prediction_rows(horizon=horizon, split="test", data=test, y_pred=test_pred)
        )
        all_metrics.append(
            calculate_metrics(
                horizon=horizon,
                split="train",
                data=train,
                y_pred=train_pred,
                objective=objective,
            )
        )
        all_metrics.append(
            calculate_metrics(
                horizon=horizon,
                split="test",
                data=test,
                y_pred=test_pred,
                objective=objective,
            )
        )
        all_importance.extend(feature_importance_rows(model, horizon, objective))
        print(
            f"Horizon {horizon}: train_rows={len(train):,}, "
            f"test_rows={len(test):,}, objective={objective}, model={model_path}"
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    importance = pd.DataFrame(all_importance)
    quantiles = quantile_table(predictions)

    write_csv(format_timestamp_columns(predictions), args.predictions_out)
    write_csv(format_timestamp_columns(metrics), args.metrics_csv_out)
    write_csv(importance, args.feature_importance_out)
    write_csv(quantiles, args.quantiles_out)
    args.metrics_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json_out.write_text(
        json.dumps(json_safe({"metrics": all_metrics}), indent=2),
        encoding="utf-8",
    )
    write_report(
        metrics=metrics,
        quantiles=quantiles,
        importance=importance,
        output_path=args.report_out,
    )

    best = (
        metrics[metrics["split"] == "test"]
        .sort_values(["spearman_ic", "directional_accuracy"], ascending=[False, False])
        .iloc[0]
    )
    print(
        "Best test horizon by Spearman IC: "
        f"N={int(best['horizon'])}, "
        f"spearman_ic={best['spearman_ic']:.6f}, "
        f"directional_accuracy={best['directional_accuracy']:.6f}"
    )
    print(f"Wrote predictions: {args.predictions_out}")
    print(f"Wrote metrics: {args.metrics_csv_out}")
    print(f"Wrote feature importance: {args.feature_importance_out}")
    print(f"Wrote quantiles: {args.quantiles_out}")
    print(f"Wrote report: {args.report_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
