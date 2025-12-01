"""Train and optimize a TiDE model against budget scenarios."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel
from scipy.optimize import minimize

import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def _build_timeseries(df: pd.DataFrame, column: str) -> TimeSeries:
    """Helper that keeps arguments consistent."""
    return TimeSeries.from_dataframe(df, time_col="Period", value_cols=column)


def _scale_series(series: TimeSeries) -> tuple[TimeSeries, Scaler]:
    scaler = Scaler()
    return scaler.fit_transform(series), scaler


def _prepare_series(df: pd.DataFrame) -> tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, Scaler, Scaler, Scaler]:
    target_series = _build_timeseries(df, "# of cases cleared")
    future_covs = _build_timeseries(df, "Budget")
    past_covs = _build_timeseries(df, "Cost Per Warrant")
    target_goal_series = _build_timeseries(df, "Target")

    scaled_target, target_scaler = _scale_series(target_series)
    scaled_future_covs, future_scaler = _scale_series(future_covs)
    scaled_past_covs, past_scaler = _scale_series(past_covs)

    return (
        scaled_target,
        scaled_future_covs,
        scaled_past_covs,
        target_goal_series,
        target_scaler,
        future_scaler,
        past_scaler,
    )


def _extend_future_covariates(
    df: pd.DataFrame,
    last_budget: float,
    forecast_end_date: pd.Timestamp,
    future_scaler: Scaler,
) -> TimeSeries:
    last_hist_date = df["Period"].max()
    future_dates = pd.date_range(start=last_hist_date + pd.DateOffset(months=1), end=forecast_end_date, freq="MS")

    future_df = pd.DataFrame({
        "Period": future_dates,
        "Budget": last_budget,
        "Target": df["Target"].iloc[-1],
        "Cost Per Warrant": float("nan"),
        "# of cases cleared": float("nan"),
    })

    extended_df = pd.concat([df, future_df], ignore_index=True)
    extended_future_covs = _build_timeseries(extended_df, "Budget")
    return future_scaler.transform(extended_future_covs)


def _scenario_future_covariates(
    changes_x: Sequence[float],
    variables: Sequence[str],
    model: TiDEModel,
    scaled_extended_future_covs: TimeSeries,
) -> TimeSeries:
    horizon = len(changes_x) // len(variables)
    input_len = model.input_chunk_length
    required_len = input_len + horizon

    base_covs_series_slice = scaled_extended_future_covs.tail(required_len)
    base_covs_df = base_covs_series_slice.to_dataframe().reset_index(names="Period")

    start_index_of_changes = input_len
    changes_matrix = np.array(changes_x, dtype=float).reshape(horizon, len(variables))

    for var_index, var in enumerate(variables):
        historical_part = base_covs_df.loc[: start_index_of_changes - 1, var]
        adjustment_part = (
            base_covs_df.loc[start_index_of_changes:, var].to_numpy()
            * (1 + changes_matrix[:, var_index])
        )
        base_covs_df.loc[: start_index_of_changes - 1, var] = historical_part.values
        base_covs_df.loc[start_index_of_changes:, var] = adjustment_part

    return TimeSeries.from_dataframe(base_covs_df, time_col="Period", value_cols=list(variables))


def _predict_from_changes(
    changes_x: Sequence[float],
    variables: Sequence[str],
    model: TiDEModel,
    series_to_predict: TimeSeries,
    past_covs_hist: TimeSeries,
    scaled_extended_future_covs: TimeSeries,
    target_scaler: Scaler,
    horizon: int,
) -> TimeSeries:
    scenario_future_covs = _scenario_future_covariates(
        changes_x, variables, model, scaled_extended_future_covs
    )

    past_input = series_to_predict.tail(model.input_chunk_length)
    past_covs_input = past_covs_hist.tail(model.input_chunk_length)

    scaled_forecast = model.predict(
        n=horizon,
        series=past_input,
        past_covariates=past_covs_input,
        future_covariates=scenario_future_covs,
    )

    return target_scaler.inverse_transform(scaled_forecast)


def objective_path(
    changes_x: Sequence[float],
    variables: Sequence[str],
    model: TiDEModel,
    series_to_predict: TimeSeries,
    past_covs_hist: TimeSeries,
    scaled_extended_future_covs: TimeSeries,
    target_goal_series: TimeSeries,
    target_scaler: Scaler,
    horizon: int,
) -> float:
    forecast = _predict_from_changes(
        changes_x,
        variables,
        model,
        series_to_predict,
        past_covs_hist,
        scaled_extended_future_covs,
        target_scaler,
        horizon,
    )

    forecast_df = forecast.to_dataframe().reset_index(drop=True)
    goal_df = target_goal_series.tail(horizon).to_dataframe().reset_index(drop=True)

    errors = forecast_df["# of cases cleared"].to_numpy() - goal_df["Target"].to_numpy()
    squared_error = errors ** 2
    undershoot_mask = errors < 0
    penalty_factor = 3.0

    asymmetric_loss = squared_error.sum()
    asymmetric_loss += (penalty_factor - 1) * squared_error[undershoot_mask].sum()

    # Mean asymmetric squared error
    return asymmetric_loss / len(squared_error)


def _run_optimization(
    model: TiDEModel,
    scaled_target: TimeSeries,
    scaled_past_covs: TimeSeries,
    scaled_extended_future_covs: TimeSeries,
    target_goal_series: TimeSeries,
    target_scaler: Scaler,
    variables: Sequence[str],
    horizon: int,
) -> np.ndarray:
    total_changes = len(variables) * horizon
    initial_guess = np.zeros(total_changes)
    bounds = [(-3.0, 3.0)] * total_changes

    res = minimize(
        objective_path,
        x0=initial_guess,
        args=(
            variables,
            model,
            scaled_target,
            scaled_past_covs,
            scaled_extended_future_covs,
            target_goal_series,
            target_scaler,
            horizon,
        ),
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return res.x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and optimize a TiDE model scenario.")
    parser.add_argument(
        "--file",
        type=str,
        default=r"C:\\Users\\SANCHEZ\\Downloads\\Metric 3.01 and 3.05.xlsx",
        help="Path to the Excel workbook with MC.03.01 and MC.03.05 sheets.",
    )
    parser.add_argument("--goal-change", type=float, default=600.0, help="Increment applied to Target column.")
    parser.add_argument("--forecast-end", type=str, default="2026-03-01", help="Forecast end date (YYYY-MM-DD).")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/validation split ratio.")
    parser.add_argument("--input-chunk", type=int, default=12, help="Look-back window (months).")
    parser.add_argument("--output-chunk", type=int, default=6, help="Forecast horizon (months).")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for TiDE.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for TiDE.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for TiDE training.")
    parser.add_argument(
        "--sheet",
        type=str,
        default="MC.03.01",
        help="Sheet name containing the main metric.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    file_path = Path(args.file).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found at {file_path}")

    df = pd.read_excel(file_path, sheet_name=args.sheet).dropna().copy()
    df.rename(columns={"Cost": "Budget", "Overall Target (if any)": "Target"}, inplace=True)
    df["Period"] = pd.to_datetime(df["Period"])
    df.sort_values("Period", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Target"] = df["Target"] + args.goal_change

    (
        scaled_target,
        scaled_future_covs,
        scaled_past_covs,
        target_goal_series,
        target_scaler,
        future_scaler,
        _,
    ) = _prepare_series(df)

    train_target, val_target = scaled_target.split_after(args.train_ratio)
    train_future, val_future = scaled_future_covs.split_after(args.train_ratio)
    train_past, val_past = scaled_past_covs.split_after(args.train_ratio)

    model = TiDEModel(
        input_chunk_length=args.input_chunk,
        output_chunk_length=args.output_chunk,
        n_epochs=args.epochs,
        random_state=42,
        likelihood=None,
        optimizer_kwargs={"lr": 1e-3},
        dropout=args.dropout,
        batch_size=args.batch_size,
        add_encoders=None,
        save_checkpoints=True,
    )

    model.fit(
        series=train_target,
        past_covariates=train_past,
        future_covariates=train_future,
        val_series=val_target,
        val_past_covariates=val_past,
        val_future_covariates=val_future,
        verbose=True,
    )

    forecast_end_date = pd.to_datetime(args.forecast_end)
    scaled_extended_future_covs = _extend_future_covariates(
        df,
        last_budget=float(df["Budget"].iloc[-1]),
        forecast_end_date=forecast_end_date,
        future_scaler=future_scaler,
    )

    variables = ["Budget"]
    horizon = args.output_chunk
    best_changes = _run_optimization(
        model,
        scaled_target,
        scaled_past_covs,
        scaled_extended_future_covs,
        target_goal_series,
        target_scaler,
        variables,
        horizon,
    )

    best_forecast = _predict_from_changes(
        best_changes,
        variables,
        model,
        scaled_target,
        scaled_past_covs,
        scaled_extended_future_covs,
        target_scaler,
        horizon,
    )

    forecast_df = best_forecast.to_dataframe().reset_index().rename(columns={"index": "Period"})
    goal_df = target_goal_series.tail(horizon).to_dataframe().reset_index().rename(columns={"index": "Period"})
    goal_df_shifted = goal_df.copy()
    goal_df_shifted["Period"] = goal_df_shifted["Period"] + pd.DateOffset(months=horizon)

    print("\nOptimal Scenario Adjustments:")
    for idx, var in enumerate(variables):
        var_changes = best_changes[idx:: len(variables)]
        print(f"{var}: {np.round(var_changes * 100, 2)}% per period")

    comparison_df = pd.concat(
        [forecast_df[["Period", "# of cases cleared"]], goal_df[["Target"]]], axis=1
    )
    print("\nForecast vs Target Goal:")
    print(comparison_df)

    plt.figure(figsize=(10, 5))
    plt.plot(goal_df_shifted["Period"], goal_df_shifted["Target"], label="Target Goal", marker="o")
    plt.plot(
        forecast_df["Period"],
        forecast_df["# of cases cleared"],
        label="Forecast (Optimized)",
        marker="x",
    )
    plt.title("Optimized Forecast vs Goal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
