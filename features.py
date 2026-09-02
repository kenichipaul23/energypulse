"""
EnergyPulse - Feature Engineering
===================================
Turns the canonical (timestamp, consumption, building_id, temperature)
dataframe into a model-ready feature table:
  - calendar / time-based features
  - lag features (t-1h, t-24h, t-168h)
  - rolling statistics
  - peak-load class label (target for the classifier)
"""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["is_business_hour"] = ((df["hour_of_day"] >= 8) & (df["hour_of_day"] <= 18)).astype(int)

    # cyclical encodings so the model understands 23:00 is "close to" 00:00
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    group_col: str = "building_id",
    target_col: str = "consumption",
    lags=(1, 24, 168),          # 1hr, 1 day, 1 week
    rolling_windows=(24, 168),
) -> pd.DataFrame:
    df = df.copy().sort_values([group_col, "timestamp"])
    grouped = df.groupby(group_col)[target_col]

    for lag in lags:
        df[f"lag_{lag}h"] = grouped.shift(lag)

    for w in rolling_windows:
        df[f"rollmean_{w}h"] = (
            df.groupby(group_col)[target_col]
            .shift(1)
            .rolling(window=w, min_periods=max(1, w // 4))
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"rollstd_{w}h"] = (
            df.groupby(group_col)[target_col]
            .shift(1)
            .rolling(window=w, min_periods=max(1, w // 4))
            .std()
            .reset_index(level=0, drop=True)
        )
    return df


def add_peak_load_label(
    df: pd.DataFrame,
    target_col: str = "consumption",
    group_col: str = "building_id",
    off_peak_q: float = 0.40,
    peak_q: float = 0.80,
) -> pd.DataFrame:
    """
    Labels each hourly reading into a 3-class peak-load category based on
    that building's own historical consumption distribution:
        Off-Peak : bottom 40%
        Standard : middle 40-80%
        Peak     : top 20%
    This is the target the classification model will learn to predict.
    """
    df = df.copy()

    def _label(group):
        low, high = group[target_col].quantile([off_peak_q, peak_q])
        conditions = [group[target_col] <= low, group[target_col] >= high]
        choices = ["Off-Peak", "Peak"]
        group["load_class"] = np.select(conditions, choices, default="Standard")
        return group

    df = df.groupby(group_col, group_keys=False).apply(_label)
    return df


def build_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One-shot pipeline: raw canonical df -> full feature table with label."""
    df = add_time_features(raw_df)
    df = add_lag_and_rolling_features(df)
    df = add_peak_load_label(df)
    df = df.dropna().reset_index(drop=True)  # drop rows with NaN lags (start of series)
    return df


FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_business_hour",
    "lag_1h", "lag_24h", "lag_168h",
    "rollmean_24h", "rollstd_24h", "rollmean_168h", "rollstd_168h",
    "temperature",
]

if __name__ == "__main__":
    from data_utils import generate_synthetic_dataset

    raw = generate_synthetic_dataset(periods_days=120)
    feat = build_feature_table(raw)
    print(feat[["timestamp", "consumption", "load_class"] + FEATURE_COLUMNS].head())
    print("\nClass balance:\n", feat["load_class"].value_counts())
    print("\nRows before/after feature build:", len(raw), "->", len(feat))
