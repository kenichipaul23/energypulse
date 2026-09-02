"""
EnergyPulse - Data Utilities
=============================
Handles:
  1) Loading real commercial power consumption datasets (e.g. Kaggle
     ASHRAE - Great Energy Predictor III, PJM Interconnection, etc.)
  2) Generating a realistic SYNTHETIC dataset so the whole pipeline can be
     built, tested, and demoed before real data is on hand.

Expected "canonical" schema after loading (used everywhere else in the app):
    timestamp   : datetime64[ns]   - hourly timestamp
    consumption : float            - power consumption in kW
    building_id : str/int          - optional, defaults to 'BLDG_1'
    temperature : float            - optional outdoor temp (deg C), helps forecasts
"""

import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    start="2023-01-01",
    periods_days=365 * 2,
    freq="h",
    building_id="BLDG_1",
    base_load=120.0,
    seed=42,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic hourly commercial power consumption
    dataset with:
      - daily seasonality (low at night, ramps up during business hours)
      - weekly seasonality (lower on weekends)
      - yearly seasonality (higher load in hot/cold months -> HVAC use)
      - random peak-demand spike events (e.g. equipment startup, heat waves)
      - gaussian noise

    Returns a DataFrame with columns: timestamp, consumption, building_id, temperature
    """
    rng = np.random.default_rng(seed)
    n = periods_days * 24
    timestamps = pd.date_range(start=start, periods=n, freq=freq)

    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values           # 0=Mon ... 6=Sun
    doy = timestamps.dayofyear.values
    is_weekend = (dow >= 5).astype(float)

    # --- daily pattern: business hours 8am-6pm ramp up ---
    daily_pattern = 1.0 + 0.9 * np.exp(-((hour - 13) ** 2) / (2 * 4.0 ** 2))
    daily_pattern *= np.where((hour >= 6) & (hour <= 20), 1.0, 0.45)  # night setback

    # --- weekly pattern: weekends ~55% of weekday load ---
    weekly_factor = np.where(is_weekend == 1, 0.55, 1.0)

    # --- yearly / seasonal pattern: HVAC load higher in summer & winter ---
    yearly_factor = 1.0 + 0.35 * np.cos((doy - 200) / 365 * 2 * np.pi) ** 2

    # --- outdoor temperature (drives HVAC), roughly sinusoidal + noise ---
    temperature = 24 + 8 * np.sin((doy - 105) / 365 * 2 * np.pi) + rng.normal(0, 2.0, n)
    temperature += rng.normal(0, 0.5, n)  # hourly jitter

    # --- base consumption model ---
    consumption = base_load * daily_pattern * weekly_factor * yearly_factor

    # extra load when temperature is extreme (cooling/heating demand)
    hvac_extra = 0.6 * base_load * np.clip(np.abs(temperature - 23) / 10, 0, 2.5)
    consumption += hvac_extra

    # --- random peak-load spike events (rare, e.g. 1.5% of hours) ---
    spike_mask = rng.random(n) < 0.015
    consumption[spike_mask] *= rng.uniform(1.3, 1.8, spike_mask.sum())

    # --- gaussian noise ---
    consumption += rng.normal(0, base_load * 0.05, n)
    consumption = np.clip(consumption, a_min=5, a_max=None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "consumption": consumption.round(2),
            "building_id": building_id,
            "temperature": temperature.round(2),
        }
    )
    return df


def load_csv_dataset(
    filepath: str,
    timestamp_col: str = None,
    consumption_col: str = None,
    building_col: str = None,
    temperature_col: str = None,
) -> pd.DataFrame:
    """
    Loads a real-world CSV (e.g. downloaded from Kaggle: ASHRAE, PJM, etc.)
    and normalizes it to the canonical EnergyPulse schema.

    If column names aren't given, this attempts to auto-detect common ones.
    """
    df = pd.read_csv(filepath)

    def _find(col_candidates, given):
        if given and given in df.columns:
            return given
        for c in df.columns:
            if c.lower() in col_candidates:
                return c
        return None

    ts_col = _find(
        ["timestamp", "datetime", "date", "time", "meter_timestamp"], timestamp_col
    )
    cons_col = _find(
        ["consumption", "meter_reading", "load", "power", "kwh", "mw", "energy"],
        consumption_col,
    )
    bldg_col = _find(["building_id", "site_id", "meter_id", "building"], building_col)
    temp_col = _find(
        ["temperature", "air_temperature", "outdoor_temp", "temp"], temperature_col
    )

    if ts_col is None or cons_col is None:
        raise ValueError(
            f"Could not auto-detect timestamp/consumption columns. "
            f"Found columns: {list(df.columns)}. "
            f"Pass timestamp_col= and consumption_col= explicitly."
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[ts_col])
    out["consumption"] = pd.to_numeric(df[cons_col], errors="coerce")
    out["building_id"] = df[bldg_col] if bldg_col else "BLDG_1"
    out["temperature"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan

    out = out.dropna(subset=["timestamp", "consumption"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp", "building_id"])
    return out.reset_index(drop=True)


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print(df.head())
    print(df.describe())
    df.to_csv("synthetic_energypulse_data.csv", index=False)
    print("Saved synthetic_energypulse_data.csv with", len(df), "rows")
