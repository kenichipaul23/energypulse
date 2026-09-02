"""
EnergyPulse - Modeling
========================
Two models are trained on the feature table produced by features.py:

  1. FORECASTING MODEL (regression)
     Predicts next-hour `consumption` (continuous kW value).
     Tries XGBoost first (better accuracy); falls back to
     GradientBoostingRegressor from scikit-learn if xgboost isn't installed.

  2. PEAK-LOAD CLASSIFICATION MODEL
     Predicts `load_class` in {Off-Peak, Standard, Peak} using RandomForest.

Both use a chronological (walk-forward) train/test split -- NEVER a random
split -- because shuffling time series causes data leakage (the model would
"see the future").
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split a time-ordered dataframe into train/test without shuffling."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ---------------------------------------------------------------------------
# 1) FORECASTING
# ---------------------------------------------------------------------------
def train_forecasting_model(train_df, feature_cols, target_col="consumption"):
    X_train, y_train = train_df[feature_cols], train_df[target_col]

    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
        )

    model.fit(X_train, y_train)
    return model


def evaluate_forecasting_model(model, test_df, feature_cols, target_col="consumption"):
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = float(np.mean(np.abs((y_test.values - preds) / np.clip(y_test.values, 1e-6, None))) * 100)

    metrics = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE_%": round(mape, 3)}
    return metrics, preds


# ---------------------------------------------------------------------------
# 2) PEAK-LOAD CLASSIFICATION
# ---------------------------------------------------------------------------
def train_classification_model(train_df, feature_cols, target_col="load_class"):
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, test_df, feature_cols, target_col="load_class"):
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    report = classification_report(y_test, preds, zero_division=0)

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision_macro": round(precision, 4),
        "Recall_macro": round(recall, 4),
        "F1_macro": round(f1, 4),
    }
    return metrics, preds, cm, report


# ---------------------------------------------------------------------------
# Convenience: run the full pipeline end-to-end
# ---------------------------------------------------------------------------
def run_full_pipeline(feature_df, feature_cols):
    train_df, test_df = chronological_split(feature_df, test_size=0.2)

    fc_model = train_forecasting_model(train_df, feature_cols)
    fc_metrics, fc_preds = evaluate_forecasting_model(fc_model, test_df, feature_cols)

    clf_model = train_classification_model(train_df, feature_cols)
    clf_metrics, clf_preds, cm, report = evaluate_classification_model(
        clf_model, test_df, feature_cols
    )

    results = {
        "forecasting_model": fc_model,
        "forecasting_metrics": fc_metrics,
        "forecasting_preds": fc_preds,
        "classification_model": clf_model,
        "classification_metrics": clf_metrics,
        "classification_preds": clf_preds,
        "confusion_matrix": cm,
        "classification_report": report,
        "train_df": train_df,
        "test_df": test_df,
        "using_xgboost": HAS_XGBOOST,
    }
    return results


if __name__ == "__main__":
    from data_utils import generate_synthetic_dataset
    from features import build_feature_table, FEATURE_COLUMNS

    raw = generate_synthetic_dataset(periods_days=365)
    feat = build_feature_table(raw)

    results = run_full_pipeline(feat, FEATURE_COLUMNS)

    print("Using XGBoost:", results["using_xgboost"])
    print("\n--- Forecasting metrics ---")
    print(results["forecasting_metrics"])
    print("\n--- Classification metrics ---")
    print(results["classification_metrics"])
    print("\n--- Classification report ---")
    print(results["classification_report"])
