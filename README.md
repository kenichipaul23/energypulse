# EnergyPulse
**A Web-Based Commercial Power Consumption Forecasting and Peak-Load Classification System**

This is a working prototype you can run locally, demo to your panel, and
gradually replace synthetic data with a real dataset (ASHRAE, PJM, OPSD, or
your own smart-meter export).

## 1. Setup

```bash
# create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 2. Run the dashboard

```bash
streamlit run app.py
```

This opens a browser window at `http://localhost:8501` with:
- **Overview** — EDA charts (consumption over time, by hour of day, by day of week)
- **Forecasting** — trains the regression model, shows MAE/RMSE/MAPE, actual vs
  predicted chart, and feature importance
- **Peak-Load Classification** — trains the Random Forest classifier, shows
  Accuracy/Precision/Recall/F1, confusion matrix, and predicted classes over time
- **About** — pipeline summary

By default it uses realistic **synthetic data** (`data_utils.generate_synthetic_dataset`)
so you can develop and demo the whole system before your real dataset is ready.
Switch to **"Upload CSV"** in the sidebar once you have real data.

## 3. Using a real dataset

Download one of these (see the "Dataset Acquisition" notes from your proposal):
- Kaggle: **ASHRAE - Great Energy Predictor III**
- Kaggle: **PJM Interconnection Hourly Energy Consumption**
- **Open Power System Data (OPSD)**

Your CSV needs at minimum a timestamp column and a consumption/load column.
`data_utils.load_csv_dataset()` auto-detects common column names
(`timestamp`/`datetime`, `consumption`/`meter_reading`/`load`, etc.), or you
can pass them explicitly:

```python
from data_utils import load_csv_dataset
df = load_csv_dataset("my_export.csv",
                       timestamp_col="datetime",
                       consumption_col="kwh_total")
```

## 4. Project structure

```
energypulse/
├── app.py            # Streamlit web dashboard (entry point)
├── data_utils.py      # synthetic data generator + real CSV loader
├── features.py        # calendar/lag/rolling features + peak-load labeling
├── models.py           # forecasting (regression) + classification training/eval
├── requirements.txt
└── README.md
```

## 5. Running the pipeline outside the web app (for your paper's Chapter 4)

Each module also runs standalone and prints results, useful for generating
tables/figures for your paper:

```bash
python data_utils.py     # generates + previews synthetic dataset
python features.py       # shows feature table + class balance
python models.py         # trains both models, prints MAE/RMSE/MAPE + classification report
```

## 6. Methodology notes for your paper

- **Train/test split is chronological (80/20), not random** — this avoids
  data leakage in time-series problems (the model must never "see" future
  values during training). Mention this explicitly in your methodology.
- **Peak-load classes** are defined per building from its own historical
  consumption quantiles: bottom 40% = Off-Peak, middle 40–80% = Standard,
  top 20% = Peak. This threshold is adjustable in `features.add_peak_load_label()`
  and is a good place to justify your choice with literature or utility
  demand-charge tariff structures.
- **Forecasting model**: XGBoost Regressor if installed, else scikit-learn's
  GradientBoostingRegressor (automatic fallback, both are gradient-boosted
  tree ensembles — cite whichever is actually used from `models.HAS_XGBOOST`).
- **Classification model**: Random Forest, `class_weight="balanced"` to
  handle the natural imbalance (Peak hours are rarer than Off-Peak/Standard).
- **Evaluation metrics**: MAE, RMSE, MAPE (forecasting) and Accuracy,
  Precision, Recall, F1-score (classification) — matches Objective 5 in your
  proposal. You can additionally run a System Usability Scale (SUS) or
  ISO/IEC 25010-based questionnaire with end users for the "user acceptability"
  part of Objective 5.
