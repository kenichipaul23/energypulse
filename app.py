"""
EnergyPulse - Web Dashboard
==============================
A Streamlit web app for:
  - Loading a commercial power dataset (upload CSV or use built-in demo data)
  - Exploratory visualizations (daily/weekly/seasonal patterns)
  - Training + evaluating the forecasting model
  - Training + evaluating the peak-load classification model
  - Forecasting the next N hours and flagging predicted peak periods

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import generate_synthetic_dataset, load_csv_dataset
from features import build_feature_table, add_time_features, add_lag_and_rolling_features, FEATURE_COLUMNS
from models import run_full_pipeline, chronological_split
import theme
from theme import COLORS

st.set_page_config(page_title="EnergyPulse", page_icon="⚡", layout="wide")
theme.inject_css()

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
        <span style="font-size:1.4rem;">⚡</span>
        <span style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                     font-size:1.25rem; color:{COLORS['text']};">EnergyPulse</span>
    </div>
    <p class="ep-muted" style="font-size:0.85rem; margin-top:0;">
        Commercial power forecasting & peak-load classification
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("###### Data source")
data_source = st.sidebar.radio(
    "Data source",
    ["Use demo (synthetic) data", "Upload CSV"],
    label_visibility="collapsed",
    help="Demo data is realistic fake data generated on the fly, so you can "
         "explore the whole app before you have a real dataset.",
)

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload your dataset (CSV)", type=["csv"],
        help="Needs at least a timestamp column and a consumption/load column. "
             "Column names are auto-detected.",
    )
    if uploaded is None:
        st.sidebar.info("Upload a CSV with timestamp + consumption columns, "
                         "or switch to demo data to explore the app first.")
        st.stop()
    raw_df = load_csv_dataset(uploaded)
else:
    days = st.sidebar.slider(
        "Days of synthetic history", 60, 730, 365, step=30,
        help="More days = more history for the models to learn daily, weekly, "
             "and seasonal patterns from.",
    )
    raw_df = generate_synthetic_dataset(periods_days=days)

buildings = sorted(raw_df["building_id"].unique().tolist())
selected_building = st.sidebar.selectbox(
    "Building", buildings,
    help="Each building gets its own forecast and its own peak-load thresholds.",
)
building_df = raw_df[raw_df["building_id"] == selected_building].copy()

st.sidebar.markdown(
    f"""
    <div class="ep-sidebar-note">
        📄 Loaded <b style="color:{COLORS['text']};">{len(building_df):,}</b> hourly
        records for <code>{selected_building}</code>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
theme.hero(
    title="EnergyPulse",
    tagline="Web-based commercial power consumption forecasting & peak-load classification",
    badge="Live dashboard · demo data" if data_source != "Upload CSV" else "Live dashboard · your data",
)

tab_overview, tab_forecast, tab_classify, tab_about = st.tabs(
    ["📊  Overview", "🔮  Forecasting", "🚦  Peak-Load Classification", "ℹ️  About"]
)

# ---------------------------------------------------------------------------
# TAB 1: Overview / EDA
# ---------------------------------------------------------------------------
with tab_overview:
    theme.callout(
        "<b>What this tab shows:</b> a first look at the raw power data — how much "
        "electricity this building used, and when. No predictions yet, just the "
        "facts as recorded.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚡ Avg Consumption", f"{building_df['consumption'].mean():.1f} kW")
    col2.metric("🔺 Peak Consumption", f"{building_df['consumption'].max():.1f} kW")
    col3.metric("🔻 Min Consumption", f"{building_df['consumption'].min():.1f} kW")
    col4.metric("🗂️ Records", f"{len(building_df):,}")

    theme.section(
        "Consumption over time",
        "Every hourly reading in the dataset, left to right. Look for the daily "
        "'spikes' (business hours) and any longer-term rise or fall across months.",
    )
    fig = px.line(building_df, x="timestamp", y="consumption",
                   labels={"consumption": "Consumption (kW)", "timestamp": "Time"})
    fig.update_traces(line=dict(color=COLORS["amber"], width=1.2))
    st.plotly_chart(theme.style_fig(fig, height=340), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        theme.section(
            "Average load by hour of day",
            "Which hours typically draw the most power — useful for spotting the "
            "building's daily operating rhythm.",
        )
        hourly = building_df.assign(hour=building_df["timestamp"].dt.hour).groupby("hour")["consumption"].mean()
        fig2 = px.bar(hourly, labels={"value": "Avg kW", "hour": "Hour of Day"})
        fig2.update_traces(marker_color=COLORS["cyan"])
        st.plotly_chart(theme.style_fig(fig2, height=300).update_layout(showlegend=False),
                         use_container_width=True)
    with c2:
        theme.section(
            "Average load by day of week",
            "Weekday vs. weekend usage — a quick sanity check that the data "
            "reflects real operating patterns.",
        )
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = building_df.assign(dow=building_df["timestamp"].dt.dayofweek).groupby("dow")["consumption"].mean()
        dow.index = [dow_names[i] for i in dow.index]
        fig3 = px.bar(dow, labels={"value": "Avg kW", "index": "Day"})
        fig3.update_traces(marker_color=COLORS["cyan"])
        st.plotly_chart(theme.style_fig(fig3, height=300).update_layout(showlegend=False),
                         use_container_width=True)

# ---------------------------------------------------------------------------
# Build features + train models once, share across tabs (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _build_features(df):
    return build_feature_table(df)


@st.cache_resource(show_spinner=False)
def _train_models(cache_key, _feature_df):
    # `_feature_df` is underscore-prefixed so Streamlit skips hashing it
    # (large DataFrame); `cache_key` is the cheap hashable value that
    # actually determines cache invalidation.
    return run_full_pipeline(_feature_df, FEATURE_COLUMNS)


feature_df = _build_features(building_df)

if len(feature_df) < 200:
    st.warning("Not enough data after feature engineering (need lag history of at least 1 week). "
               "Upload more data or increase the synthetic history length.")
    st.stop()

with st.spinner("Training forecasting & classification models..."):
    cache_key = f"{selected_building}_{len(feature_df)}_{feature_df['timestamp'].iloc[-1]}"
    results = _train_models(cache_key, feature_df)

# ---------------------------------------------------------------------------
# TAB 2: Forecasting
# ---------------------------------------------------------------------------
with tab_forecast:
    theme.callout(
        "<b>What this tab shows:</b> a model trained on this building's own "
        "history, tested on data it never saw during training (the most recent "
        "20% of the timeline). Lower MAE/RMSE/MAPE = a more accurate forecast.",
        accent=COLORS["cyan"],
    )

    m = results["forecasting_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{m['MAE']} kW", help="Mean Absolute Error — on average, how far off each prediction is, in kW.")
    c2.metric("RMSE", f"{m['RMSE']} kW", help="Root Mean Squared Error — like MAE, but penalizes big misses more.")
    c3.metric("MAPE", f"{m['MAPE_%']}%", help="Mean Absolute Percentage Error — the average error as a % of actual usage.")
    st.markdown(
        f'<p class="ep-muted" style="font-size:0.85rem;">Model: '
        f'<b style="color:{COLORS["text"]};">'
        f'{"XGBoost" if results["using_xgboost"] else "Gradient Boosting (scikit-learn)"}</b> '
        f'Regressor · trained on a chronological 80/20 split (no shuffling, so the '
        f'test period is always in the future relative to training).</p>',
        unsafe_allow_html=True,
    )

    test_df = results["test_df"].copy()
    test_df["predicted"] = results["forecasting_preds"]

    theme.section(
        "Actual vs. predicted",
        "The solid line is what really happened; the dashed line is what the model "
        "guessed beforehand. The closer they track, the better the model.",
    )
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=test_df["timestamp"], y=test_df["consumption"],
                               name="Actual", line=dict(color=COLORS["amber"], width=1.4)))
    fig4.add_trace(go.Scatter(x=test_df["timestamp"], y=test_df["predicted"],
                               name="Predicted", line=dict(color=COLORS["cyan"], width=1.4, dash="dot")))
    fig4.update_layout(yaxis_title="Consumption (kW)")
    st.plotly_chart(theme.style_fig(fig4, height=380), use_container_width=True)

    st.markdown('<hr class="ep-hr">', unsafe_allow_html=True)

    theme.section(
        "What drives the forecast",
        "Which inputs the model relied on most. Longer bars = that feature had "
        "more influence on the prediction.",
    )
    fc_model = results["forecasting_model"]
    if hasattr(fc_model, "feature_importances_"):
        imp = pd.Series(fc_model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
        fig5 = px.bar(imp, orientation="h", labels={"value": "Importance", "index": "Feature"})
        fig5.update_traces(marker_color=COLORS["cyan"])
        st.plotly_chart(theme.style_fig(fig5, height=400).update_layout(showlegend=False),
                         use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Peak-Load Classification
# ---------------------------------------------------------------------------
with tab_classify:
    badges_html = (
        theme.badge("Off-Peak", COLORS["offpeak"])
        + theme.badge("Standard", COLORS["standard"])
        + theme.badge("Peak", COLORS["peak"])
    )
    theme.callout(
        f"<b>What this tab shows:</b> every hour gets sorted into one of three "
        f"buckets based on how much power was used, relative to this building's "
        f"own history: {badges_html}"
        f"<br><br>Off-Peak = bottom 40% of usage · Standard = middle 40% · "
        f"Peak = top 20% (the hours facility managers most want to anticipate "
        f"and manage).",
        accent=COLORS["peak"],
    )

    cm_metrics = results["classification_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", cm_metrics["Accuracy"], help="% of hours the model classified correctly.")
    c2.metric("Precision", cm_metrics["Precision_macro"], help="Of the hours predicted as a class, how many really were.")
    c3.metric("Recall", cm_metrics["Recall_macro"], help="Of the hours that really were a class, how many were caught.")
    c4.metric("F1 Score", cm_metrics["F1_macro"], help="Balance between precision and recall — higher is better.")
    st.markdown(
        f'<p class="ep-muted" style="font-size:0.85rem;">Model: '
        f'<b style="color:{COLORS["text"]};">Random Forest Classifier</b> · '
        f'evaluated on the same held-out future period as the forecast.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="ep-hr">', unsafe_allow_html=True)

    clf_model = results["classification_model"]
    labels = list(clf_model.classes_)
    cm = results["confusion_matrix"]

    col_left, col_right = st.columns([1, 1.3])
    with col_left:
        theme.section(
            "Confusion matrix",
            "Rows = what actually happened, columns = what the model guessed. "
            "A bright diagonal means the model is getting it right most of the time.",
        )
        fig6 = px.imshow(cm, x=labels, y=labels, text_auto=True,
                          color_continuous_scale=[COLORS["surface"], COLORS["amber"]],
                          labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(theme.style_fig(fig6, height=360), use_container_width=True)

    with col_right:
        theme.section(
            "Predicted class over the test period",
            "Each dot is one hour, colored by its predicted class — a quick visual "
            "check of when the model expects Peak demand to occur.",
        )
        test_df2 = results["test_df"].copy()
        test_df2["predicted_class"] = results["classification_preds"]
        color_map = {"Off-Peak": COLORS["offpeak"], "Standard": COLORS["standard"], "Peak": COLORS["peak"]}
        fig7 = px.scatter(test_df2, x="timestamp", y="consumption", color="predicted_class",
                           color_discrete_map=color_map,
                           labels={"consumption": "Consumption (kW)", "timestamp": "Time",
                                   "predicted_class": "Predicted class"})
        fig7.update_traces(marker=dict(size=4))
        st.plotly_chart(theme.style_fig(fig7, height=360), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: About
# ---------------------------------------------------------------------------
with tab_about:
    theme.callout(
        "<b>What this tab shows:</b> a plain-language walkthrough of how "
        "EnergyPulse works under the hood, for anyone new to the project.",
    )

    theme.section("What EnergyPulse does")
    st.markdown(
        "EnergyPulse forecasts how much power a commercial building will use, "
        "and flags which hours are likely to be **peak-load** — the periods "
        "facility managers most want to see coming, since that's when costs and "
        "grid strain are highest."
    )

    theme.section("How it's built, step by step")
    steps = [
        ("1", "Load the data", "data_utils.py",
         "Reads a real dataset (Kaggle ASHRAE / PJM / OPSD) or generates realistic "
         "synthetic data so the app works before real data is ready."),
        ("2", "Engineer features", "features.py",
         "Builds calendar features (hour, day of week, season), lag features "
         "(usage 1h / 24h / 1 week ago), and labels each hour's peak-load class."),
        ("3", "Train the models", "models.py",
         "A regression model forecasts next-hour usage; a Random Forest "
         "classifies each hour into Off-Peak / Standard / Peak. Both are tested "
         "on a future period they never trained on."),
        ("4", "Show the results", "app.py",
         "This dashboard — the Overview, Forecasting, and Peak-Load Classification "
         "tabs you've been looking at."),
    ]
    for num, title, file, desc in steps:
        st.markdown(
            f"""
            <div style="display:flex; gap:14px; margin-bottom:14px;">
                <div style="min-width:28px; height:28px; border-radius:8px;
                            background:{COLORS['surface_raised']}; border:1px solid {COLORS['border']};
                            display:flex; align-items:center; justify-content:center;
                            font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:{COLORS['amber']}; font-size:0.85rem;">{num}</div>
                <div>
                    <span style="font-weight:600;">{title}</span>
                    <code style="margin-left:8px; font-size:0.8rem; color:{COLORS['text_muted']};">{file}</code>
                    <div class="ep-muted" style="font-size:0.88rem; margin-top:2px;">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    theme.section("Evaluation metrics used")
    st.markdown(
        "- **Forecasting:** MAE, RMSE, MAPE\n"
        "- **Classification:** Accuracy, Precision, Recall, F1-score"
    )

    theme.callout(
        "Swap in a real dataset any time from the sidebar — click "
        "<b>\"Upload CSV\"</b>.",
        accent=COLORS["offpeak"],
    )