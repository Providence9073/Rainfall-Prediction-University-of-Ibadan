import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta, datetime

st.set_page_config(
    page_title="UI Rainfall Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "tuned_gradient.pkl"
APP_VERSION = "1.2"

# University of Ibadan
LATITUDE = 7.3912
LONGITUDE = 3.9167
LOCATION_NAME = "University of Ibadan, Ibadan, Nigeria"
TIMEZONE = "Africa/Lagos"

FEATURES = ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'relative_humidity_2m_mean', 'relative_humidity_2m_max', 'relative_humidity_2m_min', 'pressure_msl_mean', 'pressure_msl_max', 'pressure_msl_min', 'dew_point_2m_mean', 'wind_speed_10m_mean', 'wind_speed_10m_max', 'cloud_cover_mean', 'rain_sum', 'rain_yesterday', 'rain_2_days_ago', 'rain_3_days_ago', 'rain_3_day_total', 'rain_7_day_total', 'rain_7_day_average', 'temperature_7_day_average', 'humidity_7_day_average', 'pressure_7_day_average', 'month', 'day_of_year', 'day_of_week']

st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.block-container { padding-top: 1.2rem; max-width: 1450px; }
.hero {
    padding: 2rem 2.2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f3d56 0%, #176b87 55%, #3aa6b9 100%);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 2.35rem; }
.hero p { margin: .45rem 0 0; opacity: .92; font-size: 1.05rem; }
.card {
    background: white; padding: 1.15rem; border-radius: 16px;
    box-shadow: 0 3px 15px rgba(15,61,86,.08);
    border: 1px solid #e7edf3; margin-bottom: 1rem;
}
.metric {
    background: white; padding: 1rem; border-radius: 15px;
    border: 1px solid #e7edf3; text-align: center;
}
.metric .value { font-size: 1.65rem; font-weight: 750; color: #0f3d56; }
.metric .label { color: #667788; font-size: .88rem; }
.prediction {
    background: linear-gradient(135deg, #ffffff, #eef8fb);
    padding: 1.5rem; border-radius: 18px; border: 2px solid #b9e1ea;
    text-align: center;
}
.prediction h2 { margin-bottom: .2rem; }
.small-note { color:#687786; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=3600)
def load_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": ",".join([
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "relative_humidity_2m_mean", "relative_humidity_2m_max",
            "relative_humidity_2m_min", "pressure_msl_mean",
            "pressure_msl_max", "pressure_msl_min", "dew_point_2m_mean",
            "wind_speed_10m_mean", "wind_speed_10m_max", "cloud_cover_mean",
            "precipitation_sum", "rain_sum", "precipitation_hours"
        ]),
        "timezone": TIMEZONE
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df

def engineer_features(df):
    """Reproduce the temporal features used by the supplied model."""
    x = df.copy().sort_values("date").reset_index(drop=True)

    x["year"] = x["date"].dt.year
    x["month"] = x["date"].dt.month
    x["month_name"] = x["date"].dt.month_name()
    x["day_of_year"] = x["date"].dt.dayofyear
    x["day_of_week"] = x["date"].dt.dayofweek

    # Seasonal labels used in the project.
    x["season"] = np.select(
        [
            x["month"].isin([12, 1, 2]),
            x["month"].isin([3, 4, 5]),
            x["month"].isin([6, 7, 8])
        ],
        ["Dry", "Early Rainy", "Peak Rainy"],
        default="Late Rainy"
    )

    x["rain_yesterday"] = x["rain_sum"].shift(1)
    x["rain_2_days_ago"] = x["rain_sum"].shift(2)
    x["rain_3_days_ago"] = x["rain_sum"].shift(3)

    x["rain_3_day_total"] = x["rain_sum"].rolling(3).sum().shift(1)
    x["rain_7_day_total"] = x["rain_sum"].rolling(7).sum().shift(1)
    x["rain_7_day_average"] = x["rain_sum"].rolling(7).mean().shift(1)

    x["temperature_7_day_average"] = (
        x["temperature_2m_mean"].rolling(7).mean().shift(1)
    )
    x["humidity_7_day_average"] = (
        x["relative_humidity_2m_mean"].rolling(7).mean().shift(1)
    )
    x["pressure_7_day_average"] = (
        x["pressure_msl_mean"].rolling(7).mean().shift(1)
    )

    return x

def prepare_model_input(df, selected_date):
    """Build the exact feature row expected by the supplied trained model."""

    row = df.loc[df["date"] == pd.Timestamp(selected_date)].copy()
    if row.empty:
        raise ValueError("Selected date is not available in the downloaded data.")

    missing = [c for c in FEATURES if c not in row.columns]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))

    X = row[FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Guard against accidentally changing the model interface before hosting.
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is not None and X.shape[1] != expected_n:
        raise ValueError(
            f"Model expects {expected_n} input features, but the app prepared "
            f"{X.shape[1]} features."
        )

    # The model requires complete engineered rows. For the first few historical
    # dates, lag/rolling values do not exist, so prediction is disabled there.
    if X.isna().any().any():
        raise ValueError(
            "This date does not have enough preceding days to calculate all "
            "lag/rolling features required by the model."
        )
    return X, row.iloc[0]

def metric_card(label, value):
    st.markdown(
        f'<div class="metric"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("""
<div class="hero">
<h1>🌧️ Rainfall Prediction System</h1>
<p>Interactive rainfall prediction and weather analytics for the University of Ibadan using the trained Gradient Boosting model.</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the supplied model: {e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ System")
    st.write("**Location**")
    st.info(LOCATION_NAME)
    st.caption(f"Coordinates: {LATITUDE:.4f}° N, {LONGITUDE:.4f}° E")
    st.divider()
    st.write("**Prediction rule**")
    st.caption("Rainfall is classified as rain when the target-day total is ≥ 1.0 mm.")
    st.divider()
    st.write("**Data source**")
    st.caption("Open-Meteo historical weather data")
    st.divider()
    st.write("**System status**")
    st.success("Model loaded")
    st.caption(f"App version {APP_VERSION}")
    with st.expander("About this system"):
        st.write("This application uses the supplied trained Gradient Boosting classifier. Historical weather observations are transformed into the temporal and rolling features expected by the model before inference.")
        st.warning("The prediction shown is a machine-learning estimate for the following day based on the selected historical observation. It is not an official meteorological warning.")

today = date.today()
end_date = today - timedelta(days=1)

with st.spinner("Loading historical weather data..."):
    try:
        raw = load_weather(date(2020, 1, 1), end_date)
        data = engineer_features(raw)
    except Exception as e:
        st.error(f"Unable to retrieve weather data: {e}")
        st.warning("The weather service could not be reached. Please refresh the page or try again later.")
        st.stop()

# Create the target only for evaluation/analytics. It is not used as an input.
data["rain_tomorrow"] = (data["rain_sum"].shift(-1) >= 1.0).astype(int)

valid = data.dropna(subset=FEATURES).copy()
if valid.empty:
    st.error("There are not enough valid observations after feature engineering.")
    st.stop()

tabs = st.tabs(["🏠 Dashboard", "🌧️ Predict", "📊 Analytics"])
st.caption(f"📡 Latest available weather date: {data['date'].max().strftime('%d %b %Y')}  •  Data retrieved: {datetime.now().strftime('%d %b %Y, %H:%M')}")

with tabs[0]:
    st.subheader("Rainfall Overview")

    latest = data.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Latest rainfall", f"{latest['rain_sum']:.1f} mm")
    with c2: metric_card("Temperature", f"{latest['temperature_2m_mean']:.1f} °C")
    with c3: metric_card("Humidity", f"{latest['relative_humidity_2m_mean']:.0f}%")
    with c4: metric_card("Wind speed", f"{latest['wind_speed_10m_mean']:.1f} km/h")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recent Rainfall")
    recent = data.tail(60)
    fig = px.line(
        recent, x="date", y="rain_sum",
        labels={"date":"Date", "rain_sum":"Rainfall (mm)"},
        title="Daily rainfall — last 60 available days"
    )
    fig.update_layout(height=390, margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        monthly = data.groupby("month", as_index=False)["rain_sum"].mean()
        monthly["Month"] = pd.to_datetime(monthly["month"], format="%m").dt.month_name().str[:3]
        fig = px.bar(monthly, x="Month", y="rain_sum",
                     labels={"rain_sum":"Average rainfall (mm)"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        seasonal = data.groupby("season", as_index=False)["rain_sum"].mean()
        fig = px.bar(seasonal, x="season", y="rain_sum",
                     labels={"rain_sum":"Average rainfall (mm)", "season":"Season"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This dashboard uses historical weather data for analysis. "
        "The prediction tab performs model inference on dates for which the "
        "model's required lag and rolling features can be calculated."
    )

with tabs[1]:
    st.subheader("🌧️ Make a Rainfall Prediction")
    st.caption(
        "Historical next-day inference: the model uses a completed daily observation "
        "to estimate whether rainfall will reach at least 1.0 mm on the following day."
    )

    available_dates = valid["date"].dt.date.tolist()
    selected = st.selectbox(
        "Select a date to predict whether rainfall occurs on the following day",
        options=available_dates[::-1]
    )

    try:
        X, row = prepare_model_input(data, selected)
        pred = int(model.predict(X)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0, 1])
        else:
            probability = None

        st.markdown('<div class="prediction">', unsafe_allow_html=True)
        if pred == 1:
            st.markdown("## 🌧️ Rainfall Expected")
            st.success("The model predicts rainfall of at least 1.0 mm for the following day.")
        else:
            st.markdown("## ☀️ No Significant Rainfall Expected")
            st.warning("The model predicts less than 1.0 mm for the following day.")

        if probability is not None:
            st.progress(probability)
            st.write(f"**Probability of rainfall:** {probability:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

        target_date = pd.Timestamp(selected) + pd.Timedelta(days=1) 
        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1: metric_card("Observation date", pd.Timestamp(selected).strftime("%d %b %Y"))
        with result_col2: metric_card("Target date", target_date.strftime("%d %b %Y"))
        with result_col3: metric_card("Predicted target", "Rain ≥ 1.0 mm" if pred == 1 else "Rain < 1.0 mm")

        st.markdown("### Weather conditions used by the model")
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("Temperature", f"{row['temperature_2m_mean']:.1f} °C")
        with c2: metric_card("Humidity", f"{row['relative_humidity_2m_mean']:.0f}%")
        with c3: metric_card("Pressure", f"{row['pressure_msl_mean']:.0f} hPa")
        with c4: metric_card("Cloud cover", f"{row['cloud_cover_mean']:.0f}%")

        st.markdown("### Rainfall history")
        c1,c2,c3 = st.columns(3)
        with c1: metric_card("Yesterday", f"{row['rain_yesterday']:.1f} mm")
        with c2: metric_card("Previous 3 days", f"{row['rain_3_day_total']:.1f} mm")
        with c3: metric_card("Previous 7 days", f"{row['rain_7_day_total']:.1f} mm")

        with st.expander("View all model input features"):
            display = X.T.rename(columns={X.index[0]: "Value"})
            st.dataframe(display, use_container_width=True)

        st.info(
            f"This is a historical next-day prediction for {target_date.strftime('%d %b %Y')}. "
            "It uses observed weather from the selected completed day and its preceding "
            "rainfall history. It should not be presented as a live meteorological forecast."
        )

    except ValueError as e:
        st.warning(str(e))

with tabs[2]:
    st.subheader("📊 Rainfall Analytics")

    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Total rainfall", f"{data['rain_sum'].sum():,.1f} mm")
    with c2: metric_card("Average daily rainfall", f"{data['rain_sum'].mean():.2f} mm")
    with c3: metric_card("Maximum daily rainfall", f"{data['rain_sum'].max():.1f} mm")

    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            data, x="rain_sum", nbins=35,
            labels={"rain_sum":"Rainfall (mm)"},
            title="Rainfall distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        annual = data.assign(year=data["date"].dt.year).groupby("year", as_index=False)["rain_sum"].sum()
        fig = px.bar(annual, x="year", y="rain_sum",
                     labels={"rain_sum":"Total rainfall (mm)", "year":"Year"},
                     title="Annual rainfall")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rainfall Categories")
    categories = pd.cut(
        data["rain_sum"],
        bins=[-0.01, 0, 2.5, 10, 50, np.inf],
        labels=["No rain", "Light", "Moderate", "Heavy", "Very heavy"]
    )
    counts = categories.value_counts().reindex(
        ["No rain","Light","Moderate","Heavy","Very heavy"]
    ).fillna(0).reset_index()
    counts.columns = ["Category","Days"]
    fig = px.bar(counts, x="Category", y="Days")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model evaluation note")
    st.info(
        "This page presents descriptive rainfall analytics from real historical weather data. "
        "A formal unseen-test score is not claimed here because the saved model file does not "
        "contain the original train/test split metadata. For the final research report, evaluate "
        "the saved model on a strictly chronological holdout dataset that was not used during training."
    )

    st.subheader("Correlation of selected weather variables")
    corr_cols = [
        "rain_sum", "temperature_2m_mean", "relative_humidity_2m_mean",
        "pressure_msl_mean", "dew_point_2m_mean",
        "wind_speed_10m_mean", "cloud_cover_mean"
    ]
    corr = data[corr_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        zmin=-1, zmax=1, text=np.round(corr.values,2), texttemplate="%{text}"
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


st.divider()
with st.expander("📘 Methodology and hosting notes"):
    st.markdown(
        """
**Prediction target:** rainfall on the following day, classified as rain when total rainfall is **≥ 1.0 mm**.

**Model:** the supplied, already-trained **Gradient Boosting Classifier** is loaded from `_tuned_gradient.pkl`. The application does **not** retrain the model.

**Input preparation:** the app reconstructs the rainfall lag and rolling-window features required by the saved model, checks the feature count, and sends the features in the fixed training order.

**Data:** historical daily observations are retrieved from Open-Meteo for the University of Ibadan location.

**Important hosting limitation:** the current interface is a reproducible **historical next-day inference** system. It is not yet a true real-time weather forecast service. A genuine tomorrow forecast would require a forecast-time data pipeline whose available variables and timing match the way the model was trained.

**Research evaluation:** do not report predictions made on training data as model accuracy. For a defensible performance result, use the original chronological holdout/test set or recreate a strictly time-ordered holdout from the original dataset.
        """
    )

st.divider()
st.caption(
    "University of Ibadan Rainfall Prediction System • "
    "Trained Gradient Boosting model • Streamlit • For research and educational use"
)
