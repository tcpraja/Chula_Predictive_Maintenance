
import sys
import warnings
import logging
import asyncio
import base64
import mimetypes
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
SRC_PATH = APP_ROOT / "src"

# Add src to path for imports
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(SRC_PATH))

# Data source priority - SIMPLIFIED
DEFAULT_DATA_CANDIDATES = [
    APP_ROOT / "data" / "compressor_sample.csv",
    APP_ROOT / "data" / "compressor_deploy.csv",
    APP_ROOT / "110kw_compressor_raw_real_time.csv",
]

# Reduce noisy Streamlit/Tornado console messages
for _logger_name in [
    "tornado.access",
    "tornado.application",
    "tornado.general",
    "streamlit.runtime",
    "streamlit.web.server.browser_websocket_handler",
]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

def _quiet_websocket_closed(loop, context):
    """Suppress harmless WebSocketClosedError / StreamClosedError messages."""
    exc = context.get("exception")
    msg = str(context.get("message", ""))
    exc_name = exc.__class__.__name__ if exc else ""
    exc_text = str(exc) if exc else ""
    noisy = (
        "WebSocketClosedError" in exc_name
        or "StreamClosedError" in exc_name
        or "WebSocketClosedError" in msg
        or "StreamClosedError" in msg
        or "Stream is closed" in exc_text
        or "Stream is closed" in msg
    )
    if noisy:
        return
    loop.default_exception_handler(context)

try:
    _loop = asyncio.get_event_loop()
    _loop.set_exception_handler(_quiet_websocket_closed)
except Exception:
    pass

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    warnings.filterwarnings("ignore", message=r"Trying to unpickle estimator .*", category=UserWarning)

warnings.filterwarnings("ignore", message=r"Trying to unpickle estimator .*", category=UserWarning)

import json
import joblib

# FIXED: Import from src module
try:
    from src.feature_engineering import engineer_features, estimate_rul_days, component_recommendation
except ImportError:
    try:
        from feature_engineering import engineer_features, estimate_rul_days, component_recommendation
    except ImportError as e:
        st.error(f"Failed to import feature_engineering: {e}")
        st.stop()

MODEL_DIR = APP_ROOT / "models"

def get_first_existing_default_data_path():
    for candidate in DEFAULT_DATA_CANDIDATES:
        try:
            if Path(candidate).exists():
                return Path(candidate)
        except Exception:
            continue
    return None

def create_sample_data():
    """Create sample compressor data if no file exists."""
    np.random.seed(42)
    n_rows = 500
    
    data = {
        "timestamp": pd.date_range(start="2026-01-01", periods=n_rows, freq="1H"),
        "rpm": np.random.normal(1470, 15, n_rows),
        "motor_power": np.random.normal(80000, 5000, n_rows),
        "torque": np.random.normal(500, 50, n_rows),
        "outlet_pressure_bar": np.random.normal(7.5, 1.0, n_rows),
        "air_flow": np.random.normal(12000, 1000, n_rows),
        "noise_db": np.random.normal(82, 5, n_rows),
        "outlet_temp": np.random.normal(90, 8, n_rows),
        "wpump_outlet_press": np.random.normal(3.0, 0.5, n_rows),
        "water_inlet_temp": np.random.normal(25, 3, n_rows),
        "water_outlet_temp": np.random.normal(35, 3, n_rows),
        "wpump_power": np.random.normal(3000, 300, n_rows),
        "water_flow": np.random.normal(70, 10, n_rows),
        "oilpump_power": np.random.normal(1500, 200, n_rows),
        "oil_tank_temp": np.random.normal(85, 7, n_rows),
        "gaccx": np.random.normal(0.8, 0.3, n_rows),
        "gaccy": np.random.normal(0.8, 0.3, n_rows),
        "gaccz": np.random.normal(3.0, 1.0, n_rows),
        "haccx": np.random.normal(1.5, 0.5, n_rows),
        "haccy": np.random.normal(1.5, 0.5, n_rows),
        "haccz": np.random.normal(5.0, 1.5, n_rows),
        "kw_consumption": np.random.normal(85, 10, n_rows),
        "load_pct": np.random.normal(75, 15, n_rows),
        "compressor_size_kw": np.full(n_rows, 110),
        "condition_label": np.random.choice(["Normal", "Warning", "Abnormal"], n_rows),
    }
    
    return pd.DataFrame(data)

class CompressorPredictor:
    """Predictor for updated model files."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.schema = self._load_schema()

        self.base_features = self.schema.get(
            "model_features",
            self.schema.get("base_features", [])
        )
        self.energy_features = self.schema.get("energy_features", self.base_features)
        self.degradation_features = self.schema.get("degradation_features", self.base_features)

        self.failure = self._safe_load("failure_prediction_model.joblib")
        self.condition = self._safe_load("condition_classification_model.joblib")
        self.energy = self._safe_load("energy_prediction_model.joblib")
        self.rul = self._safe_load("rul_prediction_model.joblib")
        self.degradation = self._safe_load("degradation_stage_model.joblib")

    def _load_schema(self):
        schema_path = self.model_dir / "feature_schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _safe_load(self, filename):
        path = self.model_dir / filename
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")
            return None

    def _unpack_model(self, obj, default_features):
        if obj is None:
            return None, default_features, None

        if isinstance(obj, dict):
            model = obj.get("model")
            features = obj.get("features", default_features)
            encoder = obj.get("label_encoder", None)
            return model, features, encoder

        return obj, default_features, None

    def _prepare_X(self, df, model, features):
        features = list(features) if features else []

        if not features and model is not None and hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)

        features = [c for c in features if c in df.columns]

        if not features:
            # Use all numeric columns as fallback
            features = [c for c in df.columns if c not in ["timestamp", "id", "condition_label"]]

        return df[features] if features else df

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = engineer_features(raw_df)

        # Failure probability
        model, features, _ = self._unpack_model(self.failure, self.base_features)
        if model is not None:
            try:
                X = self._prepare_X(df, model, features)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    df["failure_probability"] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                else:
                    pred = model.predict(X)
                    df["failure_probability"] = np.clip(pred, 0, 1)
            except Exception as e:
                st.warning(f"Failure prediction failed: {e}")
                df["failure_probability"] = (df["degradation_index"] / 100).clip(0, 1)
        else:
            df["failure_probability"] = (df["degradation_index"] / 100).clip(0, 1)

        # Condition classification
        model, features, encoder = self._unpack_model(self.condition, self.base_features)
        if model is not None:
            try:
                X = self._prepare_X(df, model, features)
                pred = model.predict(X)
                df["predicted_condition"] = encoder.inverse_transform(pred) if encoder is not None else pred.astype(str)
                df["condition_confidence"] = model.predict_proba(X).max(axis=1) if hasattr(model, "predict_proba") else 0.80
            except Exception as e:
                st.warning(f"Condition classification failed: {e}")
                df["predicted_condition"] = df["condition_label"] if "condition_label" in df.columns else "Normal"
                df["condition_confidence"] = 0.80
        else:
            df["predicted_condition"] = df["condition_label"] if "condition_label" in df.columns else "Normal"
            df["condition_confidence"] = 0.80

        # Energy prediction
        model, features, _ = self._unpack_model(self.energy, self.energy_features)
        if model is not None:
            try:
                X = self._prepare_X(df, model, features)
                df["predicted_kw"] = model.predict(X)
            except Exception as e:
                st.warning(f"Energy prediction failed: {e}")
                df["predicted_kw"] = df["kw_consumption"]
        else:
            df["predicted_kw"] = df["kw_consumption"]

        # RUL prediction
        model, features, _ = self._unpack_model(self.rul, self.base_features)
        if model is not None:
            try:
                X = self._prepare_X(df, model, features)
                df["rul_days"] = np.clip(model.predict(X), 1, 365)
            except Exception as e:
                st.warning(f"RUL prediction failed: {e}")
                df["rul_days"] = estimate_rul_days(df)
        else:
            df["rul_days"] = estimate_rul_days(df)

        # Degradation stage prediction
        model, features, encoder = self._unpack_model(self.degradation, self.degradation_features)
        if model is not None:
            try:
                X = self._prepare_X(df, model, features)
                pred = model.predict(X)
                df["predicted_degradation_stage"] = encoder.inverse_transform(pred) if encoder is not None else pred.astype(str)
            except Exception as e:
                st.warning(f"Degradation prediction failed: {e}")
                df["predicted_degradation_stage"] = df["degradation_stage"]
        else:
            df["predicted_degradation_stage"] = df["degradation_stage"]

        df["power_saving_opportunity_kw"] = (
            df["kw_consumption"] - df["predicted_kw"]
        ).clip(lower=0)

        df["condition_score"] = (100 - df["degradation_index"]).clip(0, 100)

        return df

    def predict_latest_summary(self, raw_df: pd.DataFrame) -> dict:
        df = self.predict_batch(raw_df)
        latest = df.iloc[-1]

        failure_risk = float(latest["failure_probability"])
        condition = str(latest["predicted_condition"])
        rul_days = float(latest["rul_days"])
        degradation_stage = str(latest["predicted_degradation_stage"])

        rec = component_recommendation(condition, failure_risk, rul_days, degradation_stage)

        return {
            "asset_id": "AC-110-01",
            "asset_name": "Air Compressor 110 kW",
            "location": "Plant 1 - Compressor Room",
            "failure_probability": failure_risk,
            "failure_risk_pct": round(failure_risk * 100, 1),
            "predicted_kw": round(float(latest["predicted_kw"]), 2),
            "actual_kw": round(float(latest["kw_consumption"]), 2),
            "condition": condition,
            "condition_score": round(float(latest["condition_score"]), 1),
            "condition_confidence": round(float(latest["condition_confidence"]), 3),
            "rul_days": round(rul_days, 1),
            "degradation_stage": degradation_stage,
            "degradation_index": round(float(latest["degradation_index"]), 1),
            "recommendation": rec,
            "power_saving_opportunity_kw": round(float(df["power_saving_opportunity_kw"].tail(200).mean()), 2),
            "data": df
        }


def calculate_business_roi(df: pd.DataFrame) -> dict:
    electricity_cost_usd_per_kwh = 0.12
    operating_hours_per_year = 8000
    downtime_cost_usd_per_hour = 2500
    maintenance_cost_per_event = 5000

    avg_saving_kw = (
        float(df["power_saving_opportunity_kw"].tail(500).mean())
        if "power_saving_opportunity_kw" in df
        else 2.0
    )
    energy_saving = avg_saving_kw * operating_hours_per_year * electricity_cost_usd_per_kwh

    failure_rate = (
        float(df["failure_probability"].tail(500).mean())
        if "failure_probability" in df
        else 0.25
    )
    avoided_downtime_hours = max(8, failure_rate * 80)
    downtime_saving = avoided_downtime_hours * downtime_cost_usd_per_hour

    maintenance_saving = maintenance_cost_per_event * 0.20 * 6
    total_saving = energy_saving + downtime_saving + maintenance_saving
    implementation_cost = 35000
    roi = (total_saving - implementation_cost) / implementation_cost * 100

    return {
        "energy_saving_usd": round(energy_saving, 0),
        "downtime_saving_usd": round(downtime_saving, 0),
        "maintenance_saving_usd": round(maintenance_saving, 0),
        "total_annual_saving_usd": round(total_saving, 0),
        "implementation_cost_usd": implementation_cost,
        "roi_pct": round(roi, 1),
        "payback_months": round(12 * implementation_cost / max(total_saving, 1), 1)
    }


st.set_page_config(
    page_title="Air Compressor 110 kW | Maintenance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.main {background: #f6f8fb;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #061a35 0%, #08284e 100%);}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {color: #ffffff !important;}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section *,
[data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] div *,
[data-baseweb="select"] *,
[data-testid="stTextInput"] *,
[data-testid="stNumberInput"] * {color: #0b1c3d !important;}
[data-testid="stRadio"] label,
[data-testid="stRadio"] label * {color: #ffffff !important;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.top-header {
    background: linear-gradient(90deg, #071a3a 0%, #082b5d 60%, #071a3a 100%);
    color: white; border-radius: 0px 0px 14px 14px;
    padding: 18px 24px; margin-bottom: 14px;
}
.header-title {font-size: 30px; font-weight: 800; margin: 0;}
.header-subtitle {color: #c9d7ee; font-size: 13px; margin-top: 3px;}
.status-dot {height:10px;width:10px;background-color:#2ecc71;border-radius:50%;display:inline-block;margin-left:10px;}
.kpi-card {
    background:#ffffff;border:1px solid #dfe6f1;border-radius:14px;padding:16px;
    min-height:130px;box-shadow:0 2px 8px rgba(9,30,66,0.04);
}
.kpi-label {font-size:12px;color:#31415f;text-transform:uppercase;font-weight:800;margin-bottom:8px;}
.kpi-value {font-size:31px;color:#1254d8;font-weight:800;line-height:1.1;}
.kpi-sub {font-size:12px;color:#60708f;margin-top:8px;}
.panel {
    background:#ffffff;border:1px solid #dfe6f1;border-radius:14px;padding:16px;
    box-shadow:0 2px 8px rgba(9,30,66,0.04);margin-bottom:14px;
}
.panel-title {font-size:13px;color:#1a2942;font-weight:800;text-transform:uppercase;margin-bottom:8px;}
.recommendation {background:#f2f6ff;border:1px solid #d8e6ff;border-radius:12px;padding:14px;}
.alert-box {background:#fff8ec;border-left:5px solid #f5a400;padding:10px 12px;border-radius:8px;margin-bottom:8px;}
.danger-box {background:#fff1f1;border-left:5px solid #e53935;padding:10px 12px;border-radius:8px;margin-bottom:8px;}
.good-box {background:#effaf3;border-left:5px solid #2eaf5f;padding:10px 12px;border-radius:8px;margin-bottom:8px;}
.info-box {background:#eef5ff;border-left:5px solid #1254d8;padding:10px 12px;border-radius:8px;margin-bottom:8px;}
.small-muted {color:#71809b;font-size:12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    return CompressorPredictor()

@st.cache_data(show_spinner=True)
def load_dataframe_from_path(path_str):
    path = Path(path_str)
    path_text = str(path).lower()

    if path_text.endswith(".gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    if path_text.endswith(".zip"):
        return pd.read_csv(path, compression="zip", low_memory=False)
    if path_text.endswith(".parquet"):
        return pd.read_parquet(path)

    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner=True)
def load_uploaded_data(uploaded_file_name, uploaded_file_bytes):
    import io
    name = uploaded_file_name.lower()
    bio = io.BytesIO(uploaded_file_bytes)

    if name.endswith(".gz"):
        return pd.read_csv(bio, compression="gzip", low_memory=False)
    if name.endswith(".zip"):
        return pd.read_csv(bio, compression="zip", low_memory=False)
    if name.endswith(".parquet"):
        return pd.read_parquet(bio)

    return pd.read_csv(bio, low_memory=False)


def downsample_for_plot(dataframe, max_points):
    if dataframe is None or len(dataframe) <= max_points:
        return dataframe
    positions = np.linspace(0, len(dataframe) - 1, max_points).astype(int)
    return dataframe.iloc[positions].copy()


def sample_for_stats(dataframe, max_rows=100000):
    if dataframe is None or len(dataframe) <= max_rows:
        return dataframe
    return dataframe.sample(max_rows, random_state=42)

st.sidebar.markdown("## 📊 Industrial Analytics")
st.sidebar.markdown("**110 kW Compressor App**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Overview",
        "📈 Asset Monitor",
        "📊 Analytics",
        "🔮 Predictions",
        "⚠️ Alerts",
        "🧰 Work Orders",
        "🔧 Maintenance Planning",
        "📄 Maintenance Engineer Report"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Source")

uploaded = st.sidebar.file_uploader(
    "Upload compressor file",
    type=["csv", "gz", "zip", "parquet"],
    help="For 1.8M rows, .csv.gz or .parquet is recommended."
)

default_data_path = get_first_existing_default_data_path()

local_file_path = st.sidebar.text_input(
    "Local / app data file path",
    value=str(default_data_path) if default_data_path else "",
    help="Leave empty to use sample data"
)

MAX_CHART_POINTS = st.sidebar.slider(
    "Maximum chart points",
    min_value=1000,
    max_value=25000,
    value=6000,
    step=1000
)

st.sidebar.info("Load your compressor data or use the auto-generated sample data.")

# LOAD DATA - WITH FALLBACK
try:
    if uploaded is not None:
        data_source = f"Uploaded file: {uploaded.name}"
        df_raw = load_uploaded_data(uploaded.name, uploaded.getvalue())
    elif local_file_path.strip():
        path = Path(local_file_path.strip().strip('"'))
        if path.exists():
            data_source = f"Local file: {path}"
            df_raw = load_dataframe_from_path(str(path))
        else:
            st.warning(f"File not found: {path}. Using sample data instead.")
            data_source = "Sample data (auto-generated)"
            df_raw = create_sample_data()
    else:
        data_source = "Sample data (auto-generated)"
        df_raw = create_sample_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Using sample data instead...")
    data_source = "Sample data (auto-generated)"
    df_raw = create_sample_data()

# PREDICTIONS
try:
    predictor = get_predictor()
    with st.spinner(f"Running predictions for {len(df_raw):,} rows..."):
        summary = predictor.predict_latest_summary(df_raw)
    df = summary["data"]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.info("Reloading with sample data...")
    df_raw = create_sample_data()
    predictor = get_predictor()
    summary = predictor.predict_latest_summary(df_raw)
    df = summary["data"]

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
roi = calculate_business_roi(df)

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

st.sidebar.markdown("---")
st.sidebar.markdown("### Date Range")
selected_dates = st.sidebar.slider(
    "From beginning to latest date",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)
start_date, end_date = selected_dates
view = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)].copy()
if len(view) < 20:
    view = df.tail(min(len(df), 500)).copy()

st.sidebar.success(
    f"✅ **Data Loaded Successfully**\n\n"
    f"Source: {data_source}\n"
    f"Raw rows: {len(df_raw):,}\n"
    f"Predicted rows: {len(df):,}\n"
    f"Selected period: {len(view):,}\n"
    f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
)

def condition_band(score):
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"

def risk_band(risk_pct):
    if risk_pct < 30:
        return "Low"
    if risk_pct < 60:
        return "Medium"
    if risk_pct < 80:
        return "High"
    return "Critical"

def maintenance_priority():
    risk = summary["failure_risk_pct"]
    rul = summary["rul_days"]
    deg = summary["degradation_index"]
    if risk >= 80 or rul <= 3 or deg >= 75:
        return "Critical"
    if risk >= 60 or rul <= 7 or deg >= 55:
        return "High"
    if risk >= 30 or rul <= 21 or deg >= 35:
        return "Medium"
    return "Low"

def render_header():
    min_d = view["timestamp"].min()
    max_d = view["timestamp"].max()
    st.markdown(f"""
    <div class="top-header">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="header-title">Air Compressor 110 kW <span class="status-dot"></span>
                <span style="font-size:14px;font-weight:600;color:#52e381;">Online</span></div>
                <div class="header-subtitle">Asset ID: {summary['asset_id']} &nbsp;&nbsp; | &nbsp;&nbsp; Location: {summary['location']}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-weight:700;">{min_d.strftime('%b %d, %Y')} – {max_d.strftime('%b %d, %Y')}</div>
                <div class="header-subtitle">Selected period | {len(view):,} rows</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(label, value, sub="", color="#1254d8"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def render_kpis():
    risk_pct = summary["failure_risk_pct"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("🛡️ Failure Risk", f"{risk_pct:.0f}%", f"{risk_band(risk_pct)} risk")
    with c2: kpi_card("⚡ Predicted Power", f"{summary['predicted_kw']:.1f}", "kW")
    with c3: kpi_card("〰️ Condition Score", f"{summary['condition_score']:.0f}", "/100", "#2c9a42")
    with c4: kpi_card("📅 RUL Estimate", f"{summary['rul_days']:.0f}", "Days")
    with c5: kpi_card("📉 Degradation", f"{summary['degradation_index']:.0f}", "Index /100", "#6a35d8")
    with c6: kpi_card("💲 Annual Saving", f"${roi['total_annual_saving_usd']:,.0f}", f"ROI {roi['roi_pct']:.0f}%", "#168b9b")

def power_trend_chart():
    chart_view = downsample_for_plot(view.sort_values("timestamp"), MAX_CHART_POINTS)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["kw_consumption"], mode="lines", name="Actual Power (kW)"))
    if "predicted_kw" in chart_view.columns:
        fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["predicted_kw"], mode="lines", name="Predicted Power (kW)", line=dict(dash="dot")))
    fig.add_hline(y=110, line_dash="dash", annotation_text="Rated Power 110 kW")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=25, b=10), yaxis_title="kW", legend=dict(orientation="h"))
    return fig

def risk_gauge():
    risk_pct = summary["failure_risk_pct"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#0b1c3d"},
            "steps": [
                {"range": [0, 30], "color": "#27ae60"},
                {"range": [30, 60], "color": "#f1c40f"},
                {"range": [60, 80], "color": "#f39c12"},
                {"range": [80, 100], "color": "#e74c3c"}
            ],
            "threshold": {"line": {"color": "#111", "width": 3}, "value": risk_pct}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=25, b=10))
    return fig

def page_overview():
    render_header()
    render_kpis()
    left, mid, right = st.columns([2, 1.1, 1.2])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Power Consumption Trend</div>', unsafe_allow_html=True)
        st.plotly_chart(power_trend_chart(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with mid:
        st.markdown('<div class="panel"><div class="panel-title">Risk of Failure</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Maintenance Recommendation</div>', unsafe_allow_html=True)
        rec = summary["recommendation"]
        st.markdown(f"""
        <div class="recommendation">
            <h4>🔧 Recommended Action</h4>
            <b>Priority:</b> {rec['priority']}<br>
            <b>RUL:</b> {summary['rul_days']:.0f} Days<br>
            <b>Condition:</b> {summary['condition']}<br><br>
            <span style="font-size:13px;">{rec['component_action']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def page_asset_monitor():
    render_header()
    render_kpis()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Operating Parameters")
        cols = ["timestamp", "rpm", "load_pct", "kw_consumption", "outlet_pressure_bar", "air_flow", "outlet_temp", "oil_tank_temp"]
        st.dataframe(view[[c for c in cols if c in view.columns]].sort_values("timestamp", ascending=False).head(300), use_container_width=True)
    with c2:
        st.subheader("Key Parameter Trends")
        chart_view = downsample_for_plot(view.sort_values("timestamp"), MAX_CHART_POINTS)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["outlet_temp"], name="Outlet Temp °C"))
        fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["total_vibration_rms"] * 10, name="Vibration RMS x10"))
        fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["outlet_pressure_bar"] * 10, name="Pressure x10"))
        fig.update_layout(height=390, legend=dict(orientation="h"), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

def page_analytics():
    render_header()
    st.markdown("## Exploratory Data Analysis")
    st.caption("Visual diagnostics for power, pressure, flow, temperature, vibration and degradation behavior.")

    a, b, c, d = st.columns(4)
    with a: kpi_card("Rows selected", f"{len(view):,}", "records", "#1254d8")
    with b: kpi_card("Avg Power", f"{view['kw_consumption'].mean():.1f}", "kW", "#1254d8")
    with c: kpi_card("Avg Vibration", f"{view['total_vibration_rms'].mean():.2f}", "RMS", "#6a35d8")
    with d: kpi_card("Avg Degradation", f"{view['degradation_index'].mean():.1f}", "index", "#d97706")

    st.markdown("### Relationship and Trend Analysis")
    c1, c2 = st.columns([1.35, 1.0])
    with c1:
        plot_view = downsample_for_plot(view, MAX_CHART_POINTS)
        fig = px.scatter(
            plot_view,
            x="load_pct",
            y="kw_consumption",
            color="predicted_condition" if "predicted_condition" in view.columns else None,
            size="degradation_index" if "degradation_index" in view.columns else None,
            hover_data=["timestamp", "outlet_temp", "total_vibration_rms", "air_flow"],
            title="Load vs Power Consumption"
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        corr_cols = ["kw_consumption", "load_pct", "outlet_pressure_bar", "air_flow", "outlet_temp", "oil_tank_temp", "total_vibration_rms", "degradation_index"]
        corr_cols = [c for c in corr_cols if c in view.columns]
        corr_view = sample_for_stats(view[corr_cols], max_rows=100000)
        fig = px.imshow(corr_view.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Matrix")
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

def page_predictions():
    render_header()
    render_kpis()
    st.markdown("## Prediction Summary")
    rec = summary["recommendation"]
    prediction_summary = pd.DataFrame({
        "Output": ["Failure risk", "Predicted condition", "Condition score", "RUL", "Degradation stage", "Predicted power", "Priority"],
        "Value": [
            f"{summary['failure_risk_pct']:.1f}% ({risk_band(summary['failure_risk_pct'])})",
            summary["condition"],
            f"{summary['condition_score']:.1f}/100",
            f"{summary['rul_days']:.1f} days",
            summary["degradation_stage"],
            f"{summary['predicted_kw']:.1f} kW",
            maintenance_priority()
        ]
    })
    st.dataframe(prediction_summary, use_container_width=True)

def page_alerts():
    render_header()
    st.subheader("Active Alerts")
    latest = view.iloc[-1]
    alerts = []
    if latest.get("failure_probability", 0) > 0.5:
        alerts.append(("High Failure Risk", "Failure probability is above threshold.", "High"))
    if latest.get("outlet_temp", 0) > view["outlet_temp"].quantile(0.9):
        alerts.append(("High Outlet Temperature", "Temperature is above normal band.", "Medium"))
    if latest.get("total_vibration_rms", 0) > view["total_vibration_rms"].quantile(0.9):
        alerts.append(("Vibration Increasing", "Vibration RMS is above normal band.", "Medium"))
    if not alerts:
        alerts.append(("No Critical Alert", "Asset is within operating limits.", "Low"))
    for title, msg, pri in alerts:
        box = "danger-box" if pri == "High" else "alert-box" if pri == "Medium" else "good-box"
        st.markdown(f"""<div class="{box}">⚠️ <b>{title}</b> — Priority: <b>{pri}</b><br><span class="small-muted">{msg}</span></div>""", unsafe_allow_html=True)

def page_work_orders():
    render_header()
    st.subheader("Work Order Recommendation")
    rec = summary["recommendation"]
    work_order = pd.DataFrame({
        "Field": ["Work Order ID", "Asset ID", "Asset Name", "Priority", "Condition", "Failure Risk %", "RUL Days", "Status"],
        "Value": ["WO-AC110-PDM-001", summary["asset_id"], summary["asset_name"], rec["priority"], summary["condition"], summary["failure_risk_pct"], summary["rul_days"], "Pending"]
    })
    st.dataframe(work_order, use_container_width=True)
    if st.button("Create Work Order"):
        st.success("Work order draft created!")

def page_maintenance_planning():
    render_header()
    st.markdown("## Maintenance Planning")
    plan = pd.DataFrame({
        "Area": ["Bearings", "Radiator", "Cooling", "Electrical", "Air Delivery"],
        "Trigger": ["High vibration", "High temperature", "Low water flow", "Voltage imbalance", "Low pressure"],
        "Action": ["Inspect & lubricate", "Clean cooler", "Check pump", "Check voltage", "Inspect valve"]
    })
    st.dataframe(plan, use_container_width=True)

def page_maintenance_engineer_report():
    render_header()
    st.markdown("## Maintenance Engineer Report")
    rec = summary["recommendation"]
    report_cards = pd.DataFrame({
        "Section": ["Reliability Status", "Failure Mode", "Priority", "Power Opportunity", "Reliability Action"],
        "Observation": [
            f"Condition score: {summary['condition_score']:.0f}/100 ({condition_band(summary['condition_score'])})",
            f"Predicted condition: {summary['condition']} (degradation: {summary['degradation_index']:.0f}/100)",
            f"Failure risk: {summary['failure_risk_pct']:.1f}%, RUL: {summary['rul_days']:.1f} days",
            f"Actual: {summary['actual_kw']:.1f} kW, Predicted: {summary['predicted_kw']:.1f} kW",
            rec["component_action"]
        ]
    })
    st.dataframe(report_cards, use_container_width=True)

# PAGE ROUTING
if page == "🏠 Home":
    st.markdown("# 🏠 Home Dashboard")
    render_header()
    render_kpis()
    st.success("✅ Application is running successfully!")
elif page == "📊 Overview":
    page_overview()
elif page == "📈 Asset Monitor":
    page_asset_monitor()
elif page == "📊 Analytics":
    page_analytics()
elif page == "🔮 Predictions":
    page_predictions()
elif page == "⚠️ Alerts":
    page_alerts()
elif page == "🧰 Work Orders":
    page_work_orders()
elif page == "🔧 Maintenance Planning":
    page_maintenance_planning()
elif page == "📄 Maintenance Engineer Report":
    page_maintenance_engineer_report()
