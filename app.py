
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
# Data source priority:
# 1) app-local full/smaller deployment data
# 2) user's local Windows 1.8M training/export files
# 3) app-local sample data as final fallback
DEFAULT_DATA_CANDIDATES = [
    APP_ROOT / "data" / "110kw_compressor_raw_real_time.csv",
    APP_ROOT / "data" / "110kw_fixed_speed_screw_compressor_1_8M_2020_2026.csv",
    APP_ROOT / "data" / "compressor_deploy.csv",
    APP_ROOT / "110kw_compressor_raw_real_life_dirty_1.8M_plus.csv",
    APP_ROOT / "data" / "compressor_sample.csv",
    Path(r"C:\Users\ASUS\OneDrive - Indorama Ventures PCL\ChulaLGO\Data\comp project\Updated\data\110kw_compressor_raw_real_time.csv"),
    Path(r"C:\Users\ASUS\OneDrive - Indorama Ventures PCL\ChulaLGO\Data\comp project\Updated\110kw_compressor_raw_real_life_dirty_1.8M_plus.csv"),
    Path(r"C:\Users\ASUS\OneDrive - Indorama Ventures PCL\ChulaLGO\Data\comp project\Updated\data\compressor_deploy.csv"),
]
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(SRC_PATH))


# Reduce noisy Streamlit/Tornado console messages when the browser tab refreshes,
# closes, or reconnects during reruns. This does not hide real app errors.
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

from feature_engineering import engineer_features, estimate_rul_days, component_recommendation


MODEL_DIR = APP_ROOT / "models"


def get_first_existing_default_data_path():
    for candidate in DEFAULT_DATA_CANDIDATES:
        try:
            if Path(candidate).exists():
                return Path(candidate)
        except Exception:
            continue
    return APP_ROOT / "data" / "compressor_sample.csv"


class CompressorPredictor:
    """
    App-side compatible predictor for updated model files.

    Supports both saved formats:
    1. Direct sklearn Pipeline/model saved with joblib.dump(model, path)
    2. Dict format saved as {"model": model, "features": [...], "label_encoder": encoder}

    This avoids the common KeyError: 'features' when updated models were saved
    directly as sklearn Pipeline objects.
    """

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
        return joblib.load(path)

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

        # Direct sklearn Pipeline/model may retain original training feature names.
        if not features and model is not None and hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)

        features = [c for c in features if c in df.columns]

        if not features:
            raise ValueError(
                "No model feature columns matched engineered data. "
                "Check models/feature_schema.json and src/feature_engineering.py."
            )

        return df[features]

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = engineer_features(raw_df)

        # Failure probability
        model, features, _ = self._unpack_model(self.failure, self.base_features)
        if model is not None:
            X = self._prepare_X(df, model, features)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                df["failure_probability"] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                pred = model.predict(X)
                df["failure_probability"] = np.clip(pred, 0, 1)
        else:
            df["failure_probability"] = (df["degradation_index"] / 100).clip(0, 1)

        # Condition classification
        model, features, encoder = self._unpack_model(self.condition, self.base_features)
        if model is not None:
            X = self._prepare_X(df, model, features)
            pred = model.predict(X)
            df["predicted_condition"] = encoder.inverse_transform(pred) if encoder is not None else pred.astype(str)
            df["condition_confidence"] = model.predict_proba(X).max(axis=1) if hasattr(model, "predict_proba") else 0.80
        else:
            df["predicted_condition"] = df["condition_label"] if "condition_label" in df.columns else "Normal"
            df["condition_confidence"] = 0.80

        # Energy prediction
        model, features, _ = self._unpack_model(self.energy, self.energy_features)
        if model is not None:
            X = self._prepare_X(df, model, features)
            df["predicted_kw"] = model.predict(X)
        else:
            df["predicted_kw"] = df["kw_consumption"]

        # RUL prediction: updated notebook trains this in days.
        model, features, _ = self._unpack_model(self.rul, self.base_features)
        if model is not None:
            X = self._prepare_X(df, model, features)
            df["rul_days"] = np.clip(model.predict(X), 1, 365)
        else:
            df["rul_days"] = estimate_rul_days(df)

        # Degradation stage prediction
        model, features, encoder = self._unpack_model(self.degradation, self.degradation_features)
        if model is not None:
            X = self._prepare_X(df, model, features)
            pred = model.predict(X)
            df["predicted_degradation_stage"] = encoder.inverse_transform(pred) if encoder is not None else pred.astype(str)
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
.footer-bar {
    background:linear-gradient(90deg,#071a3a 0%,#082b5d 100%);
    color:white;border-radius:14px;padding:16px;margin-top:8px;
}
.small-muted {color:#71809b;font-size:12px;}
.animation-card {
    background: radial-gradient(circle at 30% 20%, #143c73 0%, #061a35 70%);
    border-radius: 18px;
    padding: 22px;
    color: white;
    min-height: 610px;
    position: relative;
    overflow: hidden;
    border: 1px solid #1d4b82;
}
.compressor-title {
    margin: 0;
    color: #f8fbff !important;
    -webkit-text-fill-color: #f8fbff !important;
    font-size: 34px;
    font-weight: 900;
    letter-spacing: 0.2px;
    line-height: 1.15;
    text-shadow: 0 0 4px rgba(255,255,255,0.70), 0 0 18px rgba(74,179,255,0.55);
}
.compressor-title .kw-title-accent {
    color: #72c7ff !important;
    -webkit-text-fill-color: #72c7ff !important;
}
.animation-card h2, .animation-card h2 * {
    color: #f8fbff !important;
    -webkit-text-fill-color: #f8fbff !important;
}
.rotor {
    width: 120px;
    height: 120px;
    border: 12px solid #4ab3ff;
    border-top: 12px solid #16e0a1;
    border-radius: 50%;
    animation: spin 1.2s linear infinite;
    margin: 30px auto 10px auto;
    box-shadow: 0 0 35px rgba(74,179,255,0.6);
}
.pulse {
    height: 16px;
    width: 16px;
    background: #2ecc71;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1s infinite;
}
@keyframes spin {100% {transform: rotate(360deg);}}
@keyframes pulse {
    0% {box-shadow: 0 0 0 0 rgba(46,204,113,0.8);}
    70% {box-shadow: 0 0 0 18px rgba(46,204,113,0);}
    100% {box-shadow: 0 0 0 0 rgba(46,204,113,0);}
}
.metric-chip {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(110,184,255,0.32);
    border-radius: 12px;
    padding: 10px 12px;
    margin: 6px;
    animation: valueCardBreath 2.8s ease-in-out infinite;
}

.compressor-visual-wrap {
    position: relative;
    width: 100%;
    height: 520px;
    margin-top: 16px;
    border-radius: 18px;
    background: radial-gradient(circle at 50% 45%, rgba(74,179,255,0.20), rgba(6,26,53,0.18) 58%, rgba(6,26,53,0.04) 80%);
    overflow: hidden;
    border: 1px solid rgba(110, 184, 255, 0.28);
}
.compressor-visual-wrap:before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(5,18,42,0.06), rgba(5,18,42,0.16));
    pointer-events: none;
    z-index: 2;
}
.compressor-image {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    object-fit: contain;
    object-position: center center;
    margin: 0;
    animation: none;
    filter: drop-shadow(0 0 22px rgba(74,179,255,0.26));
    z-index: 1;
}
.overlay-chip {
    position: absolute;
    z-index: 5;
    min-width: 132px;
    background: linear-gradient(180deg, rgba(14, 42, 82, 0.94), rgba(5, 22, 52, 0.94));
    border: 1px solid rgba(110, 184, 255, 0.52);
    border-radius: 12px;
    padding: 8px 10px;
    box-shadow: 0 0 18px rgba(74,179,255,0.20);
    backdrop-filter: blur(7px);
    overflow: hidden;
    isolation: isolate;
}
.overlay-chip:not(.chip-condition) {
    animation: valueCardBreath 2.8s ease-in-out infinite;
}
.overlay-chip:not(.chip-condition)::before {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: 14px;
    padding: 2px;
    background: conic-gradient(
        from 0deg,
        rgba(74,179,255,0.00),
        rgba(74,179,255,0.95),
        rgba(57,255,136,0.90),
        rgba(255,255,255,0.85),
        rgba(74,179,255,0.00),
        rgba(74,179,255,0.00)
    );
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    animation: valueBorderRotate 2.4s linear infinite;
    pointer-events: none;
    z-index: 1;
}
.overlay-chip:not(.chip-condition)::after {
    content: "";
    position: absolute;
    left: -65%;
    top: 0;
    width: 55%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
    transform: skewX(-18deg);
    animation: valueBorderSweep 3.1s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
.overlay-chip b {
    display: block;
    color: #d9ebff;
    font-size: 12px;
    line-height: 1.1;
    position: relative;
    z-index: 2;
}
.overlay-chip span {
    display: block;
    color: #ffffff;
    font-size: 20px;
    font-weight: 800;
    line-height: 1.2;
    margin-top: 3px;
    animation: valuePulse 1.8s ease-in-out infinite;
    position: relative;
    z-index: 2;
}
@keyframes valuePulse {
    0%, 100% {opacity: 1; text-shadow: 0 0 0 rgba(255,255,255,0);}
    50% {opacity: 0.86; text-shadow: 0 0 12px rgba(255,255,255,0.35);}
}
@keyframes valueBorderRotate {
    0% {transform: rotate(0deg);}
    100% {transform: rotate(360deg);}
}
@keyframes valueBorderSweep {
    0%, 32% {left: -70%; opacity: 0;}
    45% {opacity: 1;}
    70%, 100% {left: 120%; opacity: 0;}
}
@keyframes valueCardBreath {
    0%, 100% {
        border-color: rgba(110, 184, 255, 0.50);
        box-shadow: 0 0 16px rgba(74,179,255,0.24), inset 0 0 0 rgba(74,179,255,0.0);
    }
    50% {
        border-color: rgba(88, 230, 255, 0.95);
        box-shadow: 0 0 28px rgba(74,179,255,0.52), inset 0 0 14px rgba(74,179,255,0.16);
    }
}
.chip-power {left: 3%; top: 9%;}
.chip-predpower {left: 3%; top: 29%;}
.chip-pressure {right: 3%; top: 9%;}
.chip-flow {right: 3%; top: 29%;}
.chip-temp {left: 5%; bottom: 13%;}
.chip-vibration {right: 5%; bottom: 13%;}
.chip-risk {left: 50%; top: 4%; transform: translateX(-50%);}
.chip-rul {left: 50%; bottom: 4%; transform: translateX(-50%);}
.chip-condition {left: 50%; top: 23%; transform: translateX(-50%); min-width: 176px;}
.chip-condition.condition-green {background: rgba(18, 138, 71, 0.96); border-color: rgba(80, 235, 150, 0.92); animation: none !important; box-shadow: 0 0 18px rgba(57,255,136,0.22);}
.chip-condition.condition-yellow {background: rgba(180, 120, 12, 0.92); border-color: rgba(255, 210, 80, 0.78); animation: none !important; box-shadow: 0 0 18px rgba(255,202,40,0.22);}
.chip-condition.condition-red {background: rgba(165, 40, 40, 0.92); border-color: rgba(255, 120, 120, 0.78); animation: none !important; box-shadow: 0 0 18px rgba(255,120,120,0.22);}
.chip-condition b, .chip-condition span {color: #ffffff;}
.chip-condition span {animation: none !important;}
.running-condition {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 5px;
    animation: none !important;
}
.running-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #39ff88;
    box-shadow: 0 0 0 rgba(57,255,136,0.75);
    animation: runningPulse 1s infinite !important;
    flex: 0 0 10px;
    display: inline-block !important;
}
.running-text {
    color: #89ffb8 !important;
    font-size: 12px !important;
    font-weight: 900 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    animation: runningBlink 1.2s ease-in-out infinite !important;
    display: inline-block !important;
}
.condition-value {
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: 900 !important;
    letter-spacing: 0.01em;
    animation: none !important;
    display: inline-block !important;
}
@keyframes runningPulse {
    0% {box-shadow: 0 0 0 0 rgba(57,255,136,0.85);}
    70% {box-shadow: 0 0 0 12px rgba(57,255,136,0);}
    100% {box-shadow: 0 0 0 0 rgba(57,255,136,0);}
}
@keyframes runningBlink {
    0%, 100% {opacity: 1;}
    50% {opacity: 0.62;}
}
@keyframes conditionGlow {
    0%, 100% {filter: drop-shadow(0 0 0 rgba(57,255,136,0));}
    50% {filter: drop-shadow(0 0 9px rgba(57,255,136,0.72));}
}
@keyframes runningCardPulse {
    0%, 100% {box-shadow: 0 0 16px rgba(57,255,136,0.22), inset 0 0 0 rgba(57,255,136,0.0);}
    50% {box-shadow: 0 0 28px rgba(57,255,136,0.58), inset 0 0 16px rgba(57,255,136,0.18);}
}
@keyframes warningCardPulse {
    0%, 100% {box-shadow: 0 0 14px rgba(255,202,40,0.22);}
    50% {box-shadow: 0 0 26px rgba(255,202,40,0.55);}
}
@keyframes compressorFloat {
    0%, 100% {transform: translateY(0px) scale(1.0);}
    50% {transform: translateY(-8px) scale(1.01);}
}
@keyframes scan {
    0% {left: -45%;}
    100% {left: 115%;}
}


/* v10: border animation only. No rotating corner / diagonal conic border. */
.overlay-chip:not(.chip-condition)::before,
.overlay-chip:not(.chip-condition)::after {
    content: none !important;
    display: none !important;
    animation: none !important;
    background: none !important;
}
.overlay-chip:not(.chip-condition) {
    border: 2px solid rgba(110, 184, 255, 0.68) !important;
    animation: valueBorderOnlyGlow 1.9s ease-in-out infinite !important;
    box-shadow:
        0 0 12px rgba(74,179,255,0.28),
        inset 0 0 10px rgba(74,179,255,0.06) !important;
}
.metric-chip {
    border: 2px solid rgba(110, 184, 255, 0.62) !important;
    animation: valueBorderOnlyGlow 1.9s ease-in-out infinite !important;
}
@keyframes valueBorderOnlyGlow {
    0%, 100% {
        border-color: rgba(110,184,255,0.58);
        box-shadow:
            0 0 10px rgba(74,179,255,0.22),
            0 0 0 0 rgba(74,179,255,0.0),
            inset 0 0 8px rgba(74,179,255,0.05);
    }
    50% {
        border-color: rgba(120,255,210,0.98);
        box-shadow:
            0 0 18px rgba(74,179,255,0.55),
            0 0 30px rgba(120,255,210,0.22),
            inset 0 0 14px rgba(74,179,255,0.15);
    }
}


/* v11: fit the original 16:9 compressor image cleanly inside the visualization frame. */
.animation-card {
    min-height: auto !important;
}
.compressor-visual-wrap {
    height: auto !important;
    aspect-ratio: 1672 / 941;
    max-height: 640px;
    min-height: unset !important;
    background: #ffffff !important;
    border-radius: 18px;
}
.compressor-visual-wrap:before {
    background: linear-gradient(180deg, rgba(5,18,42,0.02), rgba(5,18,42,0.06)) !important;
}
.compressor-image {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    object-position: center center !important;
    filter: drop-shadow(0 0 18px rgba(74,179,255,0.24));
}
/* keep the value boxes readable after the frame is widened */
.overlay-chip {
    min-width: 126px;
}
.chip-risk {top: 2.2%;}
.chip-condition {top: 22%;}
.chip-rul {bottom: 3.5%;}



/* v12: Bigger dark-green RUNNING condition button. */
.chip-condition {
    min-width: 285px !important;
    padding: 18px 22px !important;
    border-radius: 22px !important;
    top: 20.5% !important;
    border-width: 2px !important;
}
.chip-condition.condition-green {
    background: linear-gradient(180deg, rgba(7, 92, 39, 0.98), rgba(3, 63, 28, 0.98)) !important;
    border-color: rgba(11, 110, 48, 0.95) !important;
    box-shadow:
        0 0 20px rgba(0, 90, 38, 0.48),
        0 0 42px rgba(0, 70, 30, 0.22),
        inset 0 0 14px rgba(0, 45, 20, 0.30) !important;
    animation: conditionDarkGreenButton 2.4s ease-in-out infinite !important;
}
.chip-condition b {
    font-size: 18px !important;
    line-height: 1.1 !important;
    margin-bottom: 12px !important;
    color: #ffffff !important;
}
.running-condition {
    gap: 13px !important;
    margin-top: 4px !important;
}
.running-dot {
    width: 18px !important;
    height: 18px !important;
    flex-basis: 18px !important;
    background: #075f2a !important;
    border: 2px solid #0f7a38 !important;
    box-shadow: 0 0 0 rgba(5, 95, 42, 0.70) !important;
    animation: runningDarkGreenPulse 1.05s ease-out infinite !important;
}
.running-text {
    color: #d8ffe2 !important;
    font-size: 18px !important;
    font-weight: 950 !important;
    letter-spacing: 0.12em !important;
    animation: runningDarkGreenText 1.15s ease-in-out infinite !important;
}
.condition-value {
    font-size: 32px !important;
    font-weight: 950 !important;
    color: #ffffff !important;
    animation: none !important;
}
@keyframes conditionDarkGreenButton {
    0%, 100% {
        background: linear-gradient(180deg, rgba(7, 92, 39, 0.98), rgba(3, 63, 28, 0.98));
        border-color: rgba(9, 100, 44, 0.90);
        box-shadow:
            0 0 16px rgba(0, 80, 34, 0.40),
            0 0 32px rgba(0, 64, 28, 0.18),
            inset 0 0 12px rgba(0, 45, 20, 0.25);
    }
    50% {
        background: linear-gradient(180deg, rgba(5, 78, 34, 1.0), rgba(2, 50, 23, 1.0));
        border-color: rgba(8, 130, 55, 0.98);
        box-shadow:
            0 0 26px rgba(0, 105, 45, 0.56),
            0 0 52px rgba(0, 78, 33, 0.28),
            inset 0 0 18px rgba(0, 55, 25, 0.36);
    }
}
@keyframes runningDarkGreenPulse {
    0% {
        transform: scale(1.0);
        background: #064f24;
        box-shadow: 0 0 0 0 rgba(7, 95, 42, 0.78);
    }
    55% {
        transform: scale(1.12);
        background: #087333;
        box-shadow: 0 0 0 13px rgba(7, 95, 42, 0.00);
    }
    100% {
        transform: scale(1.0);
        background: #064f24;
        box-shadow: 0 0 0 0 rgba(7, 95, 42, 0.00);
    }
}
@keyframes runningDarkGreenText {
    0%, 100% {opacity: 1; text-shadow: 0 0 0 rgba(0, 85, 38, 0.0);}
    50% {opacity: 0.70; text-shadow: 0 0 9px rgba(0, 80, 34, 0.90);}
}


/* v14: Make RUNNING animation clearly visible inside the Condition button.
   Overrides earlier generic chip-condition span animation reset. */
.chip-condition .running-condition {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
    overflow: visible !important;
}
.chip-condition .running-dot {
    width: 20px !important;
    height: 20px !important;
    min-width: 20px !important;
    flex: 0 0 20px !important;
    border-radius: 50% !important;
    background: #4dff8a !important;
    border: 2px solid #d9ffe5 !important;
    box-shadow: 0 0 14px rgba(77,255,138,0.95) !important;
    animation: runningVisibleDotPulse 0.95s ease-out infinite !important;
}
.chip-condition .running-text {
    position: relative !important;
    display: inline-block !important;
    color: #effff2 !important;
    font-size: 19px !important;
    font-weight: 950 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    animation: runningVisibleTextPulse 0.95s ease-in-out infinite !important;
    padding-bottom: 4px !important;
}
.chip-condition .running-text::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: 0;
    width: 100%;
    height: 3px;
    border-radius: 8px;
    background: linear-gradient(90deg, rgba(160,255,180,0.12), rgba(160,255,180,1), rgba(160,255,180,0.12));
    transform-origin: left center;
    animation: runningVisibleUnderline 0.95s linear infinite !important;
}
.chip-condition .condition-value {
    animation: none !important;
    transform: none !important;
    text-shadow: none !important;
    opacity: 1 !important;
}
@keyframes runningVisibleDotPulse {
    0% {
        transform: scale(0.92);
        box-shadow: 0 0 0 0 rgba(77,255,138,0.95), 0 0 12px rgba(77,255,138,0.95);
    }
    55% {
        transform: scale(1.18);
        box-shadow: 0 0 0 18px rgba(77,255,138,0), 0 0 24px rgba(77,255,138,0.95);
    }
    100% {
        transform: scale(0.92);
        box-shadow: 0 0 0 0 rgba(77,255,138,0), 0 0 12px rgba(77,255,138,0.75);
    }
}
@keyframes runningVisibleTextPulse {
    0%, 100% {
        opacity: 1;
        transform: translateY(0) scale(1.0);
        text-shadow: 0 0 4px rgba(160,255,180,0.45), 0 0 12px rgba(40,255,110,0.25);
    }
    50% {
        opacity: 0.72;
        transform: translateY(-1px) scale(1.03);
        text-shadow: 0 0 10px rgba(160,255,180,0.95), 0 0 22px rgba(40,255,110,0.55);
    }
}
@keyframes runningVisibleUnderline {
    0% {transform: scaleX(0.12); opacity: 0.25;}
    50% {transform: scaleX(1.0); opacity: 1;}
    100% {transform: scaleX(0.12); opacity: 0.25;}
}


/* v15: Reduce only the Condition button size; keep same dark-green style and visible RUNNING animation. */
.chip-condition {
    min-width: 215px !important;
    padding: 10px 14px !important;
    border-radius: 16px !important;
    top: 21.5% !important;
    border-width: 2px !important;
}
.chip-condition b {
    font-size: 14px !important;
    line-height: 1.05 !important;
    margin-bottom: 7px !important;
}
.chip-condition .running-condition {
    gap: 10px !important;
    margin-top: 2px !important;
}
.chip-condition .running-dot {
    width: 14px !important;
    height: 14px !important;
    min-width: 14px !important;
    flex: 0 0 14px !important;
    border-width: 2px !important;
}
.chip-condition .running-text {
    font-size: 14px !important;
    letter-spacing: 0.13em !important;
    padding-bottom: 3px !important;
}
.chip-condition .running-text::after {
    height: 2px !important;
}
.chip-condition .condition-value {
    font-size: 24px !important;
    font-weight: 950 !important;
}
@keyframes runningVisibleDotPulse {
    0% {
        transform: scale(0.92);
        box-shadow: 0 0 0 0 rgba(77,255,138,0.95), 0 0 10px rgba(77,255,138,0.95);
    }
    55% {
        transform: scale(1.16);
        box-shadow: 0 0 0 11px rgba(77,255,138,0), 0 0 18px rgba(77,255,138,0.92);
    }
    100% {
        transform: scale(0.92);
        box-shadow: 0 0 0 0 rgba(77,255,138,0), 0 0 10px rgba(77,255,138,0.75);
    }
}

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
    """Keep charts fast for 1.8M rows while retaining the full dataframe for KPIs/tables."""
    if dataframe is None or len(dataframe) <= max_points:
        return dataframe

    positions = np.linspace(0, len(dataframe) - 1, max_points).astype(int)
    return dataframe.iloc[positions].copy()


def sample_for_stats(dataframe, max_rows=100000):
    """Use a stable representative sample for expensive EDA operations."""
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
    help="For 1.8M rows, .csv.gz or .parquet is recommended. For local testing, use the path box below."
)

default_data_path = get_first_existing_default_data_path()
local_file_path = st.sidebar.text_input(
    "Local / app data file path",
    value=str(default_data_path),
    help="For the full 1.8M-row dataset, paste the local CSV/GZ/ZIP/Parquet path here."
)

MAX_CHART_POINTS = st.sidebar.slider(
    "Maximum chart points",
    min_value=1000,
    max_value=25000,
    value=6000,
    step=1000,
    help="The app can load and predict all 1.8M rows, but charts are downsampled for speed."
)

st.sidebar.info(
    "The app now supports the full 1.8M-row dataset locally and updated sklearn Pipeline models. "
    "For GitHub/Streamlit Cloud, use a smaller deploy CSV or compressed/parquet file."
)

try:
    if uploaded is not None:
        data_source = f"Uploaded file: {uploaded.name}"
        df_raw = load_uploaded_data(uploaded.name, uploaded.getvalue())
    elif local_file_path.strip():
        path = Path(local_file_path.strip().strip('"'))
        data_source = f"Local/app file: {path}"
        if not path.exists():
            st.error(f"Data file path not found: {path}")
            st.stop()
        df_raw = load_dataframe_from_path(str(path))
    else:
        fallback_path = APP_ROOT / "data" / "compressor_sample.csv"
        data_source = f"Default sample file: {fallback_path}"
        df_raw = load_dataframe_from_path(str(fallback_path))
except Exception as e:
    st.error(f"Unable to load data: {e}")
    st.stop()

try:
    predictor = get_predictor()
    with st.spinner(f"Running feature engineering and predictions for {len(df_raw):,} rows..."):
        summary = predictor.predict_latest_summary(df_raw)
    df = summary["data"]
except Exception as e:
    st.error(f"Prediction pipeline failed: {e}")
    st.info(
        "Check that models/ contains the updated .joblib files, feature_schema.json, "
        "and src/feature_engineering.py. This app supports both Pipeline-saved models "
        "and dictionary-saved models."
    )
    st.stop()

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
    f"Source: {data_source}\n"
    f"Loaded raw rows: {len(df_raw):,}\n"
    f"Model/output rows: {len(df):,}\n"
    f"Selected period rows: {len(view):,}\n"
    f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
)
if "compressor_sample.csv" in data_source and len(df_raw) <= 20000:
    st.sidebar.warning("Sample file is active. Paste/upload the full 1.8M-row dataset to use all rows.")
elif len(df) == len(df_raw):
    st.sidebar.success("Full row prediction active: model output rows match raw data rows.")
elif len(df) < len(df_raw):
    st.sidebar.warning(f"Raw file has {len(df_raw):,} rows, but model output has {len(df):,} rows.")

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

def get_bom_table():
    return pd.DataFrame({
        "BOM ID": ["BOM-AC110-001", "BOM-AC110-002", "BOM-AC110-003", "BOM-AC110-004", "BOM-AC110-005", "BOM-AC110-006", "BOM-AC110-007", "BOM-AC110-008"],
        "Item / Spare": ["Airend bearing kit", "Oil filter", "Air filter element", "Oil separator element", "Radiator / cooler cleaning kit", "Water pump seal kit", "Pressure sensor", "Vibration sensor"],
        "Specification": ["GA110 compatible bearing set", "110 kW screw compressor oil filter", "High-efficiency intake filter", "Oil carryover separator", "Alkaline cleaner + compressed air wash", "Mechanical seal + gasket set", "0–10 bar, 4–20 mA", "Industrial accelerometer"],
        "Qty": [1, 2, 2, 1, 1, 1, 1, 2],
        "Criticality": ["Critical", "High", "High", "High", "Medium", "Medium", "Medium", "High"],
        "Lead Time": ["2–4 weeks", "1 week", "1 week", "1–2 weeks", "Stock", "1–2 weeks", "1 week", "1–2 weeks"],
        "Use Condition": ["High vibration / bearing alarm", "Scheduled PM / high oil temp", "Pressure drop / low air flow", "High oil carryover / service due", "High outlet temp", "Low cooling flow", "Pressure instability", "Vibration trend increase"]
    })

def get_requirements_table():
    return pd.DataFrame({
        "Requirement ID": ["REQ-01", "REQ-02", "REQ-03", "REQ-04", "REQ-05", "REQ-06"],
        "Requirement": ["Maintenance manpower", "Tools", "Safety", "Shutdown window", "Inspection method", "Post-maintenance validation"],
        "Details": ["1 mechanical technician, 1 electrical/instrument technician, 1 maintenance engineer", "Vibration meter, IR thermometer, pressure gauge, torque wrench, leak detector", "LOTO, depressurize air receiver, PPE, hot surface control", "4 hours planned stoppage; avoid production-critical window", "Bearing vibration, cooling circuit, filter condition, pressure stability, motor load", "Run test, monitor kW, vibration RMS, outlet temp, pressure and air flow for 24 hours"]
    })

def get_optimization_table():
    return pd.DataFrame({
        "Optimization Area": ["Power consumption", "Air pressure", "Air flow", "Cooling efficiency", "Vibration control", "Maintenance timing", "Reliability improvement"],
        "Current Signal": [
            f"Actual kW {summary['actual_kw']:.1f} vs predicted {summary['predicted_kw']:.1f}",
            f"Outlet pressure avg {view['outlet_pressure_bar'].mean():.2f} bar",
            f"Air flow avg {view['air_flow'].mean():,.0f} L/min",
            f"Outlet temp avg {view['outlet_temp'].mean():.1f} °C",
            f"Vibration RMS avg {view['total_vibration_rms'].mean():.2f}",
            f"RUL indicator {summary['rul_days']:.0f} days",
            f"Condition score {summary['condition_score']:.0f}/100"
        ],
        "Recommended Action": [
            "Check leak/load mismatch, reduce unloaded running, verify compressor setpoint and filter restriction.",
            "Avoid excessive pressure setpoint; review line pressure requirement and pressure drop.",
            "Inspect inlet filter, downstream restriction, leakage and demand-side variation.",
            "Clean cooler/radiator, check water flow and cooling delta temperature.",
            "Inspect bearings, coupling alignment, foundation looseness and lubrication.",
            "Plan inspection before RUL becomes critical; prepare spares and shutdown slot.",
            "Use weekly trend review and convert alerts into CMMS work orders."
        ],
        "Expected Benefit": [
            "Lower kWh/air output and improved energy cost.",
            "Lower power draw and reduced mechanical stress.",
            "Improved delivery efficiency and stable production air.",
            "Reduced thermal stress and oil degradation.",
            "Reduced failure risk and better bearing life.",
            "Lower unplanned downtime.",
            "Higher asset reliability and maintenance discipline."
        ]
    })

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
            title="Load vs Power Consumption — colored by predicted condition"
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="Condition")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        corr_cols = ["kw_consumption", "load_pct", "outlet_pressure_bar", "air_flow", "outlet_temp", "oil_tank_temp", "total_vibration_rms", "degradation_index"]
        corr_cols = [c for c in corr_cols if c in view.columns]
        labels = {"kw_consumption": "Power", "load_pct": "Load", "outlet_pressure_bar": "Pressure", "air_flow": "Air flow", "outlet_temp": "Outlet temp", "oil_tank_temp": "Oil temp", "total_vibration_rms": "Vibration", "degradation_index": "Degradation"}
        corr_view = sample_for_stats(view[corr_cols], max_rows=100000)
        fig = px.imshow(corr_view.rename(columns=labels).corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Matrix — key operating drivers")
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        chart_view = downsample_for_plot(view.sort_values("timestamp"), MAX_CHART_POINTS)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["kw_consumption"], name="Power kW", mode="lines"))
        fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["outlet_temp"], name="Outlet temp °C", mode="lines"))
        fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["total_vibration_rms"] * 10, name="Vibration RMS x10", mode="lines"))
        fig.update_layout(title="Time-series operating trend", height=380, legend=dict(orientation="h"), margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        dist_col = st.selectbox("Select variable for distribution", ["kw_consumption", "load_pct", "outlet_pressure_bar", "air_flow", "outlet_temp", "oil_tank_temp", "total_vibration_rms", "degradation_index"], index=0)
        hist_view = sample_for_stats(view, max_rows=100000)
        fig = px.histogram(hist_view, x=dist_col, nbins=45, marginal="box", title=f"Distribution and spread: {dist_col}")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Statistics")
    num_cols = ["rpm", "kw_consumption", "load_pct", "outlet_pressure_bar", "air_flow", "outlet_temp", "oil_tank_temp", "total_vibration_rms", "degradation_index"]
    num_cols = [c for c in num_cols if c in view.columns]
    stats = view[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T[["min", "25%", "50%", "75%", "max", "mean", "std"]]
    stats.columns = ["Minimum", "Q1", "Median", "Q3", "Maximum", "Mean", "Std Dev"]
    st.dataframe(stats.round(3), use_container_width=True)

def page_predictions():
    render_header()
    render_kpis()

    st.markdown("## Prediction Summary and Optimization Actions")
    rec = summary["recommendation"]
    prediction_summary = pd.DataFrame({
        "Prediction Output": [
            "Failure risk", "Predicted condition", "Condition score", "RUL indicator",
            "Degradation stage", "Power prediction", "Maintenance priority"
        ],
        "Current Value": [
            f"{summary['failure_risk_pct']:.1f}% ({risk_band(summary['failure_risk_pct'])})",
            summary["condition"],
            f"{summary['condition_score']:.1f}/100",
            f"{summary['rul_days']:.1f} days",
            summary["degradation_stage"],
            f"{summary['predicted_kw']:.1f} kW",
            maintenance_priority()
        ],
        "Engineering Recommendation": [
            "Review risk trend daily; trigger inspection if risk crosses medium/high band.",
            rec["component_action"],
            "Improve condition score by reducing abnormal vibration, thermal load and inefficient operation.",
            "Plan spares and shutdown window before RUL becomes critical.",
            "Use degradation stage to prioritize PM and avoid run-to-failure.",
            "Compare actual kW against predicted kW to identify power loss or inefficiency.",
            rec["action"]
        ]
    })
    st.dataframe(prediction_summary, use_container_width=True)

    st.markdown("### Optimization Action Plan")
    st.dataframe(get_optimization_table(), use_container_width=True)

    st.markdown("### Prediction Output — Latest First")
    cols = ["timestamp", "kw_consumption", "predicted_kw", "failure_probability", "predicted_condition", "condition_confidence", "condition_score", "rul_days", "predicted_degradation_stage", "power_saving_opportunity_kw"]
    cols = [c for c in cols if c in view.columns]
    prediction_table = view[cols].sort_values("timestamp", ascending=False).head(500).reset_index(drop=True)
    st.caption("Latest prediction records are shown first.")
    st.dataframe(prediction_table, use_container_width=True)

    chart_view = downsample_for_plot(view.sort_values("timestamp"), MAX_CHART_POINTS)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(chart_view, x="timestamp", y="failure_probability", title="Failure Probability Trend"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(chart_view, x="timestamp", y="rul_days", title="RUL Days Trend"), use_container_width=True)

def page_alerts():
    render_header()
    st.subheader("Active Alerts")
    latest = view.iloc[-1]
    alerts = []
    if latest.get("failure_probability", 0) > 0.5:
        alerts.append(("High Failure Risk", "Failure probability is above threshold.", "High"))
    if latest.get("outlet_temp", 0) > view["outlet_temp"].quantile(0.9):
        alerts.append(("High Outlet Temperature", "Outlet temperature is above normal band.", "Medium"))
    if latest.get("total_vibration_rms", 0) > view["total_vibration_rms"].quantile(0.9):
        alerts.append(("Vibration Increasing", "Vibration RMS is above normal operating band.", "Medium"))
    if not alerts:
        alerts.append(("No Critical Alert", "Asset is within current operating limits.", "Low"))
    for title, msg, pri in alerts:
        box = "danger-box" if pri == "High" else "alert-box" if pri == "Medium" else "good-box"
        st.markdown(f"""<div class="{box}">⚠️ <b>{title}</b> — Priority: <b>{pri}</b><br><span class="small-muted">{msg}</span></div>""", unsafe_allow_html=True)

def page_work_orders():
    render_header()
    st.subheader("Work Order Recommendation")
    rec = summary["recommendation"]
    work_order = pd.DataFrame({
        "Field": ["Work Order ID", "Asset ID", "Asset Name", "Priority", "Condition", "Failure Risk %", "RUL Days", "Recommended Action", "Component Action", "Estimated Downtime", "Status"],
        "Value": ["WO-AC110-PDM-001", summary["asset_id"], summary["asset_name"], rec["priority"], summary["condition"], summary["failure_risk_pct"], summary["rul_days"], rec["action"], rec["component_action"], "4 Hours", "Draft / Ready for CMMS"]
    })
    st.dataframe(work_order, use_container_width=True)
    if st.button("Create Work Order"):
        st.success("Work order draft created. In production, this can be integrated with SAP PM / CMMS.")

def page_maintenance_planning():
    render_header()
    st.markdown("## Maintenance Planning for Reliability Improvement")
    st.caption("Maintenance engineering plan for improving compressor availability, reliability and energy performance.")

    plan = pd.DataFrame({
        "Maintenance Area": ["Bearings and airend", "Radiator / cooler", "Water pump and cooling circuit", "Motor and electrical stability", "Compressed air delivery", "Energy optimization", "Instrumentation"],
        "Trigger": ["High vibration, noise, bearing condition", "High outlet temperature, high cooling delta", "Low water flow, abnormal pump pressure", "Motor instability, high power variation", "Low air flow with high power", "Actual kW higher than predicted kW", "Sensor drift or missing data"],
        "Action": ["Inspect lubrication, alignment and bearing wear", "Clean cooler, inspect fouling and airflow restriction", "Check pump, blockage, valves and flow", "Check voltage balance, load fluctuation and overload", "Inspect filter, leak, restriction and pressure setpoint", "Review setpoint, unload running, leak and filter pressure drop", "Calibrate pressure, temperature and vibration sensors"],
        "Decision": ["Plan PM if trend increases", "Clean during next stoppage", "Inspect within 7–21 days", "Electrical inspection", "Energy optimization action", "Reduce kWh/air output", "Improve prediction confidence"]
    })
    st.dataframe(plan, use_container_width=True)

    st.markdown("### BOM / Spare Parts Requirement")
    st.dataframe(get_bom_table(), use_container_width=True)

    st.markdown("### Manpower, Tools, Safety and Execution Requirements")
    st.dataframe(get_requirements_table(), use_container_width=True)

def page_maintenance_engineer_report():
    render_header()
    st.markdown("## Maintenance Engineer Report")
    st.caption("Reliability-focused engineering report for improving asset performance and reducing unplanned downtime.")

    rec = summary["recommendation"]
    report_cards = pd.DataFrame({
        "Report Section": [
            "Asset reliability status",
            "Failure mode indication",
            "Current maintenance priority",
            "Power and energy opportunity",
            "Reliability improvement action",
            "Maintenance execution plan",
            "Expected business impact"
        ],
        "Engineering Observation": [
            f"Condition score is {summary['condition_score']:.0f}/100; reliability band is {condition_band(summary['condition_score'])}.",
            f"Predicted condition is {summary['condition']} with degradation index {summary['degradation_index']:.0f}/100.",
            f"Failure risk is {summary['failure_risk_pct']:.1f}% and RUL indicator is {summary['rul_days']:.1f} days.",
            f"Actual power is {summary['actual_kw']:.1f} kW and predicted power is {summary['predicted_kw']:.1f} kW.",
            rec["component_action"],
            "Prepare BOM, manpower, tools, LOTO and 4-hour planned shutdown window.",
            f"Estimated annual value pool is ${roi['total_annual_saving_usd']:,.0f}; ROI {roi['roi_pct']:.1f}%; payback {roi['payback_months']} months."
        ],
        "Application Point": [
            "Use dashboard as daily health review and weekly reliability meeting input.",
            "Use predicted condition to direct inspection to the right subsystem.",
            "Convert medium/high priority prediction into CMMS work order.",
            "Use power deviation to identify leakage, fouling, restriction or inefficient operation.",
            "Execute corrective inspection before the condition moves to severe degradation.",
            "Plan spare availability and safety readiness before maintenance window.",
            "Justify predictive maintenance scale-up with downtime and energy saving."
        ]
    })
    st.dataframe(report_cards, use_container_width=True)

    st.markdown("### Current Prediction Summary with Recommendations")
    prediction_summary = pd.DataFrame({
        "Parameter": ["Failure risk", "Condition", "Condition score", "RUL", "Degradation", "Predicted kW", "Priority"],
        "Current Prediction": [
            f"{summary['failure_risk_pct']:.1f}%",
            summary["condition"],
            f"{summary['condition_score']:.1f}/100",
            f"{summary['rul_days']:.1f} days",
            summary["degradation_stage"],
            f"{summary['predicted_kw']:.1f} kW",
            maintenance_priority()
        ],
        "Maintenance Recommendation": [
            "Monitor risk trend; inspect if risk increases or crosses threshold.",
            rec["component_action"],
            "Improve by addressing vibration, cooling, filter and pressure losses.",
            "Schedule maintenance before RUL becomes critical.",
            "Use stage to prioritize planned maintenance.",
            "Optimize load, pressure setpoint, filters, leak and cooling system.",
            rec["action"]
        ]
    })
    st.dataframe(prediction_summary, use_container_width=True)

    st.markdown("### BOM / Required Spares")
    st.dataframe(get_bom_table(), use_container_width=True)

    st.markdown("### Execution Requirements")
    st.dataframe(get_requirements_table(), use_container_width=True)

    st.markdown("### Reliability KPI Targets")
    kpi_targets = pd.DataFrame({
        "KPI": ["Availability", "Unplanned downtime", "Specific power", "Vibration RMS", "Outlet temperature", "PM compliance", "Prediction review frequency"],
        "Current Application Focus": ["Improve operating availability", "Avoid sudden failure", "Reduce kWh per air output", "Reduce bearing/mechanical stress", "Reduce thermal stress", "Close actions on time", "Use dashboard weekly/daily"],
        "Target Direction": ["Increase", "Decrease", "Decrease", "Decrease", "Decrease", ">95%", "Daily for operations, weekly for reliability"]
    })
    st.dataframe(kpi_targets, use_container_width=True)


def get_compressor_image_path():
    """Resolve compressor image from the user's Windows path first, then app-local fallbacks."""
    candidate_paths = [
        Path(r"C:\Users\10071404\OneDrive - Indorama Ventures PCL\ChulaLGO\Data\comp project\Updated\assets\atlas_copco_compressor.png"),
        APP_ROOT / "assets" / "atlas_copco_compressor.png",
        APP_ROOT / "atlas_copco_compressor.png",
        APP_ROOT / "assets" / "Comp Image.png",
        APP_ROOT / "Comp Image.png",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None

@st.cache_data(show_spinner=False)
def image_file_to_data_uri(image_path_str):
    image_path = Path(image_path_str)
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def page_compressor_animation():
    render_header()
    st.markdown("## Compressor Animation and Anomaly View")
    st.caption("Visual operating status of compressor health, alerts, and anomalies.")
    latest = view.iloc[-1]
    risk = summary["failure_risk_pct"]
    condition_score = summary["condition_score"]
    degradation = summary["degradation_index"]
    alert_status = "NORMAL"
    alert_class = "good-box"
    if risk >= 60 or degradation >= 70:
        alert_status = "CRITICAL ANOMALY"
        alert_class = "danger-box"
    elif risk >= 30 or degradation >= 45:
        alert_status = "WARNING / EARLY ANOMALY"
        alert_class = "alert-box"

    condition_text = str(summary.get("condition", "Normal"))
    condition_lower = condition_text.lower()
    if "normal" in condition_lower or "good" in condition_lower:
        condition_chip_class = "condition-green"
    elif "fair" in condition_lower or "warning" in condition_lower or "medium" in condition_lower:
        condition_chip_class = "condition-yellow"
    else:
        condition_chip_class = "condition-red"

    c1, c2 = st.columns([3.0, 1.0])
    with c1:
        compressor_image_path = get_compressor_image_path()
        if compressor_image_path is not None:
            compressor_image_uri = image_file_to_data_uri(str(compressor_image_path))
            st.markdown(f"""
            <div class="animation-card">
                <div>
                    <h2 class="compressor-title" style="color:#f8fbff !important;-webkit-text-fill-color:#f8fbff !important;"><span class="kw-title-accent">110 kW</span> Screw Compressor</h2>
                    <p style="color:#c9d7ee;">Digital twin health view with live value overlay</p>
                </div>
                <div class="compressor-visual-wrap">
                    <img class="compressor-image" src="{compressor_image_uri}" />
                    <div class="overlay-chip chip-power"><b>Power</b><span>{latest.get('kw_consumption', 0):.1f} kW</span></div>
                    <div class="overlay-chip chip-predpower"><b>Pred Power</b><span>{summary['predicted_kw']:.1f} kW</span></div>
                    <div class="overlay-chip chip-pressure"><b>Pressure</b><span>{latest.get('outlet_pressure_bar', 0):.2f} bar</span></div>
                    <div class="overlay-chip chip-flow"><b>Air Flow</b><span>{latest.get('air_flow', 0):,.0f} L/min</span></div>
                    <div class="overlay-chip chip-temp"><b>Outlet Temp</b><span>{latest.get('outlet_temp', 0):.1f} °C</span></div>
                    <div class="overlay-chip chip-vibration"><b>Vibration</b><span>{latest.get('total_vibration_rms', 0):.2f} RMS</span></div>
                    <div class="overlay-chip chip-risk"><b>Failure Risk</b><span>{risk:.0f}%</span></div>
                    <div class="overlay-chip chip-rul"><b>RUL</b><span>{summary['rul_days']:.0f} Days</span></div>
                    <div class="overlay-chip chip-condition {condition_chip_class}"><b>Condition</b><div class="running-condition"><span class="running-dot"></span><span class="running-text">RUNNING</span><span class="condition-value">{condition_text}</span></div></div>
                </div>
                <div style="text-align:center;font-size:14px;color:#c9d7ee;margin-top:6px;">Compressor running — sensor stream active</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="animation-card">
                <div style="display:flex;justify-content:space-between;">
                    <div>
                        <h2 class="compressor-title" style="color:#f8fbff !important;-webkit-text-fill-color:#f8fbff !important;"><span class="kw-title-accent">110 kW</span> Screw Compressor</h2>
                        <p style="color:#c9d7ee;">Digital twin style health animation</p>
                    </div>
                    <div style="text-align:right;">
                        <span class="pulse"></span>
                        <b style="margin-left:8px;">{alert_status}</b>
                    </div>
                </div>
                <div class="rotor"></div>
                <div style="text-align:center;font-size:14px;color:#c9d7ee;">Compressor image not found. Place <b>atlas_copco_compressor.png</b> in the assets folder or keep it at the configured local path.</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;margin-top:24px;">
                    <div class="metric-chip"><b>Power</b><br>{latest.get('kw_consumption', 0):.1f} kW</div>
                    <div class="metric-chip"><b>Pred Power</b><br>{summary['predicted_kw']:.1f} kW</div>
                    <div class="metric-chip"><b>Pressure</b><br>{latest.get('outlet_pressure_bar', 0):.2f} bar</div>
                    <div class="metric-chip"><b>Air Flow</b><br>{latest.get('air_flow', 0):,.0f} L/min</div>
                    <div class="metric-chip"><b>Outlet Temp</b><br>{latest.get('outlet_temp', 0):.1f} °C</div>
                    <div class="metric-chip"><b>Vibration</b><br>{latest.get('total_vibration_rms', 0):.2f} RMS</div>
                    <div class="metric-chip"><b>Failure Risk</b><br>{risk:.0f}%</div>
                    <div class="metric-chip"><b>RUL</b><br>{summary['rul_days']:.0f} Days</div>
                    <div class="metric-chip"><b>Condition</b><br><span style="color:#39ff88;font-weight:900;">RUNNING</span> - {condition_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="{alert_class}"><b>Status:</b> {alert_status}<br>
        <span class="small-muted">The animation highlights compressor running condition based on failure risk, degradation index, vibration and temperature.</span></div>""", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=condition_score,
            number={"suffix": "/100"},
            title={"text": "Condition Score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1254d8"}, "steps": [{"range": [0, 40], "color": "#f8d7da"}, {"range": [40, 70], "color": "#fff3cd"}, {"range": [70, 100], "color": "#d4edda"}]}
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Anomaly Signals")
    anomaly_df = pd.DataFrame({
        "Signal": ["Failure risk", "Degradation index", "Outlet temperature", "Vibration RMS", "Power consumption"],
        "Current Value": [f"{risk:.1f}%", f"{degradation:.1f}/100", f"{latest.get('outlet_temp', 0):.1f} °C", f"{latest.get('total_vibration_rms', 0):.2f}", f"{latest.get('kw_consumption', 0):.1f} kW"],
        "Std. Operating Condition": [
            "Normal: <30%; monitor 30–60%; high >60%",
            "Normal: <35/100; warning 35–55; high >55",
            "Normal: 75–95 °C; investigate >95 °C",
            "Normal: ≤7.0 RMS; monitor >7.0 RMS",
            "Normal: 80–110 kW; avoid sustained >110 kW"
        ],
        "Application Meaning": ["Probability-based maintenance priority", "Overall stress and wear indicator", "Cooling and thermal load condition", "Bearing / mechanical wear indicator", "Energy efficiency and load condition"]
    })
    st.dataframe(anomaly_df, use_container_width=True)

    chart_view = view.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["failure_probability"], name="Failure risk", mode="lines"))
    fig.add_trace(go.Scatter(x=chart_view["timestamp"], y=chart_view["degradation_index"] / 100, name="Degradation index scaled", mode="lines"))
    fig.update_layout(title="Anomaly evolution over selected period", height=330, legend=dict(orientation="h"), yaxis_title="Scaled value")
    st.plotly_chart(fig, use_container_width=True)

if page == "🏠 Home":
    page_compressor_animation()
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
