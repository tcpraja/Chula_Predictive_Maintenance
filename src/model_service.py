
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from feature_engineering import engineer_features, estimate_rul_days, component_recommendation

APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = APP_ROOT / "models"


class CompressorPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)

        self.schema = self._load_schema()

        self.base_features = self.schema.get("model_features", self.schema.get("base_features", []))
        self.energy_features = self.schema.get("energy_features", self.base_features)
        self.degradation_features = self.schema.get("degradation_features", self.base_features)

        self.failure = self._safe_load("failure_prediction_model.joblib")
        self.condition = self._safe_load("condition_classification_model.joblib")
        self.energy = self._safe_load("energy_prediction_model.joblib")
        self.degradation = self._safe_load("degradation_stage_model.joblib")
        self.rul = self._safe_load("rul_prediction_model.joblib")

    def _load_schema(self):
        schema_path = self.model_dir / "feature_schema.json"
        if schema_path.exists():
            with open(schema_path, "r") as f:
                return json.load(f)
        return {}

    def _safe_load(self, name):
        path = self.model_dir / name
        return joblib.load(path) if path.exists() else None

    def _get_model_and_features(self, obj, default_features):
        """
        Supports both formats:
        1. {"model": trained_model, "features": feature_list, "label_encoder": encoder}
        2. trained_model directly as Pipeline or sklearn model
        """
        if obj is None:
            return None, default_features, None

        if isinstance(obj, dict):
            model = obj.get("model", None)
            features = obj.get("features", default_features)
            label_encoder = obj.get("label_encoder", None)
            return model, features, label_encoder

        return obj, default_features, None

    def _safe_features(self, df, features):
        features = [c for c in features if c in df.columns]

        if not features:
            raise ValueError(
                "No valid model features found in dataframe. "
                "Check feature_schema.json and engineer_features output."
            )

        return df[features]

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = engineer_features(raw_df)

        # Failure probability
        failure_model, failure_features, _ = self._get_model_and_features(
            self.failure, self.base_features
        )

        if failure_model is not None:
            X = self._safe_features(df, failure_features)
            if hasattr(failure_model, "predict_proba"):
                proba = failure_model.predict_proba(X)
                df["failure_probability"] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                df["failure_probability"] = failure_model.predict(X)
        else:
            df["failure_probability"] = (df["degradation_index"] / 100).clip(0, 1)

        # Condition prediction
        condition_model, condition_features, condition_encoder = self._get_model_and_features(
            self.condition, self.base_features
        )

        if condition_model is not None:
            X = self._safe_features(df, condition_features)
            pred = condition_model.predict(X)

            if condition_encoder is not None:
                df["predicted_condition"] = condition_encoder.inverse_transform(pred)
            else:
                df["predicted_condition"] = pred.astype(str)

            if hasattr(condition_model, "predict_proba"):
                df["condition_confidence"] = condition_model.predict_proba(X).max(axis=1)
            else:
                df["condition_confidence"] = 0.80
        else:
            df["predicted_condition"] = df.get("condition_label", "Normal")
            df["condition_confidence"] = 0.80

        # Energy prediction
        energy_model, energy_features, _ = self._get_model_and_features(
            self.energy, self.energy_features
        )

        if energy_model is not None:
            X = self._safe_features(df, energy_features)
            df["predicted_kw"] = energy_model.predict(X)
        else:
            df["predicted_kw"] = df["kw_consumption"]

        # Degradation stage prediction
        degradation_model, degradation_features, degradation_encoder = self._get_model_and_features(
            self.degradation, self.degradation_features
        )

        if degradation_model is not None:
            X = self._safe_features(df, degradation_features)
            pred = degradation_model.predict(X)

            if degradation_encoder is not None:
                df["predicted_degradation_stage"] = degradation_encoder.inverse_transform(pred)
            else:
                df["predicted_degradation_stage"] = pred.astype(str)
        else:
            df["predicted_degradation_stage"] = df["degradation_stage"]

        # RUL prediction
        rul_model, rul_features, _ = self._get_model_and_features(
            self.rul, self.base_features
        )

        if rul_model is not None:
            X = self._safe_features(df, rul_features)
            rul_pred = rul_model.predict(X)

            # Your current notebook validates RUL in days, so do not divide by 24.
            df["rul_days"] = np.clip(rul_pred, 1, 365)
        else:
            df["rul_days"] = estimate_rul_days(df)

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

    avg_saving_kw = float(df["power_saving_opportunity_kw"].tail(500).mean()) if "power_saving_opportunity_kw" in df else 2.0
    energy_saving = avg_saving_kw * operating_hours_per_year * electricity_cost_usd_per_kwh

    failure_rate = float(df["failure_probability"].tail(500).mean()) if "failure_probability" in df else 0.25
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
