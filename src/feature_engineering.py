
import numpy as np
import pandas as pd

DOMAIN_LIMITS = {
    "rpm": (1440, 1505),
    "motor_power": (22000, 125000),
    "torque": (130, 850),
    "outlet_pressure_bar": (5.0, 9.5),
    "air_flow": (4000, 25000),
    "noise_db": (60, 105),
    "outlet_temp": (40, 135),
    "wpump_outlet_press": (1.0, 4.8),
    "water_inlet_temp": (15, 50),
    "water_outlet_temp": (20, 75),
    "wpump_power": (1800, 5000),
    "water_flow": (35, 130),
    "oilpump_power": (800, 2200),
    "oil_tank_temp": (40, 125),
    "gaccx": (0.05, 3.0),
    "gaccy": (0.05, 3.0),
    "gaccz": (0.5, 12.0),
    "haccx": (0.1, 5.0),
    "haccy": (0.1, 5.0),
    "haccz": (0.5, 16.0),
    "kw_consumption": (20, 130),
    "load_pct": (20, 105),
    "compressor_size_kw": (110, 110),
}

RAW_SENSOR_COLUMNS = [
    "rpm", "motor_power", "torque", "outlet_pressure_bar", "air_flow", "noise_db",
    "outlet_temp", "wpump_outlet_press", "water_inlet_temp", "water_outlet_temp",
    "wpump_power", "water_flow", "oilpump_power", "oil_tank_temp",
    "gaccx", "gaccy", "gaccz", "haccx", "haccy", "haccz",
    "kw_consumption", "load_pct", "compressor_size_kw"
]

def _normalize_0_100(series, lo, hi):
    return ((series - lo) / (hi - lo) * 100).clip(0, 100)

def clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean incoming plant-historian or uploaded data."""
    df = df.copy()

    if "known_quality_issue" in df.columns:
        df = df.drop(columns=["known_quality_issue"])

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.Timestamp.now()

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    else:
        df["id"] = np.arange(1, len(df) + 1)

    df = df.drop_duplicates()

    category_defaults = {
        "bearings": "Ok",
        "wpump": "Ok",
        "radiator": "Clean",
        "exvalve": "Clean",
        "acmotor": "Stable",
        "condition_label": "Normal",
    }
    for col, default in category_defaults.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": default, "": default})

    for col in df.columns:
        if col not in ["timestamp", "bearings", "wpump", "radiator", "exvalve", "acmotor", "condition_label"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "motor_power" in df.columns:
        mask = df["motor_power"].between(20, 130)
        df.loc[mask, "motor_power"] = df.loc[mask, "motor_power"] * 1000

    if "kw_consumption" in df.columns:
        mask = df["kw_consumption"] > 1000
        df.loc[mask, "kw_consumption"] = df.loc[mask, "kw_consumption"] / 1000

    for col, (lo, hi) in DOMAIN_LIMITS.items():
        if col not in df.columns:
            df[col] = np.nan
        if lo == hi:
            df.loc[df[col].notna() & (df[col] != lo), col] = lo
        else:
            bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            df.loc[bad, col] = np.nan

    for col in RAW_SENSOR_COLUMNS:
        if col in df.columns:
            med = df[col].median()
            if pd.isna(med):
                lo, hi = DOMAIN_LIMITS.get(col, (0, 1))
                med = lo if lo == hi else (lo + hi) / 2
            df[col] = df[col].fillna(med)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model and dashboard features."""
    df = clean_input_data(df)

    df["motor_power_kw"] = df["motor_power"] / 1000
    df["calculated_kw_from_torque"] = df["rpm"] * df["torque"] / 9550
    df["power_calc_gap_kw"] = df["motor_power_kw"] - df["calculated_kw_from_torque"]

    df["specific_power_kw_per_lpm"] = df["kw_consumption"] / df["air_flow"].replace(0, np.nan)
    df["power_load_ratio"] = df["kw_consumption"] / df["load_pct"].replace(0, np.nan)
    df["power_utilization_pct"] = df["kw_consumption"] / df["compressor_size_kw"].replace(0, np.nan) * 100

    df["cooling_delta_temp"] = df["water_outlet_temp"] - df["water_inlet_temp"]
    df["oil_discharge_temp_gap"] = df["outlet_temp"] - df["oil_tank_temp"]

    df["g_vibration_rms"] = np.sqrt((df["gaccx"] ** 2 + df["gaccy"] ** 2 + df["gaccz"] ** 2) / 3)
    df["h_vibration_rms"] = np.sqrt((df["haccx"] ** 2 + df["haccy"] ** 2 + df["haccz"] ** 2) / 3)
    df["total_vibration_rms"] = df["g_vibration_rms"] + df["h_vibration_rms"]

    df["pressure_flow_ratio"] = df["outlet_pressure_bar"] / df["air_flow"].replace(0, np.nan)
    df["water_flow_per_kw_pump"] = df["water_flow"] / (df["wpump_power"].replace(0, np.nan) / 1000)

    df["degradation_index"] = pd.concat([
        _normalize_0_100(df["total_vibration_rms"], 4.0, 9.5),
        _normalize_0_100(df["noise_db"], 70, 98),
        _normalize_0_100(df["outlet_temp"], 80, 125),
        _normalize_0_100(df["oil_tank_temp"], 70, 115),
        _normalize_0_100(df["cooling_delta_temp"], 7, 30),
        _normalize_0_100(df["specific_power_kw_per_lpm"], 0.0030, 0.0085),
    ], axis=1).mean(axis=1)

    df["degradation_stage"] = pd.cut(
        df["degradation_index"],
        bins=[-0.01, 25, 50, 75, 100.01],
        labels=["Stage 1 - Healthy", "Stage 2 - Early Degradation", "Stage 3 - Degradation", "Stage 4 - Severe"]
    ).astype(str)

    if "failure_next_24h" not in df.columns:
        df["failure_next_24h"] = (df["degradation_index"] > 55).astype(int)

    if "condition_label" not in df.columns:
        df["condition_label"] = "Normal"

    return df

def estimate_rul_days(df: pd.DataFrame) -> float:
    """Risk-based RUL estimate for dashboard display."""
    latest = df.iloc[-1]
    degradation = float(latest.get("degradation_index", 40))
    vibration = float(latest.get("total_vibration_rms", 5.0))
    temp = float(latest.get("outlet_temp", 90))
    failure_risk = float(latest.get("failure_probability", 0.25))

    health_factor = max(0.05, 1 - degradation / 100)
    stress_factor = max(0.2, 1 - min(1, (vibration - 4) / 7) * 0.35 - min(1, (temp - 80) / 45) * 0.30 - failure_risk * 0.35)
    rul = 240 * health_factor * stress_factor
    return float(np.clip(rul, 1, 365))

def component_recommendation(condition, failure_risk, rul_days, degradation_stage):
    if failure_risk >= 0.75 or rul_days <= 7:
        priority = "Critical"
        action = "Create urgent work order; inspect compressor before next production-critical run."
    elif failure_risk >= 0.50 or rul_days <= 21:
        priority = "High"
        action = "Plan maintenance within 7–21 days; prepare spares and shutdown window."
    elif failure_risk >= 0.30 or "Stage 3" in degradation_stage:
        priority = "Medium"
        action = "Inspect during next planned stoppage and increase monitoring frequency."
    else:
        priority = "Low"
        action = "Continue normal operation with trend monitoring."

    component_actions = {
        "Bearing_Degradation": "Inspect airend bearings, vibration trend, lubrication condition and coupling alignment.",
        "Water_Pump_Issue": "Check water pump pressure, flow restriction, suction blockage and pump efficiency.",
        "Radiator_Dirty": "Clean radiator/cooler, inspect fouling and verify cooling delta temperature.",
        "Expansion_Valve_Issue": "Inspect valve response, pressure stability and air delivery restriction.",
        "Motor_Instability": "Check voltage balance, motor current, load fluctuation and overload events.",
        "Normal": "No immediate component abnormality detected."
    }

    return {
        "priority": priority,
        "action": action,
        "component_action": component_actions.get(condition, "Inspect abnormal sensor group indicated by analytics."),
    }
