# Complete Pipeline

## 1. Data ingestion
- Load compressor sensor CSV or live historian export.
- Required sensor groups: power, pressure, flow, temperature, vibration, cooling and component status.

## 2. Data cleaning
- Remove duplicate IDs and duplicate rows.
- Repair timestamp parsing.
- Correct motor power W/kW unit-entry mistakes.
- Cap impossible engineering values.
- Impute missing numeric values using median.

## 3. Feature engineering
- Specific power
- Cooling delta temperature
- Oil-discharge temperature gap
- Vibration RMS
- Pressure-flow ratio
- Water pump efficiency proxy
- Degradation index
- Degradation stage

## 4. Model layer
- Failure risk classifier
- Condition classifier
- Energy regression model
- RUL estimator
- Degradation stage classifier

## 5. Decision layer
- Failure probability threshold
- RUL threshold
- Degradation threshold
- Component-based recommendation

## 6. Application layer
- Streamlit dashboard for leadership and maintenance team
- FastAPI endpoint for system integration
- Batch prediction script for offline scoring

## 7. Monitoring
- Data drift
- Prediction drift
- Model performance
- Alert precision
- CMMS feedback loop
