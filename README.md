[README.md](https://github.com/user-attachments/files/29368243/README.md)
# Air Compressor 110 kW Industrial Analytics Application

A complete predictive maintenance dashboard and ML pipeline for a **110 kW fixed-speed screw compressor**, designed similar to the executive dashboard reference.

## What is included

- Executive dashboard similar to the attached design
- Failure risk prediction
- Predicted power consumption
- Compressor condition score
- RUL estimate
- Degradation stage
- Maintenance recommendation
- Alerts panel
- Work-order style action button
- ROI and savings panel
- Batch prediction script
- FastAPI prediction service
- Training script
- Dockerfile

## Folder structure

```text
air_compressor_industrial_analytics_app/
├── app.py
├── api.py
├── train_models.py
├── predict_batch.py
├── requirements.txt
├── Dockerfile
├── data/
│   └── compressor_sample.csv
├── models/
│   ├── failure_prediction_model.joblib
│   ├── condition_classification_model.joblib
│   ├── energy_prediction_model.joblib
│   ├── rul_prediction_model.joblib
│   └── degradation_stage_model.joblib
├── src/
│   ├── feature_engineering.py
│   └── model_service.py
└── assets/
    └── dashboard_reference.jpeg
```

## Run the dashboard

```bash
cd air_compressor_industrial_analytics_app
pip install -r requirements.txt
streamlit run app.py
```

Windows:

```bash
run_app_windows.bat
```

## Run the API

```bash
cd air_compressor_industrial_analytics_app
pip install -r requirements.txt
uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Retrain models

```bash
python train_models.py
```

## Batch prediction

```bash
python predict_batch.py --input data/compressor_sample.csv --output prediction_output.csv
```

## Business outcome logic

The app estimates annual savings from:

1. Energy reduction from abnormal power deviation
2. Avoided unplanned downtime
3. Lower emergency maintenance cost

The dashboard presents ROI, payback, active alerts and planned maintenance actions.
