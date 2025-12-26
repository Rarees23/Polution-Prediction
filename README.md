# AQI Prediction System

Machine learning system for predicting daily Air Quality Index (AQI)
using historical pollution and meteorological data.

## Problem
Air quality has direct health impacts. This project predicts daily AQI
for multiple cities using supervised regression.

## Model
- CatBoost Regressor

## Features
- Pollutant lag features & rolling averages
- Meteorological variables
- Temporal encodings (day, week, season)

## Data Sources
- AirNow (US EPA)
- Luchtmeetnet (NL)
- European Air Quality Portal (EEA)

## Evaluation Metrics
- MAE
- RMSE
- R²

## Ethics & Limitations
- Public environmental data
- No personal data (GDPR-safe)
- Not for medical or regulatory use

## How to Run
```bash
pip install -r requirements.txt
python src/models/train.py
