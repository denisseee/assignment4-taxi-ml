# ============================================================
# app.py - FastAPI Prediction Service
# NYC Taxi Tip Amount Predictor
# ============================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
import joblib
import json
import numpy as np
import pandas as pd
import uuid
import time
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))



MODEL_PATH    = os.getenv("MODEL_PATH",    os.path.join(BASE_DIR, "models/taxi_tip_model.pkl"))
SCALER_PATH   = os.getenv("SCALER_PATH",   os.path.join(BASE_DIR, "models/scaler.pkl"))
FEATURES_PATH = os.getenv("FEATURES_PATH", os.path.join(BASE_DIR, "models/feature_columns.json"))

print(f" Loading from: {BASE_DIR}")

ml_model        = joblib.load(MODEL_PATH)
scaler          = joblib.load(SCALER_PATH)
with open(FEATURES_PATH, 'r') as f:
    feature_columns = json.load(f)

start_time    = time.time()
MODEL_VERSION = "1.0.0"
MODEL_NAME    = "taxi-tip-regressor"

print(f" Model loaded: {type(ml_model).__name__}")
print(f" Features loaded: {len(feature_columns)}")


app = FastAPI(
    title="NYC Taxi Tip Predictor",
    description="Predicts tip amount for NYC Yellow Taxi trips",
    version="1.0.0",
)


class TripInput(BaseModel):
    
    passenger_count:       float = Field(..., ge=1,   le=9,    description="Number of passengers (1-9)")
    trip_distance:         float = Field(..., gt=0,   le=100,  description="Trip distance in miles (must be positive)")
    fare_amount:           float = Field(..., ge=0,   le=500,  description="Fare amount in dollars")
    extra:                 float = Field(0.0, ge=0,   le=10,   description="Extra charges")
    mta_tax:               float = Field(0.5, ge=0,   le=1,    description="MTA tax")
    tolls_amount:          float = Field(0.0, ge=0,   le=100,  description="Tolls amount")
    improvement_surcharge: float = Field(0.3, ge=0,   le=1,    description="Improvement surcharge")
    congestion_surcharge:  float = Field(2.5, ge=0,   le=10,   description="Congestion surcharge")
    Airport_fee:           float = Field(0.0, ge=0,   le=20,   description="Airport fee")
    trip_duration_minutes: float = Field(..., gt=0,   le=300,  description="Trip duration in minutes")
    trip_speed_mph:        float = Field(..., gt=0,   le=150,  description="Average speed in mph")
    pickup_hour:           int   = Field(..., ge=0,   le=23,   description="Hour of pickup (0-23)")
    pickup_day_of_week:    int   = Field(..., ge=0,   le=6,    description="Day of week (0=Monday, 6=Sunday)")
    is_weekend:            int   = Field(..., ge=0,   le=1,    description="1 if weekend, 0 if weekday")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "passenger_count": 1,
                "trip_distance": 2.5,
                "fare_amount": 12.0,
                "extra": 0.5,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 0.3,
                "congestion_surcharge": 2.5,
                "Airport_fee": 0.0,
                "trip_duration_minutes": 15.0,
                "trip_speed_mph": 10.0,
                "pickup_hour": 14,
                "pickup_day_of_week": 2,
                "is_weekend": 0
            }]
        }
    }


class PredictionResponse(BaseModel):
    tip_amount:    float
    model_version: str
    prediction_id: str


class BatchInput(BaseModel):
    records: List[TripInput] = Field(..., max_length=100,
                                     description="List of trips (max 100)")


class BatchResponse(BaseModel):
    predictions:        List[PredictionResponse]
    count:              int
    processing_time_ms: float



def prepare_features(trip: TripInput) -> pd.DataFrame:
   
    data = {
        'passenger_count':        trip.passenger_count,
        'trip_distance':          trip.trip_distance,
        'fare_amount':            trip.fare_amount,
        'extra':                  trip.extra,
        'mta_tax':                trip.mta_tax,
        'tolls_amount':           trip.tolls_amount,
        'improvement_surcharge':  trip.improvement_surcharge,
        'congestion_surcharge':   trip.congestion_surcharge,
        'Airport_fee':            trip.Airport_fee,
        'trip_duration_minutes':  trip.trip_duration_minutes,
        'trip_speed_mph':         trip.trip_speed_mph,
        'pickup_hour':            trip.pickup_hour,
        'pickup_day_of_week':     trip.pickup_day_of_week,
        'is_weekend':             trip.is_weekend,
        'log_trip_distance':      np.log1p(trip.trip_distance),
        'fare_per_mile':          trip.fare_amount / trip.trip_distance
                                  if trip.trip_distance > 0 else 0,
        'fare_per_minute':        trip.fare_amount / trip.trip_duration_minutes
                                  if trip.trip_duration_minutes > 0 else 0,
        'pickup_boro_Bronx':          0,
        'pickup_boro_Brooklyn':        0,
        'pickup_boro_Manhattan':       1,
        'pickup_boro_Other':           0,
        'pickup_boro_Queens':          0,
        'pickup_boro_Staten Island':   0,
        'dropoff_boro_Bronx':          0,
        'dropoff_boro_Brooklyn':       0,
        'dropoff_boro_Manhattan':      1,
        'dropoff_boro_Other':          0,
        'dropoff_boro_Queens':         0,
        'dropoff_boro_Staten Island':  0,
    }

    df = pd.DataFrame([data])
    df = df[feature_columns]  

    
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_columns
    )
    return df_scaled



@app.get("/health")
def health_check():
    return {
        "status":         "healthy",
        "model_loaded":   ml_model is not None,
        "model_version":  MODEL_VERSION,
        "uptime_seconds": round(time.time() - start_time, 1)
    }


@app.get("/model/info")
def model_info():
    return {
        "model_name":    MODEL_NAME,
        "version":       MODEL_VERSION,
        "model_type":    type(ml_model).__name__,
        "feature_names": feature_columns,
        "n_features":    len(feature_columns),
        "metrics": {
            "MAE":  1.19,
            "RMSE": 2.33,
            "R2":   0.633
        },
        "trained_on": "NYC Yellow Taxi - January 2024",
        "task":       "Regression - predict tip_amount"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripInput):
    
    features   = prepare_features(trip)
    prediction = ml_model.predict(features)[0]
    prediction = max(0.0, float(prediction))

    return PredictionResponse(
        tip_amount=    round(prediction, 2),
        model_version= MODEL_VERSION,
        prediction_id= str(uuid.uuid4())
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    
    start       = time.time()
    predictions = []

    for trip in batch.records:
        features   = prepare_features(trip)
        prediction = ml_model.predict(features)[0]
        prediction = max(0.0, float(prediction))
        predictions.append(PredictionResponse(
            tip_amount=    round(prediction, 2),
            model_version= MODEL_VERSION,
            prediction_id= str(uuid.uuid4())
        ))

    return BatchResponse(
        predictions=        predictions,
        count=              len(predictions),
        processing_time_ms= round((time.time() - start) * 1000, 2)
    )



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error":  "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        }
    )