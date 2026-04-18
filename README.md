# Assignment 4: MLOps & Model Deployment
## NYC Taxi Tip Prediction Service

**Course:** COMP 3610 - Big Data Analytics   

## Project Structure
assignment4/
├── assignment4.ipynb   
├── app.py              
├── test_app.py         
├── Dockerfile          
├── docker-compose.yml  
├── requirements.txt    
├── README.md           
├── .gitignore          
├── .dockerignore       
└── models/            

## Quick Start

### Option 1: Docker Compose (Recommended)

# Start API + MLflow services
docker compose up --build

# API available at:  http://localhost:8000
# MLflow UI at:      http://localhost:5001
# API docs at:       http://localhost:8000/docs

# Stop everything
docker compose down

### Option 2: Run Locally

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app:app --reload --port 8000


pytest test_app.py -v

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | API health check |
| GET | /model/info | Model metadata |
| POST | /predict | Single trip prediction |
| POST | /predict/batch | Batch predictions (max 100) |
| GET | /docs | Swagger UI documentation |

## Example Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

## Docker Image Size
- Image size: 1.22GB
- Base image: python:3.11-slim

## Model Information
- Model: Random Forest Regressor
- Task: Predict taxi tip amount (regression)
- Features: 29 engineered features
- Performance: MAE=$1.19, RMSE=$2.33, R²=0.633
- Training data: NYC Yellow Taxi January 2024