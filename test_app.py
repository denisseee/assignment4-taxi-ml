from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_TRIP = {
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
}


#TEST 1: Health check returns healthy status

def test_health_check():
    """API should report healthy with model loaded"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    assert "model_version" in data
    assert "uptime_seconds" in data
    print(" test_health_check passed!")


#TEST 2: Valid single prediction returns correct format

def test_predict_valid_input():
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    data = response.json()
    assert "tip_amount"    in data
    assert "model_version" in data
    assert "prediction_id" in data
    
    assert data["tip_amount"] >= 0
    
    assert len(data["prediction_id"]) == 36
    print(f" test_predict_valid_input passed! Predicted tip: ${data['tip_amount']}")


#TEST 3: Missing required field returns 422 error

def test_predict_missing_field():
    """Missing trip_distance should return HTTP 422"""
    incomplete_trip = VALID_TRIP.copy()
    del incomplete_trip["trip_distance"]  
    response = client.post("/predict", json=incomplete_trip)
    assert response.status_code == 422
    print(" test_predict_missing_field passed!")

#TEST 4: Invalid field type returns 422 error

def test_predict_invalid_type():
    bad_trip = VALID_TRIP.copy()
    bad_trip["fare_amount"] = "not_a_number"  
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422
    print("test_predict_invalid_type passed!")


#TEST 5: Out of range value returns 422 error

def test_predict_out_of_range():
    bad_trip = VALID_TRIP.copy()
    bad_trip["trip_distance"] = -5.0  
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422
    print(" test_predict_out_of_range passed!")


#TEST 6: Invalid pickup hour returns 422 error  

def test_predict_invalid_hour():
    bad_trip = VALID_TRIP.copy()
    bad_trip["pickup_hour"] = 25  
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422
    print(" test_predict_invalid_hour passed!")


#TEST 7: Batch prediction works correctly

def test_batch_prediction():
    batch = {"records": [VALID_TRIP] * 3}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert "processing_time_ms" in data
    
    for pred in data["predictions"]:
        assert "tip_amount"    in pred
        assert "prediction_id" in pred
    print(f" test_batch_prediction passed! Got {data['count']} predictions")


# TEST 8: Batch exceeding 100 records returns 422

def test_batch_too_large():
    
    big_batch = {"records": [VALID_TRIP] * 101}  
    response = client.post("/predict/batch", json=big_batch)
    assert response.status_code == 422
    print(" test_batch_too_large passed!")


# TEST 9: Model info endpoint returns correct structure

def test_model_info():
    
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name"    in data
    assert "version"       in data
    assert "feature_names" in data
    assert "metrics"       in data
    assert len(data["feature_names"]) == 29
    print(" test_model_info passed!")

#TEST 10: Edge case - zero distance trip

def test_edge_case_zero_distance():
    
    zero_trip = VALID_TRIP.copy()
    zero_trip["trip_distance"] = 0.0  
    response = client.post("/predict", json=zero_trip)
    assert response.status_code == 422
    print("test_edge_case_zero_distance passed!")


#TEST 11: Edge case - very high fare amount

def test_edge_case_high_fare():
    
    expensive_trip = VALID_TRIP.copy()
    expensive_trip["fare_amount"] = 490.0  
    response = client.post("/predict", json=expensive_trip)
    assert response.status_code == 200
    data = response.json()
    assert data["tip_amount"] >= 0
    print(f" test_edge_case_high_fare passed! Tip: ${data['tip_amount']}")